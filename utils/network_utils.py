import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tinycudann as tcnn

class Dir_Encoding(torch.nn.Module):
    """
    Modified based on the implementation of Gaussian Fourier feature mapping.

    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

    """

    def __init__(self, input_c, output_c, scale=25):
        super().__init__()

        self._B = nn.Parameter(torch.randn(
            (input_c, output_c)) * scale)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        assert x.dim() == 2, 'Expected 2D input (got {}D input)'.format(x.dim())
        x = x @ self._B.to(x.device)
        x = (self.sigmoid(x) * 2 - 1) * torch.pi
        return torch.sin(x)
    


def get_encoder(encoding, input_dim=3,
                degree=4, n_bins=16, n_frequencies=12,
                n_levels=16, level_dim=2, 
                base_resolution=16, log2_hashmap_size=19, 
                desired_resolution=512):
    
    # Dense grid encoding
    if 'dense' in encoding.lower():
        n_levels = 4
        per_level_scale = np.exp2(np.log2(desired_resolution  / base_resolution) / (n_levels - 1))
        embed = tcnn.Encoding(
            n_input_dims=input_dim,
            encoding_config={
                    "otype": "Grid",
                    "type": "Dense",
                    "n_levels": n_levels,
                    "n_features_per_level": level_dim,
                    "base_resolution": base_resolution,
                    "per_level_scale": per_level_scale,
                    "interpolation": "Linear"},
                dtype=torch.float
        )
        out_dim = embed.n_output_dims
    
    # Sparse grid encoding
    elif 'hash' in encoding.lower() or 'tiled' in encoding.lower():
        print('Hash size', log2_hashmap_size)
        per_level_scale = np.exp2(np.log2(desired_resolution  / base_resolution) / (n_levels - 1))
        embed = tcnn.Encoding(
            n_input_dims=input_dim,
            encoding_config={
                "otype": 'HashGrid',
                "n_levels": n_levels,
                "n_features_per_level": level_dim,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": per_level_scale
            },
            dtype=torch.float
        )
        out_dim = embed.n_output_dims

    # Spherical harmonics encoding
    elif 'spherical' in encoding.lower():
        embed = tcnn.Encoding(
                n_input_dims=input_dim,
                encoding_config={
                "otype": "SphericalHarmonics",
                "degree": degree,
                },
                dtype=torch.float
            )
        out_dim = embed.n_output_dims
    
    # OneBlob encoding
    elif 'blob' in encoding.lower():
        print('Use blob')
        embed = tcnn.Encoding(
                n_input_dims=input_dim,
                encoding_config={
                "otype": "OneBlob", #Component type.
	            "n_bins": n_bins
                },
                dtype=torch.float
            )
        out_dim = embed.n_output_dims
    
    # Frequency encoding
    elif 'freq' in encoding.lower():
        print('Use frequency')
        embed = tcnn.Encoding(
                n_input_dims=input_dim,
                encoding_config={
                "otype": "Frequency", 
                "n_frequencies": n_frequencies
                },
                dtype=torch.float
            )
        out_dim = embed.n_output_dims
    
    # Identity encoding
    elif 'identity' in encoding.lower():
        embed = tcnn.Encoding(
                n_input_dims=input_dim,
                encoding_config={
                "otype": "Identity"
                },
                dtype=torch.float
            )
        out_dim = embed.n_output_dims

    return embed, out_dim

      
class Pos_Encoding(nn.Module):
    def __init__(self, cfg, bound):
        super().__init__()
        # Coordinate encoding
        self.pe_fn, self.pe_dim = get_encoder(cfg['pos']['method'], 
                                              n_bins=cfg['pos']['n_bins'])

        # Sparse parametric grid encoding
        dim_max = (bound[:,1] - bound[:,0]).max()
        self.resolution = int(dim_max / cfg['grid']['voxel_size'])
        self.grid_fn, self.grid_dim = get_encoder(cfg['grid']['method'], 
                                                  log2_hashmap_size=cfg['grid']['hash_size'], 
                                                  desired_resolution=self.resolution)
        print('Grid size:', self.grid_dim)
    
    def forward(self, pts):
        pe = self.pe_fn(pts)
        grid = self.grid_fn(pts)
        return pe, grid
    
    
class SDF(nn.Module):
    def __init__(self, pts_dim, hidden_dim=32, feature_dim=32):
        super().__init__()

        self.decoder = tcnn.Network(n_input_dims=pts_dim+feature_dim,
                                    n_output_dims=1,
                                    network_config={
                                        "otype": "FullyFusedMLP",
                                        "activation": "ReLU",
                                        "output_activation": "None",
                                        "n_neurons": hidden_dim,
                                        "n_hidden_layers": 1})
        
    def forward(self, x, f):
        return self.decoder(torch.cat((x, f), -1))
    

class SH(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=32):
        super().__init__()

        self.decoder = tcnn.Network(n_input_dims=in_dim,
                                          n_output_dims=out_dim,
                                          network_config={
                                            "otype": "FullyFusedMLP",
                                            "activation": "ReLU",
                                            "output_activation": "None",
                                            "n_neurons": hidden_dim,
                                            "n_hidden_layers": 1})       
        
    def forward(self, x, d, f):
        out = self.decoder(torch.cat((x, d, f), -1))
        square_sum = torch.sum(out**2, dim=-1)
        out = out / torch.sqrt(square_sum)
        return out


class Density(nn.Module):
    def __init__(self, params_init={}):
        super().__init__()
        for p in params_init:
            param = nn.Parameter(torch.tensor(params_init[p]))
            setattr(self, p, param)

    def forward(self, sdf, beta=None):
        return self.density_func(sdf, beta=beta)


class LaplaceDensity(Density):  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
    def __init__(self, params_init={}, beta_min=0.0001):
        super().__init__(params_init=params_init)
        self.beta_min = torch.tensor(beta_min).cuda()

    def density_func(self, sdf, beta=None):
        if beta is None:
            beta = self.get_beta()

        alpha = 1 / beta
        return alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))

    def get_beta(self):
        beta = self.beta.abs() + self.beta_min
        return beta
    

class BellDensity(Density):  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
    def __init__(self, params_init={}, beta_min=0.0001):
        super().__init__(params_init=params_init)
        self.beta_min = torch.tensor(beta_min).cuda()

    def density_func(self, sdf, beta=None):
        if beta is None:
            beta = self.get_beta()

        return beta * torch.exp(-beta * sdf) / (1 + torch.exp(-beta * sdf)) ** 2

    def get_beta(self):
        beta = self.beta + self.beta_min
        return beta


class AbsDensity(Density):  # like NeRF++
    def density_func(self, sdf, beta=None):
        return torch.abs(sdf)


class SimpleDensity(Density):  # like NeRF
    def __init__(self, params_init={}, noise_std=1.0):
        super().__init__(params_init=params_init)
        self.noise_std = noise_std

    def density_func(self, sdf, beta=None):
        if self.training and self.noise_std > 0.0:
            noise = torch.randn(sdf.shape).cuda() * self.noise_std
            sdf = sdf + noise
        return torch.relu(sdf)

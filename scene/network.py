import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.network_utils import SDF, LaplaceDensity, BellDensity, Pos_Encoding, Dir_Encoding, SH
import os
import numpy as np
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_expon_lr_func, get_linear_noise_func, coordinates

class Model(nn.Module):
    def __init__(self, cfg, bounding_box, sh_out_dim):
        super().__init__()     
        self.pos_embed = Pos_Encoding(cfg, bounding_box)
        self.pe_dim = self.pos_embed.pe_dim
        self.grid_dim = self.pos_embed.grid_dim

        self.query_sdf = SDF(pts_dim=self.pe_dim, hidden_dim=cfg['hidden_dim'], feature_dim=self.grid_dim)

        self.sdf2opacity = BellDensity(**cfg['density'])

        self.dir_embed = Dir_Encoding(3, 32)
        self.query_sh = SH(32 + self.pe_dim + self.grid_dim, sh_out_dim * 3)
    

class SpecModel:
    def __init__(self, cfg, xyz, ref_sh_degree):
        self.cfg = cfg
        self.set_bbox(xyz, enlarge=1.02)

        self.specular = Model(cfg, self.bounding_box, (ref_sh_degree + 1) ** 2).cuda()
        self.optimizer = None


    def set_bbox(self, point_cloud, enlarge=1.0):
        '''
        Get bounding box of current scene, enlarge a little bit.
        '''
        with torch.no_grad():
            min_values, _ = torch.min(point_cloud, dim=0)
            max_values, _ = torch.max(point_cloud, dim=0)

            min_point = min_values * enlarge
            max_point = max_values * enlarge

            self.bounding_box = torch.stack([min_point, max_point], dim=-1)


    def smoothness(self, sample_points=32, voxel_size=0.1, margin=0.05):
        '''
        Smoothness loss of feature grid
        '''
        volume = self.bounding_box[:, 1] - self.bounding_box[:, 0]

        grid_size = (sample_points-1) * voxel_size
        offset_max = self.bounding_box[:, 1]-self.bounding_box[:, 0] - grid_size - 2 * margin

        offset = torch.rand(3).to(offset_max) * offset_max + margin
        coords = coordinates(sample_points - 1, 'cpu', flatten=False).float().to(volume)
        pts = (coords + torch.rand((1,1,1,3)).to(volume)) * voxel_size + self.bounding_box[:, 0] + offset

        pts_tcnn = (pts - self.bounding_box[:, 0]) / (self.bounding_box[:, 1] - self.bounding_box[:, 0])
        
        pe, grid_pts = self.specular.pos_embed(pts_tcnn.reshape(-1, pts_tcnn.shape[-1]))
        sdf = self.specular.query_sdf(pe, grid_pts)[..., :1].reshape(pts.shape[:-1], -1)
        tv_x = torch.pow(sdf[1:,...]-sdf[:-1,...], 2).sum()
        tv_y = torch.pow(sdf[:,1:,...]-sdf[:,:-1,...], 2).sum()
        tv_z = torch.pow(sdf[:,:,1:,...]-sdf[:,:,:-1,...], 2).sum()

        loss = (tv_x + tv_y + tv_z)/ (sample_points**3)

        return loss
    
    def query_alpha_sdf(self, pts):
        pts_tcnn = (pts - self.bounding_box[:, 0]) / (self.bounding_box[:, 1] - self.bounding_box[:, 0])
        pe, grid_pts = self.specular.pos_embed(pts_tcnn)
        sdf = self.specular.query_sdf(pe, grid_pts)
        alpha = self.specular.sdf2opacity(sdf)
        return alpha, sdf
    
    def query_sdf(self, pts):
        pts_tcnn = (pts - self.bounding_box[:, 0]) / (self.bounding_box[:, 1] - self.bounding_box[:, 0])
        pe, grid_pts = self.specular.pos_embed(pts_tcnn)
        sdf = self.specular.query_sdf(pe, grid_pts)
        return sdf
        
    def query_sh(self, pts, normals):
        pts_tcnn = (pts - self.bounding_box[:, 0]) / (self.bounding_box[:, 1] - self.bounding_box[:, 0])
        pe, grid_pts = self.specular.pos_embed(pts_tcnn)
        de = self.specular.dir_embed(normals)
        sh = self.specular.query_sh(pe, de, grid_pts)
        return sh
    
    def query_sdf_gradient_sh(self, pts, normals):
        pts_tcnn = (pts - self.bounding_box[:, 0]) / (self.bounding_box[:, 1] - self.bounding_box[:, 0])
        pe, grid_pts = self.specular.pos_embed(pts_tcnn)
        sdf = self.specular.query_sdf(pe, grid_pts)
        gradient = self.gradient(pts)
        alpha = self.specular.sdf2opacity(sdf)

        de = self.specular.dir_embed(normals)
        sh = self.specular.query_sh(pe, de, grid_pts)
        return alpha, sdf, gradient, sh

    def gradient(self, x):
        x = torch.reshape(x, [-1, x.shape[-1]]).float()
        normal_eps = self.cfg['grid']['voxel_size']
        eps = normal_eps / np.sqrt(3)
        k1 = torch.tensor([1, -1, -1], dtype=x.dtype, device=x.device)  
        k2 = torch.tensor([-1, -1, 1], dtype=x.dtype, device=x.device)  
        k3 = torch.tensor([-1, 1, -1], dtype=x.dtype, device=x.device)  
        k4 = torch.tensor([1, 1, 1], dtype=x.dtype, device=x.device)  
        
        sdf1 = self.query_sdf(x + k1 * eps)
        sdf2 = self.query_sdf(x + k2 * eps)
        sdf3 = self.query_sdf(x + k3 * eps)
        sdf4 = self.query_sdf(x + k4 * eps)
        gradients = (k1*sdf1 + k2*sdf2 + k3*sdf3 + k4*sdf4) / (4.0 * eps)

        # rotation_matrix = torch.tensor([[-1, 0, 0],
        #                                 [0, 1, 0],
        #                                 [0, 0, -1]], dtype=x.dtype, device=x.device)
        # gradients = torch.matmul(rotation_matrix, gradients.T).T

        return gradients

    def train_setting(self, training_args):
        l = [
            {'params': list(self.specular.parameters()),
             'lr': training_args.network_feature_lr / 10,
             "name": "specular"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.specular_scheduler_args = get_linear_noise_func(lr_init=training_args.network_feature_lr,
                                                             lr_final=training_args.network_feature_lr / 20,
                                                             lr_delay_mult=training_args.position_lr_delay_mult,
                                                             max_steps=training_args.specular_lr_max_steps)

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "specular/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.specular.state_dict(), os.path.join(out_weights_path, 'specular.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "specular"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "specular/iteration_{}/specular.pth".format(loaded_iter))
        self.specular.load_state_dict(torch.load(weights_path))

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "specular":
                lr = self.specular_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr



#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def tv_loss(map):
    horizontal_diff = torch.abs(map[:, :-1] - map[:, 1:])
    vertical_diff = torch.abs(map[:-1, :] - map[1:, :])
    return horizontal_diff.mean() + vertical_diff.mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    
def sdf2weights(sdf, z_vals, trunc=0.1, sc_factor=1):
    '''
    Convert signed distance function to weights.

    Params:
        sdf: [N_rays, N_samples]
        z_vals: [N_rays, N_samples]
    Returns:
        weights: [N_rays, N_samples]
    '''
    weights = torch.sigmoid(sdf / trunc) * torch.sigmoid(-sdf / trunc)

    signs = sdf[:, 1:] * sdf[:, :-1]
    mask = torch.where(signs < 0.0, torch.ones_like(signs), torch.zeros_like(signs))
    inds = torch.argmax(mask, axis=1)
    inds = inds[..., None]
    z_min = torch.gather(z_vals, 1, inds) # The first surface
    mask = torch.where(z_vals < z_min + sc_factor * trunc, torch.ones_like(z_vals), torch.zeros_like(z_vals))

    weights = weights * mask
    return weights / (torch.sum(weights, axis=-1, keepdims=True) + 1e-8)


def get_masks(z_vals, target_d, truncation):
    '''
    Params:
        z_vals: torch.Tensor, (Bs, N_samples)
        target_d: torch.Tensor, (Bs,)
        truncation: float
    Return:
        front_mask: torch.Tensor, (Bs, N_samples)
        sdf_mask: torch.Tensor, (Bs, N_samples)
        fs_weight: float
        sdf_weight: float
    '''

    # before truncation
    front_mask = torch.where(z_vals < (target_d - truncation), torch.ones_like(z_vals), torch.zeros_like(z_vals))
    # after truncation
    back_mask = torch.where(z_vals > (target_d + truncation), torch.ones_like(z_vals), torch.zeros_like(z_vals))
    # valid mask
    depth_mask = torch.where(target_d > 0.0, torch.ones_like(target_d), torch.zeros_like(target_d))
    # Valid sdf regionn
    sdf_mask = (1.0 - front_mask) * (1.0 - back_mask) * depth_mask

    return front_mask, sdf_mask

def compute_loss(prediction, target, loss_type='l2'):
    '''
    Params: 
        prediction: torch.Tensor, (Bs, N_samples)
        target: torch.Tensor, (Bs, N_samples)
        loss_type: str
    Return:
        loss: torch.Tensor, (1,)
    '''

    if loss_type == 'l2':
        return F.mse_loss(prediction, target)
    elif loss_type == 'l1':
        return F.l1_loss(prediction, target)

    raise Exception('Unsupported loss type')
    
def normal_laplacian_loss(predicted_normals):
    predicted_normals = predicted_normals.unsqueeze(0)
    # Define Laplacian operator
    laplacian_operator = torch.tensor([[0, 1, 0],
                                       [1, -4, 1],
                                       [0, 1, 0]], dtype=torch.float32, device=predicted_normals.device)
    laplacian_operator = laplacian_operator.view(1, 1, 3, 3)  # Reshape for convolution

    # Apply Laplacian operator
    laplacian_normals = F.conv2d(predicted_normals, laplacian_operator.repeat(1,3,1,1), padding=1)

    # Compute L1 or L2 loss between original normals and Laplacian-processed normals
    # For example, using L2 loss
    loss = torch.mean(torch.pow(predicted_normals - laplacian_normals, 2))

    return loss

def get_loss(render_pkg, opt, mlp_warm_up):
    '''
    Params:
        z_vals: torch.Tensor, (Bs, N_samples)
        target_d: torch.Tensor, (Bs,)
        predicted_sdf: torch.Tensor, (Bs, N_samples)
        truncation: float
    Return:
        fs_loss: torch.Tensor, (1,)
        sdf_loss: torch.Tensor, (1,)
        eikonal_loss: torch.Tensor, (1,)
    '''

    min_scales = render_pkg["min_scales"]
    depth_map, render_normal_map = render_pkg["depth_map"], render_pkg["render_normal_map"]
    gaussian_sdf, disk_sdf = render_pkg["gaussian_sdf"], render_pkg["disk_sdf"]
    gaussian_normal, sdf_gradient, dirs = render_pkg["gaussian_normal"], render_pkg["sdf_gradient"], render_pkg["dir_pp_normalized"]
    gaussian_sdf2normal = sdf_gradient / torch.norm(sdf_gradient, dim=-1)[:, None]
    
    volume_fs_loss, volume_sdf_loss, volume_depth_loss = render_pkg["volume_fs_loss"], render_pkg["volume_sdf_loss"], render_pkg["volume_depth_loss"]
    
    # gaussian normal loss
    normal_loss = (torch.abs(gaussian_normal - gaussian_sdf2normal)).mean() + (torch.abs((1 - torch.sum(gaussian_normal*gaussian_sdf2normal, dim=-1)))).mean()
    # eikonal loss
    eikonal_loss = ((sdf_gradient.norm(2, dim=-1) - 1) ** 2).sum()
    # gaussian sdf loss
    # sdf_loss = torch.abs(gaussian_sdf).mean() + torch.abs(disk_sdf).mean() + 10*(disk_sdf).var()
    sdf_loss = torch.var(disk_sdf, dim=-1).sum() + torch.abs(gaussian_sdf).mean()

    # depth smooth loss
    depth_smooth_loss = tv_loss(depth_map) + tv_loss(render_normal_map.permute(1,2,0))

    # minimum scale loss
    min_scale_loss = torch.relu(min_scales).mean()

    if mlp_warm_up:
        gaussian_opacity = render_pkg["gaussian_opacity"]
        mlp_opacity = render_pkg["mlp_opacity"]

        opacity_loss = F.l1_loss(gaussian_opacity, mlp_opacity)
        loss = opt.lambda_fs * volume_fs_loss + opt.lambda_vloume_sdf * volume_sdf_loss + opt.lambda_volume_depth * volume_depth_loss +\
            opt.lambda_eik_loss * eikonal_loss + opt.lambda_sdf * sdf_loss +\
            opt.lambda_depth_smooth * depth_smooth_loss + opt.lambda_scale * min_scale_loss + opt.lambda_opacity * opacity_loss
    else:      
        loss = 0.1*opt.lambda_fs * volume_fs_loss + 0.1*opt.lambda_vloume_sdf * volume_sdf_loss + 0.1*opt.lambda_volume_depth * volume_depth_loss +\
            opt.lambda_gaussian_normal * normal_loss + opt.lambda_eik_loss * eikonal_loss + opt.lambda_sdf * sdf_loss +\
            opt.lambda_depth_smooth * depth_smooth_loss + opt.lambda_scale * min_scale_loss

    return loss


def get_loss_v2(render_pkg, depth_map, render_normal_map, opt):
    gaussian_sdf, disk_sdf = render_pkg["gaussian_sdf"], render_pkg["disk_sdf"]
    gaussian_normal, sdf_gradient = render_pkg["gaussian_normal"], render_pkg["sdf_gradient"]
    gaussian_sdf2normal = sdf_gradient / (torch.norm(sdf_gradient, dim=-1)+1e-9)[:, None]

    # gaussian normal loss
    normal_loss = (torch.abs(gaussian_normal - gaussian_sdf2normal)).mean() + (torch.abs((1 - torch.sum(gaussian_normal*gaussian_sdf2normal, dim=-1)))).mean()
    # eikonal loss
    eikonal_loss = ((sdf_gradient.norm(2, dim=-1) - 1) ** 2).sum()
    # gaussian sdf loss
    # sdf_loss = torch.var(disk_sdf, dim=-1).sum() + torch.abs(gaussian_sdf).mean()
    sdf_loss = torch.abs(gaussian_sdf).mean() + 0.5 * torch.abs(gaussian_sdf[:, None, :] - disk_sdf).sum()

    # depth smooth loss
    depth_smooth_loss = tv_loss(depth_map) + tv_loss(render_normal_map.permute(1,2,0))
   
    loss = opt.lambda_gaussian_normal * normal_loss + \
           opt.lambda_eik_loss * eikonal_loss + \
           opt.lambda_sdf * sdf_loss +\
           opt.lambda_depth_smooth * depth_smooth_loss

    return loss




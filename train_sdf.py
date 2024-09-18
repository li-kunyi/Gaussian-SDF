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

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from datetime import datetime
import numpy as np
import open3d as o3d
import cv2
import torch
import torchvision
import math
import random
from random import randint
import matplotlib.pyplot as plt
from PIL import Image
from utils.loss_utils import l1_loss, ssim, get_loss_v2, gradient_consistency_loss, smoothness
from gaussian_renderer import sdf_render_v2, sdf_render_v3, network_gui, get_sdf_loss_with_gaussian_depth, gradient
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
# import marching_cubes as mcubes
from skimage.measure import marching_cubes
from scipy.interpolate import griddata
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter, distance_transform_edt
import trimesh
from plyfile import PlyData, PlyElement
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import time
from utils.vis_utils import apply_depth_colormap, save_points, colormap
from utils.depth_utils import depths_to_points, depth_to_normal

TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

@torch.no_grad()
def create_offset_gt(image, offset):
    height, width = image.shape[1:]
    meshgrid = np.meshgrid(range(width), range(height), indexing='xy')
    id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
    id_coords = torch.from_numpy(id_coords).cuda()
    
    id_coords = id_coords.permute(1, 2, 0) + offset
    id_coords[..., 0] /= (width - 1)
    id_coords[..., 1] /= (height - 1)
    id_coords = id_coords * 2 - 1
    
    image = torch.nn.functional.grid_sample(image[None], id_coords[None], align_corners=True, padding_mode="border")[0]
    return image
    
def get_edge_aware_distortion_map(gt_image, distortion_map):
    grad_img_left = torch.mean(torch.abs(gt_image[:, 1:-1, 1:-1] - gt_image[:, 1:-1, :-2]), 0)
    grad_img_right = torch.mean(torch.abs(gt_image[:, 1:-1, 1:-1] - gt_image[:, 1:-1, 2:]), 0)
    grad_img_top = torch.mean(torch.abs(gt_image[:, 1:-1, 1:-1] - gt_image[:, :-2, 1:-1]), 0)
    grad_img_bottom = torch.mean(torch.abs(gt_image[:, 1:-1, 1:-1] - gt_image[:, 2:, 1:-1]), 0)
    max_grad = torch.max(torch.stack([grad_img_left, grad_img_right, grad_img_top, grad_img_bottom], dim=-1), dim=-1)[0]
    # pad
    max_grad = torch.exp(-max_grad)
    max_grad = torch.nn.functional.pad(max_grad, (1, 1, 1, 1), mode="constant", value=0)
    return distortion_map * max_grad
    

def L1_loss_appearance(image, gt_image, gaussians, view_idx, return_transformed_image=False):
    appearance_embedding = gaussians.get_apperance_embedding(view_idx)
    # center crop the image
    origH, origW = image.shape[1:]
    H = origH // 32 * 32
    W = origW // 32 * 32
    left = origW // 2 - W // 2
    top = origH // 2 - H // 2
    crop_image = image[:, top:top+H, left:left+W]
    crop_gt_image = gt_image[:, top:top+H, left:left+W]
    
    # down sample the image
    crop_image_down = torch.nn.functional.interpolate(crop_image[None], size=(H//32, W//32), mode="bilinear", align_corners=True)[0]
    
    crop_image_down = torch.cat([crop_image_down, appearance_embedding[None].repeat(H//32, W//32, 1).permute(2, 0, 1)], dim=0)[None]
    mapping_image = gaussians.appearance_network(crop_image_down)
    transformed_image = mapping_image * crop_image
    if not return_transformed_image:
        return l1_loss(transformed_image, crop_gt_image)
    else:
        transformed_image = torch.nn.functional.interpolate(transformed_image, size=(origH, origW), mode="bilinear", align_corners=True)[0]
        return transformed_image
    
def training(dataset, opt, pipe, testing_iterations, saving_iterations, save_ckpt, ckpt_pth, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.network)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if ckpt_pth:
        (model_params, first_iter) = torch.load(ckpt_pth)
        gaussians.restore(model_params, opt)
        gaussians.load_model(f"{scene.model_path}/point_cloud/iteration_{first_iter}/model.pt")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    trainCameras = scene.getTrainCameras().copy()
    testCameras = scene.getTestCameras().copy()
    allCameras = trainCameras + testCameras
    for idx, camera in enumerate(scene.getTrainCameras() + scene.getTestCameras()):
        camera.idx = idx

    # highresolution index
    highresolution_index = []
    for index, camera in enumerate(trainCameras):
        if camera.image_width >= 800:
            highresolution_index.append(index)

    gaussians.compute_3D_filter(cameras=trainCameras)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), initial=first_iter, total=opt.iterations, desc="Training progress")
    first_iter += 1
    align = {}
    densify_grad_scheduler = get_expon_lr_func(lr_init=0.0002,
                                    lr_final=0.0002,
                                    max_steps=15_000)
    for iteration in range(first_iter, opt.iterations + 1):        
        torch.cuda.empty_cache()
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        # Pick a random high resolution camera
        if random.random() < 0.3 and dataset.sample_more_highres:
            viewpoint_cam = trainCameras[highresolution_index[randint(0, len(highresolution_index)-1)]]

        iter_start.record()

        gaussians.optimizer.zero_grad(set_to_none = True)
        gaussians.network_optimizer.zero_grad()
        gaussians.s2o_optimizer.zero_grad()

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        render_pkg = sdf_render_v3(viewpoint_cam, gaussians, pipe, background, kernel_size=dataset.kernel_size)
        rendering, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        image = rendering[:3, :, :]
        
        # rgb Loss
        gt_image = viewpoint_cam.original_image.cuda()
        
        Ll1 = l1_loss(image, gt_image)
        # use L1 loss for the transformed image if using decoupled appearance
        if dataset.use_decoupled_appearance:
            Ll1 = L1_loss_appearance(image, gt_image, gaussians, viewpoint_cam.idx)
        
        rgb_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        # depth distortion regularization
        distortion_map = rendering[8, :, :]
        # distortion_map = get_edge_aware_distortion_map(gt_image, distortion_map)
        distortion_loss = distortion_map.mean()
        
        # depth normal consistency
        depth = rendering[6, :, :]
        depth_normal, _, depth_grad_mag, dx, dy = depth_to_normal(viewpoint_cam, depth[None, ...], return_depth_grad=True)
        depth_normal = depth_normal.permute(2, 0, 1)

        render_normal = rendering[3:6, :, :]
        render_normal = torch.nn.functional.normalize(render_normal, p=2, dim=0)
        
        c2w = (viewpoint_cam.world_view_transform.T).inverse()
        normal2 = c2w[:3, :3] @ render_normal.reshape(3, -1)
        render_normal_world = normal2.reshape(3, *render_normal.shape[1:])
        
        normal_error = torch.abs(1 - (render_normal_world * depth_normal).sum(dim=0)) 
        depth_normal_loss = normal_error.mean()

        render_alpha = rendering[7, :, :]
        
        lambda_distortion = opt.lambda_distortion if iteration >= opt.distortion_from_iter else 0.0
        lambda_depth_normal = opt.lambda_depth_normal if iteration >= opt.depth_normal_from_iter else 0.0

        loss = rgb_loss + depth_normal_loss * lambda_depth_normal + distortion_loss * lambda_distortion 

        # consistency loss
        consistency_loss, img_grad_weight = gradient_consistency_loss(depth_grad_mag, dx, dy, gt_image, image, normal=None, return_weight=True)
        
        lambda_gradient_consistency = opt.lambda_gradient_consistency if iteration >= opt.densify_until_iter else 0.0
        loss += lambda_gradient_consistency * consistency_loss

        # SDF training
        if iteration > opt.start_train_sdf:
            H, W = depth.shape
            fx = W / (2 * math.tan(viewpoint_cam.FoVx / 2))
            fy = H / (2 * math.tan(viewpoint_cam.FoVy / 2))
            c2w = (viewpoint_cam.world_view_transform.T).inverse()

            n_pixel = 1024 if iteration < opt.densify_until_iter else opt.n_pixel
            n_sample = 0 if iteration < opt.densify_until_iter else opt.n_sample
            n_sample_surface = opt.n_sample_surface
            
            fs_loss, sdf_loss = get_sdf_loss_with_gaussian_depth(gaussians, c2w, fx, fy, 
                                                                depth.clone().detach(),
                                                                n_pixel=n_pixel, n_sample=n_sample, n_sample_surface=n_sample_surface, 
                                                                truncation=opt.truncation,
                                                                full_image=False, ray_sampling=True)
            lambda_sdf = 1000.0 if iteration < opt.densify_until_iter else opt.lambda_sdf
            lambda_fs = 0. if iteration < opt.densify_until_iter else opt.lambda_fs
            loss += lambda_sdf * sdf_loss + lambda_fs * fs_loss

            # if opt.lambda_smooth > 0:
            #     loss += opt.lambda_smooth * smoothness(gaussians, sample_points=opt.smooth_sample_point, voxel_size=opt.smooth_voxel_size)

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)

            if iteration == opt.iterations:
                progress_bar.close()

            # plot
            is_save_images = True
            if is_save_images and (iteration % 100 == 0):            
                depth_map = apply_depth_colormap(depth[..., None], None, near_plane=None, far_plane=None)
                depth_map = depth_map.permute(2, 0, 1)
                
                accumlated_alpha = rendering[7, :, :, None]
                colored_accum_alpha = apply_depth_colormap(accumlated_alpha, None, near_plane=0.0, far_plane=1.0)
                colored_accum_alpha = colored_accum_alpha.permute(2, 0, 1)
                
                distortion_map = rendering[8, :, :]
                distortion_map = colormap(distortion_map.detach().cpu().numpy()).to(render_normal_world.device)

                _depth_normal = (depth_normal + 1.) / 2.
                _render_normal_world = (render_normal_world + 1) / 2
            
                row0 = torch.cat([gt_image, image, _depth_normal, _render_normal_world], dim=2)
                row1 = torch.cat([depth_map, colored_accum_alpha, distortion_map, torch.abs(gt_image - image)*10], dim=2)
                
                image_to_show = torch.cat([row0, row1], dim=1)
                image_to_show = torch.clamp(image_to_show, 0, 1)
                
                os.makedirs(f"{dataset.model_path}/{TIMESTAMP}log_images", exist_ok = True)
                torchvision.utils.save_image(image_to_show, f"{dataset.model_path}/{TIMESTAMP}log_images/{iteration}.jpg")
               
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 and visibility_filter.shape[0] < 3e6:
                    size_threshold = None
                    densify_grad_threshold = densify_grad_scheduler(iteration)

                    gaussians.densify_and_prune(densify_grad_threshold, 0.05, scene.cameras_extent, size_threshold, use_sdf_normal=False)
                    gaussians.compute_3D_filter(cameras=trainCameras)
                    
            if iteration % 100 == 0 and iteration > opt.densify_until_iter:
                if iteration < opt.iterations - 100:
                    # don't update in the end of training
                    gaussians.compute_3D_filter(cameras=trainCameras)

        # Optimizer step
        if iteration < opt.iterations:
            gaussians.optimizer.step()
            gaussians.network_optimizer.step()

            if iteration > opt.start_train_sdf:
                gaussians.s2o_optimizer.step()

        # Log and save
        with torch.no_grad():
            if (iteration in saving_iterations) and save_ckpt:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)  # save gaussians
                gaussians.save_model(f"{scene.model_path}/point_cloud/iteration_{iteration}/model.pt")  # save network

                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                if not os.path.exists(f"{scene.model_path}/ckpt"):
                    os.makedirs(f"{scene.model_path}/ckpt")
                torch.save((gaussians.capture(), iteration), f"{scene.model_path}/ckpt/ckpt_{iteration}.pth")  # save gaussian model
            
            if (iteration + 1) % 10 == 0:
                training_report(tb_writer, iteration+1, rgb_loss, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, sdf_render_v3, (pipe, background, dataset.kernel_size))
        
        torch.cuda.empty_cache()

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_dir = args.model_path +'/' + TIMESTAMP
        os.makedirs(tb_dir, exist_ok = True)
        tb_writer = SummaryWriter(tb_dir)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, rgb_loss, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/rgb_loss', rgb_loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        tb_writer.add_scalar('sdf2opacity_beta', scene.gaussians.sdf2opacity.get_beta().item(), iteration)
        tb_writer.add_scalar('sdf2opacity_alpha', scene.gaussians.sdf2opacity.get_alpha().item(), iteration)

    # Report test and samples of training set
    # if iteration in testing_iterations:
    if iteration % 2000 == 0:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    rendering = renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"]
                    image = rendering[:3, :, :]
                    normal = rendering[3:6, :, :]
                    image = torch.clamp(image, 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    # if tb_writer and (idx < 5):
                    #     tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                    #     if iteration == testing_iterations[0]:
                    #         tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        # if tb_writer:
        #     gaussian_sdf = scene.gaussians.query_sdf(scene.gaussians.get_xyz)
        #     opacity = scene.gaussians.opacity_activation(gaussian_sdf)
        #     tb_writer.add_histogram("scene/sdf_histogram", gaussian_sdf.detach(), global_step=iteration)
        #     tb_writer.add_histogram("scene/opacity_histogram", opacity.detach(), global_step=iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[2_000, 4_000, 6_000, 8_000, 10_000, 15_000, 20_000, 25_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[5_000, 10_000, 20_000, 25_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--save_ckpt", action='store_true', default=False)  # save checkpoint 
    parser.add_argument("--ckpt_pth", type=str, default = None)  # load checkpoint

    args = parser.parse_args(sys.argv[1:])

    # args.source_path = '/home/kunyi/work/data/NeRF/lego'
    # args.model_path = 'outputs/blender/lego'
    
    # args.source_path = '/home/kunyi/work/data/tnt/TrainingSet/Barn'
    # args.model_path = 'outputs/tnt/Barn'
    # args.source_path = '/home/kunyi/work/data/360_v2/garden'
    # args.model_path = 'outputs/360/garden'
    # args.resolution = 2
    # args.eval = True
    # args.save_ckpt = True
    # args.ckpt_pth = 'outputs/360/garden/ckpt/ckpt_1.pth'

    print("Optimizing " + args.model_path)
    if args.save_ckpt:
        print("Save CKPT")

    # Initialize system state (RNG)
    # safe_state(args.quiet)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
    
    # # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.save_ckpt, args.ckpt_pth, args.debug_from)

    # All done
    print("\nTraining complete.")

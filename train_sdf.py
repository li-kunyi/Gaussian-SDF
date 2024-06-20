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
import numpy as np
import open3d as o3d
import cv2
import torch
import torchvision
import random
from random import randint
from utils.loss_utils import l1_loss, ssim, get_loss
from gaussian_renderer import sdf_render, network_gui
import sys
from scene import Scene, GaussianModel, SpecModel
from utils.general_utils import safe_state
import uuid
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
    
    
def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, dataset.ref_sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    specular = SpecModel(opt.network, gaussians._xyz, dataset.ref_sh_degree)
    specular.train_setting(opt)

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
    # highresolution_index = []
    # for index, camera in enumerate(trainCameras):
    #     if camera.image_width >= 800:
    #         highresolution_index.append(index)

    gaussians.compute_3D_filter(cameras=trainCameras)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        # if iteration % 1000 == 0:
        #     gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        
        # # Pick a random high resolution camera
        # if random.random() < 0.3 and dataset.sample_more_highres:
        #     viewpoint_cam = trainCameras[highresolution_index[randint(0, len(highresolution_index)-1)]]
            
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        if iteration >= 5000:
            mlp = specular
            opt_opacity = False
            if iteration == 5000:
                gaussians.training_reset(opt)
            mlp_warm_up = False
        elif iteration < 5000 and iteration > 500:
            mlp = specular
            opt_opacity = True
            mlp_warm_up = True
        else:
            mlp = None
            opt_opacity = True
            mlp_warm_up = True
        
        render_pkg = sdf_render(viewpoint_cam, gaussians, pipe, background, kernel_size=dataset.kernel_size, 
                                mlp=mlp, mlp_warm_up=mlp_warm_up)
        rendering, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        image = rendering[:3, :, :]
        
        # rgb Loss
        gt_image = viewpoint_cam.original_image.cuda()
        
        Ll1 = l1_loss(image, gt_image)
        
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        # depth distortion regularization
        distortion_map = rendering[8, :, :]
        distortion_map = get_edge_aware_distortion_map(gt_image, distortion_map)
        distortion_loss = distortion_map.mean()
        
        # depth normal consistency
        depth = rendering[6, :, :]
        depth_normal, _ = depth_to_normal(viewpoint_cam, depth[None, ...])
        depth_normal = depth_normal.permute(2, 0, 1)

        render_normal = rendering[3:6, :, :]
        render_normal = torch.nn.functional.normalize(render_normal, p=2, dim=0)
        
        c2w = (viewpoint_cam.world_view_transform.T).inverse()
        normal2 = c2w[:3, :3] @ render_normal.reshape(3, -1)
        render_normal_world = normal2.reshape(3, *render_normal.shape[1:])
        
        normal_error = 1 - (render_normal_world * depth_normal).sum(dim=0)
        depth_normal_loss = normal_error.mean()

        distortion_weight = opt.lambda_distortion if iteration >= opt.distortion_from_iter else 0.0
        depth_normal_weight = opt.lambda_depth_normal if iteration >= opt.depth_normal_from_iter else 0.0
        
        # Final loss
        loss += depth_normal_weight * depth_normal_loss + distortion_weight * distortion_loss
        
        if not mlp == None:  
            if iteration % 100 == 0:
                mlp_smooth_loss = specular.smoothness()
            else:
                mlp_smooth_loss = torch.tensor(0).to(gt_image.device)

            loss += opt.lambda_mlp_smooth * mlp_smooth_loss + get_loss(render_pkg, opt, mlp_warm_up)

            sdf_gradient = render_pkg["sdf_gradient"]
            gaussian_sdf = render_pkg["gaussian_sdf"]

            # print(f"min gaussian {gaussian_opacity.min()}, max gaussian {gaussian_opacity.max()}")
        
        loss.backward()
        
        iter_end.record()

        is_save_images = True
        if is_save_images and (iteration % opt.densification_interval == 0):
            with torch.no_grad():
                eval_cam = allCameras[random.randint(0, len(allCameras) -1)]
                
                render_pkg = sdf_render(eval_cam, gaussians, pipe, background, kernel_size=dataset.kernel_size, 
                                        mlp=mlp, full_image=True, mlp_warm_up=mlp_warm_up)
                rendering = render_pkg["render"]
                image = rendering[:3, :, :]
                # transformed_image = image

                normal = rendering[3:6, :, :]
                normal = torch.nn.functional.normalize(normal, p=2, dim=0)

                volume_depth_map = render_pkg["volume_depth_map"]
                gaussian_density_map = render_pkg["gaussian_density_map"]
    
            # transform to world space
            c2w = (eval_cam.world_view_transform.T).inverse()
            normal2 = c2w[:3, :3] @ normal.reshape(3, -1)
            normal = normal2.reshape(3, *normal.shape[1:])
            normal = (normal + 1.) / 2.
            
            depth = rendering[6, :, :]
            depth_normal, _ = depth_to_normal(eval_cam, depth[None, ...])
            depth_normal = (depth_normal + 1.) / 2.
            depth_normal = depth_normal.permute(2, 0, 1)
            
            gt_image = eval_cam.original_image.cuda()

            depth_diff = (depth - volume_depth_map).abs()
            depth_diff = apply_depth_colormap(depth_diff[..., None], None, near_plane=0.2, far_plane=6)
            depth_diff = depth_diff.permute(2, 0, 1)

            depth_map = apply_depth_colormap(depth[..., None], None, near_plane=0.2, far_plane=6)
            depth_map = depth_map.permute(2, 0, 1)

            volume_depth_map = apply_depth_colormap(volume_depth_map[..., None], None, near_plane=0.2, far_plane=6)
            volume_depth_map = volume_depth_map.permute(2, 0, 1)
            
            accumlated_alpha = rendering[7, :, :, None]
            colored_accum_alpha = apply_depth_colormap(accumlated_alpha, None, near_plane=0.0, far_plane=1.0)
            colored_accum_alpha = colored_accum_alpha.permute(2, 0, 1)

            gaussian_density_map = apply_depth_colormap(gaussian_density_map[..., None], None, near_plane=0.0, far_plane=1.0)
            gaussian_density_map = gaussian_density_map.permute(2, 0, 1)
            
            distortion_map = rendering[8, :, :]
            distortion_map = colormap(distortion_map.detach().cpu().numpy()).to(normal.device)
        
            row0 = torch.cat([gt_image, image, depth_normal, normal], dim=2)
            row1 = torch.cat([gaussian_density_map, depth_map, volume_depth_map, depth_diff], dim=2)
            
            image_to_show = torch.cat([row0, row1], dim=1)
            image_to_show = torch.clamp(image_to_show, 0, 1)
            
            os.makedirs(f"{dataset.model_path}/log_images", exist_ok = True)
            torchvision.utils.save_image(image_to_show, f"{dataset.model_path}/log_images/{iteration}.jpg")
            
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, sdf_render, (pipe, background, dataset.kernel_size, mlp))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    sdf_th = max((0.2 - (iteration/20000)), 0.1)
                    if mlp_warm_up == False and not mlp == None:
                        sdf_gradient_value = torch.norm(sdf_gradient, dim=-1)
                        sdf_value = gaussian_sdf.squeeze(-1)
                    else:
                        sdf_gradient_value = None
                        sdf_value = None
                    
                    sdf_gradient_value = None
                    sdf_value = None

                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.05, scene.cameras_extent, size_threshold, sdf_th=sdf_th, sdf_value=sdf_value, sdf_gradient=sdf_gradient_value, opt_opacity=opt_opacity)
                    # gaussians.compute_3D_filter(cameras=trainCameras)

                if mlp == None and (iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter)):
                    gaussians.reset_opacity()

            # if iteration % 100 == 0 and iteration > opt.densify_until_iter:
            #     if iteration < opt.iterations - 100:
            #         # don't update in the end of training
            #         gaussians.compute_3D_filter(cameras=trainCameras)
        
            # Optimizer step
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none = True)

            specular.optimizer.step()
            specular.update_learning_rate(iteration)
            specular.optimizer.zero_grad()

            if not mlp == None and iteration % 1000 == 0:
                with torch.no_grad():
                    print(f"Num of gaussians: {gaussians.get_xyz.shape[0]}")
                    mlp.extract_mesh(mesh_savepath=f"{dataset.model_path}/sdf_mesh_{iteration}.ply")

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            
    
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
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
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
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)

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
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    # args.source_path = '/home/kunyi/work/data/NeRF/lego'
    # args.model_path = 'outputs/blender/lego'
    args.source_path = '/home/kunyi/work/data/TNT_GOF/TrainingSet/Barn'
    args.model_path = 'outputs/tnt/barn'
    args.eval = True
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    # safe_state(args.quiet)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
    
    # # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")

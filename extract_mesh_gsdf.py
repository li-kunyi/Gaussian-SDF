import torch
import torch.nn.functional as F
from scene import Scene
import math
import os
from os import makedirs
from gaussian_renderer import render, integrate, integrate_sdf, sdf_render_v3
import random
from tqdm import tqdm
from argparse import ArgumentParser
from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args
# from gaussian_renderer import GaussianModel
from scene.sdf_gaussian_model_v3 import GaussianModel
import numpy as np
import trimesh
from tetranerf.utils.extension import cpp
from utils.tetmesh import marching_tetrahedra
from skimage.measure import marching_cubes

@torch.no_grad()
# def filter_points_in_bounding_box(points, bounding_box):
#     bbox_min = bounding_box[:, 0]
#     bbox_max = bounding_box[:, 1]
    
#     mask = (points >= bbox_min) & (points <= bbox_max)
#     mask = mask.all(dim=1)
    
#     filtered_points = points[mask]
    
#     return filtered_points, mask

@torch.no_grad()
def filter_points_in_bounding_box(pts, bbx=None):
    min_values, _ = torch.min(pts, dim=0)
    max_values, _ = torch.max(pts, dim=0)
    bounding_box = torch.stack([min_values, max_values], dim=-1)

    for i in range(3):
        if i == 1:
            continue
        selected_pts = pts[:, i]
        selected_pts_max = selected_pts[selected_pts > 0]

        if selected_pts_max.numel() > 0:
            # max_point = torch.quantile(selected_pts_max, 0.6, dim=0)
            max_point = selected_pts_max.mean() * 1.2
        else:
            max_point = selected_pts.max()
        selected_pts_min = selected_pts[selected_pts < 0]
        if selected_pts_min.numel() > 0:
            # min_point = -torch.quantile(-selected_pts_min, 0.6, dim=0)
            min_point = selected_pts_min.mean() * 1.2
        else:
            min_point = selected_pts.min()
        bounding_box[i, 0] = min_point
        bounding_box[i, 1] = max_point

    bbox_min = bounding_box[:, 0]  # Minimum boundary in each dimension
    bbox_max = bounding_box[:, 1]  # Maximum boundary in each dimension
    
    mask = (pts >= bbox_min) & (pts <= bbox_max)
    mask = mask.all(dim=1)
    
    filtered_points = pts[mask]
    
    return filtered_points, mask

def depth_test(points, views, gaussians, pipeline, background, kernel_size):
    # Initialize a mask tensor for outlier filtering
    mask = torch.zeros((points.shape[0])).bool().to(points.device)
    
    with torch.no_grad():
        for _, view in enumerate(tqdm(views, desc="Depth Test")):
            # Render depth information for the current view
            rendering = sdf_render_v3(view, gaussians, pipeline, background, kernel_size=kernel_size)["render"]
            depth = rendering[6, :, :]
            
            # Compute intrinsic parameters
            H, W = depth.shape
            fx = W / (2 * math.tan(view.FoVx / 2))
            fy = H / (2 * math.tan(view.FoVy / 2))
            K = torch.tensor([[fx, 0, W / 2],
                              [0, fy, H / 2],
                              [0, 0, 1]], dtype=torch.float32, device="cuda")

            # Transform points to the camera coordinate system
            w2c = (view.world_view_transform.T).float().to("cuda")
            ones = torch.ones((points.shape[0], 1), dtype=torch.float32, device="cuda")
            homo_points = torch.cat([points, ones], dim=1).reshape(-1, 4, 1)
            cam_cord_homo = w2c @ homo_points
            cam_cord = cam_cord_homo[:, :3]

            # Convert to pixel coordinates
            uv = K @ cam_cord.float()
            z = uv[:, -1:] + 1e-8  # Avoid division by zero
            uv = uv[:, :2] / z

            # Check bounds within the image frame
            edge = 0
            in_bounds = (uv[:, 0] < W - edge) & (uv[:, 0] > edge) & (uv[:, 1] < H - edge) & (uv[:, 1] > edge)
            in_front = (z[:, :, 0] > 0)
            mask_in_view = in_bounds & in_front
            mask_in_view = mask_in_view.squeeze()

            # Normalize the grid coordinates for grid sampling
            vgrid = uv.reshape(1, 1, -1, 2)
            vgrid[..., 0] = (vgrid[..., 0] / (W - 1) * 2.0 - 1.0)
            vgrid[..., 1] = (vgrid[..., 1] / (H - 1) * 2.0 - 1.0)

            # Sample depth at the projected points
            depth_sample = F.grid_sample(depth.reshape(1, 1, H, W), vgrid, padding_mode='zeros', align_corners=True)
            depth_sample = depth_sample.reshape(-1)

            # Define depth thresholds and mask points
            proj_depth = cam_cord[mask_in_view, 2].reshape(-1)

            # Mark points as valid based on depth threshold criteria
            valid_depth = (depth_sample[mask_in_view] - 2.5 < proj_depth) & (proj_depth < depth_sample[mask_in_view] + 0.02)
            mask_in_view = mask_in_view.reshape(-1)
            mask_in_view[mask_in_view.clone()] &= valid_depth
            mask |= mask_in_view

    return mask


@torch.no_grad()
def marching_tetrahedra_with_binary_search(model_path, name, iteration, cams, gaussians, pipeline, background, kernel_size):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "fusion")

    makedirs(render_path, exist_ok=True)
    
    # generate tetra points here
    points, points_scale = gaussians.get_tetra_points()
    points, mask = filter_points_in_bounding_box(points, gaussians.bounding_box)
    points_scale = points_scale[mask]

    # mask = depth_test(points, cams, gaussians, pipeline, background, kernel_size)
    # points = points[mask]
    # points_scale = points_scale[mask]

    # points_np = points.cpu().numpy()
    # filename = render_path + '/point_cloud.ply'
    # with open(filename, 'w') as file:
    #     file.write('ply\n')
    #     file.write('format ascii 1.0\n')
    #     file.write(f'element vertex {points_np.shape[0]}\n')
    #     file.write('property float x\n')
    #     file.write('property float y\n')
    #     file.write('property float z\n')
    #     file.write('end_header\n')

    #     for point in points_np:
    #         file.write(f'{point[0]} {point[1]} {point[2]}\n')

    # load cell if exists
    # if os.path.exists(os.path.join(render_path, "cells.pt")):
    if False:
        print("load existing cells")
        cells = torch.load(os.path.join(render_path, "cells.pt"))
    else:
        # create cell and save cells
        print("create cells and save")
        cells = cpp.triangulate(points)
        # we should filter the cell if it is larger than the gaussians
        torch.save(cells, os.path.join(render_path, "cells.pt"))

    sdf = gaussians.query_sdf(points.cuda())[None]

    vertices = points.cuda()[None]
    tets = cells.cuda().long()
    print(vertices.shape, tets.shape)

    torch.cuda.empty_cache()
    verts_list, scale_list, faces_list, _ = marching_tetrahedra(vertices, tets, sdf, points_scale[None])
    torch.cuda.empty_cache()
    
    end_points, end_sdf = verts_list[0]
    end_scales = scale_list[0]
    
    faces=faces_list[0].cpu().numpy()
    points = (end_points[:, 0, :] + end_points[:, 1, :]) / 2.
        
    left_points = end_points[:, 0, :]
    right_points = end_points[:, 1, :]
    left_sdf = end_sdf[:, 0, :]
    right_sdf = end_sdf[:, 1, :]
    left_scale = end_scales[:, 0, 0]
    right_scale = end_scales[:, 1, 0]
    distance = torch.norm(left_points - right_points, dim=-1)
    scale = left_scale + right_scale
    
    n_binary_steps = 8
    for step in range(n_binary_steps):
        print("binary search in step {}".format(step))
        mid_points = (left_points + right_points) / 2
        mid_sdf = gaussians.query_sdf(mid_points.cuda()).squeeze().unsqueeze(-1)

        ind_low = ((mid_sdf < 0) & (left_sdf < 0)) | ((mid_sdf > 0) & (left_sdf > 0))

        left_sdf[ind_low] = mid_sdf[ind_low]
        right_sdf[~ind_low] = mid_sdf[~ind_low]
        left_points[ind_low.flatten()] = mid_points[ind_low.flatten()]
        right_points[~ind_low.flatten()] = mid_points[~ind_low.flatten()]
    
        points = (left_points + right_points) / 2
        if step not in [7]:
            continue

        mesh = trimesh.Trimesh(vertices=points.cpu().numpy(), faces=faces, process=False)
        
        # filter
        # mask = (distance <= scale).cpu().numpy()
        if cams is not None:
            mask = depth_test(points, cams, gaussians, pipeline, background, kernel_size).cpu().numpy()
            face_mask = mask[faces].all(axis=1)
            mesh.update_vertices(mask)
            mesh.update_faces(face_mask)
        # mesh.fill_holes()
        mesh.export(os.path.join(render_path, f"mesh_binary_search_{step}.ply"))

    # linear interpolation
    # right_sdf *= -1
    # points = (left_points * left_sdf + right_points * right_sdf) / (left_sdf + right_sdf)
    # mesh = trimesh.Trimesh(vertices=points.cpu().numpy(), faces=faces)
    # mesh.export(os.path.join(render_path, f"mesh_binary_search_interp.ply"))
    

def extract_mesh(dataset : ModelParams, opt, iteration : int, pipeline : PipelineParams):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, opt.network)
        gaussians.load_ply(os.path.join(dataset.model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply"))
        gaussians.load_model(os.path.join(dataset.model_path, "point_cloud", f"iteration_{iteration}", "model.pt"))
        
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        kernel_size = dataset.kernel_size

        if opt.depth_test:
            scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
            cams = scene.getTrainCameras()
        else:
            cams = None
        marching_tetrahedra_with_binary_search(dataset.model_path, "test", iteration, cams, gaussians, pipeline, background, kernel_size)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--depth_test", action="store_true", default=False)
    args = get_combined_args(parser)

    print("Rendering " + args.model_path)
    op.depth_test = args.depth_test

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
    
    extract_mesh(model.extract(args), op.extract(args), args.iteration, pipeline.extract(args))
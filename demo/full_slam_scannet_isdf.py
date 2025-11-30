import argparse
import os
from os.path import join
import time
import torch
import numpy as np
import open3d as o3d
from torch.utils.data import DataLoader
from copy import deepcopy
import json

from grid_opt.configs import load_config
from grid_opt.models.isdf.isdf import iSDF
from grid_opt.loss_isdf import iSDFLoss
from grid_opt.utils.utils_sdf import save_mesh
import grid_opt.utils.utils_scannet as utils_scannet
import grid_opt.utils.utils_eval as utils_eval
import grid_opt.utils.utils_geometry as utils_geometry
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./configs/rgbd/scannet.yaml')
parser.add_argument('--default_config', type=str, default='./configs/base.yaml')
parser.add_argument('--save_dir', type=str, default='./results/demo/isdf_mapping')
parser.add_argument('--scannet_root', type=str, default='data/ScanNet/scans')
parser.add_argument('--scene', type=str, default='0000_00')
parser.add_argument('--epochs', type=int, default=100, help="Number of iterations per frame (default: 100 for iSDF)")
parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")

def run_isdf_mapping(args):
    # 1. Load Configuration
    cfg = load_config(args.config, args.default_config)
    scene = utils_scannet.scannet_scenes()[args.scene]
    
    # Configure for iSDF (Single MLP, Global Map)
    cfg['dataset']['path'] = scene.path
    cfg['dataset']['intrinsics_file'] = scene.intrinsics_file
    cfg['model']['pose']['num_poses'] = scene.num_kfs
    cfg['model']['pose']['optimize'] = False  # GT Pose mode (Table 1)
    
    device = cfg['device']
    log_dir = join(args.save_dir, scene.name)
    os.makedirs(log_dir, exist_ok=True)

    # 2. Initialize Model
    logger.info("Initializing iSDF model...")
    model = iSDF(
        cfg['model'], 
        device=device, 
        hidden_size=256, 
        hidden_layers_block=2 
    ).to(device)

    # 3. Initialize Dataset
    # Using specific params for iSDF similar to paper (e.g. n_rays)
    dataset = utils_scannet.create_scannet_dataset(
        args.scannet_root, 
        scene.name,
        n_rays=75,
        n_strat_samples=12,
        n_surf_samples=12,
        frame_downsample=5  # Consistent with MISO experiment
    )
    
    # Initialize all keyframe poses in the model with GT
    for kf_id in range(dataset.num_kfs):
        R, t = dataset.true_kf_pose_in_world(kf_id)
        model.set_initial_kf_pose(kf_id, R, t)

    # 4. Initialize Loss and Optimizer
    loss_fn = iSDFLoss(
        model_name="isdf",
        trunc_weight=5.0,
        trunc_distance=0.1,  # typical iSDF trunc dist
        eik_weight=0.1,
        loss_type="L1",
        slam_mode=True
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 5. Incremental Mapping Loop
    logger.info(f"Starting Incremental Mapping for {dataset.num_kfs} frames...")
    active_kfs = []
    total_time = 0
    
    for kf_id in range(dataset.num_kfs):
        active_kfs.append(kf_id)
        
        # Select active keyframes for this step (replay buffer style or all)
        # For strict iSDF reproduction, we often sample from all past frames
        # or a sliding window. Here we sample from all active KFs.
        dataset.select_keyframes(active_kfs)
        train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
        
        t_start = time.perf_counter()
        
        # Optimization Loop for current frame addition
        for i in range(args.epochs):
            try:
                model_input, gt = next(iter(train_loader))
            except StopIteration:
                break
                
            # Move to device
            for k, v in model_input.items():
                if isinstance(v, torch.Tensor): model_input[k] = v.to(device)
            for k, v in gt.items():
                if isinstance(v, torch.Tensor): gt[k] = v.to(device)

            optimizer.zero_grad()
            
            # Compute Loss
            loss_dict = loss_fn.compute(model, model_input, gt)
            loss = loss_dict['sdf']
            if 'eikonal_loss' in loss_dict:
                loss += loss_dict['eikonal_loss']
            
            loss.backward()
            optimizer.step()
            
        torch.cuda.synchronize()
        step_time = time.perf_counter() - t_start
        total_time += step_time
        
        if kf_id % 10 == 0:
            logger.info(f"Frame {kf_id}/{dataset.num_kfs} done. Time: {step_time:.2f}s. Loss: {loss.item():.4f}")

    # 6. Save Results
    logger.info(f"Mapping finished. Total time: {total_time:.2f}s")
    
    # Save Model
    torch.save(model.state_dict(), join(log_dir, 'isdf_model.pth'))
    
    # Extract and Save Mesh
    logger.info("Extracting mesh...")
    mesh_path = join(log_dir, 'mesh.ply')
    # iSDF is global, so we use the scene bound defined in metadata
    save_mesh(model, torch.tensor(scene.bound).to(device), mesh_path, resolution=256)
    
    # 7. Evaluate
    logger.info("Evaluating...")
    mesh_est = o3d.io.read_triangle_mesh(mesh_path)
    mesh_gt = o3d.io.read_triangle_mesh(scene.gt_mesh)
    
    # Align and Compute Metrics
    mesh_est_aligned = utils_scannet.align_mesh_to_ref(mesh_est, mesh_gt)
    o3d.io.write_triangle_mesh(join(log_dir, 'mesh_aligned.ply'), mesh_est_aligned)
    
    verts_pred = utils_eval.sample_points_from_mesh(join(log_dir, 'mesh_aligned.ply'), 200000)
    verts_trgt = utils_eval.sample_points_from_mesh(scene.gt_mesh, 200000)
    
    # Filter by OBB
    obb = mesh_gt.get_minimal_oriented_bounding_box()
    verts_pred = utils_eval.filter_points_by_oriented_bound(verts_pred, obb)
    verts_trgt = utils_eval.filter_points_by_oriented_bound(verts_trgt, obb)
    
    metrics = utils_eval.compute_chamfer_metrics(verts_pred, verts_trgt)
    print("Evaluation Results:")
    print(metrics)

    json_path = join(log_dir, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Metrics saved to {json_path}")

if __name__ == "__main__":
    args = parser.parse_args()
    run_isdf_mapping(args)
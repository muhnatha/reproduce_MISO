import argparse
from os.path import join
import torch
import numpy as np
import json
import open3d as o3d
from grid_opt.configs import *
from grid_opt.datasets.sdf_3d_lidar import PosedSdf3DLidar
from grid_opt.utils.utils_sdf import *
import grid_opt.utils.utils as utils
from grid_opt.slam.system import System
from grid_opt.slam.fuser import Fuser
import grid_opt.utils.utils_eval as utils_eval
import grid_opt.utils.utils_geometry as utils_geometry
from evo.core import metrics as evo_metrics
import logging

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--default_config', type=str, default='./configs/base.yaml')
parser.add_argument('--config', type=str, default='./configs/lidar/ncd_quad.yaml')
parser.add_argument('--save_dir', type=str, default='./results/demo/table4')
parser.add_argument('--method', type=str, required=True, 
                    choices=['icp_mapping', 'miso_odom', 'miso_full'],
                    help='Which method to run from Table IV')
parser.add_argument('--pose_source', type=str, default='reg_icp', 
                    choices=['reg_icp', 'kiss_icp'],
                    help='Source of poses (reg_icp = Point-to-Point, kiss_icp = KISS-ICP)')


def create_ncd_dataset(cfg, voxel_size, near_surf_std, n_near, n_free, n_behind, frame_samples, frame_batchsize):
    dataset = PosedSdf3DLidar(
        lidar_folder=cfg['dataset']['path'],
        pose_file_gt=cfg['dataset']['pose_gt'],
        pose_file_init=cfg['dataset']['pose_init'],
        num_frames=cfg['dataset']['num_frames'],
        trunc_dist=cfg['dataset']['trunc_dist'],
        frame_samples=frame_samples,
        frame_batchsize=frame_batchsize,
        voxel_size=voxel_size,
        near_surface_std=near_surf_std,
        near_surface_n=n_near,
        free_space_n=n_free,
        behind_surface_n=n_behind,
        min_dist_ratio=0.50,
        min_z=-10.0,
        max_z=60.0,
        min_range=1.5, 
        max_range=60.0,
        adaptive_range=False
    )
    return dataset

def evaluate_metrics(grid_atlas, dataset, method_name, log_dir, mesh_path):
    """Calculates Trajectory metrics only (Skipping Mesh to save RAM)"""
    
    # --- 1. Trajectory Evaluation ---
    print(f"Evaluating trajectory for {method_name}...")
    R_est, t_est, R_gt, t_gt = [], [], [], []
    
    for i in range(dataset.num_kfs):
        R_g, t_g = dataset.true_kf_pose_in_world(i)
        
        if hasattr(grid_atlas, 'updated_kf_pose_in_world'):
             R_e, t_e = grid_atlas.updated_kf_pose_in_world(i)
        else:
             submap_id = grid_atlas.submap_id_for_kf(i)
             R_s_w, t_s_w = grid_atlas.updated_submap_pose(submap_id)
             R_f_s, t_f_s = grid_atlas.get_camera_pose(i)
             T_s_w = utils_geometry.pose_matrix(R_s_w, t_s_w)
             T_f_s = utils_geometry.pose_matrix(R_f_s, t_f_s)
             T_f_w = T_s_w @ T_f_s
             R_e, t_e = T_f_w[:3, :3], T_f_w[:3, 3:]

        R_est.append(R_e.cpu())
        t_est.append(t_e.cpu())
        R_gt.append(R_g.cpu())
        t_gt.append(t_g.cpu())

    metrics_t = utils_eval.evo_trajectory_error(
        torch.stack(R_gt), torch.stack(t_gt), 
        torch.stack(R_est), torch.stack(t_est), 
        align=True, pose_relation=evo_metrics.PoseRelation.translation_part
    ).get_all_statistics()
    
    metrics_R = utils_eval.evo_trajectory_error(
        torch.stack(R_gt), torch.stack(t_gt), 
        torch.stack(R_est), torch.stack(t_est), 
        align=True, pose_relation=evo_metrics.PoseRelation.rotation_part
    ).get_all_statistics()
    
    rmse_tran = metrics_t['rmse']
    rmse_rot = utils_geometry.chordal_to_degree(metrics_R['rmse'])

    # --- 2. Mesh Evaluation (SKIPPED) ---
    print("Skipping Mesh Evaluation (Chamfer/F-score) to prevent OOM...")
    c_l1, f_score = 0.0, 0.0

    # --- 3. Save Results ---
    results = {
        "method": method_name,
        "metrics": {
            "Translation RMSE (m)": rmse_tran,
            "Rotation RMSE (deg)": rmse_rot,
            "Chamfer-L1 (cm)": c_l1,
            "F-score (%)": f_score
        }
    }
    
    json_path = join(log_dir, "metrics.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)

    print("\n" + "="*40)
    print(f"RESULTS: {method_name}")
    print(f"Tran RMSE:   {rmse_tran:.4f} m")
    print(f"Rot RMSE:    {rmse_rot:.4f} deg")
    print(f"Chamfer-L1:  SKIPPED")
    print(f"F-score:     SKIPPED")
    print(f"Saved to:    {json_path}")
    print("="*40 + "\n")

def main():
    args = parser.parse_args()
    cfg = load_config(args.config, args.default_config)
    
    # 1. Experiment Setup
    run_name = f"{args.method}_{args.pose_source}"
    log_dir = join(args.save_dir, run_name)
    
    utils.cond_mkdir(log_dir)
    print(f"Output directory created: {log_dir}")
    
    cfg['system']['log_dir'] = log_dir
    cfg['system']['submap_size'] = 400  # Matches paper
    cfg['system']['submap_local_bound'] = [[-60, 60], [-60, 60], [-5, 15]]
    
    # Select Pose Source
    if args.pose_source == 'reg_icp':
        cfg['dataset']['pose_init'] = join(cfg['dataset']['root'], 'poses_reg_icp.txt')
    elif args.pose_source == 'kiss_icp':
        cfg['dataset']['pose_init'] = join(cfg['dataset']['root'], 'poses_kiss_icp.txt')
    
    # 2. Configure Method Behavior
    if args.method == 'icp_mapping':
        print(">>> MODE: ICP + MISO Mapping (Tracking Disabled)")
        cfg['tracking']['disable'] = True
        cfg['model']['pose']['optimize'] = False
        
    elif args.method in ['miso_odom', 'miso_full']:
        print(f">>> MODE: {args.method} (Tracking Enabled)")
        cfg['tracking']['disable'] = False
        cfg['tracking']['verbose'] = False
        
    # 3. Initialize Grid Atlas
    grid_atlas = GridAtlas(cfg['model'], device=cfg['device'], dtype=torch.float32) 
    grid_atlas.to(cfg['device'])
    
    # 4. Create Datasets
    # Reduced sampling for memory safety
    dataset_track = create_ncd_dataset(cfg, 0.6, 0, 0, 0, 0, 2**17, 2**12)
    dataset_map = create_ncd_dataset(cfg, 0.08, 0.25, 4, 2, 1, 2**12, 1024)

    # 5. Run System (Tracking + Mapping loop)
    system = System(
        model=grid_atlas,
        dataset_track=dataset_track,
        dataset_map=dataset_map,
        cfg=cfg,
        verbose=True
    )
    system.run()
    
    torch.save(grid_atlas, join(log_dir, 'after_odometry.pth'))
    
    # 6. Optional: Global Alignment (Only for miso_full)
    if args.method == 'miso_full':
        print(">>> Running Global Alignment...")
        # CRITICAL: Reset frame selection so alignment sees all submaps
        dataset_map.unselect_keyframes()
        
        cfg['align']['verbose'] = True
        cfg['align']['latent_levels'] = [1] 
        cfg['align']['skip_finetune'] = False
        
        # --- FIX: Subsample points to prevent GPU OOM ---
        cfg['align']['subsample_points'] = 2048 
        
        fuser = Fuser(model=grid_atlas, dataset=dataset_map, cfg=cfg)
        fuser.align()
        torch.save(grid_atlas, join(log_dir, 'after_alignment.pth'))

    # 7. Save Mesh
    mesh_path = join(log_dir, 'final_mesh.ply')
    print(f"Saving mesh to {mesh_path}...")
    save_mesh(grid_atlas, grid_atlas.global_bound(), mesh_path, resolution=256)

    # 8. Evaluate Metrics (Trajectory Only)
    evaluate_metrics(grid_atlas, dataset_map, args.method, log_dir, mesh_path)

if __name__ == "__main__":
    main()
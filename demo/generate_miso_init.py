import argparse
from os.path import join
from copy import deepcopy
import torch
import logging
import os
# Import existing modules
from grid_opt.configs import load_config
from grid_opt.utils import utils_scannet, utils_geometry
from grid_opt.models.grid_atlas import GridAtlas
from grid_opt.slam.system import System
from grid_opt.slam.mapper import Mapper

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./configs/rgbd/scannet.yaml')
parser.add_argument('--default_config', type=str, default='./configs/base.yaml')
parser.add_argument('--save_dir', type=str, default='./results/demo/slam')
parser.add_argument('--scannet_root', type=str, default='data/ScanNet/scans')
parser.add_argument('--scene', type=str, default='0000_00')

# --- 1. THE MONKEY PATCH ---
# We capture the original mapping function
original_mapping = Mapper.mapping

# We define a new function that ignores the requested iterations and uses 0 instead
def mapping_zero_iter(self, mapping_kfs, iterations=10, level_iterations=5):
    print(f"[MISO Init Mode] Forcing 0 iterations for frames {mapping_kfs} (Skipping Optimization)")
    # Call the original function with 0 iterations
    return original_mapping(self, mapping_kfs, iterations=0, level_iterations=0)

# Apply the patch: Replace the class method with our new version
Mapper.mapping = mapping_zero_iter
# ---------------------------

def run_miso_init(args):
    # Load Config
    cfg = load_config(args.config, args.default_config)
    scene = utils_scannet.scannet_scenes()[args.scene]
    
    # Define Output Directory
    log_dir = join(args.save_dir, scene.name, 'miso_init')
    os.makedirs(log_dir, exist_ok=True)
    
    # Update Config
    cfg['dataset']['path'] = scene.path
    cfg['dataset']['intrinsics_file'] = scene.intrinsics_file
    cfg['model']['grid']['bound'] = scene.bound
    cfg['model']['pose']['num_poses'] = scene.num_kfs
    cfg['model']['pose']['optimize'] = True
    cfg['dataset']['anchor_kfs'] = scene.anchor_kfs
    cfg['train']['log_dir'] = log_dir
    
    # Initialize Model
    logger.info("Initializing GridAtlas...")
    grid_atlas = GridAtlas(cfg['model'], device=cfg['device'], dtype=torch.float32) 
    grid_atlas.to(cfg['device'])
    
    # Setup Datasets
    # Using 'noisy' setting (incremental slam) context
    dataset_track = utils_scannet.create_scannet_dataset(
        args.scannet_root, scene.name, n_rays=5000, n_strat_samples=0, n_surf_samples=1,
        voxel_size=0.05, frame_downsample=5
    )
    dataset_map = utils_scannet.create_scannet_dataset(
        args.scannet_root, scene.name, n_rays=5000, n_strat_samples=3, n_surf_samples=4,
        voxel_size=0.01, frame_downsample=5
    )
    
    # Configure System
    cfg_s = deepcopy(cfg)
    cfg_s['system']['submap_size'] = scene.num_kfs
    cfg_s['system']['log_dir'] = log_dir
    cfg_s['tracking']['disable'] = False  # Keep tracking enabled (or disable if you want pure init)
    cfg_s['mapping']['disable'] = False   # Must be False so it enters the mapping function we patched
    cfg_s['visualizer']['enable'] = False

    # Initialize System
    # Note: If poses_color_icp.txt is missing, ensure your noise injection patch is applied 
    # in grid_opt/datasets/sdf_rgbd.py, otherwise this defaults to GT init.
    R0, t0 = dataset_track.true_kf_pose_in_world(0)
    
    system = System(
        model=grid_atlas,
        dataset_track=dataset_track,
        dataset_map=dataset_map,
        cfg=cfg_s,
        R_world_origin=R0, 
        t_world_origin=t0,
        verbose=True
    )
    
    # Run the System (This will use our 0-iteration patch)
    logger.info("Running MISO Init-Only Loop...")
    system.run()

    # Save Result
    model_path = join(log_dir, 'result.pth')
    torch.save(grid_atlas, model_path)
    logger.info(f"Unoptimized MISO model saved to: {model_path}")

if __name__ == "__main__":
    args = parser.parse_args()
    run_miso_init(args)
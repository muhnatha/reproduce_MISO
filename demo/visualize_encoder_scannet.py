import argparse
from os.path import join
import torch
import numpy as np
from copy import deepcopy
import os

# Import MISO modules
from grid_opt.models.encoder import Encoder, EncoderPretrainLoss
from grid_opt.datasets.sdf_3d import BatchPosedSdf3D
from grid_opt.configs import load_config
from grid_opt.models.grid_net import GridNet
from grid_opt.trainer import Trainer
import grid_opt.utils.utils_scannet as utils_scannet
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./configs/rgbd/scannet.yaml')
parser.add_argument('--default_config', type=str, default='./configs/base.yaml')
parser.add_argument('--save_dir', type=str, default='./results/demo/encoder_vis')
parser.add_argument('--epochs', type=int, default=100, help="Training epochs (use more for better quality)")
parser.add_argument('--scene', type=str, default='0000_00')
parser.add_argument('--decoder_weights', type=str, default='./results/trained_decoders/decoder_indoor.pt')

def train_encoder_level(encoder, train_loader, cfg, target_level, save_dir):
    logger.info(f"--- Training Encoder Level {target_level} ---")
    
    # 1. Setup Loss
    loss = EncoderPretrainLoss(
        target_level=target_level,
        sdf_weight=3000.0,
        sign_weight=100.0, 
        eik_weight=50.0,
        smooth_weight=10.0, 
        smooth_std=0.01,
        trunc_dist=0.15,
        pred_std=1e-3
    )
    
    # 2. Unlock only the current level
    encoder.lock_all_params()
    encoder.unlock_encoder_at_level(level=target_level)
    
    # 3. Configure Trainer
    train_cfg = deepcopy(cfg['train'])
    train_cfg['learning_rate'] = 1e-3
    train_cfg['epochs'] = args.epochs
    train_cfg['log_dir'] = save_dir
    train_cfg['verbose'] = True
    
    trainer = Trainer(
        train_cfg,
        encoder,
        loss,
        train_loader,
        None,
        cfg['device'],
        torch.float32
    )
    
    # 4. Train
    trainer.train()
    
    # 5. Visualize
    # This generates the images for Figure 5
    logger.info(f"Generating visualizations for Level {target_level}...")
    
    # Disable noise for visualization
    loss.pred_std = 0
    # Run one eval pass to populate intermediate results
    trainer.run_eval(epoch=args.epochs)
    
    vis_dir = join(save_dir, f"level{target_level}_vis")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Save visualizations for the first (and only) model
    encoder.save_visualizations(model_id=0, save_dir=vis_dir, resolution=128)
    logger.info(f"Saved visualizations to {vis_dir}")

def main(args):
    # 1. Configuration
    cfg = load_config(args.config, args.default_config)
    cfg['model']['decoder']['pretrained_model'] = args.decoder_weights
    # Ensure grid config matches what encoder expects
    cfg['model']['grid']['n_levels'] = 2
    cfg['model']['grid']['feature_dim'] = 4
    
    device = cfg['device']
    os.makedirs(args.save_dir, exist_ok=True)

    # 2. Setup Dataset (ScanNet)
    scene_meta = utils_scannet.scannet_scenes()[args.scene]
    mesh_path = scene_meta.gt_mesh
    
    if not os.path.exists(mesh_path):
        logger.error(f"GT Mesh not found at {mesh_path}. Cannot train encoder.")
        return

    # Using BatchPosedSdf3D to generate SDF supervision from the mesh
    logger.info(f"Loading dataset from {mesh_path}...")
    dataset = BatchPosedSdf3D(
        [mesh_path],
        num_frames=10000, # Mock number of samples
        trunc_dist=0.15,
        frame_std_meter=0.005,
        frame_std_rad=0.008,
        distance_std=0.01,
        resample_poses_freq=50
    )
    train_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=1, num_workers=0)

    # 3. Setup Model
    logger.info("Initializing Encoder...")
    encoder = Encoder(cfg)
    encoder.to(device)
    
    # Register the grid model
    # We must set the grid bound to match the specific scene
    cfg_grid = deepcopy(cfg)
    cfg_grid['model']['grid']['bound'] = scene_meta.bound
    
    # We need to construct a GridNet helper to register with the encoder
    # This imports the factory function from configs.py (usually 'cfg_model')
    # But since we can't easily import that without circle deps in some projects,
    # we instantiate GridNet directly.
    grid_net = GridNet(cfg_grid['model'], device=device)
    encoder.register_grid_model(grid_net)

    # 4. Train & Visualize Levels
    # Train Level 0 (Coarse)
    train_encoder_level(encoder, train_loader, cfg, target_level=0, save_dir=args.save_dir)
    
    # Train Level 1 (Fine)
    train_encoder_level(encoder, train_loader, cfg, target_level=1, save_dir=args.save_dir)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
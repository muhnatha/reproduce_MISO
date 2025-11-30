import torch
import numpy as np
from grid_opt.models.grid_atlas import GridAtlas
from grid_opt.models.isdf.isdf import iSDF
from grid_opt.configs import load_config
import grid_opt.utils.utils_sdf as utils_sdf
import grid_opt.utils.utils_scannet as utils_scannet

# --- 1. Configuration ---
scene_name = '0000_00'
config_path = './configs/rgbd/scannet.yaml'

# TOGGLE THIS VARIABLE: 'miso' or 'isdf'
model_type = 'miso'  

if model_type == 'isdf':
    model_path = f'results/demo/isdf_mapping/{scene_name}/isdf_model.pth'
    save_path = f'results/demo/isdf_mapping/{scene_name}/sdf_plane_{scene_name}.png'
else:
    # Assuming standard MISO path
    model_path = f'results/demo/slam/0000_00/miso_init/result.pth'
    save_path = f'results/demo/slam/0000_00/miso_init/sdf_plane_{scene_name}.png'

cfg = load_config(config_path, './configs/base.yaml')
device = cfg['device']

# --- 2. Load Model ---
print(f"Loading {model_type.upper()} model from {model_path}...")

if model_type == 'miso':
    # MISO saves the entire object, so we just load it
    model = torch.load(model_path, map_location=device)

elif model_type == 'isdf':
    # iSDF saves only weights (state_dict), so we must build the structure first
    # 1. Instantiate the empty model (Must match training params!)
    model = iSDF(
        cfg['model'], 
        device=device, 
        hidden_size=256, 
        hidden_layers_block=2 
    ).to(device)
    
    # 2. Load the weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval() # Set to evaluation mode

# --- 3. Get Bounds ---
try:
    scene = utils_scannet.scannet_scenes()[scene_name]
    bounds = torch.tensor(scene.bound).to(device)
except KeyError:
    print(f"Scene {scene_name} not found in metadata. Using config bounds.")
    bounds = torch.tensor(cfg['model']['grid']['bound']).to(device)

# --- 4. Visualize ---
utils_sdf.visualize_sdf_plane(
    model, 
    bounds, 
    resolution=512, 
    axis='z', 
    fig_path=save_path, 
    show_colorbar=True
)
print(f"Saved SDF slice to {save_path}")
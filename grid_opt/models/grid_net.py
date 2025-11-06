import os
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms import so3_exp_map
import grid_opt.utils.utils as utils
from .base_net import BaseNet
from .modules import MLPNet
from .grid_modules import *
import grid_opt.utils.utils_geometry as utils_geometry
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class GridNet(BaseNet):
    def __init__(self, cfg: dict, device='cuda:0', dtype=torch.float32, initial_features=dict()):
        super(GridNet, self).__init__(cfg, device, dtype)
        self.initial_features = initial_features
        self.init_grid(cfg)
        self.init_decoder(cfg)
        self.init_poses(cfg)

    def save(self, ckpt_dir, ckpt_prefix):
        """ saves the learned weights of the decoder network"""
        ckpt_file = os.path.join(ckpt_dir, f"{ckpt_prefix}_decoder.pt")
        self.decoder.save(ckpt_file)

    def init_grid(self,cfg):
        """ Initializes the multi-resolution grid structure """
        self.num_levels = cfg['grid']['n_levels']
        self.second_order_grid_sample = 'second_order_grid_sample' in cfg['grid'] and cfg['grid'] and cfg['grid']['secod_order_grid_sample']
        base_cell_size = cfg['grid']['base_cell_size']
        scale_factor = cfg['grid']['per_level_scale']
        self.fdim = cfg['grid']['feature_dim']
        self.features = nn.ModuleList()
        self.feature_stability = nn.ModuleList()
        self.bases = nn.ModuleList()
        self.grid_type = cfg['grid']['type']
        self.cell_sizes = []

        # create each level grid from coarse to fine
        for level in range(self.num_levels):
            cell_size = base_cell_size / (scale_factor ** level) # calculate cell size at this level
            self.cell_sizes.append(cell_size)
            # check if there is initial feature for this level provided
            if level in self.initial_features.keys():
                init_feature = self.initial_features[level]
            else:
                init_feature = None
            # created the correct grid type
            if self.grid_type == 'regular': # 'regular' = a dense 3D tensor
                grid = FeatureGrid(
                    d = self.d,
                    fdim=self.fdim,
                    bound=self.bound,
                    cell_size=cell_size,
                    name=f"feat-{level}",
                    dtype=self.dtype,
                    initial_feature=init_feature,
                    init_stddev=cfg['grid']['init_stddev'],
                    secod_order_grid_sample=self.second_order_grid_sample
                )
                # This is an auxiliary grid that learns a "confidence" or "stability"
                # score (1.0 = mapped, 0.0 = empty). 
                # It is used during alignment to mask out unobserved/empty regions.
                grid_mu = FeatureGrid( 
                    d = self.d,
                    fdim=1,
                    bound=self.bound,
                    cell_size=cell_size,
                    name=f"stab-{level}",
                    dtype=self.dtype,
                    initial_feature=None,
                    init_stddev=0.0,
                    second_order_grid_sample=self.second_order_grid_sample
                )
                basis = None
            elif self.grid_type == 'VM': # 'VM' = Vector-Matrix decomposition
                grid = FeatureGridVM(
                    d = self.d,
                    fdim=self.fdim,
                    bound=self.bound,
                    cell_size=cell_size,
                    name=f"{level}",
                    dtype=self.dtype,
                    init_stddev=cfg['grid']['init_stddev'],
                    rank=cfg['grid']['VM']['rank']
                )
                grid_mu = None
                # This holds the learnable basis matrices (B_XY_Z, etc.)
                # It takes the coefficients from the FeatureGridVM and 
                # multiplies them to reconstruct the final feature vector.
                basis = BasisVM(
                    fdim=self.fdim,
                    name=f"{level}",
                    rank=cfg['grid']['VM']['rank'],
                    init_stddev=cfg['grid']['init_stddev'],
                    dtype=self.dtype,
                    pretrained_path=os.path.join(cfg['grid']['VM']['pretrained_bases_dir'], f"VM_basis_{level}.pt"),
                    no_optimize=cfg['grid']['VM']['fix_bases']
                )
            else:
                raise ValueError(f"Unknown grid type: {self.grid_type}!")
            # Add the newly created modules to their respective lists
            self.features.append(grid)
            self.feature_stability.append(grid_mu)
            self.bases.append(basis)
        self.ignore_level_ = np.zeros(self.num_levels).astype(bool)
    
    def init_decoder(self, cfg):
        """ Initializes the decoder network that maps grid features to desired outputs """
        self.decoder_hidden_dim = cfg['decoder']['hidden_dim']
        self.decoder_hidden_layers = cfg['decoder']['hidden_layers']
        self.decoder_out_dim = cfg['decoder']['out_dim']
        self.pos_invariant = cfg['decoder']['pos_invariant'] # wether the decoder will get spatial information or only position invariant
        self.decoder_fixed = cfg['decoder']['fix']
        self.decoder_type = cfg['decoder']['type']
        level_dim = self.fdim
        input_dim = self.num_levels * level_dim
        if not self.pos_invariant:
            input_dim += self.d # if there is no pose invariant, the model will be fed with (x, y, z) coordinate

        if self.decoder_type == 'mlp':
            logger.debug("Using MLP decoder")
            self.decoder = MLPNet(
                input_dim=input_dim,
                output_dim=self.decoder_out_dim,
                hidden_dim=self.decoder_hidden_dim,
                hidden_layers=self.decoder_hidden_layers,
                bias=True,
                pretrained_path=cfg['decoder']['pretrained_model'],
                no_optimize=self.decoder_fixed
            )
        elif self.decoder_type == 'none':
            logger.info("Not using decoder")
            self.decoder = None
        else:
            raise ValueError(f"Unknown decoder type: {self.decoder_type}")
        logger.debug(f"Initialized decoder:\n {self.decoder}")
    
    def init_poses(self, cfg):
        """Initialize pose correction terms.
        The pose corrections can be optimized jointly with the feature grid, i.e., bundle adjustment.
        As an example, see PosedSdfLoss3D.
        """
        self.num_poses=cfg['pose']['num_poses']
        self.optimize_pose=cfg['pose']['optimize']
        # initialize local corrections parameter
        self.rotation_corrections=torch.nn.Parameter(
            torch.zeros(self.num_poses, 3).float().to(self.device),
            requires_grad=self.optimize_pose
        )
        self.translation_corrections=torch.nn.Parameter(
            torch.zeros(self.num_poses, 3, 1).float().to(self.device),
            requires_grad=self.optimize_pose
        )
        self.pose_estimates_known=[False] * self.num_poses
        # self.Rwk = utils_geometry.identity_rotations(self.num_poses).to(self.device)
        # self.twk = torch.zeros(size=(self.num_poses, 3, 1), device=self.device)
        # untrainable paramater (base pose)
        self.register_buffer('Rwk', utils_geometry.identity_rotations(self.num_poses).to(self.device))
        self.register_buffer('twk', torch.zeros(size=(self.num_poses, 3, 1), device=self.device))   
        self.locked_pose_indices = set()
        self._pose_key_to_id = dict()
        logger.info(f"initialized {self.num_poses} pose variables (optimize={self.optimize_pose}).")
    
    def ignore_level(self, l):
        """Ignoring a feature level. The corresponding contribution from this level to the decoder will be set to zero.
        """
        self.ignore_level_[l] = True
        logger.warning(f"Ignore level: {self.ignore_level_}")

    def include_level(self, l):
        """Include a feature level """
        self.ignore_level_[l] = False
        logger.warning(f"Include level: {self.ignore_level_}") # BOOKMARK original code ("Ignore")
    
    # lock and unlock defined in grid_modules FeatureGridBase
    def lock_level(self, l):
        """Locking (fixing) the features at level l at the current value.
        """
        self.features[l].lock()
        self.feature_stability[l].lock()

    def unlock_level(self, l):
        """Unlocking (fixing) the features at level l at the current value.
        """
        self.features[l].unlock()
        self.feature_stability[l].unlock()
    
    # if we want to lock all map
    def lock_feature(self):
        for level in range(self.num_levels):
            self.lock_level(level)
    
    def unlock_feature(self):
        for level in range(self.num_levels):
            self.unlock_level(level)
    
    def lock_pose(self):
        self.rotation_corrections.require_grad_(False)
        self.translation_corrections.require_grad_(False)
        self.lock_all_pose_indices()
    
    def unlock_pose(self):
        self.rotation_corrections.require_grad_(True)
        self.translation_corrections.require_grad_(True)
        self.unlock_all_pose_indices()
    
    def lock_pose_index(self, pose_index:int):
        self.locked_pose_indices.add(pose_index) # add some specific keyframe ID to lock

    def lock_all_pose_indices(self):
        self.locked_pose_indices = set(range(self.num_poses))

    def unlock_pose_index(self, pose_index:int):
        self.locked_pose_indices.remove(pose_index) # remove some specific keyframe ID to lock

    def unlock_all_pose_indices(self):
        self.locked_pose_indices.clear()
    
    def pose_correction(self, kf_id: int):
        """Get trainable corrections (r & t) parameters for a single keyframe"""
        r = self.rotation_corrections[[kf_id], :] # (1, 3)
        t = self.translation_corrections[kf_id, :, :] # (3, 1)
        if kf_id in self.locked_pose_indices:
            r =  r.clone().detach()
            t = t.clone().detach()
        return r, t
    
    def set_initial_kf_pose(self, kf_id: int, Rwk: torch.Tensor, twk:torch.Tensor, kf_key=None):
        """Set the initial guess for the keyframe pose.
        
        # TODO: the kf_id is currently a local consecutive index, but 
        this should be replaced by the key completely in the future. 
        We keep it for now to be compatible with the previous version.

        Args:
            kf_id (int): local consecutive index / ID of the keyframe
            Rwk (torch.Tensor): rotation
            twk (torch.Tensor): translation
            kf_key: An optional key associated with this pose. Defaults to None.
        """
        assert Rwk.shape ==(3, 3)
        assert twk.shape == (3, 1)
        assert kf_id < self.num_poses, f"KF ID {kf_id} exceeds the number of poses {self.num_poses}!"
        self.pose_estimates_known[kf_id] = True # set kf ID has valid pose
        self.Rwk[kf_id, :, :] = Rwk.to(self.device) 
        self.twk[kf_id, :, :] = twk.to(self.device)
        # reset pertubation to zero
        with torch.no_grad():
            self.rotation_corrections[kf_id, :].copy_(torch.zeros(3, device=self.device))
            self.translation_corrections[kf_id, :, :].copy_(torch.zeros(3, 1, device=self.device))
        if kf_key is not None:
            self._pose_key_to_id[kf_key] = kf_id

    def pose_key_to_id(self, kf_key):
        """ map kf key to an ID"""
        assert kf_key in self._pose_key_to_id, f"Key {kf_key} not found in pose key to ID mapping!"
        return self._pose_key_to_id[kf_key]
    
    def initial_kf_pose(self, kf_id: int):
        """ get the base pose of a keyframe"""
        assert self.pose_estimates_known[kf_id], f"Initial pose estimate for KF {kf_id} is not available!"
        return self.Rwk[kf_id, :, :], self.twk[kf_id, :, :]   

    def initial_kf_pose_in_world(self, kf_id: int):

        return self.initial_kf_pose(kf_id)

    def initial_kf_pose_from_key(self, kf_key):
        """ combine pose_key_to_id action with initial_kf_pose so it can create base pose using a kf key"""
        kf_id = self.pose_key_to_id(kf_key)
        return self.initial_kf_pose(kf_id)

    def updated_kf_pose(self, kf_id: int):
        """ calculate the final optimized pose from base pose and trainable correction """
        Rwk, twk = self.initial_kf_pose_in_world(kf_id) # get the base pose
        Dr, Dt = self.pose_correction(kf_id) # get the small trainable delta
        return utils_geometry.apply_pose_correction( # add them together return optimized pose
            Rwk, twk, Dr, Dt
        ) 
    
    def updated_kf_pose_in_world(self, kf_id: int):
        return self.updated_kf_pose(kf_id)
    
    def updated_kf_pose_from_key(self, kf_key):
        kf_id = self.pose_key_to_id(kf_key)
        return self.updated_kf_pose(kf_id)
    
    def print_kf_pose_info(self):
        """ debugging tool to observe the pose corrections"""
        max_rot = torch.max(torch.linalg.norm(self.rotation_corrections, dim=1))
        max_tran = torch.max(torch.linalg.norm(self.translation_corrections.squeeze(2), dim=1))
        logger.info(f"GridNet KF pose corrections: max_rot={math.degrees(max_rot):.3f}deg, max_tran={max_tran:.3f}m.")
    
    def print_feature_info(self):
        for level in range(self.num_levels):
            logger.info(f"Level {level} norm: {self.features[level].norm():.2f}")
    
    def zero_features(self):
        for grid in self.features:
            grid.zero_features()
            
    def randn_features(self, std):
        for grid in self.features:
            grid.randn_features(std)
    
    def query_feature(self, x: torch.Tensor):
        """ trilinear interpolation to find the value of 3D coordinate X (feature field)"""
        assert x.ndim == 2, f"Invalid input coords shape {x.shape}!"
        assert x.shape[-1] == self.d

        # interpolate feature grid
        if self.grid_type == 'regular':
            feats = utils.grid_interp_regular(self.features, x, self.ignore_level_)
        elif self.grid_type == 'VM':
            feats = utils.grid_interp_VM(self.features, self.bases, x, self.ignore_level_)
        return feats
    
    def query_stability(self, x:torch.Tensor):
        """ calculate map confidence """
        assert self.grid_type == 'regular'
        assert x.ndim == 2, f"Invalid input coords shape {x.shape}!"
        assert x.shape[-1] == self.d
        mu_vec = utils.grid_interp_regular(self.feature_stability, x, None)
        return mu_vec

    def forward(self, x:torch, noise_std=0):
        """Predict value for coordinates x.

        Args:
            x (torch.tensor): Query coordinates with shape 
            (N, d) or (H, W, d), i.e., N coordinates or HxW coordinates.
            Each coordinate must fall within the bound.

        Returns:
            _type_: _description_
        """
        # interpolate feature grid
        feats = self.query_feature(x)

        # pass thru decoder
        pred = utils.grid_decode(feats, x, self.decoder, self.pos_invariant)
        if noise_std > 0:
            noise = torch.randn(pred.shape, device=x.device) * noise_std
            pred = pred + noise
        return pred    
    
    def params_for_poses(self):
        """ return a list of trainable pose"""
        return [self.rotation_corrections, self.translation_corrections]
    
    def params_for_features(self, stop_level=None):
        """ return a list of feature frid parameters"""
        if stop_level is None:
            stop_level = self.num_levels
        assert stop_level <= self.num_levels
        params = []
        for level in range(stop_level):
            params += list(self.features[level].parameters())
        return params
    
    def params_at_level(self, level):
        """ return a list of parameters based on level (coarse - fine)"""
        params = []
        target_levels = [level] if level < self.num_levels else range(self.num_levels)
        for l in target_levels:
            params += list(self.features[l].parameters())
            params += list(self.feature_stability[l].parameters())
        # always append decoder, if not fixed
        if not self.decoder_fixed:
            params += list(self.decoder.parameters())
        # always apped pose corrections, if not fixed
        if self.optimize_pose:
            params += self.params_for_poses()
        return params


    




    
    


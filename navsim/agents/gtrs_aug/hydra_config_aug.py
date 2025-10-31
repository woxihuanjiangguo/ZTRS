# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from dataclasses import dataclass
from typing import Tuple

from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.abstract_map import SemanticMapLayer
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.transfuser.transfuser_config import TransfuserConfig

NAVSIM_DEVKIT_ROOT = os.environ.get("NAVSIM_DEVKIT_ROOT")


@dataclass
class InferConfig:
    model: str = "teacher"  # teacher or student
    use_aug: bool = True  # whether using teacher augmentation


@dataclass
class RotationConfig:
    enable: bool = False
    fixed_angle: float = 0  # degree, positive: turn left
    offline_aug_angle_boundary: float = 0
    change_camera: bool = False
    crop_from_panoramic: bool = False


@dataclass
class VAConfig:
    # vel and acc perturb
    enable: bool = False
    offline_aug_boundary: float = 0


@dataclass
class EgoPerturbConfig:
    mode: str = 'fixed'  # 'fixed' or 'load_from_offline'
    ensemble_aug: bool = False
    offline_aug_file: str = '???'
    rotation: RotationConfig = RotationConfig()
    va: VAConfig = VAConfig()


@dataclass
class CameraProblemConfig:
    shutdown_enable: bool = False
    shutdown_mode: int = 1
    shutdown_probability: float = 0

    noise_enable: bool = False  # randomly set the pixel value
    noise_percentage: float = 0

    gaussian_enable: bool = False
    gaussian_mode: str = 'random'  # random or load_from_offline
    gaussian_probability: float = 0.0  # probability of applying gaussian noise to an image
    gaussian_mean: float = 0.0  # mean of gaussian noise
    gaussian_min_std: float = 0.05  # minimum std when using random std
    gaussian_max_std: float = 0.25  # maximum std when using random std
    gaussian_offline_file: str = ''  # file path for offline gaussian noise parameters

    # Weather augmentation settings
    weather_enable: bool = False
    weather_aug_mode: str = 'random'  # random or load_from_offline
    fog_prob: float = 0.2  # probability of applying fog effect
    rain_prob: float = 0.2  # probability of applying rain effect
    snow_prob: float = 0.2  # probability of applying snow effect


@dataclass
class DinoConfig:
    loss_weight: float = 1.0
    head_n_prototypes: int = 65536
    head_bottleneck_dim: int = 256
    head_nlayers: int = 3
    head_hidden_dim: int = 2048
    koleo_loss_weight: float = 0.1


@dataclass
class IbotConfig:
    loss_weight: float = 1.0
    mask_sample_probability: float = 0.5
    mask_ratio_min_max: Tuple[float, float] = (0.1, 0.5)
    separate_head: bool = True
    head_n_prototypes: int = 65536
    head_bottleneck_dim: int = 256
    head_nlayers: int = 3
    head_hidden_dim: int = 2048


@dataclass
class RefinementConfig:
    use_multi_stage: bool = False
    refinement_approach: str = "offset_decoder"
    num_refinement_stage: int = 1  # 2
    stage_layers: str = "3"  # "3+3"
    topks: str = "256"  # "256+64"

    use_mid_output: bool = True
    use_offset_refinement: bool = True  # abandoned
    use_offset_refinement_v2: bool = False
    use_separate_stage_heads: bool = True

    traj_expansion_in_infer: bool = False
    n_total_traj: int = 1024


@dataclass
class LabConfig:
    check_top_k_traj: bool = False
    num_top_k: int = 64
    test_full_vocab_pdm_score_path: str = "???"
    use_first_stage_traj_in_infer: bool = False

    change_loss_weight: bool = False
    use_imi_learning_in_refinement: bool = True
    adjust_refinement_loss_weight: bool = False  # change refinement loss weight: 256 / 8192.0
    adjust_refinement_score_weight: bool = False  # change dac, ep, lk score weight to 2 times
    ban_soft_label_loss: bool = False
    optimize_prev_frame_traj_for_ec: bool = False
    refinement_metrics: str = "all"  # 'all' or 'dac_ep_lk' or 'dac_ep_lk_pdms'
    use_higher_res_feat_in_refinement: bool = False

    use_cosine_ema_scheduler: bool = False
    ema_momentum_start: float = 0.99
    update_buffer_in_ema: bool = False
    save_pickle: bool = False


@dataclass
class HydraConfigAug(TransfuserConfig):
    seq_len: int = 2
    trajectory_imi_weight: float = 1.0
    trajectory_pdm_weight = {
        'no_at_fault_collisions': 3.0,
        'drivable_area_compliance': 3.0,
        'time_to_collision_within_bound': 4.0,
        'ego_progress': 2.0,
        'driving_direction_compliance': 1.0,
        'lane_keeping': 2.0,
        'traffic_light_compliance': 3.0,
        'history_comfort': 1.0,
    }
    progress_weight: float = 2.0
    ttc_weight: float = 2.0

    inference_imi_weight: float = 0.1
    inference_da_weight: float = 1.0
    decouple: bool = False
    vocab_size: int = 4096
    vocab_path: str = None
    normalize_vocab_pos: bool = False
    num_ego_status: int = 1

    ckpt_path: str = None
    sigma: float = 0.5
    use_pers_bev_embed: bool = False
    type: str = 'center'
    rel: bool = False
    use_nerf: bool = False
    extra_traj_layer: bool = False

    use_back_view: bool = False

    extra_tr: bool = False
    vadv2_head_nhead: int = 8
    vadv2_head_nlayers: int = 3

    trajectory_sampling: TrajectorySampling = TrajectorySampling(
        time_horizon=4, interval_length=0.1
    )

    # img backbone
    use_final_fpn: bool = False
    use_img_pretrained: bool = False
    # image_architecture: str = "vit_large_patch14_dinov2.lvd142m"
    image_architecture: str = "resnet34"
    backbone_type: str = 'resnet'
    vit_ckpt: str = ''
    intern_ckpt: str = ''
    vov_ckpt: str = ''
    eva_ckpt: str = ''
    swin_ckpt: str = ''

    sptr_ckpt: str = ''
    map_ckpt: str = ''

    lr_mult_backbone: float = 1.0
    backbone_wd: float = 0.0

    # lidar backbone
    lidar_architecture: str = "resnet34"

    max_height_lidar: float = 100.0
    pixels_per_meter: float = 4.0
    hist_max_per_pixel: int = 5

    lidar_min_x: float = -32
    lidar_max_x: float = 32
    lidar_min_y: float = -32
    lidar_max_y: float = 32

    lidar_split_height: float = 0.2
    use_ground_plane: bool = False

    # new
    lidar_seq_len: int = 1

    n_camera: int = 3  # 1 or 3 or 5

    camera_width: int = 2048
    camera_height: int = 512
    lidar_resolution_width: int = 256
    lidar_resolution_height: int = 256

    img_vert_anchors: int = camera_height // 32
    img_horz_anchors: int = camera_width // 32
    lidar_vert_anchors: int = lidar_resolution_height // 32
    lidar_horz_anchors: int = lidar_resolution_width // 32

    block_exp = 4
    n_layer = 2  # Number of transformer layers used in the vision backbone
    n_head = 4
    n_scale = 4
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    # Mean of the normal distribution initialization for linear layers in the GPT
    gpt_linear_layer_init_mean = 0.0
    # Std of the normal distribution initialization for linear layers in the GPT
    gpt_linear_layer_init_std = 0.02
    # Initial weight of the layer norms in the gpt.
    gpt_layer_norm_init_weight = 1.0

    perspective_downsample_factor = 1
    transformer_decoder_join = True
    detect_boxes = True
    use_bev_semantic = True
    use_semantic = False
    use_depth = False
    add_features = True

    # Transformer
    tf_d_model: int = 256
    tf_d_ffn: int = 1024
    tf_num_layers: int = 3
    tf_num_head: int = 8
    tf_dropout: float = 0.0

    # detection
    num_bounding_boxes: int = 30

    # loss weights
    agent_class_weight: float = 10.0
    agent_box_weight: float = 1.0
    bev_semantic_weight: float = 10.0

    # BEV mapping
    bev_semantic_classes = {
        1: ("polygon", [SemanticMapLayer.LANE, SemanticMapLayer.INTERSECTION]),  # road
        2: ("polygon", [SemanticMapLayer.WALKWAYS]),  # walkways
        3: ("linestring", [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]),  # centerline
        4: (
            "box",
            [
                TrackedObjectType.CZONE_SIGN,
                TrackedObjectType.BARRIER,
                TrackedObjectType.TRAFFIC_CONE,
                TrackedObjectType.GENERIC_OBJECT,
            ],
        ),  # static_objects
        5: ("box", [TrackedObjectType.VEHICLE]),  # vehicles
        6: ("box", [TrackedObjectType.PEDESTRIAN]),  # pedestrians
    }

    bev_pixel_width: int = lidar_resolution_width
    bev_pixel_height: int = lidar_resolution_height // 2
    bev_pixel_size: float = 1 / pixels_per_meter

    num_bev_classes = 7
    bev_features_channels: int = 64
    bev_down_sample_factor: int = 4
    bev_upsample_factor: int = 2

    # robust setting
    training: bool = True
    ego_perturb: EgoPerturbConfig = EgoPerturbConfig()
    camera_problem: CameraProblemConfig = CameraProblemConfig()
    only_ori_input: bool = False  # 如果是 True，说明是原来的训练设置
    student_rotation_ensemble: int = 3
    ori_vocab_pdm_score_full_path: str = "???"
    aug_vocab_pdm_score_dir: str = "???"
    pdm_closed_traj_path: str = "???"
    weakly_supervised_imi_learning: bool = False  # 直接不学 augmented 之后的 traj
    pdm_close_traj_for_augmented_gt: bool = False
    traj_smoothing: bool = False  # pdm_close_traj_for_augmented_gt 为 false 时，本来应该直接对原来的 traj 做旋转变换，但是 smoothing 可以将轨迹跟车的运动速度更加贴合

    only_imi_learning: bool = False

    soft_label_traj: str = 'first'  # first or final
    soft_label_imi_diff_thresh: float = 1.0
    soft_label_score_diff_thresh: float = 0.15

    use_rotation_loss: bool = False
    use_mask_loss: bool = False
    dino: DinoConfig = DinoConfig()
    ibot: IbotConfig = IbotConfig()
    refinement: RefinementConfig = RefinementConfig()

    inference: InferConfig = InferConfig()

    lab: LabConfig = LabConfig()

    @property
    def bev_semantic_frame(self) -> Tuple[int, int]:
        return (self.bev_pixel_height, self.bev_pixel_width)

    @property
    def bev_radius(self) -> float:
        values = [self.lidar_min_x, self.lidar_max_x, self.lidar_min_y, self.lidar_max_y]
        return max([abs(value) for value in values])

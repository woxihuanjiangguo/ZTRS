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
import pickle
from typing import Any, Union
from typing import Dict
from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, OneCycleLR

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.gtrs_dense.hydra_features import HydraFeatureBuilder, HydraTargetBuilder
from navsim.agents.gtrs_dense.gtrs_agent import three_to_two_classes
from navsim.agents.ztrs.hydra_config import HydraRLConfig
from navsim.agents.ztrs.hydra_model import HydraRLModel
from navsim.common.dataclasses import SensorConfig
from navsim.common.dataclasses import Trajectory
from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics import extract_features
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)

DEVKIT_ROOT = os.getenv('NAVSIM_DEVKIT_ROOT')
TRAJ_PDM_ROOT = os.getenv('NAVSIM_TRAJPDM_ROOT')


def compare_sim_feats(features_1, features_2):
    def calculate_rms(values):
        """
        Compute the Root Mean Square (RMS) of the given values along the time axis.

        :param values: Array containing values (n_batch, n_time).
        :return: RMS value per batch (n_batch,).
        """
        squared_values = values ** 2  # Square the differences
        mean_squared = torch.mean(squared_values, dim=-1)  # Compute mean along time axis
        rms_values = torch.sqrt(mean_squared)  # Square root to get RMS
        return rms_values

    acceleration_threshold: float = 0.7  # [m/s^2]
    jerk_threshold: float = 0.5  # [m/s^3]
    yaw_rate_threshold: float = 0.1  # [rad/s]
    yaw_accel_threshold: float = 0.1  # [rad/s^2]

    # Compute differences between corresponding time steps
    diff_acceleration = features_1["acceleration"] - features_2["acceleration"]
    diff_jerk = features_1["jerk"] - features_2["jerk"]
    diff_yaw_rate = features_1["yaw_rate"] - features_2["yaw_rate"]
    diff_yaw_accel = features_1["yaw_accel"] - features_2["yaw_accel"]

    # Calculate RMS differences
    rms_acceleration = calculate_rms(diff_acceleration)
    rms_jerk = calculate_rms(diff_jerk)
    rms_yaw_rate = calculate_rms(diff_yaw_rate)
    rms_yaw_accel = calculate_rms(diff_yaw_accel)

    # Compare RMS differences against thresholds
    meets_acceleration = rms_acceleration <= acceleration_threshold
    meets_jerk = rms_jerk <= jerk_threshold
    meets_yaw_rate = rms_yaw_rate <= yaw_rate_threshold
    meets_yaw_accel = rms_yaw_accel <= yaw_accel_threshold

    return meets_acceleration.logical_and(meets_jerk).logical_and(meets_yaw_rate).logical_and(meets_yaw_accel)


def get_sim_feat_dict():
    return {
        'acceleration': [],
        'jerk': [],
        'yaw_rate': [],
        'yaw_accel': [],
    }


def hydra_kd_imi_agent_rl_loss(
        targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor], config: HydraRLConfig,
        vocab_pdm_score,
        regression_ep=False,
        three2two=True,
        pg_loss=False,
        ec_target=False,
        prev_trajs=None,
        tokens=None,
        prev_ego_state=None,
        simulator=None
):
    """
    Helper function calculating complete loss of Transfuser
    :param targets: dictionary of name tensor pairings
    :param predictions: dictionary of name tensor pairings
    :param config: global Transfuser config
    :return: combined loss value
    """
    no_at_fault_collisions, drivable_area_compliance, time_to_collision_within_bound, ego_progress = (
        predictions['no_at_fault_collisions'],
        predictions['drivable_area_compliance'],
        predictions['time_to_collision_within_bound'],
        predictions['ego_progress']
    )
    driving_direction_compliance, lane_keeping, traffic_light_compliance = (
        predictions['driving_direction_compliance'],
        predictions['lane_keeping'],
        predictions['traffic_light_compliance']
    )
    history_comfort = predictions['history_comfort']
    dtype = no_at_fault_collisions.dtype
    device = no_at_fault_collisions.device
    # 2 cls
    da_loss = F.binary_cross_entropy_with_logits(drivable_area_compliance,
                                                 vocab_pdm_score['drivable_area_compliance'].to(dtype))
    ttc_loss = F.binary_cross_entropy_with_logits(time_to_collision_within_bound,
                                                  vocab_pdm_score['time_to_collision_within_bound'].to(dtype))
    if three2two:
        noc_gt = three_to_two_classes(vocab_pdm_score['no_at_fault_collisions'].to(dtype))
    else:
        noc_gt = vocab_pdm_score['no_at_fault_collisions'].to(dtype)
    noc_loss = F.binary_cross_entropy_with_logits(no_at_fault_collisions, noc_gt)

    if regression_ep:
        progress_loss = F.mse_loss(ego_progress.sigmoid(), vocab_pdm_score['ego_progress'].to(dtype))
    else:
        progress_loss = F.binary_cross_entropy_with_logits(ego_progress, vocab_pdm_score['ego_progress'].to(dtype))
    # expansion
    if three2two:
        ddc_gt = three_to_two_classes(vocab_pdm_score['driving_direction_compliance'].to(dtype))
    else:
        ddc_gt = vocab_pdm_score['driving_direction_compliance'].to(dtype)
    ddc_loss = F.binary_cross_entropy_with_logits(driving_direction_compliance, ddc_gt)
    lk_loss = F.binary_cross_entropy_with_logits(lane_keeping, vocab_pdm_score['lane_keeping'].to(dtype))
    tl_loss = F.binary_cross_entropy_with_logits(traffic_light_compliance,
                                                 vocab_pdm_score['traffic_light_compliance'].to(dtype))
    comfort_loss = F.binary_cross_entropy_with_logits(history_comfort,
                                                      vocab_pdm_score['history_comfort'].to(dtype))

    B = no_at_fault_collisions.shape[0]
    pdms = vocab_pdm_score['pdm_score'].to(dtype)

    if ec_target:
        n_overlap = 5  # 0.5 sec, 10hz
        interval_length = 0.1
        time_horizon = 4
        n_steps = int(time_horizon / interval_length)
        sim_prev_features = get_sim_feat_dict()

        # simulate prev_traj for feats
        for (traj, ego_state) in zip(prev_trajs, prev_ego_state):
            traj = traj.cpu().numpy()
            transformed_traj = transform_trajectory(Trajectory(traj, TrajectorySampling(
                time_horizon=time_horizon, interval_length=interval_length
            )), ego_state)
            prev_state = simulator.simulate_proposals(
                get_trajectory_as_array(
                    transformed_traj,
                    simulator.proposal_sampling,
                    ego_state.time_point
                )[None],
                ego_state
            )
            timepoints = np.arange(n_steps) * interval_length
            prev_feats = extract_features(prev_state, timepoints)
            for k in sim_prev_features:
                sim_prev_features[k].append(torch.from_numpy(prev_feats[k]).to(device).float())

        # cat along B
        sim_features = {}
        for k in sim_prev_features:
            sim_prev_features[k] = torch.cat(sim_prev_features[k], 0).unsqueeze(1)[..., n_overlap:]
            sim_features[k] = targets[k][..., :-n_overlap]

        meets_ec_indices = compare_sim_feats(sim_prev_features, sim_features)
        violates_ec_indices = meets_ec_indices.logical_not()
        pdms -= violates_ec_indices * config.ec_penalty

    if pg_loss:
        adv = (pdms - pdms.mean(1, keepdim=True)) / (pdms.std(1, keepdim=True) + 1e-7)
        adv = adv.clamp(min=-3.0, max=3.0)
        probs = predictions['rl'].softmax(-1)
        rl_loss = -(adv * probs).sum() / B
    else:
        gt = torch.zeros_like(predictions['rl'])
        gt[torch.arange(B), pdms.argmax(1)] = 1.0
        rl_loss = F.cross_entropy(predictions['rl'], gt)

    rl_loss_final = config.trajectory_imi_weight * rl_loss
    noc_loss_final = config.trajectory_pdm_weight['no_at_fault_collisions'] * noc_loss
    da_loss_final = config.trajectory_pdm_weight['drivable_area_compliance'] * da_loss
    ttc_loss_final = config.trajectory_pdm_weight['time_to_collision_within_bound'] * ttc_loss
    progress_loss_final = config.trajectory_pdm_weight['ego_progress'] * progress_loss
    ddc_loss_final = config.trajectory_pdm_weight['driving_direction_compliance'] * ddc_loss
    lk_loss_final = config.trajectory_pdm_weight['lane_keeping'] * lk_loss
    tl_loss_final = config.trajectory_pdm_weight['traffic_light_compliance'] * tl_loss
    comfort_loss_final = config.trajectory_pdm_weight['history_comfort'] * comfort_loss

    loss = (
            rl_loss_final
            + noc_loss_final
            + da_loss_final
            + ttc_loss_final
            + progress_loss_final
            + ddc_loss_final
            + lk_loss_final
            + tl_loss_final
            + comfort_loss_final
    )
    loss_dict = {
        'rl_loss': rl_loss_final,
        'pdm_noc_loss': noc_loss_final,
        'pdm_da_loss': da_loss_final,
        'pdm_ttc_loss': ttc_loss_final,
        'pdm_progress_loss': progress_loss_final,
        'pdm_ddc_loss': ddc_loss_final,
        'pdm_lk_loss': lk_loss_final,
        'pdm_tl_loss': tl_loss_final,
        'pdm_comfort_loss': comfort_loss_final
    }

    if config.top_k > 0:
        # stage 2
        pdms_topk, pdms_topk_idx = pdms.topk(k=config.top_k, dim=-1)
        adv_topk = (pdms_topk - pdms_topk.mean(1, keepdim=True)) / (pdms_topk.std(1, keepdim=True) + 1e-7)
        adv_topk = adv_topk.clamp(min=-3.0, max=3.0)
        logits_topk = torch.gather(predictions['rl_topk'], dim=1, index=pdms_topk_idx)
        probs_topk = logits_topk.softmax(-1)
        rl_topk_loss = -(adv_topk * probs_topk).sum() / B

        loss += rl_topk_loss
        loss_dict['rl_topk_loss'] = rl_topk_loss

    if 'bev_semantic_map' in predictions:
        bev_semantic_loss = F.cross_entropy(predictions["bev_semantic_map"], targets["bev_semantic_map"].long())
        bev_semantic_loss = bev_semantic_loss * 10.0
        loss += bev_semantic_loss
        loss_dict['bev_semantic_loss'] = bev_semantic_loss

    return loss, loss_dict


class ZTRSAgent(AbstractAgent):
    def __init__(
            self,
            config: HydraRLConfig,
            lr: float,
            checkpoint_path: str = None,
            pdm_gt_path=None,
            simulator: PDMSimulator = None
    ):
        super().__init__(
            trajectory_sampling=config.trajectory_sampling
        )
        if config.regression_ep:
            ep_lw = 50.0
        else:
            ep_lw = 2.0

        config.trajectory_pdm_weight = {
            'no_at_fault_collisions': 3.0,
            'drivable_area_compliance': 3.0,
            'time_to_collision_within_bound': 4.0,
            'ego_progress': ep_lw,
            'driving_direction_compliance': 1.0,
            'lane_keeping': 2.0,
            'traffic_light_compliance': 3.0,
            'history_comfort': 1.0,
        }
        self._config = config
        self._lr = lr
        self.metrics = list(config.trajectory_pdm_weight.keys()) + ['pdm_score']
        self._checkpoint_path = checkpoint_path
        if self._config.version == 'default':
            self.model = HydraRLModel(config)
        else:
            raise ValueError('Unsupported hydra version')
        self.vocab_size = config.vocab_size
        self.backbone_wd = config.backbone_wd
        self.scheduler = config.scheduler
        if pdm_gt_path is not None:
            self.vocab_pdm_score_full = pickle.load(
                open(pdm_gt_path, 'rb'))

        if config.ec_target:
            assert simulator is not None
            self.simulator = simulator

    def name(self) -> str:
        """Inherited, see superclass."""

        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))["state_dict"]
        keys_to_delete = [k for k in state_dict if "model._trajectory_head.vocab" in k]
        for k in keys_to_delete:
            del state_dict[k]
        msg = self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()}, strict=False)
        print(msg)

    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        return SensorConfig(
            cam_f0=[0, 1, 2, 3],
            cam_l0=[0, 1, 2, 3],
            cam_l1=[0, 1, 2, 3],
            cam_l2=[0, 1, 2, 3],
            cam_r0=[0, 1, 2, 3],
            cam_r1=[0, 1, 2, 3],
            cam_r2=[0, 1, 2, 3],
            cam_b0=[0, 1, 2, 3],
            lidar_pc=[],
        )

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return [HydraTargetBuilder(config=self._config)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [HydraFeatureBuilder(config=self._config)]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.model(features)

    def evaluate_dp_proposals(self, features: Dict[str, torch.Tensor], dp_proposals) -> Dict[str, torch.Tensor]:
        return self.model.evaluate_dp_proposals(features, dp_proposals)

    def forward_train(self, features, interpolated_traj):
        return self.model(features, interpolated_traj)

    def compute_loss(
            self,
            features: Dict[str, torch.Tensor],
            targets: Dict[str, torch.Tensor],
            predictions: Dict[str, torch.Tensor],
            tokens=None,
            prev_ego_state=None
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        # get the pdm score by tokens
        scores = {}
        for k in self.metrics:
            tmp = [self.vocab_pdm_score_full[token][k][None] for token in tokens]
            scores[k] = (torch.from_numpy(np.concatenate(tmp, axis=0))
                         .to(predictions['trajectory'].device))
        return hydra_kd_imi_agent_rl_loss(targets, predictions, self._config, scores,
                                          regression_ep=self._config.regression_ep,
                                          three2two=self._config.three2two,
                                          pg_loss=self._config.pg_loss,
                                          ec_target=self._config.ec_target,
                                          prev_trajs=predictions['prev_trajs'],
                                          tokens=tokens,
                                          prev_ego_state=prev_ego_state,
                                          simulator=self.simulator
                                          )

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        backbone_params_name = '_backbone.image_encoder'
        img_backbone_params = list(
            filter(lambda kv: backbone_params_name in kv[0], self.model.named_parameters()))
        default_params = list(filter(lambda kv: backbone_params_name not in kv[0], self.model.named_parameters()))
        params_lr_dict = [
            {'params': [tmp[1] for tmp in default_params]},
            {
                'params': [tmp[1] for tmp in img_backbone_params],
                'lr': self._lr * self._config.lr_mult_backbone,
                'weight_decay': self.backbone_wd
            }
        ]
        if self.scheduler == 'default':
            return torch.optim.Adam(params_lr_dict, lr=self._lr, weight_decay=self._config.weight_decay)
        elif self.scheduler == 'cycle':
            optim = torch.optim.Adam(params_lr_dict, lr=self._lr)
            return {
                "optimizer": optim,
                "lr_scheduler": OneCycleLR(
                    optim,
                    max_lr=0.001,
                    total_steps=20 * 196
                )
            }
        else:
            raise ValueError('Unsupported lr scheduler')

    def get_training_callbacks(self) -> List[pl.Callback]:
        return [
            # TransfuserCallback(self._config),
            ModelCheckpoint(
                save_top_k=100,
                monitor="val/loss_epoch",
                mode="min",
                dirpath=f"{os.environ.get('NAVSIM_EXP_ROOT')}/{self._config.ckpt_path}/",
                filename="{epoch:02d}-{step:04d}",
            )
        ]

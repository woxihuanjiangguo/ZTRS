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
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, OneCycleLR

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.gtrs_dense.hydra_config import HydraConfig
from navsim.agents.gtrs_dense.hydra_features import HydraFeatureBuilder, HydraTargetBuilder
from navsim.agents.gtrs_dense.hydra_model import HydraModel
from navsim.common.dataclasses import SensorConfig
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)


def three_to_two_classes(x):
    x[x == 0.5] = 0.0
    return x


def hydra_kd_imi_agent_loss_dropout(
        targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor], config: HydraConfig,
        vocab_pdm_score,
        regression_ep=False,
        three2two=True,
        include_dp=False
):
    """
    Helper function calculating complete loss of Transfuser
    :param targets: dictionary of name tensor pairings
    :param predictions: dictionary of name tensor pairings
    :param config: global Transfuser config
    :return: combined loss value
    """
    dropout_indices = predictions['dropout_indices']
    imi = predictions['imi']
    dtype = imi.dtype
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
    for k, v in vocab_pdm_score.items():
        gt_score = v.to(dtype)
        gt_score = gt_score[:, dropout_indices]
        vocab_pdm_score[k] = gt_score

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

    vocab = predictions["trajectory_vocab_dropout"]
    # B, 8 (4 secs, 0.5Hz), 3
    target_traj = targets["trajectory"]
    # 4, 9, ..., 39
    sampled_timepoints = [5 * k - 1 for k in range(1, 9)]
    B = target_traj.shape[0]
    if include_dp:
        l2_distance = -(
                (vocab[:, :, sampled_timepoints] - target_traj[:, None]) ** 2) / config.sigma
        imi_loss = F.cross_entropy(imi, l2_distance.sum((-2, -1)).softmax(1))

    else:
        l2_distance = -(
                    (vocab[:, sampled_timepoints][None].repeat(B, 1, 1, 1) - target_traj[:, None]) ** 2) / config.sigma
        imi_loss = F.cross_entropy(imi, l2_distance.sum((-2, -1)).softmax(1))

    imi_loss_final = config.trajectory_imi_weight * imi_loss
    noc_loss_final = config.trajectory_pdm_weight['no_at_fault_collisions'] * noc_loss
    da_loss_final = config.trajectory_pdm_weight['drivable_area_compliance'] * da_loss
    ttc_loss_final = config.trajectory_pdm_weight['time_to_collision_within_bound'] * ttc_loss
    progress_loss_final = config.trajectory_pdm_weight['ego_progress'] * progress_loss
    ddc_loss_final = config.trajectory_pdm_weight['driving_direction_compliance'] * ddc_loss
    lk_loss_final = config.trajectory_pdm_weight['lane_keeping'] * lk_loss
    tl_loss_final = config.trajectory_pdm_weight['traffic_light_compliance'] * tl_loss

    loss = (
            imi_loss_final
            + noc_loss_final
            + da_loss_final
            + ttc_loss_final
            + progress_loss_final
            + ddc_loss_final
            + lk_loss_final
            + tl_loss_final
    )
    loss_dict = {
        'imi_loss': imi_loss_final,
        'pdm_noc_loss': noc_loss_final,
        'pdm_da_loss': da_loss_final,
        'pdm_ttc_loss': ttc_loss_final,
        'pdm_progress_loss': progress_loss_final,
        'pdm_ddc_loss': ddc_loss_final,
        'pdm_lk_loss': lk_loss_final,
        'pdm_tl_loss': tl_loss_final,
    }
    if 'bev_semantic_map' in predictions:
        bev_semantic_loss = F.cross_entropy(predictions["bev_semantic_map"], targets["bev_semantic_map"].long())
        bev_semantic_loss = bev_semantic_loss * 10.0
        loss += bev_semantic_loss
        loss_dict['bev_semantic_loss'] = bev_semantic_loss

    return loss, loss_dict


class GTRSAgent(AbstractAgent):
    def __init__(
            self,
            config: HydraConfig,
            lr: float,
            checkpoint_path: str = None,
            pdm_gt_path=None,
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
        self.metrics = list(config.trajectory_pdm_weight.keys())
        self._checkpoint_path = checkpoint_path
        if self._config.version == 'default':
            self.model = HydraModel(config)
        else:
            raise ValueError('Unsupported hydra version')
        self.vocab_size = config.vocab_size
        self.backbone_wd = config.backbone_wd
        self.scheduler = config.scheduler
        if pdm_gt_path is not None:
            self.vocab_pdm_score_full = pickle.load(
                open(pdm_gt_path, 'rb'))

    def name(self) -> str:
        """Inherited, see superclass."""

        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))["state_dict"]
        # Remove keys containing 'model._trajectory_head.vocab'
        keys_to_delete = [k for k in state_dict if "model._trajectory_head.vocab" in k]
        for k in keys_to_delete:
            del state_dict[k]

        msg = self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()}, strict=False)
        print('Loading full GTRS model', msg)

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
            tokens=None
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        # get the pdm score by tokens
        scores = {}
        for k in self.metrics:
            tmp = [self.vocab_pdm_score_full[token][k][None] for token in tokens]
            scores[k] = (torch.from_numpy(np.concatenate(tmp, axis=0))
                         .to(predictions['trajectory'].device))
        return hydra_kd_imi_agent_loss_dropout(targets, predictions, self._config, scores,
                                               regression_ep=self._config.regression_ep,
                                               three2two=self._config.three2two)

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

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

import json
import os
import pickle
from typing import Any, Union
from typing import Dict, List

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.gtrs_aug.aug_meta_arch import AugMetaArch
from navsim.agents.gtrs_aug.hydra_config_aug import HydraConfigAug
from navsim.agents.gtrs_aug.hydra_features_aug import HydraAugFeatureBuilder, HydraAugTargetBuilder
from navsim.agents.gtrs_aug.hydra_loss_fn_aug import hydra_kd_imi_agent_loss_robust, \
    hydra_kd_imi_agent_loss_single_stage
from navsim.agents.gtrs_aug.hydra_model import HydraModel
from navsim.common.dataclasses import SensorConfig
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)


class GTRSAugAgent(AbstractAgent):
    def __init__(
            self,
            config: HydraConfigAug,
            lr: float,
            checkpoint_path: str = None,
            metrics=None,
    ):
        super().__init__(
            trajectory_sampling=config.trajectory_sampling
        )
        config.trajectory_pdm_weight = {
            'no_at_fault_collisions': 3.0,
            'drivable_area_compliance': 3.0,
            'time_to_collision_within_bound': 4.0,
            'ego_progress': 2.0,
            'driving_direction_compliance': 1.0,
            'lane_keeping': 2.0,
            'traffic_light_compliance': 3.0,
            'history_comfort': 1.0,
        }
        if config.lab.change_loss_weight:
            config.trajectory_pdm_weight = {
                'no_at_fault_collisions': 1.5,
                'drivable_area_compliance': 1.5,
                'time_to_collision_within_bound': 1.5,
                'ego_progress': 2.0,
                'driving_direction_compliance': 1.0,
                'lane_keeping': 2.0,
                'traffic_light_compliance': 1.0,
                'history_comfort': 1.0,
            }
        self._config = config
        self._lr = lr
        self.metrics = metrics
        self._checkpoint_path = checkpoint_path
        teacher_model = HydraModel(config)
        student_model = HydraModel(config)
        self.model = AugMetaArch(config, teacher_model, student_model)
        self.vocab_size = config.vocab_size
        self.backbone_wd = config.backbone_wd
        self.ensemble_aug = config.ego_perturb.ensemble_aug
        self.training = config.training

        if self.training:
            self.ori_vocab_pdm_score_full = pickle.load(
                open(f'{config.ori_vocab_pdm_score_full_path}', 'rb'))
            self.aug_vocab_pdm_score_dir = config.aug_vocab_pdm_score_dir

            with open(config.ego_perturb.offline_aug_file, 'r') as f:
                aug_data = json.load(f)
            assert aug_data['param']['rot'] == config.ego_perturb.rotation.offline_aug_angle_boundary
            self.aug_info = aug_data['tokens']

        self.only_ori_input = config.only_ori_input
        self.n_rotation_crop = config.student_rotation_ensemble

    def name(self) -> str:
        """Inherited, see superclass."""

        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))["state_dict"]
        self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()}, strict=False)

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
        return [HydraAugTargetBuilder(config=self._config)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [HydraAugFeatureBuilder(config=self._config)]

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        features, targets, tokens = batch
        kwargs = {'tokens': tokens}

        teacher_ori_features = dict()
        student_ori_features = dict()
        teacher_ori_features['camera_feature'] = features['ori_teacher']
        teacher_ori_features['status_feature'] = features['status_feature']

        student_feat_dict_lst = []

        student_ori_features['camera_feature'] = features['ori']
        student_ori_features['status_feature'] = features['status_feature']
        student_feat_dict_lst.append(student_ori_features)
        if not self.only_ori_input and self._config.training:
            for i in range(self.n_rotation_crop):
                student_feat_dict_lst.append(
                    {
                        'camera_feature': features['rotated'][i],
                        'status_feature': features['status_feature'],
                    }
                )

            if self._config.use_mask_loss:
                kwargs = {
                    'collated_masks': features['collated_masks'],
                    "mask_indices_list": features['mask_indices_list'],
                    "masks_weight": features['masks_weight'],
                    "upperbound": features['upperbound'],
                    "n_masked_patches": features['n_masked_patches'],
                }

        teacher_pred, student_preds, loss_dict = self.model(teacher_ori_features, student_feat_dict_lst, **kwargs)
        return teacher_pred, student_preds, loss_dict

    def forward_train(self, features, interpolated_traj):
        return self.vadv2_model(features, interpolated_traj)

    def compute_loss(
            self,
            features: Dict[str, torch.Tensor],
            targets: Dict[str, torch.Tensor],
            predictions: List[Dict[str, torch.Tensor]],
            tokens=None
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        # ori
        ori_targets = {'trajectory': targets['ori_trajectory']}
        ori_predictions = predictions[0]
        scores = {}
        for k in self.metrics:
            tmp = [self.ori_vocab_pdm_score_full[token][k][None] for token in tokens]
            scores[k] = (torch.from_numpy(np.concatenate(tmp, axis=0))
                         .to(ori_predictions['trajectory'].device))
        ori_loss = hydra_kd_imi_agent_loss_robust(ori_targets, ori_predictions, self._config, scores)
        if self._config.only_ori_input:
            return {"ori": ori_loss}

        # aug
        _aug_vocab_pdm_score = {}
        for token in tokens:
            with open(os.path.join(self.aug_vocab_pdm_score_dir, f'{token}.pkl'), 'rb') as f:
                _aug_vocab_pdm_score[token] = pickle.load(f)
        aug_loss = []
        for idx in range(self._config.student_rotation_ensemble):
            aug_targets = {'trajectory': targets['rotated_trajectories'][idx]}
            scores = {}
            for k in self.metrics:
                tmp = [_aug_vocab_pdm_score[token][idx][k][None] for token in tokens]
                scores[k] = (torch.from_numpy(np.concatenate(tmp, axis=0))
                             .to(predictions[idx + 1]['trajectory'].device))
            aug_loss.append(hydra_kd_imi_agent_loss_robust(aug_targets, predictions[idx + 1], self._config, scores))

        # Calculate average loss and loss dict
        avg_aug_loss = torch.mean(torch.stack([loss[0] for loss in aug_loss]))
        avg_aug_loss_dict = {}
        for key in aug_loss[0][1].keys():
            avg_aug_loss_dict[key] = torch.mean(torch.stack([loss[1][key] for loss in aug_loss]))
        return {
            "ori": ori_loss,
            "aug": (avg_aug_loss, avg_aug_loss_dict),
        }

    def compute_loss_soft_teacher(
            self,
            teacher_pred: Dict[str, torch.Tensor],
            student_pred: Dict[str, torch.Tensor],
            targets,
            tokens=None
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        sampled_timepoints = [5 * ii - 1 for ii in range(1, 9)]
        if self._config.soft_label_traj == 'first':
            traj_diff = teacher_pred['trajectory'][:, sampled_timepoints] - targets['ori_trajectory']
        elif self._config.soft_label_traj == 'final':
            traj_diff = teacher_pred['final_traj'][:, sampled_timepoints] - targets['ori_trajectory']

        clamped_traj_diff = torch.clamp(traj_diff, min=-self._config.soft_label_imi_diff_thresh,
                                        max=self._config.soft_label_imi_diff_thresh)
        # Apply clamped adjustment to original trajectory
        revised_targets = {'trajectory': targets['ori_trajectory'] + clamped_traj_diff}

        scores = {}
        revised_scores = {}
        for k in self.metrics:
            tmp = [self.ori_vocab_pdm_score_full[token][k][None] for token in tokens]
            scores[k] = torch.from_numpy(np.concatenate(tmp, axis=0)).to(teacher_pred['trajectory'].device).float()
            # Calculate difference and clamp to max 0.2
            diff = teacher_pred[k].sigmoid() - scores[k]
            _soft_label_score_diff_thresh = self._config.soft_label_score_diff_thresh
            clamped_diff = torch.clamp(diff, min=-_soft_label_score_diff_thresh, max=_soft_label_score_diff_thresh)
            # Apply clamped adjustment to original scores
            revised_scores[k] = scores[k] + clamped_diff

        soft_loss = hydra_kd_imi_agent_loss_robust(revised_targets, student_pred, self._config, revised_scores)
        return soft_loss

    def compute_loss_multi_stage(
            self,
            features,
            targets: Dict[str, torch.Tensor],
            predictions: List[Dict[str, torch.Tensor]],
            tokens=None
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        _refinement_metrics = self.metrics
        if self._config.lab.refinement_metrics == 'dac_ep_lk_pdms':
            _refinement_metrics = _refinement_metrics + ['pdm_score']

        trajectory_vocab = predictions[0]['trajectory_vocab']

        result_dict = dict()
        # ori
        ori_loss_lst = []
        ori_predictions = predictions[0]['refinement']
        num_stage = len(ori_predictions)
        for i in range(num_stage):
            pred_i = ori_predictions[i]
            selected_indices_i = pred_i['indices_absolute']
            scores = {}
            for k in _refinement_metrics:
                tmp = [self.ori_vocab_pdm_score_full[token][k][None] for token in tokens]
                full_scores = torch.from_numpy(np.concatenate(tmp, axis=0)).to(
                    selected_indices_i.device)  # [bs, vocab_size]
                # Extract scores based on selected indices [bs, topk_stage_i]
                batch_size, topk = selected_indices_i.shape
                batch_indices = torch.arange(batch_size, device=selected_indices_i.device).unsqueeze(1).expand(-1, topk)
                scores[k] = full_scores[batch_indices, selected_indices_i]  # [bs, topk_stage_i]
            _kwargs = {}
            if self._config.lab.use_imi_learning_in_refinement:
                _kwargs['targets'] = {'trajectory': targets['ori_trajectory']}
                pred_i['trajectory_vocab'] = trajectory_vocab
            ori_loss_i = hydra_kd_imi_agent_loss_single_stage(pred_i, self._config, scores, **_kwargs)
            ori_loss_lst.append(ori_loss_i)
        total_ori_loss = sum([loss_tup[0] for loss_tup in ori_loss_lst])
        total_ori_loss_dict = {}
        for i, loss_tup in enumerate(ori_loss_lst):
            loss_dict = loss_tup[1]
            for _key, _value in loss_dict.items():
                total_ori_loss_dict[f"stage_{i + 2}_{_key}"] = _value
        result_dict['ori'] = (total_ori_loss, total_ori_loss_dict)
        if self._config.only_ori_input:
            return result_dict

        # aug
        _aug_vocab_pdm_score = {}
        for token in tokens:
            with open(os.path.join(self.aug_vocab_pdm_score_dir, f'{token}.pkl'), 'rb') as f:
                _aug_vocab_pdm_score[token] = pickle.load(f)
        aug_loss_all_mode_lst = []
        for idx in range(self._config.student_rotation_ensemble):
            aug_loss_lst = []
            aug_idx_predictions = predictions[idx + 1]['refinement']
            for i in range(num_stage):
                aug_idx_pred_i = aug_idx_predictions[i]
                aug_idx_selected_indices_i = aug_idx_pred_i['indices_absolute']
                scores = {}
                for k in _refinement_metrics:
                    tmp = [_aug_vocab_pdm_score[token][idx][k][None] for token in tokens]
                    full_scores = torch.from_numpy(np.concatenate(tmp, axis=0)).to(aug_idx_selected_indices_i.device)
                    batch_size, topk = aug_idx_selected_indices_i.shape
                    batch_indices = torch.arange(batch_size, device=aug_idx_selected_indices_i.device).unsqueeze(
                        1).expand(-1, topk)
                    scores[k] = full_scores[batch_indices, aug_idx_selected_indices_i]
                _kwargs_idx = {}
                if self._config.lab.use_imi_learning_in_refinement:
                    _kwargs_idx['targets'] = {'trajectory': targets['rotated_trajectories'][idx]}
                    aug_idx_pred_i['trajectory_vocab'] = trajectory_vocab
                aug_loss_lst.append(
                    hydra_kd_imi_agent_loss_single_stage(aug_idx_pred_i, self._config, scores, **_kwargs_idx))
            aug_loss_single_mode = sum([loss_tup[0] for loss_tup in aug_loss_lst])
            aug_loss_single_mode_dict = {}
            for i, loss_tup in enumerate(aug_loss_lst):
                loss_dict = loss_tup[1]
                for _key, _value in loss_dict.items():
                    aug_loss_single_mode_dict[f"stage_{i + 2}_{_key}"] = _value
            aug_loss_all_mode_lst.append((aug_loss_single_mode, aug_loss_single_mode_dict))

        # Calculate average loss and loss dict
        avg_aug_loss = torch.mean(torch.stack([loss[0] for loss in aug_loss_all_mode_lst]))
        avg_aug_loss_dict = {}
        for key in aug_loss_all_mode_lst[0][1].keys():
            avg_aug_loss_dict[key] = torch.mean(torch.stack([loss[1][key] for loss in aug_loss_all_mode_lst]))
        result_dict['aug'] = (avg_aug_loss, avg_aug_loss_dict)

        return result_dict

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        backbone_params_name = '_backbone.image_encoder'
        img_backbone_params = list(
            filter(lambda kv: backbone_params_name in kv[0], self.model.student.model.named_parameters()))
        default_params = list(
            filter(lambda kv: backbone_params_name not in kv[0], self.model.student.model.named_parameters()))
        params_lr_dict = [
            {'params': [tmp[1] for tmp in default_params]},
            {
                'params': [tmp[1] for tmp in img_backbone_params],
                'lr': self._lr * self._config.lr_mult_backbone,
                'weight_decay': self.backbone_wd
            }
        ]
        return torch.optim.Adam(params_lr_dict, lr=self._lr)

    def get_training_callbacks(self) -> List[pl.Callback]:
        return [
            ModelCheckpoint(
                save_top_k=30,
                monitor="val/loss-ori",
                mode="min",
                dirpath=f"{os.environ.get('NAVSIM_EXP_ROOT')}/{self._config.ckpt_path}/",
                filename="{epoch:02d}-{step:04d}",
            )
        ]


def _get_norm_tensor(t):
    norms = torch.norm(t, p=2, dim=1, keepdim=True)
    return t / norms

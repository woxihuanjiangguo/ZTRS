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

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from navsim.agents.dp.dp_model import diff_traj
from navsim.agents.gtrs_dense.hydra_backbone import HydraBackbone
from navsim.agents.transfuser.transfuser_model import AgentHead
from navsim.agents.utils.attn import MemoryEffTransformer
from navsim.agents.utils.nerf import nerf_positional_encoding
from navsim.agents.ztrs.hydra_config import HydraRLConfig


class HydraRLModel(nn.Module):
    def __init__(self, config: HydraRLConfig):
        super().__init__()

        self._query_splits = [
            config.num_bounding_boxes,
        ]

        self._config = config
        self._backbone = HydraBackbone(config)

        img_num = 2 if config.use_back_view else 1
        self._keyval_embedding = nn.Embedding(
            config.img_vert_anchors * config.img_horz_anchors * img_num, config.tf_d_model
        )  # 8x8 feature grid + trajectory
        self._query_embedding = nn.Embedding(sum(self._query_splits), config.tf_d_model)

        # usually, the BEV features are variable in size.
        self.downscale_layer = nn.Conv2d(self._backbone.img_feat_c, config.tf_d_model, kernel_size=1)
        self._status_encoding = nn.Linear((4 + 2 + 2) * config.num_ego_status, config.tf_d_model)

        tf_decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.tf_d_model,
            nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn,
            dropout=config.tf_dropout,
            batch_first=True,
        )

        self._tf_decoder = nn.TransformerDecoder(tf_decoder_layer, config.tf_num_layers)
        self._agent_head = AgentHead(
            num_agents=config.num_bounding_boxes,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
        )

        self._trajectory_head = HydraTrajHead(
            num_poses=config.trajectory_sampling.num_poses,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
            nhead=config.vadv2_head_nhead,
            nlayers=config.vadv2_head_nlayers,
            vocab_path=config.vocab_path,
            config=config
        )
        if self._config.ec_target and not self._config.no_cond:
            self.prev_traj_embedding = nn.Sequential(
                nn.Linear(config.trajectory_sampling.num_poses * 3, config.tf_d_ffn),
                nn.ReLU(),
                nn.Linear(config.tf_d_ffn, config.tf_d_model),
            )

    def img_feat_blc(self, camera_feature):
        img_features = self._backbone(camera_feature)
        img_features = self.downscale_layer(img_features).flatten(-2, -1)
        img_features = img_features.permute(0, 2, 1)
        return img_features

    def get_feats_1frame(self, camera_feature):
        img_features = self._backbone(camera_feature)
        img_features = self.downscale_layer(img_features)
        return img_features

    def evaluate_dp_proposals(self, features, dp_proposals, topk=10, dp_only_inference=False, return_wo_override=False):
        status_feature: torch.Tensor = features["status_feature"][0]
        camera_feature = features["camera_feature"]

        if self._config.num_ego_status == 1 and status_feature.shape[1] == 32:
            status_encoding = self._status_encoding(status_feature[:, :8])
        else:
            status_encoding = self._status_encoding(status_feature)

        # original
        if isinstance(camera_feature, list):
            camera_feature = camera_feature[-1]
        img_features = self.img_feat_blc(camera_feature)
        if self._config.use_back_view:
            img_features_back = self.img_feat_blc(features["camera_feature_back"])
            img_features = torch.cat([img_features, img_features_back], 1)
        keyval = img_features
        keyval += self._keyval_embedding.weight[None, ...]
        output: Dict[str, torch.Tensor] = {}
        trajectory = self._trajectory_head.eval_dp_proposals(keyval, status_encoding, dp_proposals, topk=topk,
                                                             dp_only_inference=dp_only_inference,
                                                             return_wo_override=return_wo_override)
        output.update(trajectory)
        return output

    @torch.no_grad()
    def prev_frame_inference(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        status_feature: torch.Tensor = features["status_feature"][1]
        camera_feature = features["camera_feature"]
        status_encoding = self._status_encoding(status_feature)
        # original
        if isinstance(camera_feature, list):
            camera_feature = camera_feature[-2]
        img_features = self.img_feat_blc(camera_feature)
        if self._config.use_back_view:
            img_features_back = self.img_feat_blc(features["camera_feature_back"])
            img_features = torch.cat([img_features, img_features_back], 1)
        keyval = img_features
        keyval += self._keyval_embedding.weight[None, ...]
        output: Dict[str, torch.Tensor] = {}
        trajectory = self._trajectory_head(keyval, status_encoding)
        output.update(trajectory)
        return output

    def forward(self, features: Dict[str, torch.Tensor],
                interpolated_traj=None) -> Dict[str, torch.Tensor]:
        output: Dict[str, torch.Tensor] = {}

        status_feature: torch.Tensor = features["status_feature"][0]
        camera_feature = features["camera_feature"]

        if self._config.num_ego_status == 1 and status_feature.shape[1] == 32:
            status_encoding = self._status_encoding(status_feature[:, :8])
        else:
            status_encoding = self._status_encoding(status_feature)

        # original
        if isinstance(camera_feature, list):
            camera_feature = camera_feature[-1]
        img_features = self.img_feat_blc(camera_feature)
        B = img_features.shape[0]
        if self._config.use_back_view:
            img_features_back = self.img_feat_blc(features["camera_feature_back"])
            img_features = torch.cat([img_features, img_features_back], 1)
        keyval = img_features

        keyval += self._keyval_embedding.weight[None, ...]

        if self._config.ec_target:
            prev_inference = self.prev_frame_inference(features)
            output['prev_trajs'] = prev_inference['trajectory']
            if not self._config.no_cond:
                prev_traj_token = self.prev_traj_embedding(
                    prev_inference['trajectory'].view(B, -1)
                ).unsqueeze(1)
                keyval = torch.cat([
                    keyval, prev_traj_token
                ], 1)

        trajectory = self._trajectory_head(keyval, status_encoding, interpolated_traj)
        output.update(trajectory)

        return output


class HydraTrajHead(nn.Module):
    def __init__(self, num_poses: int, d_ffn: int, d_model: int, vocab_path: str,
                 nhead: int, nlayers: int, config: HydraRLConfig = None
                 ):
        super().__init__()
        assert config.activation in ['relu', 'gelu']

        activation = nn.GELU if config.activation == 'gelu' else nn.ReLU
        activation_func = F.gelu if config.activation == 'gelu' else F.relu
        self.config = config
        self._num_poses = num_poses
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model, nhead, d_ffn,
                dropout=0.0, batch_first=True,
                activation=activation_func
            ), nlayers
        )
        self.vocab = nn.Parameter(
            torch.from_numpy(np.load(vocab_path)),
            requires_grad=False
        )

        self.heads = nn.ModuleDict({
            'no_at_fault_collisions': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                activation(),
                nn.Linear(d_ffn, 1),
            ),
            'drivable_area_compliance':
                nn.Sequential(
                    nn.Linear(d_model, d_ffn),
                    activation(),
                    nn.Linear(d_ffn, 1),
                ),
            'time_to_collision_within_bound': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                activation(),
                nn.Linear(d_ffn, 1),
            ),
            'ego_progress': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                activation(),
                nn.Linear(d_ffn, 1),
            ),
            'driving_direction_compliance': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                activation(),
                nn.Linear(d_ffn, 1),
            ),
            'lane_keeping': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                activation(),
                nn.Linear(d_ffn, 1),
            ),
            'traffic_light_compliance': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                activation(),
                nn.Linear(d_ffn, 1),
            ),
            'history_comfort': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                activation(),
                nn.Linear(d_ffn, 1),
            ),
            'rl': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                activation(),
                nn.Linear(d_ffn, d_ffn),
                activation(),
                nn.Linear(d_ffn, 1),
            )
        })
        if config.top_k > 0:
            self.heads['rl_topk'] = nn.Sequential(
                nn.Linear(d_model, d_ffn),
                activation(),
                nn.Linear(d_ffn, d_ffn),
                activation(),
                nn.Linear(d_ffn, 1),
            )
        self.normalize_vocab_pos = config.normalize_vocab_pos
        if self.normalize_vocab_pos:
            self.encoder = nn.Sequential(
                *[MemoryEffTransformer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=d_model * 4,
                    dropout=0.0
                ) for _ in range(config.encoder_layers)]
            )

        self.use_nerf = config.use_nerf
        if self.config.extra_feat:
            self.pos_embed = nn.Sequential(
                nn.Linear(num_poses * (3 + 4), d_ffn),
                activation(),
                nn.Linear(d_ffn, d_model),
            )
        elif self.use_nerf:
            self.pos_embed = nn.Sequential(
                nn.Linear(1040, d_ffn),
                activation(),
                nn.Linear(d_ffn, d_model),
            )
        else:
            self.pos_embed = nn.Sequential(
                nn.Linear(num_poses * 3, d_ffn),
                activation(),
                nn.Linear(d_ffn, d_model),
            )

    def forward(self, bev_feature, status_encoding, interpolated_traj=None) -> Dict[str, torch.Tensor]:
        result = {}
        vocab = self.vocab.data
        L, HORIZON, _ = vocab.shape
        B = bev_feature.shape[0]
        num_total = vocab.size(0)  # 16384
        if self.training and self.config.vocab_dropout:
            num_select = num_total // 2  # 8192
            indices = torch.randperm(num_total, device=vocab.device)[:num_select]
            vocab = vocab[indices]
            result['dropout_indices'] = indices
            L, HORIZON, _ = vocab.shape
        else:
            result['dropout_indices'] = torch.arange(num_total, device=vocab.device)
        result['trajectory_vocab_dropout'] = vocab
        if self.use_nerf:
            vocab = torch.cat(
                [
                    nerf_positional_encoding(vocab[..., :2]),
                    torch.cos(vocab[..., -1])[..., None],
                    torch.sin(vocab[..., -1])[..., None],
                ], dim=-1
            )
        if self.config.extra_feat:
            extra_vocab = torch.cat([
                vocab, diff_traj(vocab)
            ], -1)
            embedded_vocab = self.pos_embed(extra_vocab.view(L, -1))[None]
            embedded_vocab = self.encoder(embedded_vocab).repeat(B, 1, 1)
        elif self.normalize_vocab_pos:
            embedded_vocab = self.pos_embed(vocab.view(L, -1))[None]
            embedded_vocab = self.encoder(embedded_vocab).repeat(B, 1, 1)
        else:
            embedded_vocab = self.pos_embed(vocab.view(L, -1))[None].repeat(B, 1, 1)
        tr_out = self.transformer(embedded_vocab, bev_feature)
        dist_status = tr_out + status_encoding.unsqueeze(1)

        # selected_indices: B,
        for k, head in self.heads.items():
            result[k] = head(dist_status).squeeze(-1)

        # baseline
        if self.config.top_k > 0:
            # stage 2
            scores = (
                    0.05 * result['rl'].softmax(-1).log() +
                    0.5 * result['traffic_light_compliance'].sigmoid().log() +
                    0.5 * result['no_at_fault_collisions'].sigmoid().log() +
                    0.5 * result['drivable_area_compliance'].sigmoid().log() +
                    0.5 * result['driving_direction_compliance'].sigmoid().log() +
                    8.0 * (5 * result['time_to_collision_within_bound'].sigmoid() +
                           5 * result['ego_progress'].sigmoid() +
                           5 * result['lane_keeping'].sigmoid()).log()
            )
            scores_topk_idx = scores.topk(k=self.config.top_k, dim=-1).indices
            topk_head_scores = torch.gather(result['rl_topk'], dim=1, index=scores_topk_idx)
            topk_head_scores_softmax = 0.01 * topk_head_scores.softmax(-1).log()
            addition = torch.zeros_like(scores)
            addition.scatter_add_(1, scores_topk_idx, topk_head_scores_softmax)
            scores += addition
        else:
            scores = (
                    0.05 * result['rl'].softmax(-1).log() +
                    0.5 * result['traffic_light_compliance'].sigmoid().log() +
                    0.5 * result['no_at_fault_collisions'].sigmoid().log() +
                    0.5 * result['drivable_area_compliance'].sigmoid().log() +
                    0.5 * result['driving_direction_compliance'].sigmoid().log() +
                    8.0 * (5 * result['time_to_collision_within_bound'].sigmoid() +
                           5 * result['ego_progress'].sigmoid() +
                           5 * result['lane_keeping'].sigmoid()).log()
            )

        selected_indices = scores.argmax(1)
        result["trajectory"] = self.vocab.data[selected_indices]
        result["trajectory_vocab"] = self.vocab.data
        result["selected_indices"] = selected_indices
        return result

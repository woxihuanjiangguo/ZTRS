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

import copy
import os
import pickle
from itertools import combinations
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from navsim.agents.gtrs_aug.hydra_backbone import HydraBackbone
from navsim.agents.gtrs_aug.hydra_config_aug import HydraConfigAug
from sklearn.cluster import KMeans

from navsim.agents.utils.attn import MemoryEffTransformer


class HydraModel(nn.Module):
    def __init__(self, config: HydraConfigAug):
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
        # self._query_embedding = nn.Embedding(sum(self._query_splits), config.tf_d_model)

        # usually, the BEV features are variable in size.
        self.downscale_layer = nn.Conv2d(self._backbone.img_feat_c, config.tf_d_model, kernel_size=1)
        self._status_encoding = nn.Linear((4 + 2 + 2) * config.num_ego_status, config.tf_d_model)

        self._trajectory_head = HydraTrajHead(
            num_poses=config.trajectory_sampling.num_poses,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
            nhead=config.vadv2_head_nhead,
            nlayers=config.vadv2_head_nlayers,
            vocab_path=config.vocab_path,
            config=config
        )

        self.use_multi_stage = self._config.refinement.use_multi_stage
        if self.use_multi_stage:
            if self._config.refinement.refinement_approach == 'offset_decoder':
                self._trajectory_offset_head = TrajOffsetHead(
                    d_ffn=config.tf_d_ffn,
                    d_model=config.tf_d_model,
                    nhead=config.vadv2_head_nhead,
                    d_backbone=self._backbone.img_feat_c,
                    num_stage=config.refinement.num_refinement_stage,
                    stage_layers=config.refinement.stage_layers,
                    topks=config.refinement.topks,
                    config=config
                )
            else:
                raise NotImplementedError

    def img_feat_blc(self, camera_feature):
        img_features = self._backbone(camera_feature)  # [b, c_img, h//32, w//32]
        img_features = self.downscale_layer(img_features).flatten(-2, -1)  # [b, c, h//32 * w//32]
        img_features = img_features.permute(0, 2, 1)  # [b, h//32 * w//32, c]
        return img_features

    def img_feat_blc_dict(self, camera_feature, **kwargs):
        img_feat_tup = self._backbone.forward_tup(camera_feature, **kwargs)
        img_feat_dict = {
            'patch_token': img_feat_tup[0],
            # [b, c_img, h//32, w//32], the patch size of vit is only 16, but use a avg pooling
            'class_token': img_feat_tup[1],
        }
        if self._config.lab.use_higher_res_feat_in_refinement:
            img_feat_dict['higher_res_feat'] = img_feat_tup[2]
        img_features = img_feat_dict['patch_token']
        img_features = self.downscale_layer(img_features).flatten(-2, -1)  # [b, c, h//32 * w//32]
        img_features = img_features.permute(0, 2, 1)  # [b, h//32 * w//32, c]
        img_feat_dict['avg_feat'] = img_features
        return img_feat_dict

    def forward_features_list(self, x_list, mask_list):
        return [self.forward(x, m) for x, m in zip(x_list, mask_list)]

    def forward(self,
                features: Dict[str, torch.Tensor],
                masks=None,
                interpolated_traj=None,
                tokens=None) -> Dict[str, torch.Tensor]:

        output: Dict[str, torch.Tensor] = {}

        camera_feature: torch.Tensor = features[
            "camera_feature"]  # List[torch.Tensor], len == seq_len, tensor.shape == [b, 3, h, w]
        status_feature: torch.Tensor = features["status_feature"][
            0]  # List[torch.Tensor], len == seq_len, tensor.shape == [b, 8] (the [0] picks present status)
        if isinstance(camera_feature, list):
            camera_feature = camera_feature[-1]  # [b, 3, h, w]

        img_feat_dict = self.img_feat_blc_dict(camera_feature, masks=masks, return_class_token=True)
        status_encoding = self._status_encoding(status_feature)  # [b, 8] -> [b, c]

        keyval = img_feat_dict.pop('avg_feat')  # [b, h//32 * w//32, c]
        keyval += self._keyval_embedding.weight[None, ...]  # [b, h//32 * w//32, c]

        output.update(img_feat_dict)
        trajectory = self._trajectory_head(keyval, status_encoding, interpolated_traj, tokens=tokens)

        if self.use_multi_stage:
            if self._config.lab.use_higher_res_feat_in_refinement:
                bev_feat_fg = img_feat_dict['higher_res_feat']
            else:
                bev_feat_fg = img_feat_dict['patch_token']  # [bs, c_vit, w, h]
            final_traj = self._trajectory_offset_head(bev_feat_fg, trajectory['refinement'])
            trajectory['final_traj'] = final_traj

        output.update(trajectory)
        return output


class HydraTrajHead(nn.Module):
    def __init__(self, num_poses: int, d_ffn: int, d_model: int, vocab_path: str,
                 nhead: int, nlayers: int, config: HydraConfigAug = None
                 ):
        super().__init__()
        self._config = config
        self._num_poses = num_poses
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model, nhead, d_ffn,
                dropout=0.0, batch_first=True
            ), nlayers
        )
        self.vocab = nn.Parameter(
            torch.from_numpy(np.load(vocab_path)),
            requires_grad=False
        )

        self.heads = nn.ModuleDict({
            'no_at_fault_collisions': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, 1),
            ),
            'drivable_area_compliance':
                nn.Sequential(
                    nn.Linear(d_model, d_ffn),
                    nn.ReLU(),
                    nn.Linear(d_ffn, 1),
                ),
            'time_to_collision_within_bound': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, 1),
            ),
            'ego_progress': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, 1),
            ),
            'driving_direction_compliance': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, 1),
            ),
            'lane_keeping': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, 1),
            ),
            'traffic_light_compliance': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, 1),
            ),
            'history_comfort': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, 1),
            ),
            'imi': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, 1),
            )
        })

        if self._config.lab.optimize_prev_frame_traj_for_ec:
            self.heads['imi_prev'] = nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, 1),
            )

        self.inference_imi_weight = config.inference_imi_weight
        self.inference_da_weight = config.inference_da_weight
        self.normalize_vocab_pos = config.normalize_vocab_pos
        if self.normalize_vocab_pos:
            self.encoder = MemoryEffTransformer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=0.0
            )
        self.use_nerf = config.use_nerf

        if self.use_nerf:
            self.pos_embed = nn.Sequential(
                nn.Linear(1040, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, d_model),
            )
        else:
            self.pos_embed = nn.Sequential(
                nn.Linear(num_poses * 3, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, d_model),
            )
        # load dp proposals
        dp_preds_path = os.getenv('DP_PREDS', None)
        if dp_preds_path is not None:
            print(f'Loading DP PREDS from {dp_preds_path}')
            self.dp_preds = pickle.load(open(dp_preds_path, 'rb'))
        else:
            self.dp_preds = None

    def forward(self, bev_feature, status_encoding, interpolated_traj=None, tokens=None) -> Dict[str, torch.Tensor]:
        # vocab: 4096, 40, 3
        # bev_feature: B, 32, C
        # embedded_vocab: B, 4096, C
        vocab = self.vocab.data
        L, HORIZON, _ = vocab.shape
        B = bev_feature.shape[0]

        if self.dp_preds is None:
            if self.normalize_vocab_pos:
                embedded_vocab = self.pos_embed(vocab.view(L, -1))[None]
                embedded_vocab = self.encoder(embedded_vocab).repeat(B, 1, 1)
            else:
                embedded_vocab = self.pos_embed(vocab.view(L, -1))[None].repeat(B, 1, 1)  # [b, n_vocab, c]
        else:
            # combined inference with dp
            curr_dp_preds = []
            for token in tokens:
                curr_dp_preds.append(
                    torch.from_numpy(self.dp_preds[token]['interpolated_proposal']).float().to(bev_feature.device)[
                        None])
            # B, N, 40, 3
            curr_dp_preds = torch.cat(curr_dp_preds, 0)
            NUM_PROPOSALS = curr_dp_preds.shape[1]
            dp_proposals = curr_dp_preds.view(B, NUM_PROPOSALS, -1)
            vocab = torch.cat([
                vocab.view(L, -1)[None].repeat(B, 1, 1),
                dp_proposals
            ], 1)
            embedded_vocab = self.pos_embed(vocab)
            embedded_vocab = self.encoder(embedded_vocab)
            # end of combined inference with dp


        tr_out = self.transformer(embedded_vocab, bev_feature)  # [b, n_vocab, c]
        dist_status = tr_out + status_encoding.unsqueeze(1)  # [b, n_vocab, c]
        result = {}
        for k, head in self.heads.items():
            result[k] = head(dist_status).squeeze(-1)

        scores = (
                0.02 * result['imi'].softmax(-1).log() +
                0.1 * result['traffic_light_compliance'].sigmoid().log() +
                0.5 * result['no_at_fault_collisions'].sigmoid().log() +
                0.5 * result['drivable_area_compliance'].sigmoid().log() +
                0.3 * result['driving_direction_compliance'].sigmoid().log() +
                6.0 * (5.0 * result['time_to_collision_within_bound'].sigmoid() +
                       5.0 * result['ego_progress'].sigmoid() +
                       2.0 * result['lane_keeping'].sigmoid() +
                       1.0 * result['history_comfort'].sigmoid()
                       ).log()
        )

        selected_indices = scores.argmax(1)
        scene_cnt_tensor = torch.arange(B, device=scores.device)
        if self.dp_preds is None:
            result["trajectory"] = self.vocab.data[selected_indices]
        else:
            result["trajectory"] = vocab[scene_cnt_tensor, selected_indices].view(B, HORIZON, 3)
        result["trajectory_vocab"] = self.vocab.data
        result["selected_indices"] = selected_indices

        if self._config.refinement.use_multi_stage:
            topk_str = str(self._config.refinement.topks)
            topk = int(topk_str.split('+')[0])
            topk_values, topk_indices = torch.topk(scores, k=topk, dim=1)
            result['refinement'] = []  # dicts of different refinement stages
            _dict = {}
            if self.dp_preds is None:
                _dict["trajs"] = self.vocab.data[topk_indices]
            else:
                _dict["trajs"] = vocab.gather(1, topk_indices.unsqueeze(-1).expand(-1, -1, vocab.size(-1)))
                _dict["trajs"] = _dict["trajs"].view(B, topk, HORIZON, 3)

            # Gather the statuses for the top-k trajectories
            batch_indices = torch.arange(B, device=topk_indices.device).view(-1, 1).expand(-1, topk)
            _dict["trajs_status"] = dist_status[batch_indices, topk_indices]
            _dict['indices_absolute'] = topk_indices

            # Store the scores for each top-k trajectory
            _dict['coarse_score'] = {}
            for score_key in self.heads.keys():
                _dict['coarse_score'][score_key] = result[score_key][batch_indices, topk_indices]

            if self._config.refinement.traj_expansion_in_infer and not self._config.training:
                self.forward_refinement_infer_expansion(bev_feature, status_encoding, _dict)

            result['refinement'].append(_dict)
        return result

    def forward_refinement_infer_expansion(self, bev_feature, status_encoding, dict_first_stage):
        trajs = dict_first_stage['trajs']
        n_total_traj = self._config.refinement.n_total_traj

        interp_trajs = cluster_interp_tensor_batch(trajs, n_total_traj - trajs.shape[1])
        B, L, HORIZON, _ = interp_trajs.shape
        embedded_interb_vocab = self.pos_embed(interp_trajs.view(B, L, -1))
        embedded_interb_vocab = self.encoder(embedded_interb_vocab)
        tr_out_interb = self.transformer(embedded_interb_vocab, bev_feature)

        dist_status_interb = tr_out_interb + status_encoding.unsqueeze(1)  # [b, n_vocab, c]
        score_interb = {}
        for k, head in self.heads.items():
            score_interb[k] = head(dist_status_interb).squeeze(-1)

        dict_first_stage['trajs'] = torch.cat([dict_first_stage['trajs'], interp_trajs], dim=1)
        dict_first_stage['trajs_status'] = torch.cat([dict_first_stage['trajs_status'], dist_status_interb], dim=1)
        indices_absolute = dict_first_stage['indices_absolute']
        indices_interb = torch.arange(self.vocab.shape[0], self.vocab.shape[0] + L).unsqueeze(0).expand(B, L).to(
            indices_absolute)
        dict_first_stage['indices_absolute'] = torch.cat([indices_absolute, indices_interb], dim=1)
        for k in self.heads.keys():
            dict_first_stage['coarse_score'][k] = torch.cat([dict_first_stage['coarse_score'][k], score_interb[k]],
                                                            dim=1)


class TrajOffsetHead(nn.Module):
    def __init__(self, d_ffn: int, d_model: int, nhead: int, d_backbone: int,
                 num_stage: int, stage_layers: str, topks: str,
                 config: HydraConfigAug = None
                 ):
        super().__init__()
        stage_layers = str(stage_layers)
        topks = str(topks)

        self._config = config
        self.num_stage = num_stage
        self.stage_layers = [int(sl) for sl in stage_layers.split('+')]
        self.topks = [int(topk) for topk in topks.split('+')]
        assert len(self.stage_layers) == num_stage and len(self.topks) == num_stage
        self.nlayers = sum(self.stage_layers)

        self.use_mid_output = config.refinement.use_mid_output
        self.use_offset_refinement = config.refinement.use_offset_refinement_v2
        if self.use_offset_refinement:
            assert self.use_mid_output == True
        self.use_separate_stage_heads = config.refinement.use_separate_stage_heads

        downscale_layer = nn.Conv2d(d_backbone, d_model, kernel_size=1)
        if self.use_separate_stage_heads:
            self.downscale_layers = nn.ModuleList([copy.deepcopy(downscale_layer) for _ in range(num_stage)])
        else:
            self.downscale_layers = nn.ModuleList([downscale_layer for _ in range(num_stage)])

        transformer_blocks = [TransformerDecoder_v2(
            nn.TransformerDecoderLayer(
                d_model, nhead, d_ffn,
                dropout=0.0, batch_first=True
            ), layer
        ) for layer in self.stage_layers]
        self.transformer_blocks = nn.ModuleList(transformer_blocks)

        assert self._config.lab.refinement_metrics in ('all', 'dac_ep_lk', 'dac_ep_lk_pdms')
        refinement_metrics = self._config.lab.refinement_metrics
        if refinement_metrics == 'all':
            heads = nn.ModuleDict({
                'no_at_fault_collisions': nn.Sequential(
                    nn.Linear(d_model, d_ffn),
                    nn.ReLU(),
                    nn.Linear(d_ffn, 1),
                ),
                'drivable_area_compliance':
                    nn.Sequential(
                        nn.Linear(d_model, d_ffn),
                        nn.ReLU(),
                        nn.Linear(d_ffn, 1),
                    ),
                'time_to_collision_within_bound': nn.Sequential(
                    nn.Linear(d_model, d_ffn),
                    nn.ReLU(),
                    nn.Linear(d_ffn, 1),
                ),
                'ego_progress': nn.Sequential(
                    nn.Linear(d_model, d_ffn),
                    nn.ReLU(),
                    nn.Linear(d_ffn, 1),
                ),
                'driving_direction_compliance': nn.Sequential(
                    nn.Linear(d_model, d_ffn),
                    nn.ReLU(),
                    nn.Linear(d_ffn, 1),
                ),
                'lane_keeping': nn.Sequential(
                    nn.Linear(d_model, d_ffn),
                    nn.ReLU(),
                    nn.Linear(d_ffn, 1),
                ),
                'traffic_light_compliance': nn.Sequential(
                    nn.Linear(d_model, d_ffn),
                    nn.ReLU(),
                    nn.Linear(d_ffn, 1),
                ),
                'history_comfort': nn.Sequential(
                    nn.Linear(d_model, d_ffn),
                    nn.ReLU(),
                    nn.Linear(d_ffn, 1),
                ),
            })
        elif refinement_metrics == 'dac_ep_lk':
            heads = nn.ModuleDict({
                'drivable_area_compliance':
                    nn.Sequential(
                        nn.Linear(d_model, d_ffn),
                        nn.ReLU(),
                        nn.Linear(d_ffn, 1),
                    ),
                'ego_progress': nn.Sequential(
                    nn.Linear(d_model, d_ffn),
                    nn.ReLU(),
                    nn.Linear(d_ffn, 1),
                ),
                'lane_keeping': nn.Sequential(
                    nn.Linear(d_model, d_ffn),
                    nn.ReLU(),
                    nn.Linear(d_ffn, 1),
                ),
            })
        elif refinement_metrics == 'dac_ep_lk_pdms':
            heads = nn.ModuleDict({
                'drivable_area_compliance':
                    nn.Sequential(
                        nn.Linear(d_model, d_ffn),
                        nn.ReLU(),
                        nn.Linear(d_ffn, 1),
                    ),
                'ego_progress': nn.Sequential(
                    nn.Linear(d_model, d_ffn),
                    nn.ReLU(),
                    nn.Linear(d_ffn, 1),
                ),
                'lane_keeping': nn.Sequential(
                    nn.Linear(d_model, d_ffn),
                    nn.ReLU(),
                    nn.Linear(d_ffn, 1),
                ),
                'pdm_score': nn.Sequential(
                    nn.Linear(d_model, d_ffn),
                    nn.ReLU(),
                    nn.Linear(d_ffn, d_ffn),
                    nn.ReLU(),
                    nn.Linear(d_ffn, 1),
                ),
            })

        if self._config.lab.use_imi_learning_in_refinement:
            heads['imi'] = nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, 1),
            )
        if self._config.lab.optimize_prev_frame_traj_for_ec:
            heads['imi_prev'] = nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, 1),
            )

        if self.use_separate_stage_heads:
            self.multi_stage_heads = nn.ModuleList([copy.deepcopy(heads) for _ in range(num_stage)])
        else:
            self.multi_stage_heads = nn.ModuleList([heads for _ in range(num_stage)])

        self.inference_da_weight = config.inference_da_weight
        self.normalize_vocab_pos = config.normalize_vocab_pos
        if self.normalize_vocab_pos:
            self.encoder = MemoryEffTransformer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=0.0
            )

    def forward(self, bev_feat_fg, refinement_dict) -> Dict[str, torch.Tensor]:
        B = bev_feat_fg.shape[0]

        for i in range(self.num_stage):
            _bev_feat_fg = self.downscale_layers[i](bev_feat_fg).flatten(2)
            _bev_feat_fg = _bev_feat_fg.permute(0, 2,
                                                1)
            status_encoding = refinement_dict[-1]['trajs_status']  # [bs, topk_stage_i, c]
            tr_out_lst = self.transformer_blocks[i](status_encoding,
                                                    _bev_feat_fg)  # [layer_stage_i, bs, topk_stage_i, c]

            # Compute scores for each layer
            layer_results = []
            # Initialize reference scores from coarse_score
            if self.use_offset_refinement:
                reference = refinement_dict[-1]['coarse_score']
            for j, dist_status in enumerate(tr_out_lst):
                layer_result = {}
                for k, head in self.multi_stage_heads[i].items():
                    if self.use_offset_refinement:
                        # Compute offset
                        offset = head(dist_status).squeeze(-1)
                        layer_result[k] = reference[k] + offset
                        # Update reference for next layer
                        reference[k] = layer_result[k]
                    else:
                        layer_result[k] = head(dist_status).squeeze(-1)
                layer_results.append(layer_result)

            if not self.use_mid_output:
                layer_results = layer_results[-1:]
            refinement_dict[-1]['layer_results'] = layer_results

            last_layer_result = layer_results[-1]
            refinement_metrics = self._config.lab.refinement_metrics
            if refinement_metrics == 'all':
                if self._config.lab.adjust_refinement_score_weight:
                    revised_times = 3.0
                else:
                    revised_times = 1.0
                scores = (
                        0.1 * last_layer_result['traffic_light_compliance'].sigmoid().log() +
                        0.5 * last_layer_result['no_at_fault_collisions'].sigmoid().log() +
                        revised_times * 0.5 * last_layer_result['drivable_area_compliance'].sigmoid().log() +
                        0.3 * last_layer_result['driving_direction_compliance'].sigmoid().log() +
                        6.0 * (5.0 * last_layer_result['time_to_collision_within_bound'].sigmoid() +
                               revised_times * 5.0 * last_layer_result['ego_progress'].sigmoid() +
                               revised_times * 2.0 * last_layer_result['lane_keeping'].sigmoid() +
                               1.0 * last_layer_result['history_comfort'].sigmoid()
                               ).log()
                )
            elif refinement_metrics == 'dac_ep_lk':
                scores = (
                        0.5 * last_layer_result['drivable_area_compliance'].sigmoid().log() +
                        6.0 * (
                                5.0 * last_layer_result['ego_progress'].sigmoid() +
                                2.0 * last_layer_result['lane_keeping'].sigmoid()
                        ).log()
                )
            elif refinement_metrics == 'dac_ep_lk_pdms':
                scores = (
                        0.4 * last_layer_result['pdm_score'].sigmoid().log() +
                        0.5 * last_layer_result['drivable_area_compliance'].sigmoid().log() +
                        6.0 * (
                                5.0 * last_layer_result['ego_progress'].sigmoid() +
                                2.0 * last_layer_result['lane_keeping'].sigmoid()
                        ).log()
                )
            if self._config.lab.use_imi_learning_in_refinement:
                scores += 0.02 * last_layer_result['imi'].softmax(-1).log()
            if self._config.lab.optimize_prev_frame_traj_for_ec:
                scores += 0.008 * last_layer_result['imi_prev'].softmax(-1).log()

            if i != self.num_stage - 1:
                _next_layer_dict = {}
                _next_topk = self.topks[i + 1]
                _, select_indices = torch.topk(scores, k=_next_topk, dim=1)
                batch_indices = torch.arange(B, device=select_indices.device).view(-1, 1).expand(-1, _next_topk)

                _next_layer_dict["trajs"] = refinement_dict[-1]['trajs'][batch_indices, select_indices]
                _next_layer_dict["trajs_status"] = tr_out_lst[-1][batch_indices, select_indices]
                _next_layer_dict['indices_absolute'] = refinement_dict[-1]['indices_absolute'][
                    batch_indices, select_indices]
                if self.use_offset_refinement:
                    _next_layer_dict['coarse_score'] = {}
                    for score_key in self.multi_stage_heads[0].keys():
                        _next_layer_dict['coarse_score'][score_key] = last_layer_result[score_key][
                            batch_indices, select_indices]

                refinement_dict.append(_next_layer_dict)

            else:
                select_indices = scores.argmax(1)
                batch_indices = torch.arange(B, device=select_indices.device)
                final_traj = refinement_dict[-1]['trajs'][batch_indices, select_indices]
        refinement_dict[-1]['scores'] = scores

        return final_traj


class TransformerDecoder_v2(nn.TransformerDecoder):

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        output = tgt

        output_lst = []

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

            output_lst.append(output)

        return output_lst


def _inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def cluster_interp_tensor_batch(trajs_tensor, target_num=1024, alpha_list=[0.25, 0.5, 0.75]):
    """
    trajs_tensor: [bs, topk, 40, 3]
    return: [bs, target_num, 40, 3]
    """
    assert trajs_tensor.ndim == 4 and trajs_tensor.shape[-2:] == (40, 3), "Input must be [bs, topk, 40, 3]"
    bs, N, T, D = trajs_tensor.shape
    device = trajs_tensor.device
    result = []

    n_clusters = int((target_num * 2 / len(alpha_list)) ** 0.5) + 1

    for b in range(bs):
        trajs_np = trajs_tensor[b].cpu().numpy()  # [N, 40, 3]
        flat = trajs_np.reshape(N, -1)

        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=5).fit(flat)
        centers = torch.tensor(kmeans.cluster_centers_.reshape(n_clusters, T, D), dtype=trajs_tensor.dtype)

        # 组合 (i, j)
        pair_indices = list(combinations(range(n_clusters), 2))
        pair_i = torch.tensor([i for i, j in pair_indices])
        pair_j = torch.tensor([j for i, j in pair_indices])

        interpolated = []
        for alpha in alpha_list:
            interp = (1 - alpha) * centers[pair_i] + alpha * centers[pair_j]  # [num_pairs, 40, 3]
            interpolated.append(interp)

        all_interp = torch.cat(interpolated, dim=0)  # [M, 40, 3]

        if all_interp.shape[0] >= target_num:
            final = all_interp[:target_num]
        else:
            repeat_factor = (target_num + all_interp.shape[0] - 1) // all_interp.shape[0]
            final = all_interp.repeat((repeat_factor, 1, 1))[:target_num]

        result.append(final)

    return torch.stack(result, dim=0).to(device)  # [bs, target_num, 40, 3]

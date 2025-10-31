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

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler

from navsim.agents.dp.dp_config import DPConfig
from navsim.agents.gtrs_dense.hydra_backbone_bev import HydraBackboneBEV

x_diff_min = -1.2698211669921875
x_diff_max = 7.475563049316406
x_diff_mean = 2.950225591659546

# Y difference statistics
y_diff_min = -5.012081146240234
y_diff_max = 4.8563690185546875
y_diff_mean = 0.0607292577624321

# Calculate scaling factors for differences
x_diff_scale = max(abs(x_diff_max - x_diff_mean), abs(x_diff_min - x_diff_mean))
y_diff_scale = max(abs(y_diff_max - y_diff_mean), abs(y_diff_min - y_diff_mean))

HORIZON = 8
ACTION_DIM = 4
ACTION_DIM_ORI = 3


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class SimpleDiffusionTransformer(nn.Module):
    def __init__(self, d_model, nhead, d_ffn, dp_nlayers, input_dim, obs_len):
        super().__init__()
        self.dp_transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model, nhead, d_ffn,
                dropout=0.0, batch_first=True
            ), dp_nlayers
        )
        self.input_emb = nn.Linear(input_dim, d_model)
        self.time_emb = SinusoidalPosEmb(d_model)
        self.ln_f = nn.LayerNorm(d_model)
        self.output_emb = nn.Linear(d_model, input_dim)
        token_len = obs_len + 1
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, token_len, d_model))
        self.pos_emb = nn.Parameter(torch.zeros(1, 1, d_model))
        self.apply(self._init_weights)

    def _init_weights(self, module):
        ignore_types = (nn.Dropout,
                        SinusoidalPosEmb,
                        nn.TransformerEncoderLayer,
                        nn.TransformerDecoderLayer,
                        nn.TransformerEncoder,
                        nn.TransformerDecoder,
                        nn.ModuleList,
                        nn.Mish,
                        nn.Sequential)
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                'in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)

            bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, SimpleDiffusionTransformer):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))

    def forward(self,
                sample,
                timestep,
                cond):
        B, HORIZON, DIM = sample.shape
        sample = sample.view(B, -1).float()
        input_emb = self.input_emb(sample)

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        time_emb = self.time_emb(timesteps).unsqueeze(1)
        # (B,To,n_emb)
        cond_embeddings = torch.cat([time_emb, cond], dim=1)
        tc = cond_embeddings.shape[1]
        position_embeddings = self.cond_pos_emb[
                              :, :tc, :
                              ]  # each position maps to a (learnable) vector
        x = cond_embeddings + position_embeddings
        memory = x
        # (B,T_cond,n_emb)

        # decoder
        token_embeddings = input_emb.unsqueeze(1)
        t = token_embeddings.shape[1]
        position_embeddings = self.pos_emb[
                              :, :t, :
                              ]  # each position maps to a (learnable) vector
        x = token_embeddings + position_embeddings
        # (B,T,n_emb)
        x = self.dp_transformer(
            tgt=x,
            memory=memory,
        )
        # (B,T,n_emb)
        x = self.ln_f(x)
        x = self.output_emb(x)
        return x.squeeze(1).view(B, HORIZON, DIM)


def diff_traj(traj):
    B, L, _ = traj.shape
    sin = traj[..., -1:].sin()
    cos = traj[..., -1:].cos()
    zero_pad = torch.zeros((B, 1, 1), dtype=traj.dtype, device=traj.device)
    x_diff = traj[..., 0:1].diff(n=1, dim=1, prepend=zero_pad)
    x_diff = x_diff - x_diff_mean
    x_diff_range = max(abs(x_diff_max - x_diff_mean), abs(x_diff_min - x_diff_mean))
    x_diff_norm = x_diff / x_diff_range

    zero_pad = torch.zeros((B, 1, 1), dtype=traj.dtype, device=traj.device)
    y_diff = traj[..., 1:2].diff(n=1, dim=1, prepend=zero_pad)
    y_diff = y_diff - y_diff_mean
    y_diff_range = max(abs(y_diff_max - y_diff_mean), abs(y_diff_min - y_diff_mean))
    y_diff_norm = y_diff / y_diff_range

    return torch.cat([x_diff_norm, y_diff_norm, sin, cos], -1)


def cumsum_traj(norm_trajs):
    B, L, _ = norm_trajs.shape
    sin_values = norm_trajs[..., 2:3]
    cos_values = norm_trajs[..., 3:4]
    heading = torch.atan2(sin_values, cos_values)

    # Denormalize x differences
    x_diff_range = max(abs(x_diff_max - x_diff_mean), abs(x_diff_min - x_diff_mean))
    x_diff = norm_trajs[..., 0:1] * x_diff_range + x_diff_mean

    # Denormalize y differences
    y_diff_range = max(abs(y_diff_max - y_diff_mean), abs(y_diff_min - y_diff_mean))
    y_diff = norm_trajs[..., 1:2] * y_diff_range + y_diff_mean

    # Cumulative sum to get absolute positions
    x = x_diff.cumsum(dim=1)
    y = y_diff.cumsum(dim=1)

    return torch.cat([x, y, heading], -1)


class DPHead(nn.Module):
    def __init__(self, num_poses: int, d_ffn: int, d_model: int, vocab_path: str,
                 nhead: int, nlayers: int, config: DPConfig = None
                 ):
        super().__init__()
        self.config = config
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=config.denoising_timesteps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule='squaredcos_cap_v2',
            variance_type='fixed_small',
            clip_sample=True,
            clip_sample_range=1.0,
            prediction_type='epsilon'
        )
        img_num = 2 if config.use_back_view else 1

        self.transformer_dp = SimpleDiffusionTransformer(
            d_model, nhead, d_ffn, config.dp_layers,
            input_dim=ACTION_DIM * HORIZON,
            obs_len=config.img_vert_anchors * config.img_horz_anchors * img_num + 1,
        )
        self.num_inference_steps = self.noise_scheduler.config.num_train_timesteps

    def forward(self, kv) -> Dict[str, torch.Tensor]:
        B = kv.shape[0]
        result = {}
        if not self.training:
            NUM_PROPOSALS = self.config.num_proposals

            condition = kv.repeat_interleave(NUM_PROPOSALS, dim=0)

            noise = torch.randn(
                size=(B * NUM_PROPOSALS, HORIZON, ACTION_DIM),
                dtype=condition.dtype,
                device=condition.device,
            )

            self.noise_scheduler.set_timesteps(self.num_inference_steps)

            for t in self.noise_scheduler.timesteps:
                model_output = self.transformer_dp(
                    noise,
                    t,
                    condition
                )
                noise = self.noise_scheduler.step(
                    model_output, t, noise
                ).prev_sample
            traj = cumsum_traj(noise)
            result['dp_pred'] = traj.view(B, NUM_PROPOSALS, HORIZON, ACTION_DIM_ORI)

        return result

    def get_dp_loss(self, kv, gt_trajectory):
        B = kv.shape[0]
        device = kv.device
        gt_trajectory = gt_trajectory.float()
        gt_trajectory = diff_traj(gt_trajectory)

        noise = torch.randn(gt_trajectory.shape, device=device, dtype=torch.float)
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_dp_input = self.noise_scheduler.add_noise(
            gt_trajectory, noise, timesteps
        )

        # Predict the noise residual
        pred = self.transformer_dp(
            noisy_dp_input,
            timesteps,
            kv
        )
        return F.mse_loss(pred, noise)


class DPModel(nn.Module):
    def __init__(self, config: DPConfig):
        super().__init__()
        self._config = config
        self._backbone = HydraBackboneBEV(config)

        kv_len = self._backbone.bev_w * self._backbone.bev_h
        emb_len = kv_len + 1
        if self._config.use_hist_ego_status:
            emb_len += 1
        self._keyval_embedding = nn.Embedding(
            emb_len, config.tf_d_model
        )  # 8x8 feature grid + trajectory

        # usually, the BEV features are variable in size.
        self.downscale_layer = nn.Linear(self._backbone.img_feat_c, config.tf_d_model)
        self._bev_semantic_head = nn.Sequential(
            nn.Conv2d(
                config.bev_features_channels,
                config.bev_features_channels,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1),
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                config.bev_features_channels,
                config.num_bev_classes,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.Upsample(
                size=(
                    config.lidar_resolution_height // 2,
                    config.lidar_resolution_width,
                ),
                mode="bilinear",
                align_corners=False,
            ),
        )

        self._status_encoding = nn.Linear((4 + 2 + 2) * config.num_ego_status, config.tf_d_model)
        if self._config.use_hist_ego_status:
            self._hist_status_encoding = nn.Linear((2 + 2 + 3) * 3, config.tf_d_model)

        self._trajectory_head = DPHead(
            num_poses=config.trajectory_sampling.num_poses,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
            nhead=config.vadv2_head_nhead,
            nlayers=config.vadv2_head_nlayers,
            vocab_path=config.vocab_path,
            config=config
        )
        if self._config.use_temporal_bev_kv:
            self.temporal_bev_fusion = nn.Conv2d(
                config.tf_d_model * 2,
                config.tf_d_model,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                bias=True,
            )

    def forward(self, features: Dict[str, torch.Tensor],
                interpolated_traj=None) -> Dict[str, torch.Tensor]:
        camera_feature: torch.Tensor = features["camera_feature"]
        camera_feature_back: torch.Tensor = features["camera_feature_back"]
        status_feature: torch.Tensor = features["status_feature"][0]

        batch_size = status_feature.shape[0]
        assert (camera_feature[-1].shape[0] == batch_size)

        camera_feature_curr = camera_feature[-1]
        if isinstance(camera_feature_back, list):
            camera_feature_back_curr = camera_feature_back[-1]
        else:
            camera_feature_back_curr = camera_feature_back
        img_tokens, bev_tokens, up_bev = self._backbone(camera_feature_curr, camera_feature_back_curr)
        keyval = self.downscale_layer(bev_tokens)
        assert not self._config.use_temporal_bev_kv
        if self._config.use_temporal_bev_kv:
            with torch.no_grad():
                camera_feature_prev = camera_feature[-2]
                camera_feature_back_prev = camera_feature_back[-2]
                img_tokens, bev_tokens, up_bev = self._backbone(camera_feature_prev, camera_feature_back_prev)
                keyval_prev = self.downscale_layer(bev_tokens)
            # grad for fusion layer
            C = keyval.shape[-1]
            keyval = self.temporal_bev_fusion(
                torch.cat([
                    keyval.permute(0, 2, 1).view(batch_size, C, self._backbone.bev_h, self._backbone.bev_w),
                    keyval_prev.permute(0, 2, 1).view(batch_size, C, self._backbone.bev_h, self._backbone.bev_w)
                ], 1)
            ).view(batch_size, C, -1).permute(0, 2, 1).contiguous()

        bev_semantic_map = self._bev_semantic_head(up_bev)
        if self._config.num_ego_status == 1 and status_feature.shape[1] == 32:
            status_encoding = self._status_encoding(status_feature[:, :8])
        else:
            status_encoding = self._status_encoding(status_feature)

        keyval = torch.concatenate([keyval, status_encoding[:, None]], dim=1)
        if self._config.use_hist_ego_status:
            hist_status_encoding = self._hist_status_encoding(features['hist_status_feature'])
            keyval = torch.concatenate([keyval, hist_status_encoding[:, None]], dim=1)

        keyval += self._keyval_embedding.weight[None, ...]

        output: Dict[str, torch.Tensor] = {}
        trajectory = self._trajectory_head(keyval)

        output.update(trajectory)

        output['env_kv'] = keyval
        output['bev_semantic_map'] = bev_semantic_map

        return output

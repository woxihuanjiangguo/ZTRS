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

import logging

import torch
from torch import nn

from navsim.agents.gtrs_aug.hydra_config_aug import HydraConfigAug

logger = logging.getLogger("aug_meta_arch")


class AugMetaArch(nn.Module):
    def __init__(self, cfg: HydraConfigAug, teacher_model, student_model):
        super().__init__()
        self.cfg = cfg
        self.fp16_scaler = None

        student_model_dict = dict()
        teacher_model_dict = dict()

        student_model_dict["model"] = teacher_model
        teacher_model_dict["model"] = student_model
        embed_dim = teacher_model._backbone.img_feat_c
        logger.info(f"OPTIONS -- architecture : embed_dim: {embed_dim}")

        self.embed_dim = embed_dim
        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)

        # there is no backpropagation through the teacher, so no need for gradients
        for p in self.teacher.parameters():
            p.requires_grad = False
        logger.info(f"Student and Teacher are built: they are both {type(self.student.model)} network.")

    def backprop_loss(self, loss):
        if self.fp16_scaler is not None:
            self.fp16_scaler.scale(loss).backward()
        else:
            loss.backward()

    def forward(self, teacher_ori_features, student_feat_dict_lst, **kwargs):
        # teacher output
        @torch.no_grad()
        def get_teacher_output():
            teacher_output_dict = self.teacher.model(teacher_ori_features)
            """
            teacher_backbone_output_dict:
            {
                "x_norm_clstoken": x_norm[:, 0],
                "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
                "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
                "x_prenorm": x,
                "masks": masks,
            }
            """
            return teacher_output_dict

        if self.cfg.training:
            teacher_pred = get_teacher_output()
        else:
            if self.cfg.inference.model == "teacher":
                teacher_pred = self.teacher.model(teacher_ori_features, **kwargs)
            else:
                teacher_pred = self.student.model(teacher_ori_features, **kwargs)
        if not self.cfg.training:
            return teacher_pred, [], {}

        if self.cfg.lab.optimize_prev_frame_traj_for_ec:
            teacher_pred = {'cur': teacher_pred}
            teacher_prev_feat = {
                'camera_feature': [teacher_ori_features['camera_feature'][-2], ],
                'status_feature': [teacher_ori_features['status_feature'][1], ],
            }
            prev_pred = self.teacher.model(teacher_prev_feat, **kwargs)
            teacher_pred['prev'] = prev_pred

        loss_dict = {}

        student_preds = self.student.model.forward_features_list(
            student_feat_dict_lst,
            mask_list=[kwargs.get('collated_masks', None)] + [None] * len(student_feat_dict_lst[1:]),
        )

        return teacher_pred, student_preds, loss_dict

    def update_teacher(self, m):
        with torch.no_grad():
            for k in self.student.keys():
                for stu_params, tea_params in zip(self.student[k].parameters(), self.teacher[k].parameters()):
                    tea_params.data.mul_(m).add_(stu_params.data, alpha=1 - m)
                if self.cfg.backbone_type in ('resnet34', 'resnet50') or self.cfg.lab.update_buffer_in_ema:
                    # update buffers (e.g., running_mean/var in BatchNorm)
                    for stu_buf, tea_buf in zip(self.student[k].buffers(), self.teacher[k].buffers()):
                        tea_buf.data.copy_(stu_buf.data)

    def train(self, mode=True):
        if mode:
            super().train()
            self.teacher.eval()
        else:
            self.teacher.eval()
            self.student.eval()

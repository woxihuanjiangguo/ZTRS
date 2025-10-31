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

"""
Implements the TransFuser vision backbone.
"""

from torch import nn

from navsim.agents.backbones.vov import VoVNet
from navsim.agents.gtrs_aug.hydra_config_aug import HydraConfigAug


class HydraBackbone(nn.Module):
    """
    Multi-scale Fusion Transformer for image + LiDAR feature fusion
    """

    def __init__(self, config: HydraConfigAug):

        super().__init__()
        self.config = config
        self.backbone_type = config.backbone_type
        if config.backbone_type == 'vov':
            self.image_encoder = VoVNet(
                spec_name='V-99-eSE',
                out_features=['stage4', 'stage5'],
                norm_eval=True,
                with_cp=True,
                init_cfg=dict(
                    type='Pretrained',
                    checkpoint=config.vov_ckpt,
                    prefix='img_backbone.'
                )
            )
            # scale_4_c = 1024
            vit_channels = 1024
            self.image_encoder.init_weights()
        else:
            raise ValueError

        self.avgpool_img = nn.AdaptiveAvgPool2d(
            (self.config.img_vert_anchors, self.config.img_horz_anchors)
        )
        self.img_feat_c = vit_channels

    def forward(self, image):
        B, C, H, W = image.shape
        if self.backbone_type == 'vov':
            image_features = self.image_encoder(image)[-1]
        else:
            raise ValueError('Forward wrong backbone')
        return self.avgpool_img(image_features)

    def forward_tup(self, image, **kwargs):

        image_feat = self.image_encoder(image)[-1]
        class_feat = image_feat.mean(dim=(-1, -2))
        image_feat_tup = (image_feat, class_feat)

        if self.config.lab.use_higher_res_feat_in_refinement:
            return (self.avgpool_img(image_feat_tup[0]), image_feat_tup[1], image_feat_tup[0])
        else:
            return (self.avgpool_img(image_feat_tup[0]), image_feat_tup[1])

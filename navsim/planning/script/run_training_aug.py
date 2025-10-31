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

import datetime
import logging
from functools import partial
from pathlib import Path
from typing import Tuple

import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.gtrs_aug.data.collate import collate_data_and_cast
from navsim.agents.gtrs_aug.data.masking import MaskingGenerator
from navsim.agents.gtrs_aug.hydra_config_aug import HydraConfigAug
from navsim.agents.transfuser.transfuser_agent import TransfuserAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.agent_lightning_module_aug import AgentLightningModuleAug
from navsim.planning.training.dataset_aug import CacheOnlyDataset, DatasetAug

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


def build_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[DatasetAug, DatasetAug]:
    """
    Builds training and validation datasets from omega config
    :param cfg: omegaconf dictionary
    :param agent: interface of agents in NAVSIM
    :return: tuple for training and validation dataset
    """
    train_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if train_scene_filter.log_names is not None:
        train_scene_filter.log_names = [
            log_name
            for log_name in train_scene_filter.log_names
            if log_name in cfg.train_logs
        ]
    else:
        train_scene_filter.log_names = cfg.train_logs

    val_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if val_scene_filter.log_names is not None:
        val_scene_filter.log_names = [
            log_name
            for log_name in val_scene_filter.log_names
            if log_name in cfg.val_logs
        ]
    else:
        val_scene_filter.log_names = cfg.val_logs

    data_path = Path(cfg.navsim_log_path)
    original_sensor_path = Path(cfg.original_sensor_path)

    train_scene_loader = SceneLoader(
        original_sensor_path=original_sensor_path,
        data_path=data_path,
        scene_filter=train_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    val_scene_loader = SceneLoader(
        original_sensor_path=original_sensor_path,
        data_path=data_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    train_data = DatasetAug(
        scene_loader=train_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cfg=cfg.agent.config,
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    val_data = DatasetAug(
        scene_loader=val_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cfg=cfg.agent.config,
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    return train_data, val_data


def build_dataloaders(train_data, val_data, cfg):
    agent_cfg: HydraConfigAug = cfg.agent.config

    if not agent_cfg.use_mask_loss:
        logger.info("Building Datasets")
        train_dataloader = DataLoader(train_data, **cfg.dataloader.params, shuffle=True)
        logger.info("Num training samples: %d", len(train_data))
        val_dataloader = DataLoader(val_data, **cfg.dataloader.params, shuffle=False)
        logger.info("Num validation samples: %d", len(val_data))
        return train_dataloader, val_dataloader
    else:
        mask_generator = MaskingGenerator(
            input_size=(agent_cfg.img_vert_anchors, agent_cfg.img_horz_anchors),
            max_num_patches=0.5 * agent_cfg.img_vert_anchors * agent_cfg.img_horz_anchors,
        )
        collate_fn = partial(
            collate_data_and_cast,
            mask_ratio_tuple=agent_cfg.ibot.mask_ratio_min_max,
            mask_probability=agent_cfg.ibot.mask_sample_probability,
            n_tokens=agent_cfg.img_vert_anchors * agent_cfg.img_horz_anchors,
            mask_generator=mask_generator,
            dtype=torch.float,
        )
        logger.info("Building Datasets")
        train_dataloader = DataLoader(train_data, **cfg.dataloader.params, shuffle=True, collate_fn=collate_fn)
        logger.info("Num training samples: %d", len(train_data))
        val_dataloader = DataLoader(val_data, **cfg.dataloader.params, shuffle=False, collate_fn=collate_fn)
        logger.info("Num validation samples: %d", len(val_data))
        return train_dataloader, val_dataloader


def build_model(config, only_teacher=False):
    logger.info("Building Teacher Agent")
    agent = instantiate(config.agent)
    lightening_module = AgentLightningModuleAug(
        config.agent.config,
        agent=agent,
    )
    return agent, lightening_module


def build_model_from_cfg(cfg, only_teacher=False):
    return build_model(cfg, only_teacher=only_teacher)


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for training an agent.
    :param cfg: omegaconf dictionary
    """
    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")

    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    agent, lightning_module = build_model_from_cfg(cfg)

    if cfg.use_cache_without_dataset:
        logger.info("Using cached data without building SceneLoader")
        assert (
            not cfg.force_cache_computation
        ), "force_cache_computation must be False when using cached data without building SceneLoader"
        assert (
                cfg.cache_path is not None
        ), "cache_path must be provided when using cached data without building SceneLoader"
        train_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.train_logs,
        )
        val_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.val_logs,
        )
    else:
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(cfg, agent)

    train_dataloader, val_dataloader = build_dataloaders(train_data, val_data, cfg)

    logger.info("Building Trainer")
    if isinstance(agent, TransfuserAgent):
        trainer = pl.Trainer(**cfg.trainer.params,
                             callbacks=agent.get_training_callbacks())
    else:
        trainer = pl.Trainer(**cfg.trainer.params,
                             callbacks=agent.get_training_callbacks(),
                             strategy=DDPStrategy(static_graph=True,
                                                  timeout=datetime.timedelta(seconds=3600)))

    logger.info("Starting Training")
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=cfg.get('resume_ckpt_path', None)
    )


if __name__ == "__main__":
    main()

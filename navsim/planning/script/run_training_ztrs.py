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
from pathlib import Path
from typing import Tuple

import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.transfuser.transfuser_agent import TransfuserAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from navsim.planning.training.dataset_ztrs import CacheOnlyDataset, Dataset


def custom_collate(batch):
    """
    Custom collate for Dataset that may include EgoState.
    Keeps EgoState objects as lists instead of collating into tensors.
    """
    # check if samples contain ego state
    if isinstance(batch[0], tuple):
        if len(batch[0]) == 4:  # (features, targets, token, prev_privileged_ego_state)
            features = [b[0] for b in batch]
            targets = [b[1] for b in batch]
            tokens = [b[2] for b in batch]
            ego_states = [b[3] for b in batch]  # keep as list

            return (
                default_collate(features),
                default_collate(targets),
                default_collate(tokens),
                ego_states,  # don’t collate, just return list
            )

        elif len(batch[0]) == 3:  # (features, targets, token)
            features = [b[0] for b in batch]
            targets = [b[1] for b in batch]
            tokens = [b[2] for b in batch]

            return (
                default_collate(features),
                default_collate(targets),
                default_collate(tokens),
            )

    # fallback: try default collate
    return default_collate(batch)


logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


def build_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[Dataset, Dataset]:
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
        val_scene_filter.log_names = [log_name for log_name in val_scene_filter.log_names if log_name in cfg.val_logs]
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

    return_prev_es = (hasattr(agent, '_config') and
                      hasattr(agent._config, 'ec_target') and
                      agent._config.ec_target is True)
    train_data = Dataset(
        scene_loader=train_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
        return_prev_es=return_prev_es
    )

    val_data = Dataset(
        scene_loader=val_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
        return_prev_es=return_prev_es
    )

    return train_data, val_data


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for training an agent.
    :param cfg: omegaconf dictionary
    """

    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")

    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)

    if agent._checkpoint_path is not None:
        agent.initialize()

    logger.info("Building Lightning Module")
    lightning_module = AgentLightningModule(
        agent=agent,
    )

    if cfg.use_cache_without_dataset:
        return_prev_es = (hasattr(agent, '_config') and
                          hasattr(agent._config, 'ec_target') and
                          agent._config.ec_target is True)
        logger.info("Using cached data without building SceneLoader")
        assert (
            not cfg.force_cache_computation
        ), "force_cache_computation must be False when using cached data without building SceneLoader"
        assert (
                cfg.cache_path is not None
        ), "cache_path must be provided when using cached data without building SceneLoader"
        if return_prev_es:
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
                val_scene_filter.log_names = [log_name for log_name in val_scene_filter.log_names if
                                              log_name in cfg.val_logs]
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
        else:
            train_scene_loader = None
            val_scene_loader = None
        train_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.train_logs,
            return_prev_es=return_prev_es,
            scene_loader=train_scene_loader
        )
        val_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.val_logs,
            return_prev_es=return_prev_es,
            scene_loader=val_scene_loader
        )
    else:
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(cfg, agent)

    logger.info("Building Datasets")
    train_dataloader = DataLoader(train_data, **cfg.dataloader.params, shuffle=True, collate_fn=custom_collate)
    logger.info("Num training samples: %d", len(train_data))
    val_dataloader = DataLoader(val_data, **cfg.dataloader.params, shuffle=False, collate_fn=custom_collate)
    logger.info("Num validation samples: %d", len(val_data))

    logger.info("Building Trainer")
    if isinstance(agent, TransfuserAgent):
        trainer = pl.Trainer(**cfg.trainer.params,
                             callbacks=agent.get_training_callbacks())
    else:
        trainer = pl.Trainer(**cfg.trainer.params,
                             callbacks=agent.get_training_callbacks(),
                             strategy=DDPStrategy(static_graph=True,
                                                  timeout=datetime.timedelta(seconds=3600),
                                                  find_unused_parameters=True))

    logger.info("Starting Training")
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=cfg.get('resume_ckpt_path', None)
    )


if __name__ == "__main__":
    main()

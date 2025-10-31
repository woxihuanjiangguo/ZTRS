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

import torch
import torch.nn.functional as F

from navsim.agents.gtrs_aug.hydra_config_aug import HydraConfigAug


def hydra_kd_imi_agent_loss_robust(
        targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor], config: HydraConfigAug,
        vocab_pdm_score
):
    """
    Helper function calculating complete loss of Transfuser
    :param targets: dictionary of name tensor pairings
    :param predictions: dictionary of name tensor pairings
    :param config: global Transfuser config
    :return: combined loss value
    """
    # if os.getenv('ROBUST_HYDRA_DEBUG') == 'true':
    #     import pdb; pdb.set_trace()

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
    imi = predictions['imi']
    # if os.getenv('ROBUST_HYDRA_DEBUG') == 'true':
    #     import pdb; pdb.set_trace()
    _dtype = imi.dtype

    # 2 cls
    da_loss = F.binary_cross_entropy_with_logits(drivable_area_compliance,
                                                 vocab_pdm_score['drivable_area_compliance'].to(_dtype))
    ttc_loss = F.binary_cross_entropy_with_logits(time_to_collision_within_bound,
                                                  vocab_pdm_score['time_to_collision_within_bound'].to(_dtype))
    noc_loss = F.binary_cross_entropy_with_logits(no_at_fault_collisions, three_to_two_classes(
        vocab_pdm_score['no_at_fault_collisions'].to(_dtype)))
    progress_loss = F.binary_cross_entropy_with_logits(ego_progress, vocab_pdm_score['ego_progress'].to(_dtype))
    # expansion
    ddc_loss = F.binary_cross_entropy_with_logits(driving_direction_compliance, three_to_two_classes(
        vocab_pdm_score['driving_direction_compliance'].to(_dtype)))
    lk_loss = F.binary_cross_entropy_with_logits(lane_keeping, vocab_pdm_score['lane_keeping'].to(_dtype))
    tl_loss = F.binary_cross_entropy_with_logits(traffic_light_compliance,
                                                 vocab_pdm_score['traffic_light_compliance'].to(_dtype))

    comfort_loss = F.binary_cross_entropy_with_logits(history_comfort,
                                                      vocab_pdm_score['history_comfort'].to(_dtype))
    vocab = predictions["trajectory_vocab"]
    # B, 8 (4 secs, 0.5Hz), 3
    target_traj = targets["trajectory"]
    # 4, 9, ..., 39
    sampled_timepoints = [5 * k - 1 for k in range(1, 9)]
    B = target_traj.shape[0]
    l2_distance = -((vocab[:, sampled_timepoints][None].repeat(B, 1, 1, 1) - target_traj[:, None]) ** 2) / config.sigma
    """
    vocab: [vocab_size, 40, 3]
    vocab[:, sampled_timepoints]: [vocab_size, 8, 3]
    vocab[:, sampled_timepoints][None].repeat(B, 1, 1, 1): [b, vocab_size, 8, 3]
    target_traj[:, None]: [b, 1, 8, 3]
    l2_distance: [b, vocab_size, 8, 3]
    """
    imi_loss = F.cross_entropy(imi, l2_distance.sum((-2, -1)).softmax(1))

    imi_loss_final = config.trajectory_imi_weight * imi_loss

    noc_loss_final = config.trajectory_pdm_weight['no_at_fault_collisions'] * noc_loss
    da_loss_final = config.trajectory_pdm_weight['drivable_area_compliance'] * da_loss
    ttc_loss_final = config.trajectory_pdm_weight['time_to_collision_within_bound'] * ttc_loss
    progress_loss_final = config.trajectory_pdm_weight['ego_progress'] * progress_loss
    ddc_loss_final = config.trajectory_pdm_weight['driving_direction_compliance'] * ddc_loss
    lk_loss_final = config.trajectory_pdm_weight['lane_keeping'] * lk_loss
    tl_loss_final = config.trajectory_pdm_weight['traffic_light_compliance'] * tl_loss
    comfort_loss_final = config.trajectory_pdm_weight['history_comfort'] * comfort_loss

    # agent_class_loss, agent_box_loss = _agent_loss(targets, predictions, config)

    # agent_class_loss_final = config.agent_class_weight * agent_class_loss
    # agent_box_loss_final = config.agent_box_weight * agent_box_loss
    loss = (
            imi_loss_final
            + noc_loss_final
            + da_loss_final
            + ttc_loss_final
            + progress_loss_final
            + ddc_loss_final
            + lk_loss_final
            + tl_loss_final
            + comfort_loss_final
    )
    return loss, {
        'imi_loss': imi_loss_final,
        'pdm_noc_loss': noc_loss_final,
        'pdm_da_loss': da_loss_final,
        'pdm_ttc_loss': ttc_loss_final,
        'pdm_progress_loss': progress_loss_final,
        'pdm_ddc_loss': ddc_loss_final,
        'pdm_lk_loss': lk_loss_final,
        'pdm_tl_loss': tl_loss_final,
        'pdm_comfort_loss': comfort_loss_final
    }


def hydra_kd_imi_agent_loss_single_stage(
        predictions: Dict[str, torch.Tensor], config: HydraConfigAug, vocab_pdm_score, targets=None
):
    """
    Helper function calculating complete loss of Transfuser
    :param targets: dictionary of name tensor pairings
    :param predictions: dictionary of name tensor pairings
    :param config: global Transfuser config
    :return: combined loss value
    """

    # if os.getenv('ROBUST_HYDRA_DEBUG') == 'true':
    #     import pdb; pdb.set_trace()

    layer_results = predictions['layer_results']
    losses = {}
    total_loss = 0.0

    refinement_metrics = config.lab.refinement_metrics

    for layer, layer_result in enumerate(layer_results):

        if refinement_metrics == 'all':
            no_at_fault_collisions, drivable_area_compliance, time_to_collision_within_bound, ego_progress = (
                layer_result['no_at_fault_collisions'],
                layer_result['drivable_area_compliance'],
                layer_result['time_to_collision_within_bound'],
                layer_result['ego_progress']
            )
            driving_direction_compliance, lane_keeping, traffic_light_compliance = (
                layer_result['driving_direction_compliance'],
                layer_result['lane_keeping'],
                layer_result['traffic_light_compliance']
            )
            history_comfort = layer_result['history_comfort']

            _dtype = drivable_area_compliance.dtype

            da_loss = F.binary_cross_entropy_with_logits(drivable_area_compliance,
                                                         vocab_pdm_score['drivable_area_compliance'].to(_dtype))
            ttc_loss = F.binary_cross_entropy_with_logits(time_to_collision_within_bound,
                                                          vocab_pdm_score['time_to_collision_within_bound'].to(_dtype))
            noc_loss = F.binary_cross_entropy_with_logits(no_at_fault_collisions, three_to_two_classes(
                vocab_pdm_score['no_at_fault_collisions'].to(_dtype)))
            progress_loss = F.binary_cross_entropy_with_logits(ego_progress, vocab_pdm_score['ego_progress'].to(_dtype))
            # expansion
            ddc_loss = F.binary_cross_entropy_with_logits(driving_direction_compliance, three_to_two_classes(
                vocab_pdm_score['driving_direction_compliance'].to(_dtype)))
            lk_loss = F.binary_cross_entropy_with_logits(lane_keeping, vocab_pdm_score['lane_keeping'].to(_dtype))
            tl_loss = F.binary_cross_entropy_with_logits(traffic_light_compliance,
                                                         vocab_pdm_score['traffic_light_compliance'].to(_dtype))

            comfort_loss = F.binary_cross_entropy_with_logits(history_comfort,
                                                              vocab_pdm_score['history_comfort'].to(_dtype))

            noc_loss_final = config.trajectory_pdm_weight['no_at_fault_collisions'] * noc_loss
            da_loss_final = config.trajectory_pdm_weight['drivable_area_compliance'] * da_loss
            ttc_loss_final = config.trajectory_pdm_weight['time_to_collision_within_bound'] * ttc_loss
            progress_loss_final = config.trajectory_pdm_weight['ego_progress'] * progress_loss
            ddc_loss_final = config.trajectory_pdm_weight['driving_direction_compliance'] * ddc_loss
            lk_loss_final = config.trajectory_pdm_weight['lane_keeping'] * lk_loss
            tl_loss_final = config.trajectory_pdm_weight['traffic_light_compliance'] * tl_loss
            comfort_loss_final = config.trajectory_pdm_weight['history_comfort'] * comfort_loss

            loss = (
                    noc_loss_final
                    + da_loss_final
                    + ttc_loss_final
                    + progress_loss_final
                    + ddc_loss_final
                    + lk_loss_final
                    + tl_loss_final
                    + comfort_loss_final
            )

        else:
            drivable_area_compliance, ego_progress = (
                layer_result['drivable_area_compliance'],
                layer_result['ego_progress']
            )
            lane_keeping = layer_result['lane_keeping']

            _dtype = drivable_area_compliance.dtype

            da_loss = F.binary_cross_entropy_with_logits(drivable_area_compliance,
                                                         vocab_pdm_score['drivable_area_compliance'].to(_dtype))
            progress_loss = F.binary_cross_entropy_with_logits(ego_progress, vocab_pdm_score['ego_progress'].to(_dtype))
            # expansion
            lk_loss = F.binary_cross_entropy_with_logits(lane_keeping, vocab_pdm_score['lane_keeping'].to(_dtype))

            da_loss_final = config.trajectory_pdm_weight['drivable_area_compliance'] * da_loss
            progress_loss_final = config.trajectory_pdm_weight['ego_progress'] * progress_loss
            lk_loss_final = config.trajectory_pdm_weight['lane_keeping'] * lk_loss

            loss = (
                    da_loss_final
                    + progress_loss_final
                    + lk_loss_final
            )

            if refinement_metrics == 'dac_ep_lk_pdms':
                pdm = layer_result['pdm_score']
                pdm_loss = F.binary_cross_entropy_with_logits(pdm,
                                                              vocab_pdm_score['pdm_score'].to(_dtype))
                pdm_loss_final = 2.0 * pdm_loss
                loss += pdm_loss_final

        if config.lab.use_imi_learning_in_refinement:
            imi = layer_result['imi']
            vocab = predictions["trajectory_vocab"]
            # B, 8 (4 secs, 0.5Hz), 3
            target_traj = targets["trajectory"]
            # 4, 9, ..., 39
            sampled_timepoints = [5 * k - 1 for k in range(1, 9)]
            indices_absolute = predictions['indices_absolute']
            l2_distance = -((vocab[:, sampled_timepoints][indices_absolute] - target_traj[:, None]) ** 2) / config.sigma
            """
            vocab: [vocab_size, 40, 3]
            vocab[:, sampled_timepoints]: [vocab_size, 8, 3]
            vocab[:, sampled_timepoints][None].repeat(B, 1, 1, 1): [b, vocab_size, 8, 3]
            target_traj[:, None]: [b, 1, 8, 3]
            l2_distance: [b, vocab_size, 8, 3]
            """
            imi_loss = F.cross_entropy(imi, l2_distance.sum((-2, -1)).softmax(1))

            imi_loss_final = config.trajectory_imi_weight * imi_loss

            loss += imi_loss_final

        if config.lab.adjust_refinement_loss_weight:
            n_cur_traj = drivable_area_compliance.shape[1]
            loss *= n_cur_traj / config.vocab_size

        total_loss += loss
        losses[f'layer_{layer + 1}'] = loss

    return total_loss, losses


def three_to_two_classes(x):
    x[x == 0.5] = 0.0
    return x

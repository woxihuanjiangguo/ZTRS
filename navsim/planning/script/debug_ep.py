import logging
import pickle
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from hydra import compose
from hydra.utils import instantiate
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from omegaconf import OmegaConf
from shapely import Point

from navsim.common.dataloader import MetricCacheLoader
from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
    state_array_to_coords_array,
)
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    BBCoordsIndex,
)
from navsim.traffic_agents_policies.abstract_traffic_agents_policy import AbstractTrafficAgentsPolicy

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score_gpu"

experiment_name = "debug"

Agent = "gtrs_dense_vov"
model_pred_path = "/mnt/d/workdir/GTRS/debug/model_traj_ce2f1ac423965b7a.pkl"
batch_size = 32
precision = 32

vocab_path = "/mnt/d/workdir/GTRS/traj_final/8192.npy"
metric_cache_path = "/mnt/g/navsim_exp_v2/navhard_two_stage_metric_cache"


def calculate_progress(num_proposals, ego_coords, centerline) -> None:
    """
    Re-implementation of nuPlan's progress metric (non-normalized).
    Calculates progress along the centerline.
    """

    # calculate raw progress in meter
    progress_in_meter = np.zeros(num_proposals, dtype=np.float64)
    for proposal_idx in range(num_proposals):
        start_point = Point(*ego_coords[proposal_idx, 0, BBCoordsIndex.CENTER])
        end_point = Point(*ego_coords[proposal_idx, -1, BBCoordsIndex.CENTER])
        progress = centerline.project([start_point, end_point])
        progress_in_meter[proposal_idx] = progress[1] - progress[0]

    progress_raw = np.clip(progress_in_meter, a_min=0, a_max=None)
    print(progress_raw)


# train_test_split = "navhard_two_stage"
# @hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main() -> None:
    """
    Main entrypoint for running PDMS evaluation (single-scene version).
    """
    np.set_printoptions(threshold=np.inf)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    hydra.initialize(config_path=CONFIG_PATH)
    cfg = compose(CONFIG_NAME, overrides=["train_test_split=navhard_two_stage"])
    OmegaConf.set_struct(cfg, False)

    cfg.agent = Agent
    cfg.combined_inference = True
    cfg.trainer.params.precision = precision
    cfg.trainer.params.batch_size = batch_size
    cfg.experiment_name = experiment_name
    cfg.metric_cache_path = metric_cache_path
    # cfg.train_test_split = train_test_split

    # load cache
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    token = "ce2f1ac423965b7a"  #
    metric_cache = metric_cache_loader.get_from_token(token)
    simulator = instantiate(cfg.simulator)
    traffic_agents_policy_stage_one: AbstractTrafficAgentsPolicy = instantiate(
        cfg.traffic_agents_policy.reactive, simulator.proposal_sampling
    )
    scorer: PDMScorer = instantiate(cfg.scorer)

    f = open(model_pred_path, "rb")
    model_trajectory = pickle.load(f)
    print(f"model_trajectory: {model_trajectory}")
    pred_trajectory = transform_trajectory(model_trajectory, metric_cache.ego_state)
    pdm_trajectory = metric_cache.trajectory
    print(f"pdm_trajectory: {pdm_trajectory}")
    initial_ego_state = metric_cache.ego_state
    pdm_states, pred_states = (
        get_trajectory_as_array(pdm_trajectory, simulator.proposal_sampling, initial_ego_state.time_point),
        get_trajectory_as_array(pred_trajectory, simulator.proposal_sampling, initial_ego_state.time_point),
    )
    trajectory_states = np.concatenate([pdm_states[None, ...], pred_states[None, ...]], axis=0)  #

    # simu
    simulated_states = simulator.simulate_proposals(trajectory_states, metric_cache.ego_state)
    print(f"simulated_states: {simulated_states}")
    print(simulated_states.shape)
    ego_coords = state_array_to_coords_array(simulated_states, get_pacifica_parameters())
    metric_cache.map_parameters.map_root = '/mnt/g/navsim/maps'
    simulated_agent_detections_tracks = traffic_agents_policy_stage_one.simulate_environment(simulated_states[1],
                                                                                             metric_cache)
    calculate_progress(2, ego_coords, metric_cache.centerline)

    res = scorer.score_proposals(
        simulated_states,
        metric_cache.observation,
        metric_cache.centerline,
        metric_cache.route_lane_ids,
        metric_cache.drivable_area_map,
        metric_cache.map_parameters,
        simulated_agent_detections_tracks,
        metric_cache.past_human_trajectory
    )[1]
    print(res)


if __name__ == "__main__":
    main()

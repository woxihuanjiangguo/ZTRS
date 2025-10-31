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
from enum import IntEnum
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
from navsim.agents.gtrs_aug.data.transforms import GaussianBlur
from navsim.agents.gtrs_aug.hydra_config_aug import HydraConfigAug
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.maps.abstract_map import AbstractMap, SemanticMapLayer, MapObject
from shapely import affinity
from shapely.geometry import Polygon, LineString
from torchvision import transforms

from navsim.common.dataclasses import AgentInput, Scene, Annotations
from navsim.common.enums import BoundingBoxIndex, LidarIndex
from navsim.planning.scenario_builder.navsim_scenario_utils import tracked_object_types
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)


class HydraAugFeatureBuilder(AbstractFeatureBuilder):
    def __init__(self, config: HydraConfigAug):
        self._config = config
        self.training = config.training

        config_rotation = config.ego_perturb.rotation
        if self._config.training:
            with open(config.ego_perturb.offline_aug_file, 'r') as f:
                aug_data = json.load(f)
            assert aug_data['param']['rot'] == config_rotation.offline_aug_angle_boundary  # TODO: translation boundary
            self.aug_info = aug_data['tokens']

        color_jittering = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )
        self.teacher_ori_augmentation = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.student_ori_augmentation = transforms.Compose(
            [
                transforms.ToTensor(),
                color_jittering,
                GaussianBlur(p=0.1),
                transforms.RandomSolarize(threshold=0.5, p=0.2),
            ]
        )
        self.student_rotated_augmentation = transforms.Compose(
            [
                transforms.ToTensor(),
                color_jittering,
                GaussianBlur(p=0.5),
            ]
        )

        self.va_perturb_enable = config.ego_perturb.va.enable

        config_camera = config.camera_problem
        self.gaussian_enable = config_camera.gaussian_enable
        self.gaussian_mode = config_camera.gaussian_mode
        assert self.gaussian_mode in ('random', 'load_from_offline')
        if self.gaussian_enable and self.gaussian_mode == 'load_from_offline':
            with open(config.camera_problem.gaussian_offline_file) as f:
                gaussian_data = json.load(f)
            self.gaussian_info = gaussian_data['tokens']

        self.weather_enable = config.camera_problem.weather_enable
        if self.weather_enable:
            self.weather_aug_mode = config.camera_problem.weather_aug_mode
            assert self.weather_aug_mode in ('random', 'load_from_offline')
            if self.weather_aug_mode == 'load_from_offline':
                assert self.training is False

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "hydra_feature_ssl"

    def compute_features(self, agent_input: AgentInput, scene: Scene) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        features = {}

        initial_token = scene.scene_metadata.initial_token
        if not self._config.only_ori_input and self._config.training:
            n_rotated = self._config.student_rotation_ensemble
        else:
            n_rotated = 0
        features.update(self._get_camera_feature(agent_input, initial_token,
                                                 rotation_num=n_rotated))  # List[torch.Tensor], tensor.shape == [c, h, w]
        if self._config.use_back_view:
            features["camera_feature_back"] = self._get_camera_feature_back(agent_input)


        ego_status_list = []
        for i in range(self._config.num_ego_status):
            idx = -(i + 1)

            ego_status = torch.concatenate(
                [
                    torch.tensor(agent_input.ego_statuses[idx].driving_command, dtype=torch.float32),
                    torch.tensor(agent_input.ego_statuses[idx].ego_velocity, dtype=torch.float32),
                    torch.tensor(agent_input.ego_statuses[idx].ego_acceleration, dtype=torch.float32)
                ],
            )  # [8,]

            ego_status_list.append(ego_status)

        features["status_feature"] = ego_status_list  # [seq_len, 8]

        return features

    def _get_camera_feature(self, agent_input: AgentInput, initial_token: str, rotation_num=3) -> Dict[
        str, Union[torch.Tensor, List[torch.Tensor]]]:
        """
        Extract stitched camera from AgentInput
        :param agent_input: input dataclass
        :return: stitched front view image as torch tensor, List[torch.Tensor]
        """
        res = dict()
        seq_len = self._config.seq_len
        cameras = agent_input.cameras[-seq_len:]  # List[Cameras]
        assert (len(cameras) == seq_len)

        res['ori_teacher'] = []
        res['ori'] = []
        res['rotated'] = [[] for _ in range(rotation_num)]
        for camera in cameras:
            image = camera.cam_l0.image
            if image is not None and image.size > 0 and np.any(image):
                n_camera = self._config.n_camera

                # Crop to ensure 4:1 aspect ratio
                l0 = camera.cam_l0.image[28:-28, 416:-416]
                f0 = camera.cam_f0.image[28:-28]
                r0 = camera.cam_r0.image[28:-28, 416:-416]
                l1 = camera.cam_l1.image[28:-28]
                r1 = camera.cam_r1.image[28:-28]
                if n_camera >= 5:
                    l2 = camera.cam_l2.image[28:-28, :-1100]
                    r2 = camera.cam_r2.image[28:-28, 1100:]
                    b0_left = camera.cam_b0.image[28:-28, :1080]
                    b0_right = camera.cam_b0.image[28:-28, -1080:]

                if n_camera == 1:
                    ori_image = f0
                elif n_camera == 3:
                    ori_image = np.concatenate([l0, f0, r0], axis=1)
                elif n_camera == 5:
                    ori_image = np.concatenate([l1, l0, f0, r0, r1], axis=1)
                else:
                    raise NotImplementedError

                _ori_image = cv2.resize(ori_image, (self._config.camera_width, self._config.camera_height))
                res['ori_teacher'].append(self.teacher_ori_augmentation(_ori_image))
                student_ori_img = self.student_ori_augmentation(_ori_image)
                res['ori'].append(student_ori_img)


                if n_camera < 5:
                    stitched_image = np.concatenate([l1, l0, f0, r0, r1], axis=1)
                else:
                    stitched_image = np.concatenate([b0_left, l2, l1, l0, f0, r0, r1, r2, b0_right], axis=1)

                img_3cam_w = l0.shape[1] + f0.shape[1] + r0.shape[1]
                l1_w = l1.shape[1]
                r1_w = r1.shape[1]
                whole_w = stitched_image.shape[1]
                half_view_w = img_3cam_w + l1_w // 2 + r1_w // 2
                if os.getenv('ROBUST_HYDRA_DEBUG') == 'true':
                    debug_dir = 'debug_viz'
                    os.makedirs(debug_dir, exist_ok=True)
                    _s_img = Image.fromarray(stitched_image)
                    _s_img.save(f'{debug_dir}/stitched_image.jpg')

                if n_camera == 1:
                    img_w = f0.shape[1]
                elif n_camera == 3:
                    img_w = img_3cam_w
                elif n_camera == 5:
                    img_w = img_3cam_w + l1_w + r1_w
                else:
                    raise NotImplementedError

                for i in range(rotation_num):
                    _ego_rotation_angle_degree = self.aug_info[initial_token][i]['rot']
                    offset_w = int(half_view_w / 180 * _ego_rotation_angle_degree)

                    rotated_image = stitched_image[:,
                                    int(whole_w / 2 - offset_w - img_w / 2):int(whole_w / 2 - offset_w + img_w / 2)]

                    resized_image = cv2.resize(rotated_image, (self._config.camera_width, self._config.camera_height))
                    tensor_image = self.student_rotated_augmentation(resized_image)  # [3, h, w]
                    if os.getenv('ROBUST_HYDRA_DEBUG') == 'true':
                        _img = transforms.ToPILImage()(tensor_image)
                        _img.save(f'{debug_dir}/output_tensor_image_{i}.jpg')
                    res['rotated'][i].append(tensor_image)

        return res

    def _get_camera_feature_back(self, agent_input: AgentInput) -> torch.Tensor:
        cameras = agent_input.cameras[-1]

        # Crop to ensure 4:1 aspect ratio
        l2 = cameras.cam_l2.image[28:-28, 416:-416]
        b0 = cameras.cam_b0.image[28:-28]
        r2 = cameras.cam_r2.image[28:-28, 416:-416]

        # stitch l0, f0, r0 images
        stitched_image = np.concatenate([l2, b0, r2], axis=1)
        resized_image = cv2.resize(stitched_image, (self._config.camera_width, self._config.camera_height))
        tensor_image = transforms.ToTensor()(resized_image)

        return tensor_image

    def _get_lidar_feature(self, agent_input: AgentInput, initial_token: str, rotation_num=3) -> Dict[
        str, torch.Tensor]:
        """
        Compute LiDAR feature as 2D histogram, according to Transfuser
        :param agent_input: input dataclass
        :return: LiDAR histogram as torch tensors
        """

        if os.getenv('ROBUST_HYDRA_DEBUG') == 'true':
            import pdb;
            pdb.set_trace()

        res_lidar = dict()
        # only consider (x,y,z) & swap axes for (N,3) numpy array
        lidar_pc = agent_input.lidars[-1].lidar_pc[LidarIndex.POSITION].T

        # NOTE: Code from
        # https://github.com/autonomousvision/carla_garage/blob/main/team_code/data.py#L873
        def splat_points(point_cloud):
            # 256 x 256 grid
            xbins = np.linspace(
                self._config.lidar_min_x,
                self._config.lidar_max_x,
                (self._config.lidar_max_x - self._config.lidar_min_x)
                * int(self._config.pixels_per_meter)
                + 1,
            )
            ybins = np.linspace(
                self._config.lidar_min_y,
                self._config.lidar_max_y,
                (self._config.lidar_max_y - self._config.lidar_min_y)
                * int(self._config.pixels_per_meter)
                + 1,
            )
            hist = np.histogramdd(point_cloud[:, :2], bins=(xbins, ybins))[0]
            hist[hist > self._config.hist_max_per_pixel] = self._config.hist_max_per_pixel
            overhead_splat = hist / self._config.hist_max_per_pixel
            return overhead_splat

        # Remove points above the vehicle
        lidar_pc = lidar_pc[lidar_pc[..., 2] < self._config.max_height_lidar]
        below = lidar_pc[lidar_pc[..., 2] <= self._config.lidar_split_height]
        above = lidar_pc[lidar_pc[..., 2] > self._config.lidar_split_height]
        above_features = splat_points(above)

        # Visualization for debugging
        if os.getenv('ROBUST_HYDRA_DEBUG') == 'true':
            # Convert to image format (0-255)
            viz_features = (above_features * 255).astype(np.uint8)
            # Save as grayscale image
            debug_dir = 'debug_viz'
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(f'{debug_dir}/pc_above_features.jpg', viz_features)
        if self._config.use_ground_plane:
            below_features = splat_points(below)
            features = np.stack([below_features, above_features], axis=-1)
        else:
            features = np.stack([above_features], axis=-1)
        features = np.transpose(features, (2, 0, 1)).astype(np.float32)

        res_lidar['ori'] = torch.tensor(features)
        res_lidar['ori_teacher'] = torch.tensor(features)

        rotated_features = []
        for i in range(rotation_num):
            _ego_rotation_angle_degree = self.aug_info[initial_token][i]['rot']
            # Convert to radians and negate (clockwise rotation to compensate for ego counter-clockwise rotation)
            rotation_angle_rad = -(_ego_rotation_angle_degree / 180.0 * np.pi)

            # Rotate point cloud
            rotated_pc = lidar_pc.copy()

            # Apply rotation to x,y coordinates
            cos_angle = np.cos(rotation_angle_rad)
            sin_angle = np.sin(rotation_angle_rad)

            x_orig = rotated_pc[:, 0]
            y_orig = rotated_pc[:, 1]

            rotated_pc[:, 0] = x_orig * cos_angle - y_orig * sin_angle
            rotated_pc[:, 1] = x_orig * sin_angle + y_orig * cos_angle

            # Process the rotated point cloud
            rotated_pc_above = rotated_pc[rotated_pc[:, 2] > self._config.lidar_split_height]
            rotated_above_features = splat_points(rotated_pc_above)

            if self._config.use_ground_plane:
                rotated_pc_below = rotated_pc[rotated_pc[:, 2] <= self._config.lidar_split_height]
                rotated_below_features = splat_points(rotated_pc_below)
                rotated_feature = np.stack([rotated_below_features, rotated_above_features], axis=-1)
            else:
                rotated_feature = np.stack([rotated_above_features], axis=-1)

            # Visualization for debugging
            if os.getenv('ROBUST_HYDRA_DEBUG') == 'true':
                viz_rotated = (rotated_above_features * 255).astype(np.uint8)
                cv2.imwrite(f'debug_viz/{i}_pc_rotated_features.jpg', viz_rotated)

            rotated_feature = np.transpose(rotated_feature, (2, 0, 1)).astype(np.float32)
            rotated_features.append(torch.tensor(rotated_feature))

        res_lidar['rotated'] = rotated_features

        res_lidar = {'lidar': res_lidar}
        return res_lidar


class HydraAugTargetBuilder(AbstractTargetBuilder):
    def __init__(self, config: HydraConfigAug):
        self._config = config
        self.v_params = get_pacifica_parameters()
        self.training = config.training

        config_rotation = config.ego_perturb.rotation

        if self.training:
            with open(config.ego_perturb.offline_aug_file, 'r') as f:
                aug_data = json.load(f)
                assert aug_data['param'][
                           'rot'] == config_rotation.offline_aug_angle_boundary  # TODO: translation boundary
                assert aug_data['param'].get('va', 0) == config.ego_perturb.va.offline_aug_boundary
                self.aug_info = aug_data['tokens']
                self.ensemble_aug = config.ego_perturb.ensemble_aug

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "transfuser_target"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""

        initial_token = scene.scene_metadata.initial_token
        future_traj = scene.get_future_trajectory(
            num_trajectory_frames=int(4 / 0.5)
        )  # [8, 3] (local pose, do not include present pose)

        if not self._config.only_ori_input and self._config.training:
            n_rotated = self._config.student_rotation_ensemble
            _rotations = [self.aug_info[initial_token][i]['rot'] for i in range(n_rotated)]
            _reverse_rotations = [-_rot / 180.0 * np.pi for _rot in _rotations]  # Convert degree to rad
        else:
            _rotations = [0]
            _reverse_rotations = [0]

        if os.getenv('ROBUST_HYDRA_DEBUG') == 'true':
            def _visualize_trajectories(original_poses, rotated_poses, filename, traj_name):
                import matplotlib.pyplot as plt
                import matplotlib
                zhfont1 = matplotlib.font_manager.FontProperties(
                    fname="/DATA3/yaowenhao/downloads/fonts/SourceHanSansSC-Normal.otf")
                plt.figure(figsize=(8, 8))
                ax = plt.gca()
                ax.set_aspect('equal')

                # Set plot limits with some padding
                max_x = max(abs(original_poses[:, 0].max()), abs(rotated_poses[:, 0].max()))
                max_y = max(abs(original_poses[:, 1].max()), abs(rotated_poses[:, 1].max()))
                margin = 3  # Add 5m margin
                plt.xlim(-1, max_x + 2)
                plt.ylim(-max_y - margin, max_y + margin)

                # Add grid
                plt.grid(True, linestyle='--', alpha=0.7)

                # Set ticks at reasonable intervals
                tick_spacing = 5
                x_ticks = np.arange(int(0), int(15) + 1, tick_spacing)
                y_ticks = np.arange(int(-5), int(5) + 1, tick_spacing)
                plt.xticks(x_ticks)
                plt.yticks(y_ticks)

                # Plot origin point as star
                plt.plot(0, 0, '*', color='grey', markersize=8, label='自车位置')

                # Plot trajectory points and connecting lines for both trajectories
                for poses, color, label in [(original_poses, '#FF69B4', '原始轨迹'),  # Pink
                                            (rotated_poses, '#4169E1', traj_name)]:  # Royal Blue
                    x = poses[:, 0]
                    y = poses[:, 1]
                    plt.scatter(x, y, c=color, s=50, alpha=0.7, label=label)
                    plt.plot(x, y, c=color, alpha=0.5)
                    # Draw line from origin to first point with matching color
                    plt.plot([0, x[0]], [0, y[0]], c=color, alpha=0.5)

                plt.legend(loc='upper right', prop=zhfont1)
                plt.savefig(filename, dpi=400, bbox_inches='tight')
                plt.close()

        # Original trajectory
        ori_trajectory = torch.tensor(future_traj.poses)  # [num_poses, 3]

        # Apply rotations to get multiple trajectories
        rotated_trajectories = []
        for _reverse_rotation in _reverse_rotations:
            if self.training and abs(_reverse_rotation) > 1e-5:
                # Rotate each trajectory point around origin (ego vehicle)
                rotated_poses = []
                for pose in future_traj.poses:
                    # Rotate x,y coordinates
                    rotated_xy = np_vector2_aug(pose[:2], _reverse_rotation)
                    # Adjust heading and normalize to [-pi, pi]
                    rotated_heading = (pose[2] + _reverse_rotation + np.pi) % (2 * np.pi) - np.pi
                    rotated_poses.append(np.array([rotated_xy[0], rotated_xy[1], rotated_heading]))
                rotated_poses = np.array(rotated_poses)
                rotated_trajectories.append(torch.tensor(rotated_poses))  # [num_poses, 3]
            else:
                rotated_trajectories.append(ori_trajectory)  # Use original trajectory when no rotation

        ret = {
            "ori_trajectory": ori_trajectory,
            "rotated_trajectories": rotated_trajectories,
        }

        return ret

    def _compute_agent_targets(self, annotations: Annotations) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts 2D agent bounding boxes in ego coordinates
        :param annotations: annotation dataclass
        :return: tuple of bounding box values and labels (binary)
        """

        max_agents = self._config.num_bounding_boxes
        agent_states_list: List[npt.NDArray[np.float32]] = []

        def _xy_in_lidar(x: float, y: float, config: HydraConfigAug) -> bool:
            return (config.lidar_min_x <= x <= config.lidar_max_x) and (
                    config.lidar_min_y <= y <= config.lidar_max_y
            )

        for box, name in zip(annotations.boxes, annotations.names):
            box_x, box_y, box_heading, box_length, box_width = (
                box[BoundingBoxIndex.X],
                box[BoundingBoxIndex.Y],
                box[BoundingBoxIndex.HEADING],
                box[BoundingBoxIndex.LENGTH],
                box[BoundingBoxIndex.WIDTH],
            )

            if name == "vehicle" and _xy_in_lidar(box_x, box_y, self._config):
                agent_states_list.append(
                    np.array([box_x, box_y, box_heading, box_length, box_width], dtype=np.float32)
                )

        agents_states_arr = np.array(agent_states_list)

        # filter num_instances nearest
        agent_states = np.zeros((max_agents, BoundingBox2DIndex.size()), dtype=np.float32)
        agent_labels = np.zeros(max_agents, dtype=bool)

        if len(agents_states_arr) > 0:
            distances = np.linalg.norm(agents_states_arr[..., BoundingBox2DIndex.POINT], axis=-1)
            argsort = np.argsort(distances)[:max_agents]

            # filter detections
            agents_states_arr = agents_states_arr[argsort]
            agent_states[: len(agents_states_arr)] = agents_states_arr
            agent_labels[: len(agents_states_arr)] = True

        return torch.tensor(agent_states), torch.tensor(agent_labels)

    def _compute_bev_semantic_map(
            self, annotations: Annotations, map_api: AbstractMap, ego_pose: StateSE2
    ) -> torch.Tensor:
        """
        Creates sematic map in BEV
        :param annotations: annotation dataclass
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :return: 2D torch tensor of semantic labels
        """

        bev_semantic_map = np.zeros(self._config.bev_semantic_frame, dtype=np.int64)
        for label, (entity_type, layers) in self._config.bev_semantic_classes.items():
            if entity_type == "polygon":
                entity_mask = self._compute_map_polygon_mask(map_api, ego_pose, layers)
            elif entity_type == "linestring":
                entity_mask = self._compute_map_linestring_mask(map_api, ego_pose, layers)
            else:
                entity_mask = self._compute_box_mask(annotations, layers)
            bev_semantic_map[entity_mask] = label

        return torch.Tensor(bev_semantic_map)

    def _compute_map_polygon_mask(
            self, map_api: AbstractMap, ego_pose: StateSE2, layers: List[SemanticMapLayer]
    ) -> npt.NDArray[np.bool_]:
        """
        Compute binary mask given a map layer class
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :param layers: map layers
        :return: binary mask as numpy array
        """

        map_object_dict = map_api.get_proximal_map_objects(
            point=ego_pose.point, radius=self._config.bev_radius, layers=layers
        )
        map_polygon_mask = np.zeros(self._config.bev_semantic_frame[::-1], dtype=np.uint8)
        for layer in layers:
            for map_object in map_object_dict[layer]:
                polygon: Polygon = self._geometry_local_coords(map_object.polygon, ego_pose)
                exterior = np.array(polygon.exterior.coords).reshape((-1, 1, 2))
                exterior = self._coords_to_pixel(exterior)
                cv2.fillPoly(map_polygon_mask, [exterior], color=255)
        # OpenCV has origin on top-left corner
        map_polygon_mask = np.rot90(map_polygon_mask)[::-1]
        return map_polygon_mask > 0

    def _compute_map_linestring_mask(
            self, map_api: AbstractMap, ego_pose: StateSE2, layers: List[SemanticMapLayer]
    ) -> npt.NDArray[np.bool_]:
        """
        Compute binary of linestring given a map layer class
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :param layers: map layers
        :return: binary mask as numpy array
        """
        map_object_dict = map_api.get_proximal_map_objects(
            point=ego_pose.point, radius=self._config.bev_radius, layers=layers
        )
        map_linestring_mask = np.zeros(self._config.bev_semantic_frame[::-1], dtype=np.uint8)
        for layer in layers:
            for map_object in map_object_dict[layer]:
                linestring: LineString = self._geometry_local_coords(
                    map_object.baseline_path.linestring, ego_pose
                )
                points = np.array(linestring.coords).reshape((-1, 1, 2))
                points = self._coords_to_pixel(points)
                cv2.polylines(map_linestring_mask, [points], isClosed=False, color=255, thickness=2)
        # OpenCV has origin on top-left corner
        map_linestring_mask = np.rot90(map_linestring_mask)[::-1]
        return map_linestring_mask > 0

    def _compute_box_mask(
            self, annotations: Annotations, layers: TrackedObjectType
    ) -> npt.NDArray[np.bool_]:
        """
        Compute binary of bounding boxes in BEV space
        :param annotations: annotation dataclass
        :param layers: bounding box labels to include
        :return: binary mask as numpy array
        """
        box_polygon_mask = np.zeros(self._config.bev_semantic_frame[::-1], dtype=np.uint8)
        for name_value, box_value in zip(annotations.names, annotations.boxes):
            agent_type = tracked_object_types[name_value]
            if agent_type in layers:
                # box_value = (x, y, z, length, width, height, yaw) TODO: add intenum
                x, y, heading = box_value[0], box_value[1], box_value[-1]
                box_length, box_width, box_height = box_value[3], box_value[4], box_value[5]
                agent_box = OrientedBox(StateSE2(x, y, heading), box_length, box_width, box_height)
                exterior = np.array(agent_box.geometry.exterior.coords).reshape((-1, 1, 2))
                exterior = self._coords_to_pixel(exterior)
                cv2.fillPoly(box_polygon_mask, [exterior], color=255)
        # OpenCV has origin on top-left corner
        box_polygon_mask = np.rot90(box_polygon_mask)[::-1]
        return box_polygon_mask > 0

    @staticmethod
    def _query_map_objects(
            self, map_api: AbstractMap, ego_pose: StateSE2, layers: List[SemanticMapLayer]
    ) -> List[MapObject]:
        """
        Queries map objects
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :param layers: map layers
        :return: list of map objects
        """

        # query map api with interesting layers
        map_object_dict = map_api.get_proximal_map_objects(
            point=ego_pose.point, radius=self, layers=layers
        )
        map_objects: List[MapObject] = []
        for layer in layers:
            map_objects += map_object_dict[layer]
        return map_objects

    @staticmethod
    def _geometry_local_coords(geometry: Any, origin: StateSE2) -> Any:
        """
        Transform shapely geometry in local coordinates of origin.
        :param geometry: shapely geometry
        :param origin: pose dataclass
        :return: shapely geometry
        """

        a = np.cos(origin.heading)
        b = np.sin(origin.heading)
        d = -np.sin(origin.heading)
        e = np.cos(origin.heading)
        xoff = -origin.x
        yoff = -origin.y

        translated_geometry = affinity.affine_transform(geometry, [1, 0, 0, 1, xoff, yoff])
        rotated_geometry = affinity.affine_transform(translated_geometry, [a, b, d, e, 0, 0])

        return rotated_geometry

    def _coords_to_pixel(self, coords):
        """
        Transform local coordinates in pixel indices of BEV map
        :param coords: _description_
        :return: _description_
        """

        # NOTE: remove half in backward direction
        pixel_center = np.array([[0, self._config.bev_pixel_width / 2.0]])
        coords_idcs = (coords / self._config.bev_pixel_size) + pixel_center

        return coords_idcs.astype(np.int32)


class BoundingBox2DIndex(IntEnum):
    _X = 0
    _Y = 1
    _HEADING = 2
    _LENGTH = 3
    _WIDTH = 4

    @classmethod
    def size(cls):
        valid_attributes = [
            attribute
            for attribute in dir(cls)
            if attribute.startswith("_")
               and not attribute.startswith("__")
               and not callable(getattr(cls, attribute))
        ]
        return len(valid_attributes)

    @classmethod
    @property
    def X(cls):
        return cls._X

    @classmethod
    @property
    def Y(cls):
        return cls._Y

    @classmethod
    @property
    def HEADING(cls):
        return cls._HEADING

    @classmethod
    @property
    def LENGTH(cls):
        return cls._LENGTH

    @classmethod
    @property
    def WIDTH(cls):
        return cls._WIDTH

    @classmethod
    @property
    def POINT(cls):
        # assumes X, Y have subsequent indices
        return slice(cls._X, cls._Y + 1)

    @classmethod
    @property
    def STATE_SE2(cls):
        # assumes X, Y, HEADING have subsequent indices
        return slice(cls._X, cls._HEADING + 1)


def np_vector2_aug(arr, angle):
    # angle: rad, positive means turning left
    _sin, _cos = np.sin(angle), np.cos(angle)
    x_rotated = arr[0] * _cos - arr[1] * _sin
    y_rotated = arr[0] * _sin + arr[1] * _cos
    return np.array([x_rotated, y_rotated])

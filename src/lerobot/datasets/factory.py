# #!/usr/bin/env python

# # Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# import logging
# from pprint import pformat

# import torch

# from lerobot.configs.policies import PreTrainedConfig
# from lerobot.configs.train import TrainPipelineConfig
# from lerobot.datasets.lerobot_dataset import (
#     LeRobotDataset,
#     LeRobotDatasetMetadata,
#     MultiLeRobotDataset,
# )
# from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
# from lerobot.datasets.transforms import ImageTransforms
# from lerobot.utils.constants import OBS_PREFIX

# IMAGENET_STATS = {
#     "mean": [[[0.485]], [[0.456]], [[0.406]]],  # (c,1,1)
#     "std": [[[0.229]], [[0.224]], [[0.225]]],  # (c,1,1)
# }


# def resolve_delta_timestamps(
#     cfg: PreTrainedConfig, ds_meta: LeRobotDatasetMetadata
# ) -> dict[str, list] | None:
#     """Resolves delta_timestamps by reading from the 'delta_indices' properties of the PreTrainedConfig.

#     Args:
#         cfg (PreTrainedConfig): The PreTrainedConfig to read delta_indices from.
#         ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
#             delta_timestamps against.

#     Returns:
#         dict[str, list] | None: A dictionary of delta_timestamps, e.g.:
#             {
#                 "observation.state": [-0.04, -0.02, 0]
#                 "observation.action": [-0.02, 0, 0.02]
#             }
#             returns `None` if the resulting dict is empty.
#     """
#     delta_timestamps = {}
#     for key in ds_meta.features:
#         if key == "next.reward" and cfg.reward_delta_indices is not None:
#             delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
#         if key == "action" and cfg.action_delta_indices is not None:
#             delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
#         if key.startswith(OBS_PREFIX) and cfg.observation_delta_indices is not None:
#             delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]

#     if len(delta_timestamps) == 0:
#         delta_timestamps = None

#     return delta_timestamps


# def make_dataset(cfg: TrainPipelineConfig) -> LeRobotDataset | MultiLeRobotDataset:
#     """Handles the logic of setting up delta timestamps and image transforms before creating a dataset.

#     Args:
#         cfg (TrainPipelineConfig): A TrainPipelineConfig config which contains a DatasetConfig and a PreTrainedConfig.

#     Raises:
#         NotImplementedError: The MultiLeRobotDataset is currently deactivated.

#     Returns:
#         LeRobotDataset | MultiLeRobotDataset
#     """
#     image_transforms = (
#         ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
#     )

#     if isinstance(cfg.dataset.repo_id, str):
#         ds_meta = LeRobotDatasetMetadata(
#             cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
#         )
#         delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
#         if not cfg.dataset.streaming:
#             dataset = LeRobotDataset(
#                 cfg.dataset.repo_id,
#                 root=cfg.dataset.root,
#                 episodes=cfg.dataset.episodes,
#                 delta_timestamps=delta_timestamps,
#                 image_transforms=image_transforms,
#                 revision=cfg.dataset.revision,
#                 video_backend=cfg.dataset.video_backend,
#             )
#         else:
#             dataset = StreamingLeRobotDataset(
#                 cfg.dataset.repo_id,
#                 root=cfg.dataset.root,
#                 episodes=cfg.dataset.episodes,
#                 delta_timestamps=delta_timestamps,
#                 image_transforms=image_transforms,
#                 revision=cfg.dataset.revision,
#                 max_num_shards=cfg.num_workers,
#             )
#     else:
#         raise NotImplementedError("The MultiLeRobotDataset isn't supported for now.")
#         dataset = MultiLeRobotDataset(
#             cfg.dataset.repo_id,
#             # TODO(aliberts): add proper support for multi dataset
#             # delta_timestamps=delta_timestamps,
#             image_transforms=image_transforms,
#             video_backend=cfg.dataset.video_backend,
#         )
#         logging.info(
#             "Multiple datasets were provided. Applied the following index mapping to the provided datasets: "
#             f"{pformat(dataset.repo_id_to_index, indent=2)}"
#         )

#     if cfg.dataset.use_imagenet_stats:
#         for key in dataset.meta.camera_keys:
#             for stats_type, stats in IMAGENET_STATS.items():
#                 dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

#     return dataset

#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
from pprint import pformat
from bisect import bisect_right
from typing import List, Union, Optional, Dict

import torch
from torch.utils.data import Dataset

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
    MultiLeRobotDataset,  # 保留导入，以兼容其他地方的类型引用
)
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
from lerobot.datasets.transforms import ImageTransforms
from lerobot.utils.constants import ACTION, OBS_PREFIX, REWARD


IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],  # (c,1,1)
    "std":  [[[0.229]], [[0.224]], [[0.225]]],  # (c,1,1)
}


# ====== 多源拼接包装类 ======
class _MultiConcatDataset(Dataset):
    """
    将多个 LeRobotDataset 无缝拼接为一个 Dataset，并暴露 train.py 需要的属性：
      - meta: 复用第一个数据集（含 stats）
      - num_frames / num_episodes: 求和
    注：如果要更严谨地合并 stats（mean/std），可以后续扩展在这里做聚合。
    """
    def __init__(self, datasets: List[LeRobotDataset]):
        assert len(datasets) > 0, "No sub-datasets provided to _MultiConcatDataset."
        self.datasets = datasets
        self.cum_lengths = []
        total = 0
        for ds in datasets:
            total += len(ds)
            self.cum_lengths.append(total)

        # 供日志/训练追踪
        self.num_frames = sum(getattr(ds, "num_frames", 0) for ds in datasets)
        self.num_episodes = sum(getattr(ds, "num_episodes", 0) for ds in datasets)

        # 供预处理器使用的统计信息
        self.meta = getattr(datasets[0], "meta", None)

    def __len__(self):
        return self.cum_lengths[-1] if self.cum_lengths else 0

    def __getitem__(self, idx):
        ds_idx = bisect_right(self.cum_lengths, idx)
        prev = 0 if ds_idx == 0 else self.cum_lengths[ds_idx - 1]
        inner_idx = idx - prev
        return self.datasets[ds_idx][inner_idx]


def resolve_delta_timestamps(
    cfg: PreTrainedConfig, ds_meta: LeRobotDatasetMetadata
) -> Optional[Dict[str, list]]:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the PreTrainedConfig.

    Args:
        cfg (PreTrainedConfig): The PreTrainedConfig to read delta_indices from.
        ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
            delta_timestamps against.

    Returns:
        dict[str, list] | None: A dictionary of delta_timestamps, e.g.:
            {
                "observation.state": [-0.04, -0.02, 0]
                "observation.action": [-0.02, 0, 0.02]
            }
            returns `None` if the resulting dict is empty.
    """
    delta_timestamps: Dict[str, list] = {}
    for key in ds_meta.features:
        if key == REWARD and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
        if key == ACTION and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        if key.startswith(OBS_PREFIX) and cfg.observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]

    if len(delta_timestamps) == 0:
        return None
    return delta_timestamps


def _parse_multi_repo_id(repo_id_field: Union[str, List[str], tuple]) -> List[str]:
    """
    支持以下几种形式：
      - 单个字符串：'cao/repo'
      - 逗号分隔的字符串：'path_or_repo_v1,path_or_repo_v2'
      - 列表/元组：['path_or_repo_v1', 'path_or_repo_v2']
    返回：去空白后的路径/Repo 列表。
    """
    if isinstance(repo_id_field, str):
        if "," in repo_id_field:
            parts = [p.strip() for p in repo_id_field.split(",") if p.strip()]
            return parts
        else:
            return [repo_id_field.strip()]
    elif isinstance(repo_id_field, (list, tuple)):
        return [str(p).strip() for p in repo_id_field if str(p).strip()]
    else:
        return []


def make_dataset(cfg: TrainPipelineConfig) -> Union[LeRobotDataset, MultiLeRobotDataset, _MultiConcatDataset]:
    """
    允许 cfg.dataset.repo_id 为：
      - 单个字符串（原逻辑，完全兼容）
      - 逗号分隔的多个路径/Repo 或 列表/元组（多源拼接）
    多源场景下当前不支持 streaming（如需可扩展）。
    """
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )

    # 解析 repo_id（支持逗号分隔 / 列表）
    repo_list = _parse_multi_repo_id(cfg.dataset.repo_id)

    # === 多源 ===
    if len(repo_list) > 1:
        if cfg.dataset.streaming:
            raise NotImplementedError("Multi-dataset with streaming=True is not supported yet.")

        logging.info(f"Loading multiple datasets: {repo_list}")

        # 简化处理：使用第一段数据的 meta 计算 delta_timestamps（假设多段 fps 与 features 一致）
        first_meta = LeRobotDatasetMetadata(
            repo_list[0], root=cfg.dataset.root, revision=cfg.dataset.revision
        )
        delta_timestamps = resolve_delta_timestamps(cfg.policy, first_meta)

        sub_datasets: List[LeRobotDataset] = []
        for repo in repo_list:
            ds = LeRobotDataset(
                repo_id=repo,
                root=cfg.dataset.root,
                episodes=cfg.dataset.episodes,            # 如需每段不同的 episodes 子集，可在此扩展
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=cfg.dataset.revision,
                video_backend=cfg.dataset.video_backend,
            )
            sub_datasets.append(ds)

        dataset: Union[_MultiConcatDataset, LeRobotDataset] = _MultiConcatDataset(sub_datasets)

    # === 单源（兼容原逻辑） ===
    elif len(repo_list) == 1:
        single_repo = repo_list[0]
        ds_meta = LeRobotDatasetMetadata(single_repo, root=cfg.dataset.root, revision=cfg.dataset.revision)
        delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)

        if not cfg.dataset.streaming:
            dataset = LeRobotDataset(
                single_repo,
                root=cfg.dataset.root,
                episodes=cfg.dataset.episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=cfg.dataset.revision,
                video_backend=cfg.dataset.video_backend,
            )
        else:
            dataset = StreamingLeRobotDataset(
                single_repo,
                root=cfg.dataset.root,
                episodes=cfg.dataset.episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=cfg.dataset.revision,
                max_num_shards=cfg.num_workers,
            )

    else:
        # 未能解析出有效的 repo_id
        raise ValueError(
            f"Invalid cfg.dataset.repo_id: {cfg.dataset.repo_id}. "
            "Expect a string (optionally comma-separated) or a list/tuple of repo ids."
        )

    # 可选：用 ImageNet 统计覆盖（与原版一致）
    if cfg.dataset.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return dataset

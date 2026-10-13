# Copyright 2026 The RLinf Authors.
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

"""Trusted LeRobot episodes processed with a GR00T checkpoint's saved statistics."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler, Subset


class DemonstrationCollator:
    """Map flat LeRobot features to native GR00T supervised processor inputs."""

    def __init__(self, processor, embodiment):
        self.processor = processor
        self.embodiment = embodiment
        self.modality = processor.modality_configs[embodiment.value]
        self.stats = processor.state_action_processor.statistics[embodiment.value]

    def _groups(self, values, modality):
        values = np.asarray(values, dtype=np.float32)
        groups = {}
        offset = 0
        for key in self.modality[modality].modality_keys:
            width = np.asarray(self.stats[modality][key]["mean"]).shape[-1]
            groups[key] = values[..., offset : offset + width]
            offset += width
        if offset != values.shape[-1]:
            raise ValueError(f"Dataset {modality} dimension disagrees with checkpoint")
        return groups

    def __call__(self, samples):
        from gr00t.data.types import VLAStepData

        features = []
        for sample in samples:
            if torch.as_tensor(sample["action_is_pad"]).any():
                raise ValueError("Trusted action chunks must stay within one episode")
            images = {}
            for key in self.modality["video"].modality_keys:
                image = sample[f"observation.images.{key}"]
                images[key] = [
                    image.clamp(0, 1).mul(255).to(torch.uint8).permute(1, 2, 0).numpy()
                ]
            step = VLAStepData(
                images=images,
                states=self._groups(sample["observation.state"][None], "state"),
                actions=self._groups(sample["action"], "action"),
                text=sample["task"],
                embodiment=self.embodiment,
            )
            features.append(self.processor([{"content": step}]))
        return self.processor.collator(features)["inputs"], None


def create_data_loader(actor_cfg):
    """Build rank-sharded, episode-safe SFT minibatches from a local curated set.

    ``sft_episodes`` optionally selects whole episode IDs. The saved model
    processor supplies all normalization statistics; dataset statistics are not
    used for normalization. Only complete action chunks are sampled. Augmentation
    is disabled so the supervised and PPO observation paths share preprocessing.
    """
    if (
        not actor_cfg.get("enable_sft_co_train", False)
        or actor_cfg.get("sft_loss_weight", 0.1) <= 0
    ):
        raise ValueError(
            "SFT data loading requires enabled co-training and a positive loss weight"
        )

    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    if not actor_cfg.get("sft_data_path"):
        raise ValueError("actor.sft_data_path is required for positive-weight co-training")
    root = Path(actor_cfg.sft_data_path).expanduser().resolve()
    if not (root / "meta" / "info.json").is_file():
        raise FileNotFoundError(
            "actor.sft_data_path must be a resolved local LeRobot dataset"
        )
    episodes = actor_cfg.get("sft_episodes")
    if episodes is not None:
        episodes = list(episodes)
        if not episodes or len(episodes) != len(set(episodes)):
            raise ValueError("sft_episodes must contain distinct trusted episode IDs")
    batch_size = int(actor_cfg.get("sft_batch_size", actor_cfg.micro_batch_size))
    if batch_size < 1:
        raise ValueError("sft_batch_size must be positive")
    processor = Gr00tN1d7Processor.from_pretrained(
        actor_cfg.model.model_path,
        model_name=actor_cfg.model.get("backbone_model_path"),
        transformers_loading_kwargs={"local_files_only": True},
    )
    processor.eval()
    tag = str(actor_cfg.model.embodiment_tag)
    embodiment = (
        EmbodimentTag.NEW_EMBODIMENT
        if tag in ("so101", "so100")
        else EmbodimentTag(tag)
    )
    modality = processor.modality_configs[embodiment.value]
    offsets = modality["action"].delta_indices
    if not offsets or offsets != list(range(len(offsets))):
        raise ValueError(
            "SFT loader requires consecutive action offsets starting at zero"
        )
    if any(modality[key].delta_indices != [0] for key in ("state", "video")):
        raise ValueError("SFT loader requires one current state and image per camera")
    metadata = LeRobotDatasetMetadata(root.name, root=root)
    lengths = {
        int(row["episode_index"]): int(
            row["dataset_to_index"] - row["dataset_from_index"]
        )
        for row in metadata.episodes
    }
    if episodes is not None and not set(episodes).issubset(lengths):
        raise ValueError(
            "sft_episodes contains episode IDs absent from the local dataset"
        )
    dataset = LeRobotDataset(
        root.name,
        root=root,
        episodes=episodes,
        delta_timestamps={"action": [offset / metadata.fps for offset in offsets]},
        video_backend="pyav",
    )
    indices = [
        index
        for index, (episode, frame) in enumerate(
            zip(
                dataset.hf_dataset["episode_index"],
                dataset.hf_dataset["frame_index"],
                strict=True,
            )
        )
        if int(frame) + offsets[-1] < lengths[int(episode)]
    ]
    if not indices:
        raise ValueError(
            "No selected episode supplies a complete supervised action chunk"
        )
    selected = Subset(dataset, indices)
    distributed = torch.distributed.is_initialized()
    sampler = DistributedSampler(
        selected,
        num_replicas=torch.distributed.get_world_size() if distributed else 1,
        rank=torch.distributed.get_rank() if distributed else 0,
        seed=int(actor_cfg.get("seed", 0)),
        shuffle=True,
    )
    return DataLoader(
        selected,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        collate_fn=DemonstrationCollator(processor, embodiment),
    )

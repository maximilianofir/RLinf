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

import importlib.util
import sys
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

_PATH = (
    Path(__file__).parents[2] / "rlinf/models/embodiment/gr00t/gr00t_n1d7/sft_data.py"
)
_SPEC = importlib.util.spec_from_file_location("gr00t_sft_data_test", _PATH)
SFT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(SFT)


class Embodiment(Enum):
    NEW_EMBODIMENT = "new_embodiment"


class Processor:
    def __init__(self):
        self.training = True
        self.modality_configs = {
            "new_embodiment": {
                "action": SimpleNamespace(
                    delta_indices=[0, 1], modality_keys=["arm", "gripper"]
                ),
                "state": SimpleNamespace(
                    delta_indices=[0], modality_keys=["arm", "gripper"]
                ),
                "video": SimpleNamespace(
                    delta_indices=[0], modality_keys=["wrist", "external"]
                ),
            }
        }
        self.state_action_processor = SimpleNamespace(
            statistics={
                "new_embodiment": {
                    modality: {"arm": {"mean": [1, 2]}, "gripper": {"mean": [3]}}
                    for modality in ("state", "action")
                }
            }
        )
        self.seen = []

    def eval(self):
        self.training = False

    def __call__(self, messages):
        assert not self.training
        step = messages[0]["content"]
        self.seen.append(step)
        return {"actions": torch.from_numpy(step.actions["arm"])}

    def collator(self, features):
        return {"inputs": {"action": torch.stack([x["actions"] for x in features])}}


@pytest.fixture
def dataset_setup(monkeypatch, tmp_path):
    root = tmp_path / "trusted"
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text("{}")
    processor = Processor()
    lengths = {0: 3, 1: 5}

    class Metadata:
        def __init__(self, repo_id, root):
            self.fps = 30
            self.episodes = [
                {"episode_index": 0, "dataset_from_index": 0, "dataset_to_index": 3},
                {"episode_index": 1, "dataset_from_index": 3, "dataset_to_index": 8},
            ]

    class Dataset:
        def __init__(self, repo_id, root, episodes, delta_timestamps, video_backend):
            assert delta_timestamps == {"action": [0, 1 / 30]}
            assert video_backend == "pyav"
            self.frames = [
                (ep, frame) for ep in episodes for frame in range(lengths[ep])
            ]
            self.hf_dataset = {
                "episode_index": [ep for ep, _ in self.frames],
                "frame_index": [frame for _, frame in self.frames],
            }

        def __len__(self):
            return len(self.frames)

        def __getitem__(self, index):
            ep, frame = self.frames[index]
            return {
                "observation.state": torch.tensor([ep, frame, 20.0]),
                "action": torch.tensor([[ep, frame, 20.0], [ep, frame + 1, 21.0]]),
                "action_is_pad": torch.tensor([False, frame + 1 >= lengths[ep]]),
                "observation.images.wrist": torch.full((3, 2, 2), 0.5),
                "observation.images.external": torch.ones(3, 2, 2),
                "task": "Place vial",
            }

    modules = {
        "gr00t.data.embodiment_tags": {"EmbodimentTag": Embodiment},
        "gr00t.data.types": {"VLAStepData": SimpleNamespace},
        "gr00t.model.gr00t_n1d7.processing_gr00t_n1d7": {
            "Gr00tN1d7Processor": SimpleNamespace(
                from_pretrained=lambda *a, **kw: processor
            )
        },
        "lerobot.datasets.lerobot_dataset": {
            "LeRobotDataset": Dataset,
            "LeRobotDatasetMetadata": Metadata,
        },
    }
    for name, values in modules.items():
        module = ModuleType(name)
        module.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, module)
    cfg = OmegaConf.create(
        {
            "enable_sft_co_train": True,
            "sft_loss_weight": 0.1,
            "sft_data_path": str(root),
            "sft_episodes": [1],
            "sft_batch_size": 2,
            "micro_batch_size": 1,
            "seed": 7,
            "model": {"model_path": "model", "embodiment_tag": "new_embodiment"},
        }
    )
    return cfg, processor


def test_selected_nonzero_episode_uses_relative_indices_and_excludes_tails(
    dataset_setup,
):
    cfg, processor = dataset_setup
    loader = SFT.create_data_loader(cfg)
    assert loader.dataset.indices == [0, 1, 2, 3]
    batches = list(loader)
    assert len(batches) == 2
    assert all(actions is None for _, actions in batches)
    assert {int(step.states["arm"][0, 1]) for step in processor.seen} == {0, 1, 2, 3}
    for step in processor.seen:
        assert step.states["arm"][0, 0] == 1
        assert list(step.images) == ["wrist", "external"]
        assert np.all(step.images["wrist"][0] == 127)
        assert step.images["wrist"][0].shape == (2, 2, 3)
        assert step.actions["gripper"].shape == (2, 1)
        assert step.text == "Place vial"


def test_epoch_shuffle_is_seeded_and_can_restart(dataset_setup):
    cfg, _ = dataset_setup
    loader = SFT.create_data_loader(cfg)
    first = list(loader.sampler)
    loader.sampler.set_epoch(1)
    second = list(loader.sampler)
    assert loader.sampler.epoch == 1
    loader.sampler.set_epoch(0)
    assert list(loader.sampler) == first
    assert set(second) == set(first) == {0, 1, 2, 3}


@pytest.mark.parametrize("enabled,weight", [(False, 0.1), (True, 0)])
def test_disabled_sft_never_opens_dataset(enabled, weight):
    cfg = OmegaConf.create({"enable_sft_co_train": enabled, "sft_loss_weight": weight})
    with pytest.raises(ValueError, match="enabled co-training"):
        SFT.create_data_loader(cfg)


def test_invalid_selection_rejected(dataset_setup):
    cfg, _ = dataset_setup
    cfg.sft_episodes = [9]
    with pytest.raises(ValueError, match="absent from the local dataset"):
        SFT.create_data_loader(cfg)

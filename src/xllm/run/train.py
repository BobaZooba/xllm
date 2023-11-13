# Copyright 2023 Boris Zubarev. All rights reserved.
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
from typing import Optional

from ..core.config import Config
from ..datasets.general import GeneralDataset
from ..experiments.base import Experiment
from ..experiments.registry import experiments_registry


def train(
    config: Config,
    train_dataset: Optional[GeneralDataset] = None,
    eval_dataset: Optional[GeneralDataset] = None,
) -> Experiment:
    experiment_cls = experiments_registry.get(config.experiment_key)

    if experiment_cls is None:
        raise ValueError(f"Experiment class {config.experiment_key} not found")

    additional_kwargs = {}

    if train_dataset is not None:
        additional_kwargs["train_dataset"] = train_dataset

    if eval_dataset is not None:
        additional_kwargs["train_dataset"] = eval_dataset

    experiment: Experiment = experiment_cls(config=config, **additional_kwargs)

    experiment.build()

    experiment.run()

    return experiment

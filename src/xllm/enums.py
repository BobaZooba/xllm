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

from dataclasses import dataclass


@dataclass
class General:
    text_parts: str = "text_parts"
    default_sample_field: str = "text"


@dataclass
class Transformers:
    input_ids: str = "input_ids"
    attention_mask: str = "attention_mask"
    labels: str = "labels"
    logits: str = "logits"


@dataclass
class Registry:
    datasets: str = "datasets"
    collators: str = "collators"
    trainers: str = "trainers"
    experiments: str = "experiments"


@dataclass
class Datasets:
    default: str = "default"
    general: str = "general"
    soda: str = "soda"


@dataclass
class Collators:
    lm: str = "lm"
    completion: str = "completion"


@dataclass
class Trainers:
    lm: str = "lm"


@dataclass
class Experiments:
    base: str = "base"


@dataclass
class EnvironmentVariables:
    huggingface_hub_token: str = "HUGGING_FACE_HUB_TOKEN"
    wandb_api_key: str = "WANDB_API_KEY"
    wandb_entity: str = "WANDB_ENTITY"
    wandb_project: str = "WANDB_PROJECT"
    wandb_disabled: str = "WANDB_DISABLED"
    tokenizers_parallelism: str = "TOKENIZERS_PARALLELISM"


@dataclass
class LogLevel:
    info: str = "info"
    warning: str = "warning"
    error: str = "error"
    critical: str = "critical"

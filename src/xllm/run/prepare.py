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

import json
from typing import Tuple

from loguru import logger
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from ..core.config import Config
from ..datasets.registry import datasets_registry


def prepare(config: Config) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
    json_config = json.dumps(config.__dict__, indent=2)
    logger.info(f"Config:\n{json_config}")

    if config.prepare_dataset:
        dataset_cls = datasets_registry.get(config.dataset_key)

        if dataset_cls is None:
            raise ValueError(f"Dataset with key {config.dataset_key} not found")

        dataset_cls.prepare(config=config)
        logger.info(f"Dataset {dataset_cls.__name__} loaded")
    else:
        logger.warning("Dataset is not prepared because this set in config")

    tokenizer = AutoTokenizer.from_pretrained(config.correct_tokenizer_name_or_path)
    logger.info(f"Tokenizer {config.correct_tokenizer_name_or_path} loaded")

    model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path)
    logger.info(f"Model {config.model_name_or_path} loaded")

    return tokenizer, model

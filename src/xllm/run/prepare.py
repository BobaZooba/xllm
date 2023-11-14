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
    """
    Prepares the tokenizer and model for use from the provided configuration, and optionally prepares a dataset
    if specified.

    This function is a high-level utility that aggregates the steps required for setting up the environment necessary
    to work with pre-trained language models, including loading the tokenizer and model as well
    as preparing the dataset.

    Args:
        config (`Config`):
            The configuration object encapsulating settings related to the tokenizer, model, and dataset. This object
            defines how the preparation process should be executed, which resources should be loaded, and whether the
            dataset needs to be prepared.

    Returns:
        Tuple[PreTrainedTokenizer, PreTrainedModel]:
            A tuple containing two elements:
            - The first element is the tokenizer that is loaded from the path specified in the configuration.
            - The second element is the language model that is loaded from the path specified in the configuration.

    The function goes through the following steps:
    - Serializes and logs the configuration settings to provide a transparent overview of the used parameters.
    - Checks if `prepare_dataset` is True in the configuration; if so, retrieves the dataset class using the key
      from the `datasets_registry` and prepares the dataset. Logs an error if the dataset class is not found.
    - Loads the tokenizer using `AutoTokenizer` with the provided path from the configuration. Logs the action.
    - Loads the model using `AutoModelForCausalLM` with the provided path from the configuration. Logs the action.

    Raises:
        ValueError:
            If the dataset is set to be prepared in the configuration, but no corresponding dataset class is found
            in the `datasets_registry`. This indicates the specified dataset key is not registered.

    Example usage:
        ```python
        from some_module.config import Config

        # Assuming we have a predefined Config object with model and tokenizer paths.
        config = Config(...)
        tokenizer, model = prepare(config=config)
        # Now `tokenizer` and `model` can be used for further processing or inference.
        ```

    Note:
        - The function provides informative status updates for each step of the preparation process via logging.
        - It is recommended to ensure all the necessary fields in the `config` object are set correctly
          before calling this function, as it directly influences the resources that will be prepared and loaded.
    """
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

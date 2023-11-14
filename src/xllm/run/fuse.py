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

from typing import Tuple

from loguru import logger
from transformers import PreTrainedModel, PreTrainedTokenizer

from ..core.config import Config
from ..utils.post_training import fuse_lora


def fuse(config: Config) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
    """
    Performs the model parameter fusion step for models that use LoRA (Low-Rank Adaptation) during training.

    This function specifically deals with models that were trained with the LoRA technique, where additional
    learnable parameters were introduced during training for adapting the pre-existing weights. The fusing process
    integrates these parameters into the main model weights, effectively finalizing the model before deployment or
    further usage.

    Args:
        config (`Config`):
            The configuration object that holds settings and parameters for the fusing process, including
            the model and LoRA weights paths locally or at Huggingface Hub.

    Returns:
        Tuple[PreTrainedTokenizer, PreTrainedModel]:
            A tuple containing the tokenizer and the fused model after the LoRA parameters have been integrated.

    During its execution, the `fuse` function calls the `fuse_lora` utility, which handles the intricacies of the
    LoRA fusion process based on the provided configuration. After successful fusion, it logs a message to indicate
    that the process is complete.

    Example usage:
        ```python
        from some_module.config import Config

        # Assuming we have a predefined Config object for a model trained with LoRA.
        config = Config(...)
        tokenizer, fused_model = fuse(config=config)

        # `tokenizer` and `fused_model` can now be used for inference or further steps following the fusion.
        ```

    Note:
        LoRA fusing is a critical step for models that were trained using the LoRA technique. It should be done prior
        to using such models for inference, as it ensures the trained adaptations are correctly reflected in the model's
        behavior.
    """
    tokenizer, model = fuse_lora(config=config)
    logger.info("Fusing complete")

    return tokenizer, model

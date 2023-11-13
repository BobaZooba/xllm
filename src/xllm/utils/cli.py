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

import os

import transformers
from dotenv import load_dotenv
from loguru import logger

from xllm import enums

from ..core.config import Config


def setup_cli(config: Config, logger_path: str = "xllm.log", rotation: str = "5 MB") -> None:
    logger.add(logger_path, rotation=rotation)
    load_dotenv(dotenv_path=config.path_to_env_file)
    logger.info(".env loaded")

    os.environ[enums.EnvironmentVariables.tokenizers_parallelism] = "false"

    if config.report_to_wandb and enums.EnvironmentVariables.wandb_api_key not in os.environ:
        logger.warning("W&B token not found in env vars")

    if enums.EnvironmentVariables.huggingface_hub_token not in os.environ:
        logger.warning("HuggingFaceHub token not found in env vars")

    transformers.set_seed(seed=config.seed)
    transformers.logging.set_verbosity_error()
    logger.info(f'Logger path "{logger_path}" with rotation "{rotation}"')

    return None

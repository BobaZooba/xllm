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
    """
    Sets up the command-line interface (CLI) environment for language model training and evaluation
    by initializing the logger, loading environment variables, and setting global configuration options
    for tokenization and seeding.

    Args:
        config (`Config`):
            The experiment's configuration object that contains necessary parameters,
            including the path to a `.env` file, seed value for reproducibility, and settings related
            to Weights & Biases (wandb) reporting.
        logger_path (`str`, defaults to "xllm.log"):
            The file path where the log records will be stored.
        rotation (`str`, defaults to "5 MB"):
            The policy that determines when a new log file is started. It could be a size limit (like "5 MB"),
            a time period, or a condition.

    This function performs several key setup steps:

    - Initializes the file logger with the specified `logger_path` and `rotation` policy, which manages
        log file rotation based on the file size limit or other criteria.
    - Loads environment variables from the `.env` file specified by the `config.path_to_env_file` attribute.
        This step is crucial for retrieving sensitive information, which should not be hardcoded in the code,
        such as API keys.
    - Sets tokenization-related environment variables to avoid parallelism-related warnings or issues during
        tokenization processes.
    - Checks and issues warnings if API keys for Weights & Biases or HuggingFace Hub are not found
        in the environment variables, which are essential for model reporting and uploading.
    - Seeds the random number generators for libraries like Transformers to ensure reproducibility across runs.
    - Sets the logging verbosity level for the Transformers library to suppress unnecessary messages during execution.

    The `setup_cli` function is typically called at the start of a training or evaluation run to ensure that
    the environment is correctly configured and that all requisite external dependencies are in place and
    properly initialized for the rest of the experiment's execution.
    """

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

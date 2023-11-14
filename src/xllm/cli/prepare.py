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

from typing import Tuple, Type

from transformers import HfArgumentParser, PreTrainedModel, PreTrainedTokenizer

from ..core.config import Config
from ..run.prepare import prepare
from ..utils.cli import setup_cli


def cli_run_prepare(
    config_cls: Type[Config] = Config,
) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
    """
    Provides a command-line interface (CLI) entry point for setting up a tokenizer and model based on a configuration.

    This function serves as an executable script for preparing dataset, tokenizer and model by parsing arguments from
    command line, applying the configuration, and initiating the preparation process. It also sets up the CLI-related
    configurations, such as logging.

    Args:
        config_cls (Type[Config], defaults to `Config`):
            The class type of the configuration to be used. This should be a subclass of `Config` that defines the
            necessary configurations for preparation, including the details for dataset preparation if needed.

    Returns:
        Tuple[PreTrainedTokenizer, PreTrainedModel]:
            A tuple containing the tokenizer and model that have been prepared based on the CLI-provided configuration.

    The function follows these steps:
    - Initializes a command-line argument parser based on `config_cls` to parse arguments into a configuration object.
    - Sets up command-line interactions, including logging to a specified file (`./xllm_prepare.log`).
    - If dataset preparation is configured, it prepares the dataset according to the specifications in `Config`.
    - Invokes the `prepare` function with the parsed configuration object to prepare the tokenizer and model.

    As an entry point when the script is run as a main program (`__name__ == "__main__"`), this function will:
    - Parse command-line arguments into a `Config` object.
    - Run preparation steps (including dataset preparation if applicable) while logging output to "xllm_prepare.log".
    - Return the prepared tokenizer and model.

    Example CLI usage:
        ```sh
        python cli_run_prepare.py --model_name_or_path my_model --dataset_key my_dataset
        ```

    Note:
        This function is intended to be used as part of a CLI workflow and will parse arguments from `sys.argv`.
        The `if __name__ == "__main__":` block below ensures that it runs when the script is executed directly
        from the command line.
    """
    parser = HfArgumentParser(config_cls)
    config = parser.parse_args_into_dataclasses()[0]
    setup_cli(config=config, logger_path="./xllm_prepare.log")
    tokenizer, model = prepare(config=config)
    return tokenizer, model


if __name__ == "__main__":
    cli_run_prepare(config_cls=Config)

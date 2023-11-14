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
from ..run.fuse import fuse
from ..utils.cli import setup_cli


def cli_run_fuse(
    config_cls: Type[Config] = Config,
) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
    """
    Provides a command-line interface (CLI) entry point for fusing LoRA parameters into the model after training.

    This function serves as a script for fusing parameters using the LoRA technique by parsing command-line arguments
    to configure the process and then invoking the `fuse` function. It also manages CLI-related configurations,
    including setting up logging for the process.

    Args:
        config_cls (Type[Config], defaults to `Config`):
            The configuration class type to be used for parsing the command-line arguments into a configuration object.
            This class should define the necessary parameters for the fusing process.

    Returns:
        Tuple[PreTrainedTokenizer, PreTrainedModel]:
            A tuple containing the tokenizer and the LoRA-fused model.

    The function performs the following steps:
    - Initializes an `HfArgumentParser` object with `config_cls` to handle command-line arguments.
    - Parses the arguments into a configuration object.
    - Sets up CLI interactions including logging to a file (default is `./xllm_fuse.log`).
    - Calls the `fuse` function with the parsed configuration object to begin the fusing process.
    - Returns the tokenizer and the model that have been processed.

    When the script is executed directly from the command line, it will run the following as part of the main program:
    - Parse the command-line arguments into a `Config` object.
    - Fuse the LoRA parameters in the trained model while logging the output to `xllm_fuse.log`.
    - Return the tokenizer and LoRA-fused model.

    Example CLI usage:
        ```sh
        python cli_run_fuse.py --model_name_or_path my_model
        ```

    Note:
        This function is particularly meant to be used when working with models that were trained with the LoRA
        technique. It is intended to be used as part of a CLI workflow and should be executed directly from the terminal
        or within a script.
    """
    parser = HfArgumentParser(config_cls)
    config = parser.parse_args_into_dataclasses()[0]
    setup_cli(config=config, logger_path="./xllm_fuse.log")
    tokenizer, model = fuse(config=config)
    return tokenizer, model


if __name__ == "__main__":
    cli_run_fuse(config_cls=Config)

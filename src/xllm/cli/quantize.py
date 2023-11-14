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

from typing import Type

from transformers import HfArgumentParser

from ..core.config import Config
from ..quantization.quantizer import Quantizer
from ..run.quantize import quantize
from ..utils.cli import setup_cli


def cli_run_quantize(
    config_cls: Type[Config] = Config,
) -> Quantizer:
    """
    Provides a command-line interface (CLI) entry point for the quantization process of a pre-trained language model.

    This function allows users to execute model quantization from the command line, setting up the appropriate
    configuration through parsed arguments and invoking the `quantize` function. It also sets up the CLI environment,
    including logging configurations.

    Args:
        config_cls (Type[Config], defaults to `Config`):
            The configuration class used for parsing command-line arguments into a configuration object. This class
            should contain the parameters necessary for the quantization process such as model and tokenizer paths, and
            quantization specifications.

    Returns:
        Quantizer:
            An instance of the `Quantizer` class that has conducted the quantization process and stores the resulting
            quantized model.

    The function undergoes the following procedure:
    - Initializes an `HfArgumentParser` with `config_cls` for handling command-line arguments.
    - Parses the command-line arguments into an instance of the configuration object.
    - Sets up CLI interactions, which include logging outputs to a specified file
      (defaults to `./xllm_gptq_quantize.log`).
    - Executes the `quantize` function with the parsed configuration to perform model quantization.
    - Returns the `Quantizer` instance that now holds the quantized model and associated configurations.

    When this script is run as a main program (that is, `__name__ == "__main__"`), it will perform the following:
    - Parse CLI arguments into a configuration object using the provided `config_cls`.
    - Run the quantization process with logging to the file `xllm_gptq_quantize.log`.
    - Return the `Quantizer` instance with the quantized model ready for use or distribution.

    Example CLI usage:
        ```sh
        python cli_run_quantize.py --model_name_or_path my_model --gptq_bits 4
        ```

    Note:
        This function is designed to facilitate the simplification of the model quantization workflow through the CLI,
        intended for direct execution from the terminal or within scripting environments.
    """
    parser = HfArgumentParser(config_cls)
    config = parser.parse_args_into_dataclasses()[0]
    setup_cli(config=config, logger_path="./xllm_gptq_quantize.log")
    quantizer = quantize(config=config)
    return quantizer


if __name__ == "__main__":
    cli_run_quantize(config_cls=Config)

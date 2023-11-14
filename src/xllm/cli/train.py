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

from typing import Optional, Type

from transformers import (
    HfArgumentParser,
)

from ..core.config import Config
from ..datasets.general import GeneralDataset
from ..experiments.base import Experiment
from ..run.train import train
from ..utils.cli import setup_cli


def cli_run_train(
    config_cls: Type[Config] = Config,
    train_dataset: Optional[GeneralDataset] = None,
    eval_dataset: Optional[GeneralDataset] = None,
) -> Experiment:
    """
    Provides a command-line interface (CLI) entry point for training a model based on a provided configuration and
    optional datasets.

    This function serves as an executable script for starting the training process by parsing command-line arguments,
    applying the configuration, and invoking the `train` function. It also manages CLI-related configurations,
    including setting up logging.

    Args:
        config_cls (Type[Config], defaults to `Config`):
            The type of the configuration class to use for parsing command-line arguments.
            It defaults to the `Config` class if not specified.
        train_dataset (Optional[GeneralDataset], defaults to `None`):
            An optional instance of `GeneralDataset` to be used as the training dataset. If provided, it bypasses
            the need for dataset preparation as specified in the configuration.
        eval_dataset (Optional[GeneralDataset], defaults to `None`):
            An optional instance of `GeneralDataset` to be used for evaluation during training. If provided, it is
            used instead of preparing an evaluation dataset.

    Returns:
        Experiment:
            An instance of the `Experiment` class representing the executed training experiment. The experiment object
            includes the trained model and other relevant training details.

    The function follows these steps:
    - Initializes a `HfArgumentParser` with `config_cls` and parses the command-line arguments into a configuration
      data class.
    - Sets up CLI interactions such as logging to a file specified by `logger_path` (`./xllm_train.log` by default).
    - Calls the `train` function with the parsed configuration and optional datasets to conduct the training experiment.
    - Returns the `Experiment` instance that contains all the training details, such as the final model state.

    As an entry point when the script is run as a main program (`__name__ == "__main__"`), this function will:
    - Parse command-line arguments into a `Config` object.
    - Run the training process with the provided configuration and optional datasets while logging to "xllm_train.log".
    - Return the `Experiment` instance containing the trained model and training results.

    Example CLI usage:
        ```sh
        python cli_run_train.py --model_name_or_path my_model --dataset_key my_dataset
        ```

    Note:
        This function is designed to be part of a command-line toolchain and should be run directly from a terminal or
        a script. Logging, error handling, and command-line argument parsing are integral parts of this function.
    """
    parser = HfArgumentParser(config_cls)
    config = parser.parse_args_into_dataclasses()[0]
    setup_cli(config=config, logger_path="./xllm_train.log")
    experiment = train(config=config, train_dataset=train_dataset, eval_dataset=eval_dataset)
    return experiment


if __name__ == "__main__":
    cli_run_train(config_cls=Config)

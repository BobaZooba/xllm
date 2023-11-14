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
from typing import Optional

from ..core.config import Config
from ..datasets.general import GeneralDataset
from ..experiments.base import Experiment
from ..experiments.registry import experiments_registry


def train(
    config: Config,
    train_dataset: Optional[GeneralDataset] = None,
    eval_dataset: Optional[GeneralDataset] = None,
) -> Experiment:
    """
    Initiates the training process for an experiment based on the provided configuration and optional datasets.

    Utilizing the configuration, the method selects the appropriate experiment class from the `experiments_registry`
    and orchestrates the construction, setup and execution of the experiment's training routine.

    Args:
        config (`Config`):
            The configuration object that contains settings and parameters driving the training process.
        train_dataset (`Optional[GeneralDataset]`, defaults to `None`):
            An optional dataset to be used for training. If provided, it is used instead of building a new dataset.
        eval_dataset (`Optional[GeneralDataset]`, defaults to `None`):
            An optional dataset for evaluation. If provided, it is used for evaluating the model performance
            during training.

    Returns:
        Experiment:
            An instance of the `Experiment` class representing the completed training process, including the trained
            model, training history, and any relevant outputs or metrics.

    The `train` function follows this sequence of steps:
    - Retrieves the appropriate experiment class using the key specified in the `config.experiment_key`. Raises a
      ValueError if no matching class is found in the `experiments_registry`.
    - Instantiates the experiment class, passing the configuration and additional keyword arguments for provided
      datasets.
    - Builds the experiment, setting up the necessary environment, model, tokenizer, and datasets.
    - Executes the run method of the experiment, which encompasses the actual training and evaluation routine.

    Raises:
        ValueError:
            If no experiment class corresponding to `config.experiment_key` is found in the `experiments_registry`.

    Example usage:
        ```python
        from some_module.config import Config

        # Assuming we have a predefined Config object set up for an experiment.
        config = Config(...)
        experiment = train(config=config)

        # After training, `experiment` holds the trained model and results,
        # which can then be used for further analysis or deployment.
        ```

    Note:
        - If the train or evaluation datasets are not provided, the function expects the experiment's `build` method
          to handle their construction based on the provided configuration.
        - This function abstracts away the specifics of the training routine to a higher level, allowing users to work
          with experiments through a uniform interface.
    """
    experiment_cls = experiments_registry.get(config.experiment_key)

    if experiment_cls is None:
        raise ValueError(f"Experiment class {config.experiment_key} not found")

    additional_kwargs = {}

    if train_dataset is not None:
        additional_kwargs["train_dataset"] = train_dataset

    if eval_dataset is not None:
        additional_kwargs["train_dataset"] = eval_dataset

    experiment: Experiment = experiment_cls(config=config, **additional_kwargs)

    experiment.build()

    experiment.run()

    return experiment

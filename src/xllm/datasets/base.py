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
import os
import random
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple

from loguru import logger
from torch.utils.data import Dataset

from ..core.config import Config
from ..types import RawSample


class BaseDataset(Dataset[RawSample], ABC):
    """
    `BaseDataset` is an abstract class inheriting from PyTorch's `Dataset` class, serving as
    a foundational template for creating datasets used in training and evaluating LLM's.
    It provides a standard set of functionalities to prepare, serialize, load, and access data samples.

    The class is essential for managing and manipulating the raw data that will be fed into LLM's
    during training or evaluation. Implementations of `BaseDataset` must provide concrete definitions for
    several abstract methods that handle data retrieval and individual sample access.

    The core functionalities that must be implemented by any subclass include:

    - `get_data`: A class method that fetches raw data samples from a source (like files or databases) and
        provides them in a format ready for further processing.
    - `get_sample`: An instance method used to retrieve a single data sample from the dataset by its index,
        ensuring compatibility with PyTorch's data handling ecosystem.
    - `prepare`: A class method that processes the raw data samples into a format suitable for training and
        evaluation and saves them to specified file paths.
    - `load`: A class method used to create a dataset instance from serialized data samples stored in a file.

    The `BaseDataset` class also implements the `__len__` method to return the number of samples in the dataset
    and the `__getitem__` method to make the class iterable and indexable, which is particularly useful for
    integrating with PyTorch's `DataLoader`.

    Attributes provided by the class:
    - `data` (`List[RawSample]`): A list where each item represents a raw sample included in the dataset,
        accessible by a dataset object instance.

    As an abstract class, `BaseDataset` cannot be instantiated directly. Instead, it must be subclassed
    to provide specific implementations tailored to the dataset needs of a particular language model training
    or evaluation task.
    """

    def __init__(self, data: List[RawSample]):
        """
        Initializes a base dataset object for LLM training and evaluation.

        Args:
            data (`List[RawSample]`):
                A list where each item represents a raw sample to be included in the dataset. Each `RawSample`
                is expected to be a dictionary or other data type, depending on the implementation details
                of the subclass.

        The `BaseDataset` is an abstract base class (ABC) for datasets. It provides a standard interface for
        dataset classes used in the experiment. Subclasses should provide concrete implementations for abstract
        methods defined in this class, such as `get_data` and `get_sample`, which are crucial for preparing
        and retrieving individual data samples, respectively.

        Upon initialization, the `BaseDataset` stores the provided data samples internally, making them accessible
        for later retrieval during the training or evaluation process. It is assumed that the data samples have
        already been processed and are in a usable format.

        Note: This ABC should not be instantiated directly but should be subclassed to create custom dataset
        implementations tailored to specific datasets or tasks.
        """
        super().__init__()
        self.data = data

    @classmethod
    def prepare(cls, config: Config) -> None:
        """
        Prepares and saves raw data samples for training and evaluation.

        This class method is a one-time setup operation that processes raw data and distributes it into t
        raining and evaluation datasets. It handles the shuffling of data, truncation of evaluation samples
        based on maximum limit settings, and the potential combination of evaluation data with training data
        when no separate path is provided for evaluation data.

        Args:
            config (`Config`):
                The experiment configuration object that contains paths to datasets and other dataset-related
                settings such as maximum evaluation samples, shuffle flag, and options regarding combining
                evaluation data with training data.

        The method first retrieves the raw data by calling the `get_data` class method, which must be implemented
        by subclasses. It then separates the data into training and evaluation sets based on provided configurations.
        If evaluation data is available but no separate path is provided for it, and the `add_eval_to_train_if_no_path`
        setting is `True`, then the evaluation data is added to the training data.

        If `config.eval_local_path_to_data` is specified and evaluation data is available, the method saves
        a subset of evaluation data up to a specified maximum sample size (`config.max_eval_samples`).
        Any excess evaluation data can be added back to the training data, ensuring it's not wasted.

        After optionally shuffling the training data, the method ensures there are non-zero samples available
        for training. Finally, it writes both the training and evaluation datasets to their respective files
        specified by `config.train_local_path_to_data` and `config.eval_local_path_to_data`.

        Raises:
            ValueError: If no raw data is available or if the training data is empty after preparation.
            FileNotFoundError: If the required data file path does not exist, indicating that the `.prepare`
                method needs to be called before loading data.
        """

        raw_data = cls.get_data(config=config)

        if raw_data is None:
            raise ValueError("Method get_data returned None")
        else:
            train_data, eval_data = raw_data

        if config.eval_local_path_to_data is None and eval_data is not None:
            logger.warning("eval_local_path_to_data is None, but eval_data is not None")
            if config.add_eval_to_train_if_no_path:
                train_data += eval_data
                logger.info("Add eval data to train")

        if eval_data is not None and config.eval_local_path_to_data is not None:
            if len(eval_data) > config.max_eval_samples and config.max_eval_samples > 0:
                train_data += eval_data[config.max_eval_samples :]
                eval_data = eval_data[: config.max_eval_samples]
                logger.info(f"Eval data size truncated to {config.max_eval_samples}")
            else:
                logger.info(f"Eval data size: {len(eval_data)}")

            with open(config.eval_local_path_to_data, mode="w") as file_object:
                for raw_sample in eval_data:
                    file_object.write(json.dumps(raw_sample) + "\n")
        else:
            logger.warning("eval data or eval_local_path_to_data is None")

        if config.shuffle:
            random.shuffle(train_data)
            logger.info("Train data shuffled")

        if len(train_data) == 0:
            raise ValueError("Train data length is 0 at prepare step")

        with open(config.train_local_path_to_data, mode="w") as file_object:
            for raw_sample in train_data:
                file_object.write(json.dumps(raw_sample) + "\n")

        logger.info(f"Train data size: {len(train_data)}")

        return None

    @classmethod
    def load(cls, path_to_data: str, **kwargs: Any) -> "BaseDataset":
        """
        Loads a dataset from a specified file path and returns a dataset object.

        This is a class method used for loading processed data samples from a file and creating a dataset instance.
        It reads each data sample line by line, deserializes it from JSON format, and constructs a list
        of data samples that define the dataset.

        Args:
            path_to_data (`str`):
                The file path where the dataset is stored. Each line in the file should contain a single
                serialized JSON object corresponding to a sample in the dataset.
            **kwargs (`Any`):
                Additional keyword arguments that can be passed and are handled by the subclass implementation.

        Returns:
            `BaseDataset`: An instance of `BaseDataset` or its subclass, containing the loaded data samples.

        Important: This method assumes that the file located at `path_to_data` exists and contains the serialized
        dataset. If the file does not exist, a `FileNotFoundError` is raised, suggesting that the `.prepare` method
        should be called to populate the file with data prior to loading.

        The method is typically used after calling the `.prepare` class method, which writes the processed data
        samples to a file at `path_to_data`. Once the data is prepared and stored, `load` can be used to create
        the dataset instance that will be used for training or evaluation in the machine learning workflow.

        This method should be overridden by subclasses if they require special handling or additional steps
        when loading data.
        """
        data = list()

        if not os.path.isfile(path_to_data):
            raise FileNotFoundError(f"File {path_to_data} not found. Probably you should run .prepare before")

        with open(path_to_data) as file_object:
            for line in file_object:
                sample = json.loads(line)
                data.append(sample)

        dataset = cls(data=data)

        return dataset

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> RawSample:
        sample = self.get_sample(index=index)
        return sample

    @classmethod
    @abstractmethod
    def get_data(cls, config: Config) -> Optional[Tuple[List[RawSample], Optional[List[RawSample]]]]:
        """
        Retrieves raw data samples for training and evaluation.

        This abstract class method must be implemented by subclasses of `BaseDataset`.
        It's responsible for fetching and returning raw data samples that will later be prepared
        and used to create a dataset for training or evaluation.

        Args:
            config (`Config`):
                The experiment configuration object that contains all necessary parameters and settings
                which might influence how data is fetched.

        Returns:
            `Optional[Tuple[List[RawSample], Optional[List[RawSample]]]]`:
                A tuple containing two elements:
                - The first element is a list of `RawSample` items for the training dataset.
                - The second element is an optional list of `RawSample` items for the evaluation dataset.

                If no data is available or necessary for either training or evaluation, `None` is returned
                in place of a list.

        The `get_data` method is called during the `.prepare` step and is crucial for initially splitting
        available raw data into what will be used for training and what will be reserved for evaluation.
        The methodology and source of data retrieval (e.g., files, databases, external APIs) are determined
        by the subclass implementation, which might also process or filter the raw data based
        on the provided configuration.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError

    @abstractmethod
    def get_sample(self, index: int) -> RawSample:
        """
        Retrieves a single data sample from the dataset.

        This abstract method must be implemented by subclasses of `BaseDataset`. It is called to access
        the data sample corresponding to a particular index in the dataset, allowing for iteration and random access.

        Args:
            index (`int`):
                The index of the sample to retrieve from the dataset. It must be an integer within the range
                of the dataset size.

        Returns:
            `RawSample`: The data sample at the specified index within the dataset. The structure and content
                of the `RawSample` should match the expectations of the model and other components that will
                process the dataset.

        The method is used internally by the `__getitem__` function to provide easy access and conform to the
        `Dataset` interface expected by PyTorch's `DataLoader` and other training utilities.

        Subclasses are responsible for providing the concrete implementation of this method. The implementation
        should handle the proper retrieval and formatting of data samples to ensure coherence with the training
        or evaluation processes where these samples are used.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError

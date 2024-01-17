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

from typing import List, Optional, Tuple, Dict, Any

from .. import enums
from ..core.config import Config
from ..datasets.base import BaseDataset
from ..types import RawSample
from ..utils.logger import dist_logger
import json


class InputOuput(BaseDataset):
    """
    `GeneralDataset` is a subclass of `BaseDataset` designed to handle generic text data for
    training and evaluating language models. It can manage text data in a flexible manner and
    is compatible with a variety of text-related LLM tasks.

    The class provides mechanisms for constructing datasets from plain lists of text strings,
    making it easy to convert raw text data into the required format for input into language models.
    It also supports a simple feature for splitting text data into multiple parts based on a provided separator,
    which can be utilized for tasks that require separate processing of different segments within the text.

    Core functionalities and methods provided by `GeneralDataset` include:

    - `__init__`: Initializes a new instance of the dataset with a list of raw data samples, each potentially
        containing structured text data.
    - `from_list`: A convenience class method for creating a dataset instance from a simple list of text strings.
    - `get_sample`: Retrieves a single processed data sample from the dataset based on an index value.
    - `get_data`: A placeholder method that warns about its intended non-usage in this subclass, directing users
        to instead directly initialize the dataset with data or use alternative provided methods.

    The class should be used when the data does not require extensive preprocessing or transformations beyond
    simple splitting. It serves as a straightforward way to create datasets for text data that is already
    in a processed or semi-processed state.

    Attributes:
    - `sample_field` (`str`): The field name used within each data sample's dictionary to access the text data.
    - `separator` (`Optional[str]`): If not `None`, it specifies the character or string used to split
        the text into parts within the `get_sample` method.

    `GeneralDataset` is ideal for use cases where quick experimentation with minimal data preprocessing
    is desired or when working with text data that adheres to a regular structure.
    """

    def __init__(
        self,
        data: List[RawSample],
        sample_field: str = enums.General.default_sample_field,
        separator: Optional[str] = None,
    ):
        """
        Initializes a `GeneralDataset` object, which allows for flexibility in handling various types of
        text data for language model training and evaluation.

        Args:
            data (`List[RawSample]`):
                A list of raw data samples where each `RawSample` is typically a dictionary containing
                a single key-value pair. The key is specified by `sample_field`, and the value is a string
                representing the text data.
            sample_field (`str`, defaults to `enums.General.default_sample_field`):
                The key used in each `RawSample` dictionary to access the text data. This allows the dataset
                to handle different structures of raw data.
            separator (`Optional[str]`, defaults to `None`):
                A string that specifies the delimiter used to split the text data within each sample into parts.
                If `None`, the text data is not split and is treated as a single block of text.

        The `GeneralDataset` class is a concrete implementation of `BaseDataset` that provides an interface
        to create datasets from lists of text strings. It is designed to be versatile and can be used when
        the data does not require extensive preprocessing or is already preprocessed.

        By extending `BaseDataset`, `GeneralDataset` provides out-of-the-box integration with PyTorch's data
        loading utilities, making it suitable for a wide range of text-based machine learning tasks.
        The class also includes an additional method `from_list` that offers a convenient way to create
        a dataset directly from a list of strings.
        """

        super().__init__(data=data)
        self.sample_field = sample_field
        self.separator = separator

    @classmethod
    def get_data(cls, config: Config) -> Optional[Tuple[List[RawSample], Optional[List[RawSample]]]]:
        dist_logger.warning(
            "This is a special type of dataset in which it is not supposed to get_data anything. "
            "You must pass the data here through __init__ or use from_list, "
            "or through the path in config.train_local_path_to_data and config.eval_local_path_to_data (optional)"
        )
        return None

    def get_sample(self, index: int) -> RawSample:
        """
        Retrieves a single data sample from the `GeneralDataset`.

        This method accesses the raw text data at a specific index, optionally splits it into parts
        if a separator is defined, and returns it as a `RawSample` dictionary.

        Args:
            index (`int`):
                The index of the data sample to be retrieved from the dataset. The index value should
                be within the range of the total number of samples in the dataset.

        Returns:
            `RawSample`: A dictionary containing the text data split into parts, if a separator is set,
            under the key defined by `enums.General.text_parts`. If no separator is set, the dictionary will
        contain the original text data as a single part.

        The `get_sample` method is an implementation of the abstract method from the `BaseDataset` class.
        It facilitates random access to individual samples within the dataset, enabling iterative and indexed data
        loading which is essential for training and evaluating models using PyTorch's DataLoader and similar utilities.

        Subclasses of `GeneralDataset` can override this method to customize the format of the returned sample
        or to implement additional preprocessing steps specific to the model or task at hand.
        """

        text = [
            "[INST] " + self.data[index]['input'].strip() + " [/INST]",
            self.data[index]['output'].strip() + " </s>"
            ]

        sample: RawSample = {enums.General.text_parts: text}
        
        if index == 0:
            print(sample)
        
        return sample

    @classmethod
    def load(cls, path_to_data: str, **kwargs: Any) -> "BaseDataset":
        
        with open(path_to_data, 'r') as file_object:
            data = json.load(file_object)

        dataset = cls(data=data)

        return dataset

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

from abc import ABC, abstractmethod
from typing import List

from transformers import PreTrainedTokenizer

from ..types import Batch, RawSample


class BaseCollator(ABC):
    """
    `BaseCollator` serves as an abstract base class for creating custom data collators that preprocess batches
    of data samples for language model training and evaluation. It defines the structure and expected methods that
    a data collator should implement to fit seamlessly into the training pipeline.

    Data collators are responsible for converting raw text samples into the batched format expected by the model.
    This typically involves tokenization, addition of special tokens, handling of sequence lengths, padding, and
    potentially other preprocessing steps.

    Core features of the `BaseCollator` class:

    - `__init__`: Initializes the collator with a tokenizer, a maximum sequence length, and an optional
        separator for text samples.
    - `__call__`: Serves as an entry point for calling the collation method on a batch of `RawSample` objects,
        delegating to the `parse_batch` method for the actual processing logic.
    - `parse_batch`: An abstract method that must be implemented by subclasses to define how a raw batch is
        processed into the tokenized and formatted inputs suitable for model consumption.

    The `BaseCollator` class ensures that all custom collators have a common interface, which helps standardize
    the batch preparation process across different datasets and models. Each subclass of `BaseCollator` can tailor
    the preprocessing to fit specific model requirements or data formats, providing the flexibility to handle
    a wide range of language modeling tasks.

    As an abstract class, `BaseCollator` cannot be instantiated directly. It must be extended with concrete
    implementations that fulfill the required `parse_batch` method to define the collation strategy for the raw data.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        separator: str = "\n",
    ):
        """
        Initializes the `BaseCollator`, which is an abstract base class for data collators used in preparing batches
        of raw text data for model training and evaluation.

        The `BaseCollator` handles the tokenization and any additional processing required to transform raw text
        samples into a format that can be fed into language models. Concrete subclasses of `BaseCollator` should
        implement the `parse_batch` method, which defines the specific collation logic.

        Args:
            tokenizer (`PreTrainedTokenizer`):
                An instance of `PreTrainedTokenizer`, or any subclass thereof, which will be used to tokenize
                the text data.
            max_length (`int`):
                The maximum length of the tokenized sequences. Any sequence longer than this will be truncated
                to this length.
            separator (`str`, defaults to "\n"):
                A string used as a separator to join multiple text samples within a batch, if necessary.

        The `BaseCollator` class itself does not directly tokenize or process the batches; the provided tokenizer
        and the implemented `parse_batch` method in subclasses carry out these operations. The `max_length` argument
        ensures that the tokenized input does not exceed the model's acceptable input size, and the separator can be
        used to combine multiple text pieces strategically if the collation logic demands it.

        Note: Since `BaseCollator` is an abstract class, it should not be instantiated directly. It is designed
        to be subclassed to create custom collators for specific data and model requirements.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.separator = separator

    def __call__(self, raw_batch: List[RawSample]) -> Batch:
        batch = self.parse_batch(raw_batch=raw_batch)
        return batch

    @abstractmethod
    def parse_batch(self, raw_batch: List[RawSample]) -> Batch:
        """
        An abstract method that must be implemented by subclasses of `BaseCollator`. It outlines the steps to
        process a raw batch of text samples and convert them into a suitable format for the model's input
        (a `Batch` object).

        Args:
            raw_batch (`List[RawSample]`):
                A list of `RawSample` objects, each representing an individual data sample to be included in the batch.

        Returns:
            `Batch`: A `Batch` object that contains the processed and tokenized input data ready to be passed
                to the model for training or evaluation.

        The `parse_batch` method is a key component of the data collation process and is responsible for
        interfacing directly with the tokenizer. It applies any necessary preprocessing steps, such as joining
        text pieces, adding special tokens, or performing tokenization.

        Subclasses of `BaseCollator` will provide concrete implementations for this method, defining the precise
        logic for how the raw data is transformed into model-ready inputs. This can include considerations for padding,
        truncation, and any task-specific data manipulations required for the model's training routine.

        Raises:
            NotImplementedError: If `parse_batch` is called on `BaseCollator` directly, without a subclass
            providing an implementation.
        """
        raise NotImplementedError

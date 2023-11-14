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

from typing import List, Optional, Tuple

import torch
from transformers import PreTrainedTokenizer

from .. import enums
from ..collators.base import BaseCollator
from ..types import Batch, RawSample


class CompletionCollator(BaseCollator):
    """
    `CompletionCollator` is a specialized collator class extending `BaseCollator`. It is designed to prepare
    data specifically for text completion tasks, such as language model fine-tuning and sentence completion,
    where a model generates text based on a given prompt.

    This class takes care of tokenizing the text input, generating the necessary attention masks, and creating
    targets for the language model, ensuring that the data is presented in a way that is consistent with the
    expectations of the model during training.

    This collator is needed for cases when we want to calculate the loss only for the last text in the list.
    For example, these are instances of interacting with an assistant. We don't want to train the model on
    how the user speaks. We don't need the model to be able to imitate the user, so we construct the dataset
    in such a way that at the end of the list of texts (dialogue), there is a phrase by the assistant.
    Essentially, we will be training the model to generate these completions by the assistant.

    Key features and methods provided by `CompletionCollator`:

    - `__init__`: Initializes a new `CompletionCollator` with a tokenizer, maximum sequence length, and optional
        special markers for the distinction between text prompt and completion.

    - `parse_sample`:
        Tokenizes individual text parts, differentiating between the prompt and completion sections,
        and prepares the input and target tokens.

    - `parse_batch`: Aggregates multiple samples into a single batch, padding the sequences to the same length and
        generating attention masks.

    Attributes:
    - `prefix_end` (`Optional[str]`): A special marker used to identify the end of the prefix or prompt in the text.
        If provided, the tokens following this marker are treated as the completion section for which the model
        generates predictions during training.

    - Other attributes inherited from `BaseCollator`, like `tokenizer`, `max_length`, and `separator`.

    By providing a structured way to prepare batches of data specifically for completion tasks, the `CompletionCollator`
    facilitates efficient training workflows and ensures the data adheres to the format required for
    effective fine-tuning of language models.

    The `CompletionCollator` should be employed when building training loops for models that generate completions
    to given text inputs, as it automates much of the otherwise manual batch preparation processes.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        prefix_end: Optional[str] = None,
        separator: str = "\n",
    ):
        super().__init__(tokenizer=tokenizer, max_length=max_length, separator=separator)

        self.prefix_end = prefix_end

    def parse_sample(self, sample: List[str]) -> Tuple[List[int], List[int]]:
        """
        Tokenizes a single text sample and prepares individual input and target sequences for text completion tasks.

        This method takes a list of text parts, representing the segments of a single raw text sample,
        and processes them to generate sequences of token IDs suitable for model training. It identifies
        a prefix within the text (if specified) and ensures that targets are generated only for the completion section.

        Args:
            sample (`List[str]`):
                A list of strings where each element is a part of the text to be tokenized. The last text part
                is considered the completion target when a `prefix_end` is provided.

        Returns:
            `Tuple[List[int], List[int]]`: A tuple containing two lists:
                - The first list represents the `input_ids`, token IDs for the model's input sequence.
                - The second list represents the `targets`, corresponding token IDs for labels used during
                    loss computation.

        The `parse_sample` method performs the following steps for the given text sample:

        - Appends a separator to each text part and tokenizes them, generating a list of token indices.
        - Constructs a mask to identify the target tokens within the completion section of the text.
        - Aligns the token indices and the mask to create `input_ids` and `targets`. Padding tokens are used
            for non-target positions in the `targets` list.

        The separation between the prefix and completion sections is defined by `self.prefix_end`.
        When this marker is present in the last text part, it distinguishes the section of the text where model
        predictions should align with the provided text (targets for the model).

        This method is an integral part of the `CompletionCollator` class, enabling fine-grained control over
        the tokenization and preparation of each text sample prior to batch collation.
        """

        text_parts = [text + self.separator for text in sample]
        tokenized = self.tokenizer(text_parts)[enums.Transformers.input_ids]

        token_indices = list()
        mask = list()

        if self.prefix_end and self.prefix_end in text_parts[-1]:
            prefix = text_parts[-1][: text_parts[-1].index(self.prefix_end) + len(self.prefix_end)]
            n_prefix = len(self.tokenizer(prefix)[enums.Transformers.input_ids]) - 1
        else:
            n_prefix = 0

        for n_sample, text_indices in enumerate(tokenized):
            if n_sample > 0:
                text_indices = text_indices[1:]
            if n_sample == len(tokenized) - 1:
                sample_mask = [0] * n_prefix + [1] * (len(text_indices) - n_prefix)
            else:
                sample_mask = [0] * len(text_indices)

            token_indices.extend(text_indices)
            mask.extend(sample_mask)

        input_ids = token_indices[:-1]

        targets = list()

        for token_index, flag in zip(token_indices[1:], mask[1:]):
            if flag:
                targets.append(token_index)
            else:
                targets.append(self.tokenizer.pad_token_id)

        return input_ids, targets

    def parse_batch(self, raw_batch: List[RawSample]) -> Batch:
        """
        Processes a batch of raw samples and converts them into the format expected by language models
        for completion tasks.

        This method is an implementation of the `parse_batch` abstract method from `BaseCollator`.
        It is responsible for tokenizing the text parts within each raw sample, preparing the tokens
        as model inputs (`input_ids`), and generating corresponding targets.

        Args:
            raw_batch (`List[RawSample]`):
                A list of dictionaries, each representing a raw text sample. Each sample is expected to contain
                a key-value pair, where the key is specified by `enums.General.text_parts` and the value is a list
                of text parts to be tokenized and processed.

        Returns:
            `Batch`: A dictionary that contains three keys (`input_ids`, `attention_mask`, `labels`) with their
            respective tensors, ready for use as input to a language model.
                - `input_ids` and `attention_mask` are used by the model to compute the outputs.
                - `labels` are used to calculate the loss during training.

        The `parse_batch` method performs several steps for each raw sample in the batch:

        - Splits the sample into text parts and tokenizes them.
        - Determines the prefix length to differentiate between prefix and completion parts according
            to `self.prefix_end`.
        - Constructs the `input_ids` and `labels` by including tokens from the text and assigning padding
            token IDs where necessary.
        - Generates the `attention_mask` to inform the model which tokens to pay attention to during training.
        - Pads the sequences in the batch to match the length of the longest sequence, ensuring batch
            uniformity in size.

        This collation logic is essential for text completion tasks, where models typically predict the
        continuation of a given text prompt. By handling the tokenization and the setup of inputs and targets,
        the `CompletionCollator` enables seamless integration with the model's expected input format.
        """

        input_ids = list()
        targets = list()
        attention_masks = list()

        batch_max_length = 0

        for sample in raw_batch:
            text_parts = sample[enums.General.text_parts]

            if isinstance(text_parts, list):
                sample_input_ids, sample_targets = self.parse_sample(sample=[str(item) for item in text_parts])
                sample_length = len(sample_input_ids)

                input_ids.append(sample_input_ids)
                targets.append(sample_targets)

                if sample_length > batch_max_length:
                    batch_max_length = sample_length

                attention_masks.append([1] * sample_length)

        for n_sample in range(len(input_ids)):
            pad_sequence = [self.tokenizer.pad_token_id] * (batch_max_length - len(input_ids[n_sample]))
            if pad_sequence:
                additional_attention_mask = [0] * len(pad_sequence)
                if self.tokenizer.padding_side == "left":
                    input_ids[n_sample] = pad_sequence + input_ids[n_sample]
                    targets[n_sample] = pad_sequence + targets[n_sample]
                    attention_masks[n_sample] = additional_attention_mask + attention_masks[n_sample]
                else:
                    input_ids[n_sample] += pad_sequence
                    targets[n_sample] += pad_sequence
                    attention_masks[n_sample] += additional_attention_mask

        batch = {
            enums.Transformers.input_ids: torch.tensor(input_ids),
            enums.Transformers.attention_mask: torch.tensor(attention_masks),
            enums.Transformers.labels: torch.tensor(targets),
        }

        return batch

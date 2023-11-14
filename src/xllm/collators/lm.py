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

from typing import List

from .. import enums
from ..collators.base import BaseCollator
from ..types import Batch, RawSample


class LMCollator(BaseCollator):
    """
    `LMCollator` is a data collator class specifically designed to prepare batches of data for language modeling tasks.
    Extending the `BaseCollator`, it adapts the general data collation procedure to suit the sequential nature of
    language models, where each token in an input sequence is used to predict the next token.

    The `LMCollator` provides a streamlined approach to handle the conversion of raw text data into the tokenized and
    tensor-formatted inputs required by language models during training. Its primary functionality is implemented in
    the `parse_batch` method, which oversees this conversion process.

    This collator is needed for simple language modeling. It compiles the lists of texts from each example into a
    single text by concatenating them using a separator.

    Key functionalities provided by `LMCollator`:

    - `parse_batch`: A method that processes a batch of `RawSample` objects, creating the tensors needed for training.
        It generates input token IDs (`input_ids`), an attention mask to differentiate between real tokens and padding
        (`attention_mask`), and the labels for training the language model (`labels`).

    Attributes (inherited from `BaseCollator`):

    - `tokenizer`: The tokenizer used to convert raw text into token IDs.
    - `max_length`: The maximum length allowed for tokenized sequences. Longer sequences will be truncated
        to this length.
    - `separator`: A string used to join text pieces within each raw sample, if necessary.

    The `LMCollator` is particularly useful when training Transformer-based language models like GPT on
    next-token prediction tasks. Since it already applies common preprocessing steps such as tokenization, padding,
    truncation, and sequence shifting, it makes setting up training pipelines simpler and more efficient.

    Usage of the `LMCollator` facilitates the generation of data in the precise format required by language models,
    enabling practitioners to focus on model architecture and performance rather than boilerplate data preparation code.
    """

    def parse_batch(self, raw_batch: List[RawSample]) -> Batch:
        """
        Processes a batch of raw text samples and converts them into a suitable format for language model training,
        specifically for tasks involving language modeling (LM) such as next-token prediction.

        Args:
            raw_batch (`List[RawSample]`):
                A list of dictionaries, where each `RawSample` dictionary contains a key-value pair.
                The key is defined by `enums.General.text_parts` and the value is expected to be either a string,
                a numeric value, or a list of these types representing segments of text to include in the batch.

        Returns:
            `Batch`: A dictionary containing the following keys, each associated with its corresponding tensor:
                - `enums.Transformers.input_ids`: The input tokens, with the last token removed from each sequence,
                    as input to the model.
                - `enums.Transformers.attention_mask`: The attention mask, indicating valid tokens (as 1) and padding
                    tokens (as 0) for the model, also with the last token removed from each sequence.
                - `enums.Transformers.labels`: The labels used for training the language model, obtained by shifting
                    the input IDs by one token to the right to predict the next token.

        The `parse_batch` method operates in the following steps:

        - Joins the text segments for each sample in the batch using the specified separator.
        - Tokenizes the combined texts using the provided tokenizer, with padding to the longest sequence in the batch
            and truncation to the specified maximum length.
        - Constructs the input IDs and attention mask for the model input, ensuring to remove the last token from each,
            as it has no subsequent token to predict.
        - Prepares the labels by shifting the input sequence by one token position, facilitating the LM's task of
            predicting the next token in the sequence.

        This collator is tailored for language modeling, where the inputs and labels are often closely related
        sequences, with labels simply being the input sequence offset by one token. It integrates smoothly with
        training routines that utilize PyTorch's DataLoader and other training utilities.
        """

        texts = list()

        for sample in raw_batch:
            item = sample[enums.General.text_parts]
            if isinstance(item, (str, int, float)):
                texts.append(self.separator.join(str(item)))
            else:
                texts.append(self.separator.join(str(i) for i in item))

        tokenized = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        batch = {
            enums.Transformers.input_ids: tokenized.input_ids[:, :-1],
            enums.Transformers.attention_mask: tokenized.attention_mask[:, :-1],
            enums.Transformers.labels: tokenized.input_ids[:, 1:],
        }

        return batch

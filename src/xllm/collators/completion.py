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

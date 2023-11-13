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
    def parse_batch(self, raw_batch: List[RawSample]) -> Batch:
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

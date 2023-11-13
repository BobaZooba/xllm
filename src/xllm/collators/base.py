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

from ..core.constants import BATCH_MUST_HAVE_KEYS
from ..types import Batch, RawSample
from ..utils.miscellaneous import have_missing_keys


class BaseCollator(ABC):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        separator: str = "\n",
        batch_must_have_keys: List[str] = BATCH_MUST_HAVE_KEYS,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.separator = separator
        self.batch_must_have_keys = batch_must_have_keys

    def __call__(self, raw_batch: List[RawSample]) -> Batch:
        batch = self.parse_batch(raw_batch=raw_batch)
        flag, difference = have_missing_keys(data=batch, must_have_keys=self.batch_must_have_keys)
        if flag:
            raise ValueError(f"Batch from {self.__class__.__name__} must have {difference} keys")
        return batch

    @abstractmethod
    def parse_batch(self, raw_batch: List[RawSample]) -> Batch:
        raise NotImplementedError

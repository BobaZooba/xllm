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

import pytest
from torch import Tensor
from transformers import PreTrainedTokenizer

from src.xllm import enums
from src.xllm.collators.completion import CompletionCollator
from tests.helpers.dummy_data import DATA


@pytest.mark.parametrize("prefix_end", [None, ":"])
def test_completion_collator(llama_tokenizer: PreTrainedTokenizer, prefix_end: Optional[str]):
    collator = CompletionCollator(tokenizer=llama_tokenizer, max_length=128, prefix_end=prefix_end)
    batch = collator(DATA)
    for _key, value in batch.items():
        assert isinstance(value, Tensor)

    condition_result = (batch[enums.Transformers.labels][:, :2] == llama_tokenizer.pad_token_id).unique()

    assert len(condition_result) == 1 and condition_result.item()

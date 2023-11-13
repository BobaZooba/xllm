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

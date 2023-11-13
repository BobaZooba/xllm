from torch import Tensor
from transformers import PreTrainedTokenizer

from src.xllm.collators.lm import LMCollator
from tests.helpers.dummy_data import DATA


def test_lm_collator(llama_tokenizer: PreTrainedTokenizer):
    collator = LMCollator(tokenizer=llama_tokenizer, max_length=128, separator="\n")
    batch = collator(DATA)
    for _key, value in batch.items():
        assert isinstance(value, Tensor)

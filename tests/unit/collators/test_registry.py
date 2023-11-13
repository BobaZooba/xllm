from transformers import PreTrainedTokenizer

from src.xllm import enums
from src.xllm.collators.lm import LMCollator
from src.xllm.collators.registry import collators_registry


def test_get_lm_collator(llama_tokenizer: PreTrainedTokenizer):
    collator_cls = collators_registry.get(key=enums.Collators.lm)
    collator = collator_cls(llama_tokenizer, max_length=64)
    assert isinstance(collator, LMCollator)

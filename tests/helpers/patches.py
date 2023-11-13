from contextlib import contextmanager
from typing import Any, Dict, Union

import datasets
import torch
from peft import PeftModel
from pytest import MonkeyPatch
from torch import dtype
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GPTQConfig,
    LlamaConfig,
    LlamaForCausalLM,
)

from src.xllm.trainers.lm import LMTrainer
from tests.helpers.dummy_data import RAW_SODA_DATASET


@contextmanager
def patch_from_pretrained_auto_causal_lm(monkeypatch: MonkeyPatch) -> Any:
    def from_pretrained(
        pretrained_model_name_or_path: str,
        quantization_config: Union[BitsAndBytesConfig, GPTQConfig, None] = None,
        torch_dtype: dtype = torch.float16,
        trust_remote_code: bool = True,
        device_map: Union[str, Dict[str, Any], None] = None,
        use_cache: bool = False,
        use_flash_attention_2: bool = True,
    ) -> LlamaForCausalLM:
        config = LlamaConfig(
            vocab_size=32_000,
            hidden_size=8,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            max_position_embeddings=32,
        )
        model = LlamaForCausalLM(config=config)
        return model

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", from_pretrained)
    yield True
    monkeypatch.undo()


@contextmanager
def patch_load_soda_dataset(monkeypatch: MonkeyPatch) -> Any:
    def load_dataset(dataset_name: str):
        return RAW_SODA_DATASET

    monkeypatch.setattr(datasets, "load_dataset", load_dataset)
    yield True
    monkeypatch.undo()


@contextmanager
def patch_trainer_train(monkeypatch: MonkeyPatch) -> Any:
    def train(*args, **kwargs):
        return None

    monkeypatch.setattr(LMTrainer, "train", train)
    yield True
    monkeypatch.undo()


@contextmanager
def patch_tokenizer_from_pretrained(monkeypatch: MonkeyPatch) -> Any:
    def mock_tokenizer_from_pretrained(*args, **kwargs):
        class MockTokenizer:
            def __init__(self):
                self.pad_token = None
                self.eos_token = "EOS"
                self.padding_side = "right"

            def save_pretrained(self, *args, **kwargs):
                return None

            def push_to_hub(self, *args, **kwargs):
                ...

        mock_tokenizer = MockTokenizer()

        return mock_tokenizer

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", mock_tokenizer_from_pretrained)

    yield True
    monkeypatch.undo()


@contextmanager
def patch_peft_model_from_pretrained(monkeypatch: MonkeyPatch) -> Any:
    def mock_peft_model_from_pretrained(*args, **kwargs):
        class MockPeftModel:
            def parameters(self):
                yield torch.rand(1024, 1024)

            def merge_and_unload(self, *args, **kwargs):
                return self

            def save_pretrained(self, *args, **kwargs):
                return None

            def push_to_hub(self, *args, **kwargs):
                ...

        mock_peft_model = MockPeftModel()

        return mock_peft_model

    monkeypatch.setattr(PeftModel, "from_pretrained", mock_peft_model_from_pretrained)

    yield True
    monkeypatch.undo()

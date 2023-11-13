import pytest
import torch

from src.xllm.core.config import Config
from src.xllm.core.deepspeed_configs import STAGE_2


def test_check_hub():
    config = Config(hub_model_id="llama2", push_to_hub=False)
    config.check_hub()


def test_check_none_deepspeed():
    config = Config(deepspeed_stage=None)
    config.check_deepspeed()


def test_check_not_use_flash_attention_2():
    config = Config(use_flash_attention_2=False)
    config.check_flash_attention()


def test_check_hub_push_without_repo():
    config = Config(hub_model_id=None, push_to_hub=True)
    with pytest.raises(ValueError):
        config.check_hub()


def test_tokenizer_name():
    config = Config(tokenizer_name_or_path="llama1000", model_name_or_path="llama2")
    assert config.correct_tokenizer_name_or_path == "llama1000"


def test_tokenizer_name_from_model():
    config = Config(tokenizer_name_or_path=None, model_name_or_path="llama2")
    assert config.correct_tokenizer_name_or_path == "llama2"


def test_lora_target_modules():
    config = Config(raw_lora_target_modules="q,w,e")
    assert config.lora_target_modules == ["q", "w", "e"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Running tests without GPU")
def test_dtype():
    config = Config()
    assert config.dtype == torch.float16


def test_deepspeed():
    config = Config(deepspeed_stage=2)
    assert config.deepspeed == STAGE_2


def test_check_deepspeed():
    config = Config(deepspeed_stage=0)
    config.check_deepspeed()


def test_check_deepspeed_exception():
    config = Config(deepspeed_stage=2)
    with pytest.raises(ImportError):
        config.check_deepspeed()


def test_check_flash_attention():
    config = Config(use_flash_attention_2=False)
    config.check_flash_attention()


def test_check_flash_attention_exception():
    config = Config(use_flash_attention_2=True)
    with pytest.raises(ImportError):
        config.check_flash_attention()


def test_check_auto_gptq_exception():
    config = Config()
    with pytest.raises(ImportError):
        config.check_auto_gptq()

from pytest import MonkeyPatch

from src.xllm.core.config import Config
from src.xllm.run.fuse import fuse
from tests.helpers.patches import (
    patch_from_pretrained_auto_causal_lm,
    patch_peft_model_from_pretrained,
    patch_tokenizer_from_pretrained,
)


def test_fuse(monkeypatch: MonkeyPatch, path_to_fused_model_local_path: str):
    config = Config(
        push_to_hub=True,
        hub_model_id="BobaZooba/SomeModelLoRA",
        lora_hub_model_id="BobaZooba/SomeModelLoRA",
        fused_model_local_path=path_to_fused_model_local_path,
        push_to_hub_bos_add_bos_token=False,
    )

    with patch_tokenizer_from_pretrained(monkeypatch=monkeypatch):
        with patch_from_pretrained_auto_causal_lm(monkeypatch=monkeypatch):
            with patch_peft_model_from_pretrained(monkeypatch=monkeypatch):
                fuse(config=config)

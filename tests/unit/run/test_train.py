from pytest import MonkeyPatch

from src.xllm.core.config import Config
from src.xllm.run.train import train
from tests.helpers.constants import LLAMA_TOKENIZER_DIR
from tests.helpers.patches import patch_from_pretrained_auto_causal_lm, patch_trainer_train


def test_train(monkeypatch: MonkeyPatch, path_to_train_prepared_dummy_data: str):
    config = Config(
        push_to_hub=False,
        deepspeed_stage=0,
        train_local_path_to_data=path_to_train_prepared_dummy_data,
        report_to_wandb=False,
        save_total_limit=0,
        max_steps=2,
        tokenizer_name_or_path=LLAMA_TOKENIZER_DIR,
    )
    with patch_from_pretrained_auto_causal_lm(monkeypatch=monkeypatch):
        with patch_trainer_train(monkeypatch=monkeypatch):
            train(config=config)

from pytest import MonkeyPatch

from src.xllm.core.config import Config
from src.xllm.run.prepare import prepare
from tests.helpers.constants import LLAMA_TOKENIZER_DIR
from tests.helpers.patches import patch_from_pretrained_auto_causal_lm, patch_load_soda_dataset


def test_prepare(monkeypatch: MonkeyPatch, path_to_download_result: str):
    config = Config(tokenizer_name_or_path=LLAMA_TOKENIZER_DIR, train_local_path_to_data=path_to_download_result)
    with patch_load_soda_dataset(monkeypatch=monkeypatch):
        with patch_from_pretrained_auto_causal_lm(monkeypatch=monkeypatch):
            prepare(config=config)

from pytest import MonkeyPatch

from src.xllm.core.config import Config
from src.xllm.datasets.soda import SodaDataset
from tests.helpers.patches import patch_load_soda_dataset


def test_soda_get_data(monkeypatch: MonkeyPatch, config: Config):
    with patch_load_soda_dataset(monkeypatch=monkeypatch):
        train_data, _ = SodaDataset.get_data(config=config)

from src.xllm import enums
from src.xllm.datasets.registry import datasets_registry
from src.xllm.datasets.soda import SodaDataset
from tests.helpers.dummy_data import DATA


def test_get_soda_dataset() -> None:
    dataset_cls = datasets_registry.get(key=enums.Datasets.soda)
    dataset = dataset_cls(data=DATA)
    assert isinstance(dataset, SodaDataset)

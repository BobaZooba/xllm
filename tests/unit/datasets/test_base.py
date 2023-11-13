from copy import deepcopy
from typing import List, Optional, Tuple

import pytest

from src.xllm import enums
from src.xllm.core.config import Config
from src.xllm.datasets.base import BaseDataset
from src.xllm.types import RawSample
from tests.helpers.dummy_data import DATA


class TestDataset(BaseDataset):
    @classmethod
    def get_data(cls, config: Config) -> Tuple[List[RawSample], Optional[List[RawSample]]]:
        return DATA, DATA

    def get_sample(self, index: int) -> RawSample:
        sample = deepcopy(self.data[index])
        sample["something"] = 1
        del sample[enums.General.text_parts]
        return sample


def test_base_dataset_exception(config: Config):
    train_data, _ = TestDataset.get_data(config=config)
    dataset = TestDataset(data=train_data)
    with pytest.raises(ValueError):
        _ = dataset[0]


def test_prepare(path_to_empty_train_dummy_data: str, path_to_empty_eval_dummy_data: str):
    config = Config(
        train_local_path_to_data=path_to_empty_train_dummy_data,
        eval_local_path_to_data=path_to_empty_eval_dummy_data,
    )
    TestDataset.prepare(config=config)


def test_prepare_eval_path_is_none(path_to_empty_train_dummy_data: str):
    config = Config(
        train_local_path_to_data=path_to_empty_train_dummy_data,
        add_eval_to_train_if_no_path=True,
    )
    TestDataset.prepare(config=config)


def test_prepare_eval_max_samples(path_to_empty_train_dummy_data: str, path_to_empty_eval_dummy_data: str):
    config = Config(
        train_local_path_to_data=path_to_empty_train_dummy_data,
        eval_local_path_to_data=path_to_empty_eval_dummy_data,
        add_eval_to_train_if_no_path=True,
        max_eval_samples=1,
    )
    TestDataset.prepare(config=config)

from typing import List, Optional

import pytest

from src.xllm import enums
from src.xllm.core.config import Config
from src.xllm.datasets.general import GeneralDataset


def test_init():
    GeneralDataset(data=[{"text": "hey"}], sample_field="text")


def test_get_data():
    config = Config()
    value = GeneralDataset.get_data(config=config)
    assert value is None


def test_prepare_exception():
    config = Config()
    with pytest.raises(ValueError):
        GeneralDataset.prepare(config=config)


@pytest.mark.parametrize("data", [["hey", "hello"], ["hey\nhello", "i'm john\ni don't care"]])
@pytest.mark.parametrize("separator", [None, "\n"])
def test_from_list(data: List[str], separator: Optional[str]):
    dataset = GeneralDataset.from_list(data=data, separator=separator)
    assert len(dataset) == len(data)
    sample = dataset.get_sample(index=0)
    assert isinstance(sample, dict)
    assert enums.General.text_parts in sample
    assert isinstance(sample.get(enums.General.text_parts), list)
    assert isinstance(sample.get(enums.General.text_parts)[0], str)

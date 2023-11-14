# Copyright 2023 Boris Zubarev. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

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

from copy import deepcopy
from typing import List, Optional, Tuple

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

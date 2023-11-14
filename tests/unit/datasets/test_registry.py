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

from src.xllm import enums
from src.xllm.datasets.registry import datasets_registry
from src.xllm.datasets.soda import SodaDataset
from tests.helpers.dummy_data import DATA


def test_get_soda_dataset() -> None:
    dataset_cls = datasets_registry.get(key=enums.Datasets.soda)
    dataset = dataset_cls(data=DATA)
    assert isinstance(dataset, SodaDataset)

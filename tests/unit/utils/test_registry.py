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

import pytest

from src.xllm.collators.lm import LMCollator
from src.xllm.utils.registry import Registry


def test_registry():
    registry = Registry(name="tmp", default_value=LMCollator)
    registry.add(key="a", value=1)
    a_value = registry.get(key="a")
    assert a_value == 1
    default_value = registry.get(key="b")
    default_value_from_none = registry.get(key=None)
    assert default_value == default_value_from_none


def test_registry_without_default():
    registry = Registry(name="tmp")
    with pytest.raises(ValueError):
        registry.get(key="a")

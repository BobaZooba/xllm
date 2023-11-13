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

from typing import Any, Dict, Optional

from loguru import logger


class Registry:
    DEFAULT_KEY = "default"

    def __init__(self, name: str, default_value: Optional[Any] = None):
        self.name = name

        self.mapper: Dict[str, Any] = dict()

        if default_value is not None:
            self.mapper[self.DEFAULT_KEY] = default_value

    def add(self, key: str, value: Any, override: bool = False) -> None:
        if not override and key in self.mapper:
            raise ValueError(f"Key exist in {self.name} registry")
        self.mapper[key] = value
        return None

    def get(self, key: Optional[str]) -> Any:
        if key is not None:
            value = self.mapper.get(key, None)
        else:
            value = self.mapper.get(self.DEFAULT_KEY, None)

        if value is None:
            value = self.mapper.get(self.DEFAULT_KEY, None)
            if value is not None:
                logger.warning(f"Default item {value.__name__} chosen from {self.name} registry")

        if value is None:
            raise ValueError(f"Item with key {key} not found in {self.name} registry")

        return value

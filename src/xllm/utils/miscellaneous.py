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

from typing import Any, Dict, List, Set, Tuple

import torch.distributed as distributed


def is_distributed_training() -> bool:
    return distributed.is_available() and distributed.is_initialized()


def have_missing_keys(data: Dict[str, Any], must_have_keys: List[str]) -> Tuple[bool, Set[str]]:
    flag = False

    existing_keys = set(data.keys())
    difference = set(must_have_keys).difference(existing_keys)

    if len(difference) > 0:
        flag = True

    return flag, difference

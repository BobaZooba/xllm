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

from src.xllm import enums
from src.xllm.utils.logger import dist_logger


@pytest.mark.parametrize("message", ["Hello world", "Hey!", "All good"])
def test_info(message: str):
    dist_logger.info(message=message)


@pytest.mark.parametrize("message", ["Hello world", "Hey!", "All good"])
def test_waring(message: str):
    dist_logger.warning(message=message)


@pytest.mark.parametrize("message", ["Hello world", "Hey!", "All good"])
def test_error(message: str):
    dist_logger.error(message=message)


@pytest.mark.parametrize("message", ["Hello world", "Hey!", "All good"])
def test_critical(message: str):
    dist_logger.critical(message=message)


@pytest.mark.parametrize("message", ["Hello world", "Hey!", "All good"])
@pytest.mark.parametrize(
    "level",
    [
        enums.LogLevel.info,
        enums.LogLevel.warning,
        enums.LogLevel.error,
        enums.LogLevel.critical,
        None,
        "some",
    ],
)
def test_log(message: str, level: str):
    dist_logger.log(message=message, level=level)


@pytest.mark.parametrize("message", ["Hello world", "Hey!", "All good"])
@pytest.mark.parametrize(
    "level",
    [
        enums.LogLevel.info,
        enums.LogLevel.warning,
        enums.LogLevel.error,
        enums.LogLevel.critical,
        None,
        "some",
    ],
)
def test_call(message: str, level: str):
    dist_logger(message=message, level=level)

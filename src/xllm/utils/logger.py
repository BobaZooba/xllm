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

from typing import Optional

import torch.distributed as distributed
from loguru import logger

from .. import enums
from ..utils.miscellaneous import is_distributed_training


class DistributedLogger:
    @classmethod
    def can_log(cls, local_rank: int = 0) -> bool:
        if is_distributed_training():
            if distributed.get_rank() == local_rank:
                return True
            else:
                return False
        else:
            return True

    @classmethod
    def info(cls, message: str, local_rank: int = 0) -> None:
        if cls.can_log(local_rank=local_rank):
            logger.info(message)
        return None

    @classmethod
    def warning(cls, message: str, local_rank: int = 0) -> None:
        if cls.can_log(local_rank=local_rank):
            logger.warning(message)
        return None

    @classmethod
    def error(cls, message: str, local_rank: int = 0) -> None:
        if cls.can_log(local_rank=local_rank):
            logger.error(message)
        return None

    @classmethod
    def critical(cls, message: str, local_rank: int = 0) -> None:
        if cls.can_log(local_rank=local_rank):
            logger.critical(message)
        return None

    @classmethod
    def log(cls, message: str, level: Optional[str] = None, local_rank: int = 0) -> None:
        if level == enums.LogLevel.warning:
            cls.warning(message, local_rank)
        elif level == enums.LogLevel.error:
            cls.error(message, local_rank)
        elif level == enums.LogLevel.critical:
            cls.critical(message, local_rank)
        else:
            cls.info(message, local_rank)
        return None

    @classmethod
    def __call__(
        cls,
        message: str,
        level: Optional[str] = enums.LogLevel.info,
        local_rank: int = 0,
    ) -> None:
        cls.log(message, level, local_rank)
        return None


dist_logger = DistributedLogger()

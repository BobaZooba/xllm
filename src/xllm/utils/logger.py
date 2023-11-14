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
    """
    `DistributedLogger` is a utility class for logging messages that is aware of the distributed training environment.
    It ensures that log messages are only recorded by a designated process (typically with a local rank of 0)
    to prevent duplicate entries from multiple processes.

    Upon initialization, the class sets a default logging level and stores the specified local rank. This rank
    determines the process that will be allowed to log messages when running in a distributed setting.

    The class provides the following methods:
    - `can_log`: Checks whether the current process can log based on its rank.
    - `info`: Logs an informational message if the process's rank allows it.
    - `warning`: Logs a warning message subject to the rank permissions.
    - `error`: Logs an error message if the process has the appropriate logging rights.
    - `critical`: Logs a critical message, indicating severe issues when rank conditions are met.
    - `log`: A generic method to log messages that dispatches to the appropriate severity level method.
    - `__call__`: Allows the class to be used as a callable for logging, which selects the correct logging method
        based on the severity level provided.

    By respecting the distributed training settings, `DistributedLogger` provides a consistent and controlled
    logging mechanism suitable for multi-process training scenarios, facilitating debugging and monitoring of
    the training runs.
    """

    def __init__(self, default_level: str = enums.LogLevel.info, local_rank: Optional[int] = None):
        self.default_level = default_level
        self.local_rank = local_rank

    def can_log(self, local_rank: Optional[int] = None) -> bool:
        """
        Determines if the current process is allowed to log messages based on its rank in a distributed
        training environment or when no distributed training is taking place.

        Args:
            local_rank (`Optional[int]`, defaults to `None`):
                The rank of the local process which, if specified, overrides the `local_rank` set during
                initialization of the `DistributedLogger`. If `None`, the initial `local_rank` is used
                for checking the permission to log.

        Returns:
            `bool`: `True` if the process is allowed to log, either as the main process during distributed
            training or in a non-distributed context; `False` otherwise.
        """
        if is_distributed_training():
            if distributed.get_rank() == local_rank or self.local_rank:
                return True
            else:
                return False
        else:
            return True

    def info(self, message: str, local_rank: Optional[int] = None) -> None:
        """
        Logs an informational message with the 'info' logging level in a distributed training context,
        allowing only the designated local process to record the message.

        Args:
            message (`str`): The message to be logged.
            local_rank (`Optional[int]`, defaults to `None`): If specified, overrides the object's `local_rank`.
                Otherwise, uses the `DistributedLogger`'s initialized rank to determine if logging should occur.
        """
        if self.can_log(local_rank=local_rank):
            logger.info(message)
        return None

    def warning(self, message: str, local_rank: Optional[int] = None) -> None:
        """
        Records a warning message at the 'warning' severity level, subject to rank-based permissions in
        a distributed training setting.

        Args:
            message (`str`): The warning message to be logged.
            local_rank (`Optional[int]`, defaults to `None`): If provided, takes precedence over the
                `DistributedLogger`'s rank to check if the process is allowed to log the warning.
        """
        if self.can_log(local_rank=local_rank):
            logger.warning(message)
        return None

    def error(self, message: str, local_rank: Optional[int] = None) -> None:
        """
        Logs an error message at the 'error' severity level, contingent upon the process's eligibility
        to log based on its rank in distributed training.

        Args:
            message (`str`): The error message to be documented.
            local_rank (`Optional[int]`, defaults to `None`): Overrides the `DistributedLogger`'s default rank
                if provided, to verify logging permissions.
        """
        if self.can_log(local_rank=local_rank):
            logger.error(message)
        return None

    def critical(self, message: str, local_rank: Optional[int] = None) -> None:
        """
        Posts a critical message to the log when the current process has the authority within a
        distributed training framework, indicating a severe problem that requires immediate attention.

        Args:
            message (`str`): The critical message to be logged.
            local_rank (`Optional[int]`, defaults to `None`): Used to determine if the process can log,
                taking priority over the `DistributedLogger`'s set rank if given.
        """
        if self.can_log(local_rank=local_rank):
            logger.critical(message)
        return None

    def log(self, message: str, level: Optional[str] = None, local_rank: Optional[int] = None) -> None:
        """
        Dispatches the log message to the appropriate logging level method based on the provided severity level.

        Only the main process in distributed training, or any process in a non-distributed setting, records
        the message to avoid duplicate logs.

        Args:
            message (`str`): The log message to be recorded.
            level (`Optional[str]`): The severity level of the log message (`"warning"`, `"error"`, `"critical"`),
                or defaults to `"info"` if not provided.
            local_rank (`Optional[int]`, defaults to `None`): The rank of the current process, used to determine
                logging eligibility in distributed settings.

        Each log message is checked against the current training distribution context to ensure unity in
        logging output across different distributed processes.
        """
        if level == enums.LogLevel.warning:
            self.warning(message, local_rank)
        elif level == enums.LogLevel.error:
            self.error(message, local_rank)
        elif level == enums.LogLevel.critical:
            self.critical(message, local_rank)
        else:
            self.info(message, local_rank)
        return None

    def __call__(
        self,
        message: str,
        level: Optional[str] = None,
        local_rank: Optional[int] = None,
    ) -> None:
        """
        Logs a message using the distributed logger instance, considering the specified severity level and
        the rank of the local process in a distributed training setup.

        This method enables the `DistributedLogger` object to function as a callable, ensuring that log messages
        are recorded based on the severity level provided and only by the main process
        (typically the one with `local_rank` 0) or in non-distributed environments, to prevent duplicate
        logs across processes.

        Args:
            message (`str`):
                The message to be logged.
            level (`Optional[str]`, defaults to `None`):
                The severity level assigned to the log message. If not provided, the `default_level` set during the
                `DistributedLogger`'s initialization is used. This can be set to `"info"`, `"warning"`, `"error"`,
                or `"critical"`.
            local_rank (`Optional[str]`, defaults to `None`):
                The rank of the local process, which is taken into account when using distributed training to determine
                if the current process is eligible to log. If non-distributed, logging proceeds regardless of rank.

        The method first checks whether the current process can log based on the `local_rank`. If allowed, it logs the
        message with the provided severity level or with the `default_level`. It supports various log levels, mirroring
        standard logging library functionality, to provide different verbosity levels such as info, warnings, errors,
        and critical issues.

        Using this method within a distributed training environment simplifies the process of controlled, coherent
        logging across multiple nodes by automatically handling rank checks and level designation.
        """
        level = level if level is not None else self.default_level
        self.log(message, level, local_rank)
        return None


dist_logger = DistributedLogger()

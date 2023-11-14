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


import torch.distributed as distributed


def is_distributed_training() -> bool:
    """
    Checks if the current environment is set up for distributed training with PyTorch.

    This function queries the PyTorch distributed package to determine if distributed training utilities
    are available and have been properly initialized.

    Returns:
        `bool`: `True` if the distributed training environment is available and initialized, indicating that
        the code is being executed in a distributed manner across multiple processes or nodes. Otherwise,
        returns `False`, indicating that the training is running in a single-process mode.
    """
    return distributed.is_available() and distributed.is_initialized()

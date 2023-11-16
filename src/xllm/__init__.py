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

# ruff: noqa: F401

__version__ = "0.1.6"

from . import enums, types
from .cli.fuse import cli_run_fuse
from .cli.prepare import cli_run_prepare
from .cli.train import cli_run_train
from .core.config import Config
from .run.fuse import fuse
from .run.prepare import prepare
from .run.train import train
from .utils.cli import setup_cli
from .utils.logger import dist_logger
from .utils.post_training import fuse_lora

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

import os

from dotenv import load_dotenv

load_dotenv()

PROJECT_DIR: str = os.getcwd()

STATIC_DIR: str = os.path.join(PROJECT_DIR, "tests/static/")
TOKENIZERS_DIR: str = os.path.join(STATIC_DIR, "tokenizers/")

LLAMA_TOKENIZER_DIR: str = os.path.join(TOKENIZERS_DIR, "llama/")
FALCON_TOKENIZER_DIR: str = os.path.join(TOKENIZERS_DIR, "falcon/")

LORA_FOR_LLAMA_DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

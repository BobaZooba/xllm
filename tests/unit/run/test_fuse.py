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

from pytest import MonkeyPatch

from src.xllm.core.config import Config
from src.xllm.run.fuse import fuse
from tests.helpers.patches import (
    patch_from_pretrained_auto_causal_lm,
    patch_peft_model_from_pretrained,
    patch_tokenizer_from_pretrained,
)


def test_fuse(monkeypatch: MonkeyPatch, path_to_fused_model_local_path: str):
    config = Config(
        push_to_hub=True,
        hub_model_id="BobaZooba/SomeModelLoRA",
        lora_hub_model_id="BobaZooba/SomeModelLoRA",
        fused_model_local_path=path_to_fused_model_local_path,
        push_to_hub_bos_add_bos_token=False,
    )

    with patch_tokenizer_from_pretrained(monkeypatch=monkeypatch):
        with patch_from_pretrained_auto_causal_lm(monkeypatch=monkeypatch):
            with patch_peft_model_from_pretrained(monkeypatch=monkeypatch):
                fuse(config=config)

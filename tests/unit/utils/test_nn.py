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

from peft import PeftModelForCausalLM, PeftType
from transformers import LlamaConfig, LlamaForCausalLM

from src.xllm.core.config import Config
from src.xllm.utils.nn import apply_lora, stabilize_training


def test_apply_lora(llama_model_config: LlamaConfig):
    config = Config(apply_lora=True, raw_lora_target_modules="all")
    model = LlamaForCausalLM(config=llama_model_config)
    peft_model, _ = apply_lora(config=config, model=model)
    assert isinstance(peft_model, PeftModelForCausalLM)
    assert peft_model.peft_type == PeftType.LORA


def test_stabilize_training(llama_model_config: LlamaConfig):
    model = LlamaForCausalLM(config=llama_model_config)
    stabilize_training(model=model)

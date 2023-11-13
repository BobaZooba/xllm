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

from typing import Optional, Tuple

import torch
from peft import LoraConfig, PeftModel, get_peft_model  # type: ignore
from peft.tuners.lora import LoraLayer
from torch import nn
from transformers import (
    PreTrainedModel,
)

from ..core.config import Config


def apply_lora(
    config: Config, model: PreTrainedModel, lora_config: Optional[LoraConfig] = None
) -> Tuple[PeftModel, LoraConfig]:
    lora_target_modules = config.lora_target_modules

    if lora_target_modules is None:
        target_modules = set()

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                names = name.split(".")
                target_modules.add(names[0] if len(names) == 1 else names[-1])

        if "lm_head" in target_modules:  # stabilize in 16-bit
            target_modules.remove("lm_head")

        lora_target_modules = list(target_modules)

    if lora_config is None:
        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=lora_target_modules,
        )

    model = get_peft_model(model=model, peft_config=lora_config)

    return model, lora_config


def stabilize_training(model: PreTrainedModel) -> PreTrainedModel:
    is_bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer) and is_bf16_supported:
            module.lora_A.to(torch.bfloat16)
            module.lora_A.to(torch.bfloat16)
            module.lora_embedding_A.to(torch.bfloat16)
            module.lora_embedding_B.to(torch.bfloat16)
        elif "norm" in name:
            module.to(torch.float32)
        elif (
            ("lm_head" in name or "embed_tokens" in name or "wte" in name or "wpe" in name)
            and hasattr(module, "weight")
            and is_bf16_supported
            and module.weight.dtype == torch.float32
        ):
            module.to(torch.bfloat16)

    return model

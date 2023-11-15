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

from typing import Optional, Tuple, Union

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
    """
    Applies LoRA (Low Rank Adaptation) to a pre-trained language model, enhancing it for efficient task-specific tuning.

    LoRA introduces additional trainable parameters represented as low-rank matrices that adapt the original model's
    weight matrices for new tasks with a minimal increase in the number of parameters. This is particularly useful
    for fine-tuning large pre-trained models.

    Args:
        config (`Config`):
            An instance of the Config class that contains settings for LoRA application, including attributes like
            `lora_rank`, `lora_alpha`, and `lora_dropout`. It also potentially includes `lora_target_modules`,
            which specifies the names of the modules within the model to which LoRA should be applied.
        model (`PreTrainedModel`):
            The pre-trained language model, such as BERT or GPT-3, where LoRA will be applied. It must be a Hugging Face
            `PreTrainedModel` instance.
        lora_config (`Optional[LoraConfig]`):
            A custom LoRA configuration specifying the rank and scaling of adaptations. If provided, it will be used
            instead of the default configuration generated from the `config`. If `None`, a new `LoRAConfig` will be
            created based on parameters defined in the `config`.

    Returns:
        Tuple[PeftModel, LoraConfig]:
            A tuple containing the model enhanced with LoRA layers as `PeftModel` and the `LoraConfig` used to apply
            LoRA to the model.

    Example:
        To apply LoRA to a pre-trained GPT-2 model:

        ```python
        from transformers import GPT2Model
        from my_project.core import Config

        # Load a pre-trained GPT-2 model
        gpt2 = GPT2Model.from_pretrained('gpt2')

        # Define your configuration for LoRA
        my_config = Config(lora_rank=16, lora_alpha=128, lora_dropout=0.1)

        # Apply LoRA using your configuration
        lora_model, lora_cfg = apply_lora(my_config, gpt2)
        ```

    Notes:
        If `lora_target_modules` is not defined in the `config`, the function identifies `nn.Linear` layers to
        which LoRA will be applied, excluding 'lm_head' for stability in 16-bit training environments.

        The resulting model should be used with Hugging Face's `Trainer` API or a similar training loop for fine-tuning.
    """
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


def stabilize_training(
    model: Union[PreTrainedModel, PeftModel], norm_fp32: bool = False
) -> Union[PreTrainedModel, PeftModel]:
    """
    Stabilizes the training of a neural network by adjusting the data types of the model's parameters.
    Specifically, it sets the LoRA (Low-Rank Adaptation) layers to bfloat16 precision and the normalization layers
    to float32 precision to maintain stability during training, especially when using mixed precision training.
    The function also checks if the current environment supports bfloat16, and only performs the conversion
    if bfloat16 is supported. If not, the model's parameters remain unchanged.

    Args:
        model (`Union[PreTrainedModel, PeftModel]`):
            The neural network model to be stabilized for training. This can be a `PreTrainedModel` or a
            `PeftModel` that includes LoRA layers among its submodules.
        norm_fp32 (`bool`, defaults to `False`):
            Convert norm weights to fp32 or not

    Returns:
        Union[PreTrainedModel, PeftModel]: The adjusted model with stabilized training settings ready for
            mixed precision training.

    Examples:
        >>> from transformers import GPT2LMHeadModel
        >>> from peft import PeftModel, LoraLayer
        >>> llm = GPT2LMHeadModel.from_pretrained('gpt2')
        >>> stabilized_model = stabilize_training(llm)
        >>> # Now `stabilized_model` can be used for training with mixed precision support.
    """
    is_bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer) and is_bf16_supported:
            module.lora_A.to(torch.bfloat16)
            module.lora_A.to(torch.bfloat16)
            module.lora_embedding_A.to(torch.bfloat16)
            module.lora_embedding_B.to(torch.bfloat16)
        elif "norm" in name and norm_fp32:
            module.to(torch.float32)
        elif (
            ("lm_head" in name or "embed_tokens" in name or "wte" in name or "wpe" in name)
            and hasattr(module, "weight")
            and is_bf16_supported
            and module.weight.dtype == torch.float32
        ):
            module.to(torch.bfloat16)

    return model

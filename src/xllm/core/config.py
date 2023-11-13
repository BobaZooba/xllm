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

import json
import os
from dataclasses import dataclass, field
from importlib.util import find_spec
from typing import Any, Dict, List, Optional, Union

import torch
from transformers.trainer_utils import FSDPOption

from .. import enums
from ..core.deepspeed_configs import DS_CONFIG_MAPPER
from ..utils.logger import dist_logger


@dataclass
class Config:
    # general
    experiment_key: str = field(
        default=enums.Experiments.base,
        metadata={"help": "Experiment class key"},
    )
    save_safetensors: bool = field(
        default=True,
        metadata={
            "help": "Safe serialization",
        },
    )
    max_shard_size: str = field(
        default="10GB", metadata={"help": "max_shard_size for the model pushing to the HuggingFace Hub"}
    )
    local_rank: int = field(
        default=0,
        metadata={
            "help": "Local rank for logging and saving. Works only in distributed training",
        },
    )
    use_gradient_checkpointing: bool = field(
        default=False,
        metadata={
            "help": "If True, use gradient checkpointing to save memory at the expense of slower backward pass",
        },
    )
    trainer_key: str = field(
        default=enums.Trainers.lm,
        metadata={
            "help": "Key of the trainer for loading from trainers_registry",
        },
    )
    force_fp32: bool = field(
        default=False,
        metadata={
            "help": "Force using fp32 when model loading",
        },
    )
    force_fp16: bool = field(
        default=False,
        metadata={
            "help": "Force using fp16 when model loading",
        },
    )
    from_gptq: bool = field(
        default=False,
        metadata={
            "help": "If you loadining GPTQ quantized model",
        },
    )
    huggingface_hub_token: Optional[str] = field(
        default=None,
        metadata={
            "help": "HuggingFace Hub token. You can also set this key using .env file",
        },
    )
    deepspeed_stage: int = field(
        default=0,
        metadata={
            "help": "Predifined DeepSpeed stage",
        },
    )
    deepspeed_config_path: Optional[int] = field(
        default=None,
        metadata={
            "help": "Path to DeepSpeed config",
        },
    )
    fsdp_strategy: str = field(
        default="",
        metadata={
            "help": "FSDP strategy",
        },
    )
    fsdp_offload: bool = field(
        default=True,
        metadata={
            "help": "Offload weights when using FSDP",
        },
    )
    seed: int = field(
        default=42,
        metadata={
            "help": "Seed value for random operations",
        },
    )
    stabilize: bool = field(
        default=False,
        metadata={
            "help": "Stabilize the model. Convert some weights to fp32, some to fp16/bf16",
        },
    )
    path_to_env_file: Optional[str] = field(
        default="./.env",
        metadata={"help": "Custom path to .env file"},
    )

    # prepare
    prepare_dataset: bool = field(
        default=True,
        metadata={
            "help": 'Prepare the dataset. Works only at "prepare" stage',
        },
    )

    # fuse
    lora_hub_model_id: Optional[str] = field(
        default=None,
        metadata={
            "help": "Fusing LoRA. The name of the LoRA model at the hub for fusing. Example: BobaZooba/Shurale",
        },
    )
    lora_model_local_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Fusing LoRA. Local path to the LoRA model",
        },
    )
    fused_model_local_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Local path to fused model. Useful if you want to quantize model after fusing on the same machine",
        },
    )
    fuse_after_training: bool = field(
        default=False,
        metadata={
            "help": "Fuse or not model after training",
        },
    )

    # gptq quantization
    quantization_dataset_id: Optional[str] = field(
        default=None,
        metadata={
            "help": "Dataset id for GPTQ quantization. You can install either the idi dataset, or use any dataset",
        },
    )
    quantization_max_samples: int = field(
        default=1024,
        metadata={
            "help": "Max samples for GPTQ quantization if you use custom dataset",
        },
    )
    quantized_model_path: str = field(
        default="./quantized_model/",
        metadata={
            "help": "Path to GPTQ quantized model",
        },
    )
    quantized_hub_model_id: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the model at the hub for GPTQ quantization. Example: BobaZooba/Shurale-GPTQ",
        },
    )
    quantized_hub_private_repo: bool = field(
        default=True,
        metadata={
            "help": "Private repository for GPTQ quantization model or not",
        },
    )

    # dataset
    dataset_key: str = field(
        default=enums.Datasets.soda,
        metadata={
            "help": "Key of the dataset for loading from datasets_registry",
        },
    )
    train_local_path_to_data: str = field(
        default="./train.jsonl",
        metadata={
            "help": "The path to the local training data file",
        },
    )
    eval_local_path_to_data: Optional[str] = field(
        default=None,
        metadata={
            "help": "The path to the local eval data file",
        },
    )
    shuffle: bool = field(
        default=True,
        metadata={
            "help": "Shuffle training data",
        },
    )
    max_eval_samples: int = field(
        default=1_000,
        metadata={
            "help": "Maximum number of examples for evaluation",
        },
    )
    add_eval_to_train_if_no_path: bool = field(
        default=False,
        metadata={
            "help": "Add evaluation data to training data if their number is greater than max_eval_samples",
        },
    )

    # tokenizer
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Tokenizer name or path. If the value is not set, "
            "then it will be taken from the model_name_or_path",
        },
    )
    tokenizer_use_fast: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Use fast flag for the tokenizer",
        },
    )
    tokenizer_padding_side: Optional[str] = field(
        default=None,
        metadata={
            "help": "Padding side of the collator: None, right or left",
        },
    )

    # collator
    collator_key: str = field(
        default=enums.Collators.lm,
        metadata={
            "help": "Key of the collator for loading from collators_registry",
        },
    )
    max_length: int = field(
        default=2048,
        metadata={
            "help": "Max sequence length of the model",
        },
    )

    # model
    model_name_or_path: str = field(
        default="mistralai/Mistral-7B-v0.1",
        metadata={
            "help": "Model name or path. It could be from HuggingFace or locally",
        },
    )
    push_to_hub_bos_add_bos_token: bool = field(
        default=False,
        metadata={
            "help": "Upload to the hub tokenization config with add_bos_token equals to True. Might be helpful for TGI"
        },
    )
    use_flash_attention_2: bool = field(
        default=False,
        metadata={
            "help": "Use or not flash attention 2. Requires 1) CUDA >= 11.6; 2) install flash-attn 3) compatible model",
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "Trust remote code from HuggingFace",
        },
    )
    device_map: Optional[str] = field(
        default=None,
        metadata={
            "help": "Device map for loading the model",
        },
    )
    prepare_model_for_kbit_training: bool = field(
        default=True,
        metadata={
            "help": "Prepare or not for kbit training",
        },
    )

    # bitsandbytes
    load_in_8bit: bool = field(
        default=False,
        metadata={
            "help": "Load the model in 8 bit using bitsandbytes",
        },
    )
    load_in_4bit: bool = field(
        default=False,
        metadata={
            "help": "Load the model in 4 bit using bitsandbytes",
        },
    )
    llm_int8_threshold: float = field(
        default=6.0,
        metadata={
            "help": "Threshold for outlier detection",
        },
    )
    llm_int8_has_fp16_weight: bool = field(
        default=True,
        metadata={
            "help": "LLM has weights in fp16",
        },
    )
    bnb_4bit_use_double_quant: bool = field(
        default=True,
        metadata={
            "help": "Double quantization. "
            "This will enable a second quantization after the first "
            "one to save an additional 0.4 bits per parameter",
        },
    )
    bnb_4bit_quant_type: str = field(
        default="nf4",
        metadata={
            "help": "Quantization type for 4 bit",
        },
    )
    bnb_quantize_after_model_init: bool = field(
        default=False, metadata={"help": "If False, quantization will be at model init"}
    )

    # gptq
    gptq_bits: int = field(
        default=4,
        metadata={
            "help": "Bits for GPTQ quantization",
        },
    )
    gptq_group_size: int = field(
        default=128,
        metadata={
            "help": "Group size for GPTQ quantization",
        },
    )
    gptq_disable_exllama: bool = field(
        default=True,
        metadata={
            "help": "Disable ExLlama kernels for GPTQ quantization",
        },
    )

    # lora
    apply_lora: bool = field(
        default=False,
        metadata={
            "help": "Apply LoRA to the model or not",
        },
    )
    lora_rank: int = field(
        default=8,
        metadata={
            "help": "LoRA rank value. LoRA matrices W_A x R and R x W_B, where R is LoRA rank",
        },
    )
    lora_alpha: int = field(
        default=32,
        metadata={
            "help": "LoRA alpha value. The resulting LoRA matrix will be multiplied by this value",
        },
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={
            "help": "LoRA dropout value",
        },
    )
    raw_lora_target_modules: str = field(
        default="all",
        metadata={
            "help": 'Names of modules to apply LoRA. A comma-separated string, for example: "k,q,v". '
            'When setting the value "all", LoRA will be applied to all linear layers, except for the '
            "input embeddings and the lm_head.",
        },
    )

    # training arguments
    output_dir: str = field(
        default="./outputs/",
        metadata={
            "help": "The path to the directory where the artifacts will be saved",
        },
    )
    per_device_train_batch_size: int = field(
        default=2,
        metadata={
            "help": "Batch size on each GPU",
        },
    )
    do_eval: bool = field(
        default=False,
        metadata={
            "help": "Run eval or not",
        },
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Batch size on each GPU for evaluation. "
            "If None per_device_eval_batch_size equals to per_device_train_batch_size",
        },
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={
            "help": "Number of steps to accumulate gradients",
        },
    )
    eval_accumulation_steps: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of steps to accumulate gradients at evaluation."
            "If None eval_accumulation_steps equals to gradient_accumulation_steps",
        },
    )
    eval_delay: int = field(
        default=0,
        metadata={
            "help": "Number of epochs or steps to wait for before the first "
            "evaluation can be performed, depending on the evaluation_strategy"
        },
    )
    eval_steps: Optional[int] = field(
        default=1_000, metadata={"help": "Number of update steps between two evaluations"}
    )
    warmup_steps: int = field(
        default=1_000,
        metadata={
            "help": "Number of steps to warm up",
        },
    )
    max_steps: Optional[int] = field(
        default=None,
        metadata={
            "help": "Maximum number of training steps",
        },
    )
    num_train_epochs: int = field(
        default=1,
        metadata={
            "help": "Number of training epochs",
        },
    )
    learning_rate: float = field(
        default=2e-4,
        metadata={
            "help": "Learning rate value",
        },
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={
            "help": "Clip grad value",
        },
    )
    weight_decay: float = field(
        default=0.001,
        metadata={
            "help": "Weight decay value",
        },
    )
    label_smoothing_factor: float = field(
        default=0.0,
        metadata={
            "help": "Label smoothing value",
        },
    )
    logging_steps: int = field(
        default=10,
        metadata={
            "help": "Number of steps between logging",
        },
    )
    save_steps: int = field(
        default=100,
        metadata={
            "help": "The number of training steps between saving the checkpoint and uploading to the hub",
        },
    )
    save_total_limit: int = field(
        default=1,
        metadata={
            "help": "The number of checkpoints that are saved locally",
        },
    )
    optim: Optional[str] = field(
        default="paged_adamw_8bit",
        metadata={
            "help": "Optimizer name. It will be overwritten if you use deepspeed",
        },
    )
    push_to_hub: bool = field(
        default=False,
        metadata={
            "help": "Upload the model to the hub. "
            "The model will be uploaded to the hub every save_steps. "
            "If LoRA is used, then LoRA's weights will be loaded onto the hub",
        },
    )
    hub_model_id: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the model at the hub. Example: BobaZooba/Shurale",
        },
    )
    hub_private_repo: bool = field(
        default=True,
        metadata={
            "help": "Private repository or not",
        },
    )

    # wandb
    report_to_wandb: bool = field(
        default=False,
        metadata={
            "help": "Report or not to Weight & Biases",
        },
    )
    wandb_api_key: Optional[str] = field(
        default=None,
        metadata={
            "help": "Weight & Biases API key. You can also set this key using .env file",
        },
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={
            "help": "Weight & Biases project name",
        },
    )
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={
            "help": "Weight & Biases entity name (user or company)",
        },
    )

    def __post_init__(self):
        if self.huggingface_hub_token is not None:
            os.environ[enums.EnvironmentVariables.huggingface_hub_token] = self.huggingface_hub_token
            dist_logger(message=f"Environment variable {enums.EnvironmentVariables.huggingface_hub_token} set")

        if self.report_to_wandb:
            for key, value in zip(
                [
                    enums.EnvironmentVariables.wandb_api_key,
                    enums.EnvironmentVariables.wandb_project,
                    enums.EnvironmentVariables.wandb_entity,
                ],
                [
                    self.wandb_api_key,
                    self.wandb_project,
                    self.wandb_entity,
                ],
            ):
                if value is not None:
                    os.environ[key] = value
                    dist_logger(message=f"Environment variable {key} set")
        else:
            os.environ[enums.EnvironmentVariables.wandb_disabled] = "true"

    def check_hub(self) -> None:
        if self.push_to_hub and self.hub_model_id is None:
            raise ValueError("You want to push to HF hub, but hub_model_id is None")
        elif self.hub_model_id is not None and not self.push_to_hub:
            dist_logger.warning("You set hub_model_id, but push_to_hub is False")

        return None

    def check_deepspeed(self) -> None:
        if self.deepspeed is not None:
            spec = find_spec("deepspeed")

            if spec is None:
                raise ImportError("Deepspeed is not None, but failed to import deepspeed. Please install deepspeed.")

        return None

    def check_flash_attention(self) -> None:
        if self.use_flash_attention_2:
            if not torch.cuda.is_available():
                raise ImportError("You want to use flash_attention_2, but CUDA is not available")

            spec = find_spec("flash_attn")

            if spec is None:
                raise ImportError(
                    "You want to use flash_attention_2, but flash-attn is not installed. Please install flash-attn."
                )

        return None

    def check_auto_gptq(self) -> None:
        spec = find_spec("auto_gptq")

        if spec is None:
            raise ImportError(
                "You want to quantize model using GPTQ, but auto-gptq is not installed. Please install auto-gptq."
            )

        return None

    def check(self) -> None:
        self.check_hub()
        self.check_deepspeed()
        self.check_flash_attention()

        return None

    @property
    def correct_tokenizer_name_or_path(self) -> str:
        if self.tokenizer_name_or_path is not None:
            return self.tokenizer_name_or_path
        else:
            return self.model_name_or_path

    @property
    def lora_target_modules(self) -> Optional[List[str]]:
        if self.raw_lora_target_modules == "all":
            return None
        elif self.raw_lora_target_modules is not None:
            modules_names = [module_name.strip() for module_name in self.raw_lora_target_modules.split(",")]
            return modules_names
        else:
            raise ValueError("raw_lora_target_modules doesn't set")

    @property
    def dtype(self) -> torch.dtype:
        if not torch.cuda.is_available() or self.force_fp32:
            return torch.float32
        elif self.force_fp16:
            return torch.float16
        elif torch.cuda.is_bf16_supported():
            return torch.bfloat16
        else:
            return torch.float16

    @property
    def deepspeed(self) -> Optional[Dict[str, Any]]:
        if self.deepspeed_stage in [0, "0", "stage_0"]:
            return None

        deepspeed_config: Optional[Dict[str, Any]] = None

        if self.deepspeed_stage is not None:
            deepspeed_config = DS_CONFIG_MAPPER.get(self.deepspeed_stage, None)
            if deepspeed_config is None:
                raise ValueError(
                    f'Deepspeed stage "{self.deepspeed_stage}" not found in keys: {list(DS_CONFIG_MAPPER.keys())}'
                )

        if self.deepspeed_config_path is not None:
            if os.path.isfile(self.deepspeed_config_path):
                with open(self.deepspeed_config_path) as file_object:
                    deepspeed_config = json.load(file_object)
            else:
                raise ValueError(f"deepspeed_config_path set to {self.deepspeed_config_path}, but not found")

        return deepspeed_config

    @property
    def fsdp(self) -> Union[str, List[str]]:
        fsdp_options = list()

        if self.fsdp_strategy is not None and self.fsdp_strategy != "":
            fsdp_options.append(self.fsdp_strategy)
        else:
            return ""

        if self.fsdp_offload:
            fsdp_options.append(FSDPOption.OFFLOAD)

        return fsdp_options

    @property
    def lora_model_name_or_path_for_fusing(self) -> str:
        if self.lora_hub_model_id is not None:
            return self.lora_hub_model_id
        elif self.lora_model_local_path is not None:
            return self.lora_model_local_path
        else:
            raise ValueError("Please set lora_hub_model_id or lora_model_local_path for fusing")

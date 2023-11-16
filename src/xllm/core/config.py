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
    """
    The `Config` class serves as a comprehensive configuration schema for managing various parameters required during
    the setup and execution of experiments relating to language models, such as training, quantization, and
    optimization.

    Write more here:
        - https://github.com/BobaZooba/xllm/blob/main/DOCS.md#config
        - https://github.com/BobaZooba/xllm/blob/main/DOCS.md#detailed-config-explanation

    This dataclass is used to encapsulate and standardize the configuration for a diverse range of tasks including
    dataset preparation, tokenizer and model initialization, training, evaluation, and interactions with remote services
    like the Hugging Face Model Hub.

    Attributes in this class cover aspects like model name and path, tokenizer settings, dataset paths, training
    strategies, quantization methods, hardware acceleration, logging, output directories, and more. The class provides
    properties with custom logic to resolve specific configurations and validation checks to ensure the environment is
    appropriately set up before proceeding with the workflow.

    Customization and flexibility are core to this class, as it provides reasonable defaults while also allowing for
    detailed and scalable configurations catered to advanced tasks such as leveraging LoRA, FSDP, deepspeed stage
    setups, and applying incremental quantization techniques like GPTQ and bits-and-bytes.

    Methods within the class include:
    - `check`: Performs checks across various attributes for compatibility and correctness.
    - Property getters such as `correct_tokenizer_name_or_path`, `lora_target_modules`, `dtype`, `deepspeed`, `fsdp`,
      and `lora_model_name_or_path_for_fusing` to fetch calculated or defaulted values based on attribute settings.

    Subclassing can be done to extend or modify the functionality of the `Config` class to address specific experimental
    scenarios or customized workflows. It is the central piece for orchestrating experimental setups and is intimately
    integrated with the rest of the codebase that operates on top of these configurations.

    Attributes:

    General Settings:
    - `experiment_key`: An enumeration key to specify the experiment type.
    - `save_safetensors`: A boolean value to indicate whether to use safe serialization for tensors.
    - `max_shard_size`: The maximum shard size when pushing the model to the HuggingFace Hub.
    - `local_rank`: Local rank for distributed training, used for logging and saving.
    - `use_gradient_checkpointing`: If set to `True`, enables gradient checkpointing to reduce memory usage at
        the cost of a slower backward pass.
    - `trainer_key`: An enumeration key to select the trainer using the trainers_registry.
    - `force_fp32`: Forces loading the model in fp32 precision, if set to `True`.
    - `force_fp16`: Forces loading the model in fp16 precision, if set to `True`.
    - `from_gptq`: Indicates if a GPTQ quantized model is being loaded.
    - `huggingface_hub_token`: Token for uploading models to HuggingFace Hub.
    - `deepspeed_stage`: Predefined DeepSpeed stage for optimization.
    - `deepspeed_config_path`: Path to the DeepSpeed config file.
    - `fsdp_strategy`: The strategy to be used for Fully Sharded Data Parallelism (FSDP).
    - `fsdp_offload`: If set to `True`, offloads weights to CPU when using FSDP to save memory.
    - `seed`: Seed for random number generators to ensure reproducibility.
    - `stabilize`: Converts some model weights to fp32 and others to bf16 for stabilization.
    - `path_to_env_file`: Custom path to the .env file for reading environment variables.

    Data Preparation:
    - `prepare_dataset`: Flags whether to prepare the dataset during the "prepare" stage.

    LoRA Fusing:
    - `lora_hub_model_id`: Name of the LoRA model on the hub for fusion.
    - `lora_model_local_path`: Local path to LoRA model to be fused.
    - `fused_model_local_path`: Local path to save the fused model.
    - `fuse_after_training`: If `True`, will fuse the model post-training.

    GPTQ Quantization:
    - `quantization_dataset_id`: Dataset ID for GPTQ quantization.
    - `quantization_max_samples`: Maximum number of samples to use during GPTQ quantization.
    - `quantized_model_path`: Path to save the GPTQ quantized model.
    - `quantized_hub_model_id`: Name of the model at the hub post-GPTQ quantization.
    - `quantized_hub_private_repo`: If set to `True`, creates a private repository for the quantized model.

    Dataset Related:
    - `dataset_key`: Key to select the dataset from the datasets_registry.
    - `train_local_path_to_data`: Local path to the training data file.
    - `eval_local_path_to_data`: Local path to the evaluation data file.
    - `shuffle`: If `True`, shuffles the training data.
    - `max_eval_samples`: Maximum number of examples to use for evaluation.
    - `add_eval_to_train_if_no_path`: If `True`, adds evaluation data to training if there's no separate eval path.

    Tokenizer Settings:
    - `tokenizer_name_or_path`: Name or path to the tokenizer.
    - `tokenizer_use_fast`: If `True`, uses the fast version of the tokenizer.
    - `tokenizer_padding_side`: Sets padding side to 'right' or 'left'.

    Data Collator Settings:
    - `collator_key`: Key to select the collator from the collators_registry.
    - `max_length`: Maximum sequence length for the model.

    Model Configuration:
    - `model_name_or_path`: Name or path to the model to be used.
    - `push_to_hub_bos_add_bos_token`: Adds BOS token when uploading tokenization configuration to the hub.
    - `use_flash_attention_2`: Flags the use of flash attention 2.
    - `trust_remote_code`: If `True`, trusts remote code from the HuggingFace Hub.
    - `device_map`: Device map for placing model layers on specific devices.
    - `prepare_model_for_kbit_training`: If `True`, prepares the model for k-bit training.

    BitsAndBytes Integration:
    - `load_in_8bit`: Load the model in 8-bit mode using bitsandbytes.
    - `load_in_4bit`: Load the model in 4-bit mode using bitsandbytes.
    - `llm_int8_threshold`: Threshold for detecting outliers in the model weights.
    - `llm_int8_has_fp16_weight`: If `True`, the model will have fp16 weights.
    - `bnb_4bit_use_double_quant`: If `True`, a second quantization step is used for 4-bit weights.
    - `bnb_4bit_quant_type`: Specifies the quantization type used for 4-bit weights.
    - `bnb_quantize_after_model_init`: Determines when the quantization should occur.

    GPTQ Specific Parameters:
    - `gptq_bits`: Number of bits for GPTQ quantization.
    - `gptq_group_size`: Group size for GPTQ quantization.
    - `gptq_disable_exllama`: If `True`, disables ExLlama kernels during GPTQ quantization.

    LoRA Specific Parameters:
    - `apply_lora`: If `True`, applies LoRA to the model.
    - `lora_rank`: LoRA rank to define the size of the LoRA matrices.
    - `lora_alpha`: Multiplication factor for the resulting LoRA matrix.
    - `lora_dropout`: Dropout rate for LoRA.
    - `raw_lora_target_modules`: Comma-separated string of module names to apply LoRA, or 'all' to apply broadly.

    Training Arguments:
    - `output_dir`: Path to save training outputs.
    - `per_device_train_batch_size`: Batch size per device for training.
    - `do_eval`: If `True`, performs evaluation.
    - `per_device_eval_batch_size`: Batch size per device for evaluation.
    - `gradient_accumulation_steps`: Number of steps to accumulate gradients for larger effective batch size.
    - `eval_accumulation_steps`: Number of steps to accumulate gradients during evaluation.
    - `eval_delay`: Delay before the first evaluation.
    - `eval_steps`: Number of update steps between evaluations.
    - `warmup_steps`: Number of steps for learning rate warmup.
    - `max_steps`: Maximum number of training steps.
    - `num_train_epochs`: Number of epochs for training.
    - `learning_rate`: Learning rate for the optimizer.
    - `max_grad_norm`: Gradient clipping threshold.
    - `weight_decay`: Coefficient for weight decay regularization.
    - `label_smoothing_factor`: Label smoothing factor.
    - `logging_steps`: Number of steps between logging intermediate results.
    - `save_steps`: Number of training steps between checkpoints and model upload.
    - `save_total_limit`: Maximum number of checkpoints to keep.
    - `optim`: Optimizer name, overwritten by DeepSpeed if used.
    - `push_to_hub`: If `True`, model checkpoints are uploaded to HuggingFace Hub.
    - `hub_model_id`: Name of the model on the HuggingFace Hub.
    - `hub_private_repo`: If `True`, creates a private repository on the HuggingFace Hub.

    Weights & Biases Integration:
    - `report_to_wandb`: If `True`, logs metrics to Weights & Biases.
    - `wandb_api_key`: API key for Weights & Biases.
    - `wandb_project`: Project name on Weights & Biases.
    - `wandb_entity`: Entity name (user or organization) on Weights & Biases.

    Example of creating a `Config` object:
        ```python
        config = Config(
            model_name_or_path='gpt2',
            dataset_key='my_dataset',
            gradient_accumulation_steps=8,
            max_length=512,
            deepspeed_stage=3,
        )
        ```

    Note:
        - Throughout the codebase, `Config` instances are passed around to provide a unified source of configurations
          for various components.
        - It is crucial to ensure all required settings are properly set in a `Config` object before it is utilized,
          particularly when overriding defaults or specifying custom configurations.
    """

    # general
    experiment_key: str = field(
        default=enums.Experiments.base,
        metadata={"help": "Experiment class key"},
    )
    save_safetensors: bool = field(
        default=True,
        metadata={
            "help": "Use safe serialization (safe tensors) or not",
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
            "help": "Stabilize the model. Convert some weights (e.g. LoRA) to bf16",
        },
    )
    norm_fp32: bool = field(
        default=False,
        metadata={
            "help": "Convert norm to fp32",
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
    prepare_model_for_kbit_training: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Prepare or not for kbit training",
        },
    )
    offload_folder: Optional[str] = field(
        default=None,
        metadata={
            "help": "Offloading folder. Helps for fusing in colab",
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
    neftune_noise_alpha: Optional[float] = field(
        default=None,
        metadata={
            "help": "If not None, this will activate NEFTune noise embeddings. "
            "This can drastically improve model performance for instruction fine-tuning",
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
        """
        Performs a series of checks to validate the configuration for compatibility with the training environment.

        This method is responsible for ensuring that the environment is properly set up for the actions specified in
        the configuration object, such as pushing to Hugging Face's hub, using deepspeed, and using flash attention.

        It includes the following checks:
        - Verifies that credentials for Hugging Face hub are provided if the model is intended to be pushed to the hub.
        - Validates that deepspeed is installed if it is specified in the configuration.
        - Ensures that the necessary packages are installed for using flash attention if configured to do so.

        Does not return any value.

        Raises:
            - ValueError: If the configuration for hub interaction is incorrect.
            - ImportError: If any of the required libraries (e.g., deepspeed, flash-attn, auto-gptq) are not installed.

        Example usage:
            ```python
            from xllm import Config

            config = Config(...)
            # Before proceeding with training or other operations, run checks to ensure environment compatibility.
            config.check()
            ```

        Note:
            - Always invoke this method after initializing a `Config` object and before proceeding with model training
              or other operations that rely on the configuration settings.
        """
        self.check_hub()
        self.check_deepspeed()
        self.check_flash_attention()

        return None

    @property
    def correct_tokenizer_name_or_path(self) -> str:
        """
        Resolves the tokenizer name or path to be used for initializing the tokenizer.

        This property ensures that if a specific tokenizer name or path is not provided in the configuration object,
        the model name or path is used instead, maintaining consistency between model and tokenizer.

        Returns:
            `str`: The name or path of the tokenizer to be used. If `tokenizer_name_or_path` is specified in `Config`
            object, that value is used. Otherwise, `model_name_or_path` is returned as the default tokenizer identifier.

        Example usage:
            ```python
            from xllm import Config

            config = Config(model_name_or_path="gpt2", tokenizer_name_or_path=None)
            tokenizer_name_or_path = config.correct_tokenizer_name_or_path
            # tokenizer_name_or_path now holds the value "gpt2"
            ```

        Note:
            - It is a common practice to use the same identifier for both the model and its corresponding tokenizer.
              This property handles such a case automatically when the `tokenizer_name_or_path` is not explicitly set.
        """
        if self.tokenizer_name_or_path is not None:
            return self.tokenizer_name_or_path
        else:
            return self.model_name_or_path

    @property
    def lora_target_modules(self) -> Optional[List[str]]:
        """
        Interprets the LoRA target modules setting from the configuration to determine which model modules to apply
        LoRA to.

        LoRA (Low-Rank Adaptation) is a parameter-efficient training method that modifies specific layers within a
        model. This property is responsible for parsing the `raw_lora_target_modules` configuration to identify
        the specific modules (like attention key, query, and value matrices) that LoRA will be applied to.

        Returns:
            Optional[List[str]]: A list of module names to apply LoRA to if specified, otherwise `None` if LoRA should
            be applied to all eligible modules as determined by the string "all" in `raw_lora_target_modules`.

        Raises:
            ValueError: If `raw_lora_target_modules` is not set.

        Example usage:
            ```python
            from xllm import Config

            # Assuming a Config object with LoRA targets specified.
            config = Config(raw_lora_target_modules="k,q,v")
            lora_modules = config.lora_target_modules
            # lora_modules now holds the list ['k', 'q', 'v'].
            ```

        Note:
            - The `raw_lora_target_modules` should be provided as a comma-separated string specifying the target
              modules. If LoRA should be applied broadly, the value "all" can be used.
        """
        if self.raw_lora_target_modules == "all":
            return None
        elif self.raw_lora_target_modules is not None:
            modules_names = [module_name.strip() for module_name in self.raw_lora_target_modules.split(",")]
            return modules_names
        else:
            raise ValueError("raw_lora_target_modules doesn't set")

    @property
    def dtype(self) -> torch.dtype:
        """
        Determines the appropriate PyTorch data type for the model based on availability of CUDA and configuration
        settings.

        This property assists in setting computational precision for training and inference (e.g., FP32, FP16, BF16),
        basing the decision on system capabilities and user preferences as defined in the `Config` object. The selected
        data type can impact both the computational efficiency and memory usage of the model operations.

        Returns:
            `torch.dtype`: The data type to be used for the model tensors. This can be one of the following based on the
            system's CUDA support and configuration flags: `torch.float32` (FP32), `torch.float16` (FP16), or
            `torch.bfloat16` (BF16).

        Example usage:
            ```python
            from xllm import Config

            config = Config(force_fp32=False, force_fp16=True)
            model_dtype = config.dtype
            # If CUDA is available and BF16 is supported, model_dtype will be `torch.bfloat16`.
            # Otherwise, it falls back to `torch.float16` due to the forced FP16 configuration.
            ```

        Note:
            - This property plays a critical role in memory management and computational efficiency, especially when
              working with large models or limited system resources.
        """
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
        """
        Retrieves the deepspeed configuration dictionary based on settings within the `Config` object.

        This property parses the deepspeed settings from the configuration to construct the configuration dictionary
        used for ing up deepspeed in the model's training environment. It determines whether a predefined stage
        or a custom configuration file path should be utilized.

        Returns:
            `Optional[Dict[str, Any]]`: A dictionary containing deepspeed configurations, or `None` if deepspeed is not
            to be used.

        Raises:
            ValueError: If the `deepspeed_stage` specified does not correspond to a known configuration,
                         or if a custom deepspeed configuration file path does not exist.

        Example usage:
            ```python
            from xllm import Config

            # Assuming a predefined Config object with deepspeed specifications.
            config = Config(deepspeed_stage=2)
            ds_config = config.deepspeed
            # ds_config now contains the deepspeed configuration for stage 2.
            ```

        Note:
            - A deepspeed stage is a set of predefined configurations. If this is set, the corresponding configuration
              will be used and any custom deepspeed configuration file will be ignored.
            - If a custom deepspeed configuration file path is given and it exists, that configuration will be loaded
              and used.
        """
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
        """
        Compiles the configurations for Fully Sharded Data Parallel (FSDP) based on the settings in the `Config` object.

        This property creates a list containing FSDP-related options, which informs the training process whether to
        enable FSDP and which FSDP strategy to employ.

        A list of options (fsdp_strategy) along the following:
            "full_shard": Shard parameters, gradients and optimizer states.
            "shard_grad_op": Shard optimizer states and gradients.
            "offload": Offload parameters and gradients to CPUs (only compatible with "full_shard" and "shard_grad_op").
            "auto_wrap": Automatically recursively wrap layers with FSDP using default_auto_wrap_policy.

        Returns:
            `Union[str, List[str]]`: A list of FSDP options as strings. It can be an empty string if FSDP is not used or
            a list with the specified FSDP strategy and options such as offloading.

        Example usage:
            ```python
            from xllm import Config

            # Assuming a predefined Config object with FSDP specifications.
            config = Config(fsdp_strategy="full_shard", fsdp_offload=True)
            fsdp_options = config.fsdp
            ```

        Note:
            - FSDP strategies and options improve memory efficiency during distributed training by sharding the model's
              parameters across multiple devices.
            - The FSDP settings in the configuration should match the target training environment and system
              capabilities.
        """
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
        """
        Determines the name or path of the LoRA model to be used for the fusing process.

        This property resolves which model should be fused by checking whether a model ID from the Hugging Face hub or a
        local path to a LoRA model is provided in the configuration object. It is essential for the fusing operation
        when LoRA weights need to be integrated into the base model.

        Returns:
            `str`: The Hugging Face hub model ID or the local file path to the LoRA model, depending on which is
            specified.

        Raises:
            ValueError: If neither `lora_hub_model_id` nor `lora_model_local_path` is set, indicating that there is no
                        model specified for fusing.

        Example usage:
            ```python
            from xllm import Config

            # Assuming a Config object with a specified LoRA model on Hugging Face Hub or locally.
            config = Config(lora_hub_model_id="username/model-id", lora_model_local_path=None)
            model_name_or_path = config.lora_model_name_or_path_for_fusing
            # model_name_or_path will hold the value "username/model-id".
            ```

        Note:
            - This property is specifically used during the model fusing step and should be configured correctly in
              scenarios where LoRA is utilized.
        """
        if self.lora_hub_model_id is not None:
            return self.lora_hub_model_id
        elif self.lora_model_local_path is not None:
            return self.lora_model_local_path
        else:
            raise ValueError("Please set lora_hub_model_id or lora_model_local_path for fusing")

    @property
    def need_to_prepare_model_for_kbit_training(self) -> bool:
        if self.prepare_model_for_kbit_training is not None:
            return self.prepare_model_for_kbit_training
        else:
            return self.from_gptq or self.load_in_4bit or self.load_in_8bit

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
from typing import Any, Dict, Optional, Type, Union

import torch
from peft import (  # type: ignore
    PeftModel,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GPTQConfig,
    IntervalStrategy,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)

from .. import enums
from ..collators.base import BaseCollator
from ..collators.registry import collators_registry
from ..core.config import Config
from ..datasets.base import BaseDataset
from ..datasets.registry import datasets_registry
from ..trainers.lm import LMTrainer
from ..trainers.registry import trainers_registry
from ..utils.logger import dist_logger


def build_training_arguments(config: Config) -> TrainingArguments:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported() and not config.force_fp16:
            fp16 = False
            bf16 = True
        else:
            fp16 = True
            bf16 = False
    else:
        fp16 = False
        bf16 = False

    training_arguments = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=config.warmup_steps,
        learning_rate=config.learning_rate,
        max_steps=config.max_steps if config.max_steps is not None else -1,
        num_train_epochs=config.num_train_epochs,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        label_smoothing_factor=config.label_smoothing_factor,
        fp16=fp16,
        bf16=bf16,
        logging_steps=config.logging_steps,
        report_to=["wandb"] if config.report_to_wandb else None,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        hub_model_id=config.hub_model_id,
        hub_strategy="checkpoint",
        hub_token=os.environ.get(enums.EnvironmentVariables.huggingface_hub_token, None),
        push_to_hub=config.push_to_hub,
        hub_private_repo=config.hub_private_repo,
        save_safetensors=config.save_safetensors,
        fsdp=config.fsdp,
        deepspeed=config.deepspeed,
        remove_unused_columns=False,
        log_level=enums.LogLevel.info,
        disable_tqdm=False,
        logging_first_step=True,
        optim=config.optim,  # will be overwriten by deepspeed config if exist
        do_eval=config.do_eval,
        evaluation_strategy="steps" if config.do_eval else IntervalStrategy.NO,
        per_device_eval_batch_size=config.per_device_eval_batch_size or config.per_device_train_batch_size,
        eval_accumulation_steps=config.eval_accumulation_steps or config.gradient_accumulation_steps,
        eval_delay=config.eval_delay,
        eval_steps=config.eval_steps,
        seed=config.seed,
        data_seed=config.seed,
        metric_for_best_model="eval_loss" if config.do_eval else "loss",
    )
    return training_arguments


def build_dataset(config: Config, is_train: bool = True, **kwargs: Any) -> Optional[BaseDataset]:
    if is_train:
        path_to_data = config.train_local_path_to_data
    elif config.eval_local_path_to_data is not None:
        path_to_data = config.eval_local_path_to_data
    else:
        return None

    dataset_cls: Type[BaseDataset] = datasets_registry.get(key=config.dataset_key)

    if issubclass(dataset_cls, BaseDataset):
        dataset = dataset_cls.load(path_to_data=path_to_data, **kwargs)
    else:
        raise ValueError(f"Unknown type of dataset: {dataset_cls.__name__}")

    return dataset


def build_tokenizer(config: Config, use_fast: Optional[bool] = None) -> PreTrainedTokenizer:
    kwargs = dict()

    if use_fast is not None:
        kwargs["use_fast"] = use_fast

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=config.correct_tokenizer_name_or_path, **kwargs
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        dist_logger.info(message="Tokenizer pad token set to eos token", local_rank=config.local_rank)

    if config.tokenizer_padding_side is not None:
        tokenizer.padding_side = config.tokenizer_padding_side
        dist_logger.info(
            message=f"Tokenizer padding side set to {config.tokenizer_padding_side}", local_rank=config.local_rank
        )

    return tokenizer


def build_collator(config: Config, tokenizer: PreTrainedTokenizer, **kwargs: Any) -> BaseCollator:
    collator_cls: Type[BaseCollator] = collators_registry.get(key=config.collator_key)

    if not issubclass(collator_cls, BaseCollator):
        raise ValueError(f"Unknown type of collator: {collator_cls.__name__}")

    collator = collator_cls(tokenizer=tokenizer, max_length=config.max_length, **kwargs)

    return collator


def build_quantization_config(
    config: Config,
) -> Union[BitsAndBytesConfig, GPTQConfig, None]:
    if config.from_gptq:
        quantization_config = GPTQConfig(
            bits=config.gptq_bits,
            group_size=config.gptq_group_size,
            disable_exllama=config.gptq_disable_exllama,
        )
    elif config.load_in_8bit or config.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=config.load_in_8bit,
            load_in_4bit=config.load_in_4bit,
            llm_int8_threshold=config.llm_int8_threshold,
            llm_int8_has_fp16_weight=config.llm_int8_has_fp16_weight,
            bnb_4bit_compute_dtype=config.dtype,
            bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        )
    else:
        quantization_config = None

    return quantization_config


def build_model(
    config: Config,
    quantization_config: Union[BitsAndBytesConfig, GPTQConfig, None],
    low_cpu_mem_usage: Optional[bool] = None,
) -> PreTrainedModel:
    if config.bnb_quantize_after_model_init:
        quantization_config = None
        dist_logger("bnb quantization is expected later")

    if config.use_gradient_checkpointing:
        use_cache = False
    else:
        use_cache = True

    kwargs: Dict[str, Any] = dict()

    if config.use_flash_attention_2:
        kwargs["use_flash_attention_2"] = True

    if low_cpu_mem_usage is not None:
        kwargs["low_cpu_mem_usage"] = low_cpu_mem_usage

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=config.model_name_or_path,
        quantization_config=quantization_config,
        torch_dtype=config.dtype,
        trust_remote_code=config.trust_remote_code,
        device_map=config.device_map,
        use_cache=use_cache,
        **kwargs,
    )
    model.config.pretraining_tp = 1

    if quantization_config is not None and config.prepare_model_for_kbit_training:
        model = prepare_model_for_kbit_training(
            model=model, use_gradient_checkpointing=config.use_gradient_checkpointing
        )
        dist_logger(
            message=f"Model prepared for kbit training. Gradient checkpointing: {config.use_gradient_checkpointing}",
            local_rank=config.local_rank,
        )

    return model


def build_trainer(
    config: Config,
    pad_token_id: int,
    training_arguments: TrainingArguments,
    model: Union[PreTrainedModel, PeftModel],
    train_dataset: BaseDataset,
    collator: BaseCollator,
    eval_dataset: Optional[BaseDataset] = None,
    **kwargs: Any,
) -> LMTrainer:
    trainer_cls = trainers_registry.get(key=config.trainer_key)

    if not issubclass(trainer_cls, Trainer):
        raise ValueError(f"Unknown type of trainer: {trainer_cls.__name__}")

    trainer: LMTrainer = trainer_cls(
        config=config,
        model=model,
        args=training_arguments,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        ignore_index=pad_token_id,
        **kwargs,
    )

    try:
        model.config.use_cache = False  # type: ignore
    except Exception as exception:
        dist_logger.warning(
            message=f"Can't set use cache to false. Exception: {exception}",
            local_rank=config.local_rank,
        )

    return trainer

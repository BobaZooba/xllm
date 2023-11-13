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
from time import sleep
from typing import Any, Dict, Optional, Union

import torch
import torch.distributed as distributed
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training  # type: ignore
from transformers import (
    BitsAndBytesConfig,
    GPTQConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainingArguments,
)
from transformers.integrations.bitsandbytes import replace_with_bnb_linear

from ..collators.base import BaseCollator
from ..core.config import Config
from ..core.dependencies import (
    build_collator,
    build_dataset,
    build_model,
    build_quantization_config,
    build_tokenizer,
    build_trainer,
    build_training_arguments,
)
from ..datasets.base import BaseDataset
from ..trainers.lm import LMTrainer
from ..utils.logger import dist_logger
from ..utils.miscellaneous import is_distributed_training
from ..utils.nn import apply_lora, stabilize_training
from ..utils.post_training import post_training, push_to_hub_bos_add_bos_token


class Experiment:
    def __init__(
        self,
        config: Config,
        training_arguments: Optional[TrainingArguments] = None,
        train_dataset: Optional[BaseDataset] = None,
        eval_dataset: Optional[BaseDataset] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        collator: Optional[BaseCollator] = None,
        quantization_config: Union[BitsAndBytesConfig, GPTQConfig, None] = None,
        model: Union[PreTrainedModel, PeftModel, None] = None,
        lora_config: Optional[LoraConfig] = None,
        trainer: Optional[LMTrainer] = None,
    ):
        self.config = config

        self.training_arguments = training_arguments
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.collator = collator
        self.quantization_config = quantization_config
        self.model = model
        self.lora_config = lora_config
        self.trainer = trainer

        self.internal_checks()

    def internal_checks(self) -> None:
        if self.tokenizer is not None and self.collator is not None and self.tokenizer != self.collator.tokenizer:
            dist_logger.warning("Tokenizer not equals to tokenizer in collator")

        return None

    def internal_checks_before_train(self) -> None:
        if self.training_arguments is not None and self.training_arguments.do_eval and self.eval_dataset is None:
            raise ValueError(
                f"You set do_eval at config to {self.config.do_eval}, "
                "but experiment can't run eval, because eval_dataset is None. "
                f"config.eval_local_path_to_data: {self.config.eval_local_path_to_data}"
            )

    def build(self):
        dist_logger("Experiment building has started")
        self.at_beginning()
        self.save_config()
        dist_logger.info("Config saved")

        self.before_checks()
        self.checks()
        dist_logger("Checks passed successfully")
        self.after_checks()

        if self.training_arguments is None:
            self.before_training_arguments_build()
            self.training_arguments = self.build_training_arguments()
            dist_logger(f"Training arguments was built:\n{self.training_arguments.to_json_string()}")
            self.after_training_arguments_build()

        if self.train_dataset is None:
            self.before_train_dataset_build()
            self.train_dataset = self.build_train_dataset()
            dist_logger(
                f"Train dataset {self.train_dataset.__class__.__name__} was built. Size: {len(self.train_dataset)}"
            )
            self.after_train_dataset_build()

        if self.eval_dataset is None:
            self.before_eval_dataset_build()
            self.eval_dataset = self.build_eval_dataset()
            if self.eval_dataset is not None:
                if self.training_arguments is not None:
                    self.training_arguments.do_eval = True
            if self.eval_dataset is not None:
                dist_logger(
                    f"Eval dataset {self.eval_dataset.__class__.__name__} was built. Size: {len(self.eval_dataset)}"
                )
            else:
                dist_logger("Eval dataset is None")
            self.after_eval_dataset_build()

        if self.tokenizer is None:
            self.before_tokenizer_build()
            self.tokenizer = self.build_tokenizer()
            dist_logger(f"Tokenizer {self.config.correct_tokenizer_name_or_path} was built")
            self.after_tokenizer_build()

        if self.collator is None:
            self.before_collator_build()
            self.collator = self.build_collator()
            dist_logger(f"Collator {self.collator.__class__.__name__} was built")
            self.after_collator_build()

        if self.quantization_config is None:
            self.before_quantization_config_build()
            self.quantization_config = self.build_quantization_config()
            if self.quantization_config is not None:
                dist_logger(f"Quantization config was built:\n{self.quantization_config.to_json_string()}")
            else:
                dist_logger(f"Quantization config is None. Model will be loaded using {self.config.dtype}")
            self.after_quantization_config_build()

        if self.model is None:
            self.before_model_build()
            self.model = self.build_model()
            dist_logger(f"Model {self.config.model_name_or_path} was built")
            self.after_model_build()
        elif self.quantization_config is not None:
            dist_logger.warning("quantization_config is not None, but the model was built outside of the experiment")

        if self.is_bnb_quantization and self.config.bnb_quantize_after_model_init:
            self.before_bnb_quantization()
            self.bnb_quantization()
            bnb_quantization_type = "int4" if self.quantization_config.load_in_4bit else "int8"
            dist_logger(f"Bnb quantization applyed. Type: {bnb_quantization_type}")
            self.after_bnb_quantization()

        if self.config.apply_lora:
            self.before_lora_apply()
            self.lora_config = self.apply_lora()
            dist_logger(f"LoRA applied to the model {self.config.model_name_or_path}")
            self.after_lora_apply()

        if self.config.stabilize:
            self.before_stabilize_training()
            self.stabilize_training()
            dist_logger(f"Model {self.config.model_name_or_path} is stabilized for training")
            self.after_stabilize_training()

        if self.trainer is None:
            self.before_trainer_build()
            self.trainer = self.build_trainer()
            dist_logger(f"Trainer {self.trainer.__class__.__name__} was built")
            self.after_trainer_build()
        else:
            trainer_components = self.trainer_components
            trainer_components_is_not_none = [key for key, value in trainer_components.items() if value is not None]
            dist_logger.warning(
                "Trainer was built outside of the experiment, "
                f"but this components is not None: {', '.join(trainer_components_is_not_none)}."
            )

        self.internal_checks()

        dist_logger("Experiment built successfully")

    @property
    def is_bnb_quantization(self) -> bool:
        return isinstance(self.quantization_config, BitsAndBytesConfig) and self.config.from_gptq

    @property
    def trainer_components(self) -> Dict[str, Any]:
        components = {
            "training_arguments": self.training_arguments,
            "train_dataset": self.train_dataset,
            "eval_dataset": self.eval_dataset,
            "collator": self.collator,
            "model": self.model,
        }
        return components

    # checks
    def before_checks(self) -> None:
        return None

    def checks(self) -> None:
        if not torch.cuda.is_available():
            dist_logger.warning("CUDA is not available")

        self.config.check()

        return None

    def after_checks(self) -> None:
        return None

    # training arguments
    def before_training_arguments_build(self) -> None:
        return None

    def build_training_arguments(self) -> TrainingArguments:
        training_arguments = build_training_arguments(config=self.config)
        return training_arguments

    def after_training_arguments_build(self) -> None:
        return None

    # train_dataset
    def before_train_dataset_build(self) -> None:
        return None

    def build_train_dataset_additional_kwargs(self) -> Dict[str, Any]:
        return dict()

    def build_train_dataset(self) -> BaseDataset:
        train_dataset_additional_kwargs = self.build_train_dataset_additional_kwargs()
        dataset = build_dataset(config=self.config, is_train=True, **train_dataset_additional_kwargs)
        if dataset is None:
            raise ValueError("Train dataset can't be loaded")
        return dataset

    def after_train_dataset_build(self) -> None:
        return None

    # eval_dataset
    def before_eval_dataset_build(self) -> None:
        return None

    def build_eval_dataset_additional_kwargs(self) -> Dict[str, Any]:
        return self.build_train_dataset_additional_kwargs()

    def build_eval_dataset(self) -> Optional[BaseDataset]:
        eval_dataset_additional_kwargs = self.build_eval_dataset_additional_kwargs()
        dataset = build_dataset(config=self.config, is_train=False, **eval_dataset_additional_kwargs)
        return dataset

    def after_eval_dataset_build(self) -> None:
        return None

    # tokenizer
    def before_tokenizer_build(self) -> None:
        return None

    def build_tokenizer(self) -> PreTrainedTokenizer:
        tokenizer = build_tokenizer(config=self.config, use_fast=self.config.tokenizer_use_fast)
        return tokenizer

    def after_tokenizer_build(self) -> None:
        return None

    # collator
    def before_collator_build(self) -> None:
        return None

    def build_collator_additional_kwargs(self) -> Dict[str, Any]:
        return dict()

    def build_collator(self) -> BaseCollator:
        collator_additional_kwargs = self.build_collator_additional_kwargs()
        collator = build_collator(config=self.config, tokenizer=self.tokenizer, **collator_additional_kwargs)
        return collator

    def after_collator_build(self) -> None:
        return None

    # quantization_config
    def before_quantization_config_build(self) -> None:
        return None

    def build_quantization_config(self) -> Union[BitsAndBytesConfig, GPTQConfig, None]:
        quantization_config = build_quantization_config(config=self.config)
        return quantization_config

    def after_quantization_config_build(self) -> None:
        return None

    # model
    def before_model_build(self) -> None:
        return None

    def build_model(self) -> PreTrainedModel:
        quantization_config = (
            None if self.is_bnb_quantization and self.config.bnb_quantize_after_model_init else self.quantization_config
        )
        model = build_model(config=self.config, quantization_config=quantization_config)
        return model

    def after_model_build(self) -> None:
        return None

    # bnb_quantization
    def before_bnb_quantization(self) -> None:
        return None

    def bnb_quantization(self) -> None:
        self.model = replace_with_bnb_linear(
            model=self.model,
            quantization_config=self.quantization_config,
        )
        self.model.is_loaded_in_4bit = self.config.load_in_4bit
        self.model.is_loaded_in_8bit = self.config.load_in_8bit
        if self.config.prepare_model_for_kbit_training:
            self.model = prepare_model_for_kbit_training(
                model=self.model, use_gradient_checkpointing=self.config.use_gradient_checkpointing
            )

    def after_bnb_quantization(self) -> None:
        return None

    # lora
    def before_lora_apply(self) -> None:
        return None

    def apply_lora(self) -> LoraConfig:
        self.model, lora_config = apply_lora(config=self.config, model=self.model, lora_config=self.lora_config)
        return lora_config

    def after_lora_apply(self) -> None:
        return None

    # stabilize_training
    def before_stabilize_training(self) -> None:
        return None

    def stabilize_training(self) -> None:
        self.model = stabilize_training(model=self.model)

    def after_stabilize_training(self) -> None:
        return None

    # trainer
    def before_trainer_build(self) -> None:
        return None

    def build_trainer_additional_kwargs(self) -> Dict[str, Any]:
        return dict()

    def build_trainer(self) -> LMTrainer:
        additional_trainer_kwargs = self.build_trainer_additional_kwargs()

        if self.tokenizer is None:
            raise ValueError("tokenizer is None")

        if self.train_dataset is None:
            raise ValueError("train_dataset is None")

        if self.collator is None:
            raise ValueError("collator is None")

        trainer = build_trainer(
            config=self.config,
            pad_token_id=self.tokenizer.pad_token_id,
            training_arguments=self.training_arguments,
            model=self.model,
            train_dataset=self.train_dataset,
            collator=self.collator,
            eval_dataset=self.eval_dataset,
            **additional_trainer_kwargs,
        )

        return trainer

    def after_trainer_build(self) -> None:
        return None

    def save_config(self) -> None:
        json_config = json.dumps(self.config.__dict__, indent=2)
        dist_logger(f"Config:\n{json_config}")

        if is_distributed_training():
            if distributed.get_rank() == self.config.local_rank:
                os.makedirs(self.config.output_dir, exist_ok=True)
                with open(os.path.join(self.config.output_dir, "training_config.json"), "w") as file_object:
                    file_object.write(json_config)
        else:
            os.makedirs(self.config.output_dir, exist_ok=True)
            with open(os.path.join(self.config.output_dir, "training_config.json"), "w") as file_object:
                file_object.write(json_config)

        return None

    def before_train(self) -> None:
        return None

    def run(self):
        self.before_train()

        if self.trainer is None:
            raise ValueError("trainer is None")

        if self.training_arguments is None:
            raise ValueError("training_arguments is None")

        self.internal_checks_before_train()
        dist_logger("Training will start soon")
        self.trainer.train()
        dist_logger("Training end")

        self.after_train()

        if self.config.fuse_after_training:
            self.fuse_lora()

        if is_distributed_training():
            if distributed.get_rank() == self.config.local_rank:
                post_training(config=self.config, tokenizer=self.tokenizer)
        else:
            post_training(config=self.config, tokenizer=self.tokenizer)

        dist_logger(f"Model saved to {self.training_arguments.output_dir}")

        self.at_end()

    def push_to_hub(
        self,
        repo_id: Optional[str] = None,
        private: Optional[bool] = None,
        safe_serialization: Optional[bool] = None,
        need_push_to_hub_bos_add_bos_token: Optional[bool] = None,
    ) -> None:
        repo_id = repo_id or self.config.hub_model_id

        private = private if private is not None else self.config.hub_private_repo
        safe_serialization = safe_serialization if safe_serialization is not None else self.config.save_safetensors
        need_push_to_hub_bos_add_bos_token = (
            need_push_to_hub_bos_add_bos_token
            if need_push_to_hub_bos_add_bos_token is not None
            else self.config.push_to_hub_bos_add_bos_token
        )

        if repo_id is None:
            raise ValueError("repo_id and hub_model_id is None, but you want to push to HF hub")

        if self.tokenizer is None:
            dist_logger.warning("Tokenizer is None. Can't push to the hub")
        else:
            self.tokenizer.push_to_hub(repo_id=repo_id, private=private)
            sleep(10.0)
            if need_push_to_hub_bos_add_bos_token:
                push_to_hub_bos_add_bos_token(repo_id=repo_id)

        if self.model is None:
            raise ValueError("Model is None. Can't push to the hub")
        else:
            self.model.push_to_hub(
                repo_id=repo_id,
                private=private,
                safe_serialization=safe_serialization,
                max_shard_size=self.config.max_shard_size,
            )

    def fuse_lora(self) -> PreTrainedModel:
        if not self.config.apply_lora:
            dist_logger.warning("Apply LoRA set to False at config")

        if isinstance(self.model, PeftModel):
            self.model = self.model.merge_and_unload()
        else:
            raise TypeError(f"Can't fuse model, this the model is not the PeftModel. Model type: {type(self.model)}")

        dist_logger("LoRA fused")

        self.after_fuse()

        return self.model

    def after_fuse(self) -> None:
        return None

    def after_train(self) -> None:
        return None

    def at_beginning(self) -> None:
        return None

    def at_end(self) -> None:
        return None

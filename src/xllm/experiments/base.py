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
    """
    The Experiment class orchestrates the setup, execution, and management of the training process for LLM's.
    It encapsulates the creation of datasets, models, tokenizers, collators, and trainers, alongside handling their
    respective configurations and ensuring compatibility across components. The class provides an integrated environment
    to apply model quantization, Low-Rank Adaptation (LoRA), perform training, evaluation, fusing LoRA
    and pushing the results to the Hugging Face Hub.

    The class provides methods for various stages of the experiment lifecycle:

    - `__init__`: Initializes the experiment with user-provided or default components and configurations.
    - `build`: Constructs all necessary components, setting up the environment for the training process.
    - `run`: Executes the training according to the configuration and prebuilt components, handles post-training
    activities, and optionally fuses LoRA parameters.
    - `push_to_hub`: Uploads the model and tokenizer to the Hugging Face Hub for sharing and deployment.
    - `fuse_lora`: Integrates LoRA parameters for streamlined model deployment if LoRA is applied during training.

    Throughout the experiment life cycle, several hooks (`before_*` and `after_*` methods) are provided for users
    to inject custom logic or steps into the process at defined points.

    The Experiment class is designed to be flexible and extensible, allowing users to customize the experiment
    by providing their implementations of datasets, models, collators, and trainers or relying on the defaults
    determined by the given configuration parameters.

    By handling the intricate setup and ensuring all components work harmoniously, the Experiment class provides
    a structured approach to training language models, thereby simplifying the process for users.

    Attributes:
        config (`Config`): Holds the entire configuration for the experiment, including model, dataset,
            and training parameters.
        training_arguments (`Optional[TrainingArguments]`): Encapsulates arguments for training,
            such as batch size, learning rate, and saving preferences.
        train_dataset (`Optional[BaseDataset]`): The dataset for training the model.
        eval_dataset (`Optional[BaseDataset]`): The dataset for evaluating the model.
        tokenizer (`Optional[PreTrainedTokenizer]`): Processes text data for model input.
        collator (`Optional[BaseCollator]`): Prepares batches of data for the model.
        quantization_config (`Union[BitsAndBytesConfig, GPTQConfig, None]`): Settings for model quantization
            to reduce size and improve speed.
        model (`Union[PreTrainedModel, PeftModel, None]`): The actual model object that will be trained.
        lora_config (`Optional[LoraConfig]`): Configuration for LoRA.
        trainer (`Optional[LMTrainer]`): Manages and executes the training process.

    The class requires at least a configuration object to be passed during initialization, while other components
    can be optionally specified and will otherwise be built based on the provided configuration.
    """

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
        """
        Initializes an experiment environment to set up and execute the training process of language models.

        Args:
            config (`Config`):
                A configuration object containing all necessary parameters for the experiment such as model details,
                dataset paths, training hyperparameters, etc.
            training_arguments (`Optional[TrainingArguments]`, defaults to `None`):
                Arguments relevant to the training process such as batch size, learning rate, number of epochs,
                and the device to be used for training. If `None`, it will be built based on `config`.
            train_dataset (`Optional[BaseDataset]`, defaults to `None`):
                The dataset to be used for training the model. If `None`, it will be constructed from the details
                present in `config`.
            eval_dataset (`Optional[BaseDataset]`, defaults to `None`):
                The dataset to be used for evaluating the model. It's built only if required and `None` is provided.
            tokenizer (`Optional[PreTrainedTokenizer]`, defaults to `None`):
                The tokenizer instance for text preprocessing. If `None`, it will be built based on `config`.
            collator (`Optional[BaseCollator]`, defaults to `None`):
                A collator instance that prepares data batches for input into the model. If `None`, it will be built
                based on `config`.
            quantization_config (`Union[BitsAndBytesConfig, GPTQConfig, None]`, defaults to `None`):
                Configuration object for model quantization, which can help reduce model size and improve
                inference speed. If not provided, and quantization is desired, it will be built based on `config`.
            model (`Union[PreTrainedModel, PeftModel, None]`, defaults to `None`):
                The model that will undergo training. If `None`, it will be built using the provided `config`.
            lora_config (`Optional[LoraConfig]`, defaults to `None`):
                Configuration for applying Low-Rank Adaptation (LoRA) to enhance the model's capabilities with
                minimal parameter increase. If LoRA is desired and `lora_config` is `None`, it will be constructed.
            trainer (`Optional[LMTrainer]`, defaults to `None`):
                The trainer instance responsible for managing the model's training process. If `None`, it will
                be built considering the other provided components and the `config`.

        The constructor method sets up the `Experiment` with the necessary components for training, creating
        default instances for any component not provided. It also includes checks to ensure provided components
        are compatible and satisfies internal conditions for training to proceed.
        """
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
        """
        Constructs the various components required for running the experiment, including datasets, tokenizer, collator,
        model, and trainer.

        This method handles the sequential construction and initialization of components, ensuring that each is
        configured correctly and ready for the training process. If any components have not been externally provided
        during initialization, they will be built using the configuration parameters from the `Config` object.

        Following the build sequence, if any component is not initialized, the method builds them as follows:

        - It checks for the presence of a `TrainingArguments` instance and builds it if necessary, setting training
        parameters such as directories, logging, and device usage.

        - If a training dataset is not provided, the method builds it using the training data location specified in
        the configuration object.

        - An evaluation dataset is built similarly if evaluation is required and no dataset was provided.

        - A tokenizer is built to process the text data for training and evaluation if not already supplied.

        - The data collator, which prepares model inputs, is built if absent.

        - The model quantization configuration is built if quantization is desired and no configuration was supplied.

        - The actual model to be trained is built using the model details from the configuration object if not already
        provided.

        - If quantization is requested through `BitsAndBytesConfig` and deferred until after model initialization, it
        is applied now.

        - If Low-Rank Adaptation (LoRA) is configured to be applied, the corresponding adjustments are made
        to the model.

        - If the model requires stabilization before training, it is stabilized.

        - Finally, the trainer is built, which encapsulates the training logic and handles the execution of
        the training process.

        Each step includes pre- and post-construction hooks that allow for custom operations before and after
        building each component. Additionally, the method includes checks to validate the correct assembly and setup
        of the entire experiment before proceeding with the training.

        After the build process completes, an internal consistency check is performed to ensure that all components
        are compatible and the experiment is ready to run.
        """

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
        if self.config.need_to_prepare_model_for_kbit_training:
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
        self.model = stabilize_training(model=self.model, norm_fp32=self.config.norm_fp32)

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
        """
        Executes the training process for the experiment. Before calling this method, the build method must be called

        Before beginning, this method runs a series of internal checks to validate that the experiment is set up
        correctly and all necessary components are in place. This includes verifying that the trainer and
        training arguments are not `None` and checking if evaluation is requested, ensuring that the evaluation dataset
        is available.

        The method then performs the following steps:

        - Calls the `before_train` method, which serves as a hook for any pre-training procedures or custom actions
        to be performed just before training starts.

        - Starts the training process by calling the `train` method on the trainer object.

        - Logs the completion of training and proceeds to any post-training steps.

        - If the `fuse_after_training` flag is set in the configuration, LoRA layers, if used, are integrated into
        the main model parameters.

        - Handles the post-training tasks such as model saving and optionally pushing the trained model
        to the Hugging Face Hub.

        - Calls the `after_train` method, a hook for post-training actions that need to be executed after
        the entire training process has completed.

        - Lastly, it performs any actions required at the end of the experiment via the `at_end` method.

        If the process was successful, the model will have updated weights that reflect the training it underwent,
        and all artifacts such as logs, model checkpoints, and final model files will be saved at their respective
        locations as defined in the training arguments.

        Note: This method assumes that all the necessary components are already built and the experiment is ready
        to run. If this is not the case, an appropriate `ValueError` will be raised indicating which required component
        is missing.
        """

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
        """
        Pushes the trained model and its tokenizer to the Hugging Face Hub.

        This method helps you to upload the final trained model and its tokenizer directly to the Hugging Face
        Hub, making it easily accessible for sharing and deploying.

        Args:
            repo_id (`Optional[str]`, defaults to `None`):
                The repository name for the model on the Hugging Face Hub. If `None`, it defaults to using the
                `hub_model_id` from the configuration.
            private (`Optional[bool]`, defaults to `None`):
                A boolean flag to set the repository as private or public. If `None`, it uses the `hub_private_repo`
                setting from the configuration.
            safe_serialization (`Optional[bool]`, defaults to `None`):
                A boolean flag to enable safe serialization of model weights. If `None`, it uses the `save_safetensors`
                setting from the configuration.
            need_push_to_hub_bos_add_bos_token (`Optional[bool]`, defaults to `None`):
                A boolean flag to indicate if there is a need to handle the special case for BOS tokens when the model
                uses `bos_token`. If `None`, it uses the `push_to_hub_bos_add_bos_token` setting from the configuration.

        This method checks for proper initialization of the repository ID (`repo_id`) and raises a `ValueError` if it's
        not provided and not specified in the configuration. It then proceeds to push the tokenizer and model
        to the Hugging Face Hub using the provided parameters or defaults from the configuration.

        The model is uploaded with the specified serialization method to ensure compatibility and potential sharding
        for very large models (`max_shard_size` setting from the configuration). The tokenizer is also uploaded,
        and if needed, an additional procedure is invoked to handle special cases for BOS tokens.

        Note: If the method encounters a `None` value for the tokenizer when it attempts to push it to the hub,
        a warning is logged, and no action is taken for the tokenizer.

        By the end of this method, the artifacts (model and tokenizer) are available on the Hugging Face Hub at the
        specified repository, accessible according to the privacy settings.
        """

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
        """
        Integrates Low-Rank Adaptation (LoRA) parameters into the main model parameters, effectively 'fusing' them.

        This method is called after training if the `fuse_after_training` flag is set in the configuration.
        Fusing LoRA parameters is a process of merging LoRA's low-rank matrices into the main model's
        weight matrices, which reduces the number of parameters and can potentially streamline deployment for inference.

        The method performs the following steps:

        - Checks whether LoRA was applied during the setup of the experiment. If LoRA was not used or the
        `apply_lora` flag in the configuration is set to `False`, a warning is logged, and no fusing is performed.

        - If the model is an instance of `PeftModel` (a model class that supports parameter-efficient
        fine-tuning techniques like LoRA), it proceeds to merge LoRA parameters with the main model parameter
        matrices. If the model is not of type `PeftModel`, a `TypeError` is raised.

        - Logs the completion of the LoRA fusion process.

        - Executes any custom operations or cleanup needed after the fusion process through the `after_fuse` method,
        which serves as a hook.

        Upon successful completion of this method, the model's parameters will be updated to reflect the incorporation
        of LoRA adjustments, and the model will be ready for further actions such as evaluation, saving, or deployment.

        Returns:
            `PreTrainedModel`: The updated model with LoRA parameters fused into the main model weights.
        """

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

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
    """
    Constructs `TrainingArguments` for model training from the provided configuration object.

    This function determines the appropriate training parameters based on the system's capabilities and the user's
    configuration, setting up arguments that control aspects of the training process such as batch size, learning
    rate, weight decay, and hardware acceleration options.

    Args:
        config (`Config`):
            A configuration object containing necessary specifications for setting up the training environment.

    Returns:
        `TrainingArguments`: An instance of the `TrainingArguments` class with all the provided configuration settings
        applied. This object is then utilized by the training process.

    The function checks whether training is supported using mixed precision (both FP16 and BF16) depending on CUDA
    availability and settings in the config object. It also adjusts the weight saving and evaluation strategies
    according to the specified conditions, among other settings.

    Example usage:
        ```python
        from xllm import Config

        # Assuming a predefined Config object.
        config = Config(...)
        training_args = build_training_arguments(config=config)
        # training_args is now ready to be passed to Trainer or any training loop.
        ```

    Note:
        - This function does not train the model but merely prepares the arguments required for training.
        - It is important to ensure that the `Config` object has accurate and intended values, as they will directly
          impact all aspects of the model's training strategy.
    """
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
        neftune_noise_alpha=config.neftune_noise_alpha,
    )
    return training_arguments


def build_dataset(config: Config, is_train: bool = True, **kwargs: Any) -> Optional[BaseDataset]:
    """
    Creates an instance of the dataset class specified in the configuration object.

    This function is responsible for constructing the dataset to be used for training or evaluation, leveraging the
    dataset registry to find the corresponding class and initializing it with the provided configuration path and
    arguments.

    Args:
        config (`Config`):
            The configuration object containing the dataset-related settings including the path to data and dataset key.
        is_train (`bool`, optional, defaults to `True`):
            A flag indicating whether to construct the training dataset or evaluation dataset. If `True`, the function
            constructs the training dataset; otherwise, it constructs the evaluation dataset if specified in the config.
        **kwargs (`Any`):
            Additional keyword arguments that are passed to the dataset class upon construction.

    Returns:
        Optional[BaseDataset]: An instance of the derived `BaseDataset` class if successfully created, otherwise `None`.

    The function performs the following operations:
    - Determines the path to the dataset based on whether a training or evaluation dataset is requested.
    - Retrieves the specified dataset class from the datasets registry using the key provided in the configuration.
    - Instantiates the dataset class with the determined path and any additional keyword arguments.

    Raises:
        ValueError: If the dataset class cannot be found in the registry or the type is not a subclass of `BaseDataset`.

    Example usage:
        ```python
        from xllm import Config

        # Assuming a predefined Config object with the dataset specifications.
        config = Config(...)
        train_dataset = build_dataset(config=config, is_train=True)
        eval_dataset = build_dataset(config=config, is_train=False)
        # train_dataset and eval_dataset are now ready to be used in the training process.
        ```

    Note:
        - If the path to data for the specified type of dataset does not exist in the configuration, the function will
          return `None`.
        - This function is designed to abstract away the dataset initialization, allowing for a centralized and
          consistent approach to dataset construction based on the configuration settings.
    """
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
    """
    Constructs the tokenizer for processing the text according to the specifications provided in the configuration
    object.

    This function loads the tokenizer from the path specified in the `Config` object. If requested, it ensures the
    tokenizer uses fast implementation (when available), and sets the padding token and side according to the given
    configuration.

    Args:
        config (`Config`):
            The configuration object containing tokenizer settings such as the path to the tokenizer and the desired
            padding side.
        use_fast (`bool`, optional, defaults to `None`):
            A flag indicating whether to use the fast version of the tokenizer if available. When set to `None`,
            falls back to the default behavior of tokenizer class.

    Returns:
        `PreTrainedTokenizer`: An instance of the `PreTrainedTokenizer` class loaded and configured as specified.

    The function carries out the following steps:
    - Loads the tokenizer from the pretrained path specified in the configuration.
    - If the tokenizer does not have a defined padding token, sets it to the `eos_token`.
    - If padding side settings are provided, configures the tokenizer to apply padding on the specified side.

    Example usage:
        ```python
        from xllm import Config

        # Assuming a predefined Config object with tokenizer path and padding preferences.
        config = Config(...)
        tokenizer = build_tokenizer(config=config)
        # tokenizer is now ready for text processing.
        ```

    Note:
        - This function prepares the tokenizer for use in data preparation and model inputs generation.
        - It is crucial to specify the tokenizer's path in the `Config` object for the correct initialization.
    """
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
    """
    Creates a data collator instance, which is responsible for collating batches of data when fed to the model during
    training.

    This function fetches the appropriate collator class from a registry using the key provided in the configuration.
    It then instantiates the collator with the tokenizer and any additional arguments necessary to prepare the data
    according to the model's requirements.

    Args:
        config (`Config`):
            The configuration object containing the key to identify the appropriate collator class and other related
            settings.
        tokenizer (`PreTrainedTokenizer`):
            The tokenizer instance that will be used by the collator to tokenize and encode the input data.
        **kwargs (`Any`):
            Additional keyword arguments that may be required by the specific collator class during instantiation.

    Returns:
        `BaseCollator`: An instance of a subclass of `BaseCollator` that is ready to format batches of data for the
        model.

    The function carries out the following operations:
    - Identifies the collator class using the `collator_key` from the configuration object's registry.
    - Initializes the collator class with the tokenizer along with any custom configurations or arguments.

    Raises:
        ValueError: If the type of collator found in the registry does not inherit from `BaseCollator`.

    Example usage:
        ```python
        from xllm import Config

        # Assuming a predefined Config object and a loaded tokenizer.
        config = Config(...)
        tokenizer = load_tokenizer(...)
        collator = build_collator(config=config, tokenizer=tokenizer)
        # collator is now ready to be used for creating model-ready data batches during training.
        ```

    Note:
        - The data collator prepares and formats the data in a manner suitable for the model's input requirements.
        - Ensure the correct collator key is specified in the `Config` object for proper data collator retrieval and
          initialization.
    """
    collator_cls: Type[BaseCollator] = collators_registry.get(key=config.collator_key)

    if not issubclass(collator_cls, BaseCollator):
        raise ValueError(f"Unknown type of collator: {collator_cls.__name__}")

    collator = collator_cls(tokenizer=tokenizer, max_length=config.max_length, **kwargs)

    return collator


def build_quantization_config(
    config: Config,
) -> Union[BitsAndBytesConfig, GPTQConfig, None]:
    """
    Constructs a configuration for model quantization based on the settings provided in the configuration object.

    This function generates either a `BitsAndBytesConfig` or a `GPTQConfig` instance, which are used to inform the
    quantization process for a language model. The function decides which quantization method to apply based on the
    flags set in the `Config` object.

    Args:
        config (`Config`):
            The configuration object that contains the flags and settings specifying the quantization method
            and parameters.

    Returns:
        Union[BitsAndBytesConfig, GPTQConfig, None]:
            An instance of the quantization configuration class corresponding to the method chosen based on the
            configuration settings, or `None` if quantization is not configured.

    The function inspects the configuration to determine the following:
    - If GPTQ-based quantization is specified, it sets up a `GPTQConfig` with the designated bit precision and grouping
      size.
    - If bitsandbytes (bnb) methodology is specified, returns a `BitsAndBytesConfig` with the respective settings.
    - If neither quantization approach is specified or the required settings are absent, it returns `None`.

    Example usage:
        ```python
        from xllm import Config

        # Assuming a predefined Config object with quantization settings.
        config = Config(...)
        quantization_config = build_quantization_config(config=config)
        # quantization_config is now ready to be applied in the model quantization process.
        ```

    Note:
        - Having the correct quantization settings in the `Config` object is crucial, as they dictate the behavior
          of the quantization process and impact the model size and computational speed after quantization.
    """
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
    """
    Constructs the language model from a pretrained path with potential quantization configurations and customizations.

    This function loads the model specified in the configuration object. It can also apply quantization settings as
    defined by the quantization configuration or prepare the model for k-bit training if required.

    Args:
        config (`Config`):
            The configuration object containing model-related settings such as the model's name or path and other
            options.
        quantization_config (`Union[BitsAndBytesConfig, GPTQConfig, None]`):
            A configuration object guiding the quantization behavior of the model.
        low_cpu_mem_usage (`bool`, optional, defaults to `None`):
            A flag that, when set, instructs the model to optimize memory usage on CPUs. This can be helpful when
            dealing with large models or on systems with limited CPU memory resources.

    Returns:
        `PreTrainedModel`: An instance of a subclass of `PreTrainedModel` that has been instantiated and possibly
        quantized.

    The function handles the following tasks:
    - Determines whether caching should be enabled based on the gradient checkpointing setting.
    - Loads the model using the AutoModelForCausalLM class with provided configuration settings such as `dtype` and
      `device_map`.
    - Applies k-bit training preparations if configured to do so.
    - Modifies model configurations, such as disabling caching if necessary.

    Raises:
        ValueError: If the model's type is not supported or cannot be correctly instantiated.

    Example usage:
        ```python
        from xllm import Config

        # Assuming a predefined Config object with model path and quantization settings.
        config = Config(...)
        quantization_config = build_quantization_config(config=config)
        model = build_model(config=config, quantization_config=quantization_config)
        # model is now ready for training or inference.
        ```

    Note:
        - If quantization is intended to be applied after model initialization, the `bnb_quantize_after_model_init` flag
          should be set in the `Config` object.
        - Before calling this function, ensure that the model path and any desired customization options are properly
          set in the `Config` object.
    """
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

    if quantization_config is not None and config.need_to_prepare_model_for_kbit_training:
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
    """
    Instantiates and configures a LLM trainer appropriate for the model and datasets.

    This function retrieves the trainer class based on the provided configuration, setting it up with the specified
    model, token padding information, training arguments, datasets, and data collator.

    Args:
        config (`Config`):
            The configuration object containing necessary settings for the trainer, such as trainer key and model
            configuration.
        pad_token_id (`int`):
            The ID of the padding token used by the model, necessary for correctly computing the loss during training.
        training_arguments (`TrainingArguments`):
            The training arguments specifying training and evaluation parameters, such as learning rate and batch size.
        model (`Union[PreTrainedModel, PeftModel]`):
            The language model to be trained. Can be any model that is compatible with the training process.
        train_dataset (`BaseDataset`):
            The dataset object containing the training data samples.
        collator (`BaseCollator`):
            The data collator responsible for creating model-ready batches from the data samples.
        eval_dataset (`Optional[BaseDataset]`, defaults to `None`):
            The optional dataset object for evaluation. If provided, it is used for evaluating the model's performance
            during training.
        **kwargs (`Any`):
            Additional keyword arguments that may be required by the specific trainer instantiated.

    Returns:
        `LMTrainer`:
            A trainer instance of type `LMTrainer`, which extends the `Trainer` class, configured with the provided
            settings.

    The function follows these operations:
    - Retrieves the appropriate subclass of `Trainer` from the `trainers_registry` using a key.
    - Initializes the trainer subclass with provided arguments, configuring it for the training process.
    - Modifies the model configuration, mostly to disable caching for specific model types if necessary.

    Raises:
        ValueError: If the trainer class fetched from the registry does not subclass from `Trainer`.

    Example usage:
        ```python
        from xllm import Config

        # Assuming a predefined Config object with trainer specifications and instance of TrainingArguments.
        config = Config(...)
        training_args = build_training_arguments(config)
        trainer = build_trainer(
            config=config,
            pad_token_id=tokenizer.pad_token_id,
            training_arguments=training_args,
            model=model,
            train_dataset=train_dataset,
            collator=collator,
            eval_dataset=eval_dataset
        )
        # trainer is now ready to start the training cycle.
        ```

    Note:
        - The specific subclass of `LMTrainer` depends on the `trainer_key` provided in the `Config` object,
          which allows for the use of custom training behavior if needed.
    """
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

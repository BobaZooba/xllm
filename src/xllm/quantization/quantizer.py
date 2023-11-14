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
from typing import Any, Dict, List, Optional, Union

import torch
from optimum.gptq import GPTQQuantizer
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from .. import enums
from ..core.config import Config
from ..core.dependencies import build_dataset, build_model, build_tokenizer
from ..datasets.registry import datasets_registry
from ..utils.logger import dist_logger
from ..utils.post_training import push_to_hub_bos_add_bos_token


class Quantizer:
    """
    The `Quantizer` class is responsible for the quantization of pretrained language models using the GPTQ approach
    provided by the Optimum library. Quantization is the process of reducing the precision of the model's weights,
    which can lead to decreased model size and potentially faster inference times while maintaining comparable accuracy.

    The class provides tools to set up the quantization environment, execute the quantization, and save the resultant
    quantized model.

    Thanks to Phil Schmid's post: https://www.philschmid.de/gptq-llama

    Key methods:
    - `__init__`: Initializes the Quantizer with a configuration object and optional tokenizer, model, and dataset.
    - `build`: Sets up the tokenizer, model, and dataset if not provided during initialization and prepares the
        internal `GPTQQuantizer` for the quantization process.
    - `build_dataset`: Constructs or retrieves the dataset to calibrate the model for the quantization process.
    - `quantize`: Runs the actual quantization process by fine-tuning quantization scales with the dataset and
        quantizing model weights.
    - `save`: Saves the quantized model to disk and optionally uploads it to the Hugging Face model hub.

    Attributes:
    - `config` (`Config`): Configuration parameters for building the tokenizer, model, creating the dataset, and
        quantization settings.
    - `tokenizer` (`PreTrainedTokenizer`, defaults to `None`): The tokenizer to format the input for the model
        to be quantized.
    - `model` (`PreTrainedModel`, optional): The pretrained language model to be quantized.
    - `dataset` (`Union[str, List[str], None]`, defaults to `None`): Identifier or samples for the dataset used during
        quantization calibration.
    - `low_cpu_mem_usage` (`bool`, defaults to `None`): Whether to use optimized settings to lower CPU memory usage
        during quantization.
    - `quantizer` (`GPTQQuantizer`, defaults to `None`): The Optimum library's quantizer instance to perform
        the quantization.
    - `quantized_model` (`PreTrainedModel`, defaults to `None`): The resultant quantized model post-quantization
      process.

    This class should be used when there is a need to quantize language models for deployment or when optimizing models
    for environments with resource constraints or specific performance targets.
    """

    def __init__(
        self,
        config: Config,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        model: Optional[PreTrainedModel] = None,
        dataset: Union[str, List[str], None] = None,
        low_cpu_mem_usage: Optional[bool] = None,
    ):
        """
        Initializes the Quantizer object which is responsible for the quantization of a pretrained language model.

        The quantization process aims to reduce the precision of the model's weights to a specific bit-width to optimize
        its size and, potentially, inference speed, often with minimal impact on performance.

        Args:
            config (`Config`):
                The configuration object containing parameters for model building, tokenizer, datasets,
                and quantization.
            tokenizer (`PreTrainedTokenizer`, defaults to `None`):
                The tokenizer associated with the model that will be quantized. It is required for the quantization
                process as it formats the input for the model. If not provided, it will be built from the configuration.
            model (`PreTrainedModel`, optional):
                The pretrained model to be quantized. If not provided, it will be built from the configuration.
            dataset (`Union[str, List[str], None]`, defaults to `None`):
                The dataset to be used for calibrating the quantization process. It can be either a dataset identifier
                string or a list of text samples. If not provided, it will be built from the configuration.
            low_cpu_mem_usage (`bool`, defaults to `None`):
                If set to `True`, the model will attempt to use less CPU memory during quantization which can be
                beneficial when working with large models on machines with limited resources.

        The initializer of the Quantizer sets up the necessary components such as the tokenizer, model, and dataset,
        which will be used for quantization. If any of these components are not provided, they will be constructed based
        on the specified configuration. The `low_cpu_mem_usage` parameter is an important consideration for systems with
        restricted resources, and it's recommended to be set to `True` for quantization.

        Quantizer uses the `GPTQQuantizer` class from Optimum library for the quantization process. The attributes
        `quantizer` and `quantized_model` are initialized as `None` and will be created and filled during the build and
        quantize steps, respectively.

        Upon initialization, this class also performs internal checks like the availability of CUDA and whether all
        necessary settings for automatic GPT quantization are provided within the configuration.
        """
        self.config = config

        self.tokenizer = tokenizer
        self.model = model
        self.dataset = dataset

        self.low_cpu_mem_usage = low_cpu_mem_usage

        self.quantizer: Optional[GPTQQuantizer] = None
        self.quantized_model: Optional[PreTrainedModel] = None

    def internal_checks(self) -> None:
        if not torch.cuda.is_available():
            dist_logger.warning("CUDA is not available")

        self.config.check_auto_gptq()

    def build(self) -> None:
        """
        Builds the necessary components for the quantization process by performing internal checks, constructing
        the tokenizer, model and building/verifying the dataset.

        This method prepares the `Quantizer` instance to perform the quantization process on the language model.
        It initializes internal class attributes such as tokenizer, model, and dataset if they are not already provided.

        The method performs the following steps:
        - Validates the availability of CUDA and performs checks based on the configuration settings for quantization.
        - Constructs the tokenizer if it wasn't provided during Quantizer initialization. The tokenizer is essential
          for formatting the input data for the model.
        - Constructs the model if it wasn't provided during Quantizer initialization. A warning is issued if the
          `low_cpu_mem_usage` attribute is not set since quantization can be resource-intensive.
        - Sets up the quantizer instance of `GPTQQuantizer` class, configured with bits, group size, dataset, and
          model sequence length based on the provided configuration.

        Post execution, the Quantizer is set up with a tokenizer and model ready for quantization, as well as a
        `GPTQQuantizer` instance initialized with the appropriate configuration and dataset.

        Does not return any value.

        Raises:
            - ValueError if the dataset class corresponding to the `config.dataset_key` is not found in the registry.
            - ValueError if the quantization dataset cannot be loaded.

        Note: If `low_cpu_mem_usage` is not specified or set to `False`, a warning is given to consider setting it to
        `True` for quantization, especially when dealing with large models and systems with limited resources.
        """
        self.internal_checks()

        if self.tokenizer is None:
            self.tokenizer = build_tokenizer(config=self.config, use_fast=False)
            dist_logger.info(f"Tokenizer {self.config.correct_tokenizer_name_or_path} was built")

        if self.model is None:
            if self.low_cpu_mem_usage is None or not self.low_cpu_mem_usage:
                dist_logger.warning("low_cpu_mem_usage is None. Recommended to set to True for quantization")
            self.model = build_model(
                config=self.config,
                quantization_config=None,
                low_cpu_mem_usage=self.low_cpu_mem_usage,
            )
            dist_logger.info(f"Model {self.config.model_name_or_path} was built")

        dataset = self.build_dataset() if self.dataset is None else self.dataset

        self.quantizer = GPTQQuantizer(
            bits=self.config.gptq_bits,
            group_size=self.config.gptq_group_size,
            dataset=dataset,
            model_seqlen=self.config.max_length,
        )
        dist_logger.info("Quantizer loaded")

        return None

    def build_dataset(self) -> Union[str, List[str]]:
        """
        Constructs or retrieves the dataset to be used for calibrating the quantization process of the model.

        The dataset is a critical component for the quantization process as it is used to fine-tune the quantization
        scales on actual data, ensuring model accuracy is preserved post-quantization.

        The method performs the following steps:
        - If a dataset ID is specified in the configuration for quantization, this ID is used directly.
        - If no dataset ID is provided, but `prepare_dataset` is enabled in the configuration, the dataset class
            associated
          with `config.dataset_key` is used to prepare the data.
        - If neither of the above are provided or applicable, the method attempts to build the dataset by leveraging
          the configuration settings and dataset preparation procedures.

        During the process, it will:
        - Fetch and combine text parts from the raw dataset into sample texts for quantization.
        - Limit the total number of samples to `config.quantization_max_samples` if this value is set.

        Returns:
            `Union[str, List[str]]`: A dataset identifier string if the dataset comes from a repository or a list
            of text samples for calibrating quantization.

        Raises:
            ValueError: If the dataset class for `config.dataset_key` is not found in the datasets registry or
            if the dataset cannot be built or loaded according to the configuration specifications.

        This method ensures that a suitable and correctly formatted dataset is available for quantizing the model.
        """
        dataset_id = None
        samples: List[str] = list()

        if self.config.quantization_dataset_id is not None:
            dataset_id = self.config.quantization_dataset_id
        else:
            if self.config.prepare_dataset:
                dataset_cls = datasets_registry.get(self.config.dataset_key)

                if dataset_cls is None:
                    raise ValueError(f"Dataset with key {self.config.dataset_key} not found")

                dataset_cls.prepare(config=self.config)

            raw_dataset = build_dataset(config=self.config, is_train=True)

            if raw_dataset is None:
                raise ValueError("Quantization dataset can't be loaded")

            samples = list()

            for sample_index in tqdm(
                range(len(raw_dataset)), desc="Loading quantization dataset", total=self.config.quantization_max_samples
            ):
                sample: Dict[str, Any] = raw_dataset[sample_index]
                text_parts = sample[enums.General.text_parts]
                text = "\n".join(text_parts)
                if isinstance(text, str):
                    samples.append(text)

                if 0 < self.config.quantization_max_samples == len(samples):
                    break

        return dataset_id or samples

    def quantize(self) -> None:
        """
        Executes the quantization process on the pre-loaded pretrained language model using the `GPTQQuantizer`.

        This is the core method where the quantization scales are fine-tuned based on the provided dataset and the model
        weights are quantized to the specified bit-width.

        The method performs the following steps:
        - Validates that the tokenizer, model, and quantizer have been set up by calling `build`. If any of these
          components are not initialized, a ValueError is raised.
        - Proceeds with the quantization process, during which the `GPTQQuantizer` adjusts the quantization parameters
          based on the provided dataset and quantizes the model's weights.
        - Stores the resulting quantized model in the `quantized_model` attribute.

        Does not return any value.

        Raises:
            ValueError: If the tokenizer, model, or quantizer is not initialized before calling this method.

        Note:
            The quantized model is stored within the `quantized_model` attribute of the `Quantizer` instance and can be
            accessed or saved after quantization.
        """
        if self.tokenizer is None:
            raise ValueError("tokenizer is None. It is impossible to quantize. Please run build")

        if self.model is None:
            raise ValueError("model is None. It is impossible to quantize. Please run build")

        if self.quantizer is None:
            raise ValueError("quantizer is None. It is impossible to quantize. Please run build")

        dist_logger.info("Start quantization")
        self.quantized_model = self.quantizer.quantize_model(self.model, self.tokenizer)
        dist_logger.info("Quantization complete")

        return None

    def save(self) -> None:
        """
        Saves the quantized model to a specified directory and pushes it to the Hugging Face model hub if desired.

        This method deals with the post-quantization step of persisting the quantized model for future use. It ensures
        that the model is stored in an accessible location and is properly serialized.

        The method performs the following steps:
        - Checks if the quantized model is present. If not, raises a ValueError as there is nothing to save.
        - Saves the quantized model to the path specified in `config.quantized_model_path`.
        - Saves the tokenizer associated with the quantized model to the same path.
        - Adjusts the configuration file for the quantized model to ensure that any `disable_exllama` flags are set
          correctly and in sync with the quantizer's configuration.
        - Optionally, if a `quantized_hub_model_id` is specified in the config, the method pushes the quantized model
          and tokenizer to the Hugging Face model hub using this ID.

        Does not return any value.

        Raises:
            ValueError: If `quantized_model` is not set, indicating that there is no model to save.

        Note:
            The method logs appropriate messages to keep the user informed about the saving process and any issues that
            might occur, such as the absence of a `quantized_hub_model_id` for hub uploads. It's essential to ensure
            that the model hub ID is set if pushing the model to the cloud is intended.
        """
        if self.quantized_model is None:
            raise ValueError("quantized_model is None. Nothing to save")

        dist_logger.info(f"Saving quantized model to {self.config.quantized_model_path}")
        self.quantized_model.save_pretrained(
            save_directory=self.config.quantized_model_path,
            safe_serialization=self.config.save_safetensors,
        )

        fast_tokenizer = build_tokenizer(config=self.config)

        fast_tokenizer.save_pretrained(save_directory=self.config.quantized_model_path)

        path_to_config = os.path.join(self.config.quantized_model_path, "config.json")
        path_to_quantize_config = os.path.join(self.config.quantized_model_path, "quantize_config.json")

        if self.quantizer is not None:
            with open(
                path_to_quantize_config,
                "w",
                encoding="utf-8",
            ) as file_object:
                self.quantizer.disable_exllama = False
                json.dump(self.quantizer.to_dict(), file_object, indent=2)
        else:
            dist_logger.error("quantizer is None. saved quantized model can be broken")

        with open(path_to_config, "r", encoding="utf-8") as file_object:
            model_config = json.load(file_object)
            model_config["quantization_config"]["disable_exllama"] = False

        with open(path_to_config, "w", encoding="utf-8") as file_object:
            json.dump(model_config, file_object, indent=2)

        if self.config.quantized_hub_model_id is not None:
            dist_logger.info(f"Push quantized model to the hub {self.config.quantized_hub_model_id}")
            self.quantized_model.push_to_hub(
                repo_id=self.config.quantized_hub_model_id,
                private=self.config.quantized_hub_private_repo,
                safe_serialization=self.config.save_safetensors,
                max_shard_size=self.config.max_shard_size,
            )
            if fast_tokenizer is not None:
                fast_tokenizer.push_to_hub(
                    repo_id=self.config.quantized_hub_model_id,
                    private=self.config.quantized_hub_private_repo,
                    safe_serialization=self.config.save_safetensors,
                )
                if self.config.push_to_hub_bos_add_bos_token:
                    push_to_hub_bos_add_bos_token(repo_id=self.config.quantized_hub_model_id)
        else:
            dist_logger.warning("quantized_hub_model_id is None. Model will stay locally")

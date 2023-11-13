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
    # Thanks to Phil Schmid's post: https://www.philschmid.de/gptq-llama

    def __init__(
        self,
        config: Config,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        model: Optional[PreTrainedModel] = None,
        dataset: Union[str, List[str], None] = None,
        low_cpu_mem_usage: Optional[bool] = None,
    ):
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

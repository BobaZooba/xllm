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
from typing import Tuple

from huggingface_hub import HfApi, hf_hub_download
from loguru import logger
from peft import PeftModel  # type: ignore
from transformers import AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer

from ..core.config import Config
from ..core.dependencies import build_tokenizer
from ..utils.logger import dist_logger

TOKENIZER_CONFIG_FILE = "tokenizer_config.json"


def fuse_lora(config: Config) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
    """
    Fuses a specified LoRA (Low-Rank Adaptation) model with a specified pre-trained language model and
    returns a tokenizer-model tuple. The method alters the model's parameters to integrate LoRA layers.
    Post integration, the method can also save the modified model according to the specified configuration,
    either locally or by pushing to the Hugging Face model hub.

    This function also includes the capability to ensure parameter data types match the desired configuration,
    and to append a beginning-of-sentence (BOS) token and remove an end-of-sentence (EOS) token if specified.

    Args:
        config (Config):
            The configuration object containing all necessary parameters for loading models, tokenizers,
            setting data types, local saving, pushing to the hub, and modifying tokenizer behavior.

    Returns:
        Tuple[PreTrainedTokenizer, PreTrainedModel]: A tuple containing the tokenizer and the modified pre-trained
            model with LoRA layers fused.

    Raises:
        ValueError: If `config.hub_model_id` is `None` when attempting to push the fused model to the Hugging Face hub.
    """

    model_loaded_in_int4 = False

    lora_model_name_or_path_for_fusing = config.lora_model_name_or_path_for_fusing

    model_loading_additional_kwargs = dict()

    if config.load_in_8bit:
        model_loading_additional_kwargs["config.load_in_8bit"] = config.load_in_8bit
    elif config.load_in_4bit:
        model_loading_additional_kwargs["config.load_in_4bit"] = config.load_in_4bit
        model_loaded_in_int4 = True

    tokenizer = build_tokenizer(config=config)
    logger.info(f"Tokenizer {config.correct_tokenizer_name_or_path} loaded")
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=config.model_name_or_path,
        torch_dtype=config.dtype,
        trust_remote_code=config.trust_remote_code,
        **model_loading_additional_kwargs,
    )
    logger.info(f"Model {config.model_name_or_path} loaded")
    model = PeftModel.from_pretrained(
        model,
        lora_model_name_or_path_for_fusing,
        torch_dtype=config.dtype,
    )
    logger.info(f"LoRA {lora_model_name_or_path_for_fusing} loaded")
    logger.info("Start fusing")
    model = model.merge_and_unload()
    logger.info("LoRA fused")

    model_dtype = next(iter(model.parameters())).dtype
    if model_dtype != config.dtype:
        model = model.to(config.dtype)
        logger.info(f"Model converted to: {config.dtype}")

    if model_loaded_in_int4:
        dist_logger.warning(
            "You load model in int4. It can't be serialized. There will be problems with saving the model "
            "locally and uploading it to the Hugging Face Hub"
        )

    if config.fused_model_local_path is not None:
        logger.info(f"Saving locally to {config.fused_model_local_path}")
        tokenizer.save_pretrained(
            config.fused_model_local_path,
            safe_serialization=config.save_safetensors,
        )
        model.save_pretrained(
            config.fused_model_local_path,
            safe_serialization=config.save_safetensors,
        )
        if config.push_to_hub_bos_add_bos_token:
            path_to_tokenizer_config = os.path.join(config.fused_model_local_path, TOKENIZER_CONFIG_FILE)
            with open(path_to_tokenizer_config) as file_object:
                tokenizer_config = json.load(file_object)

            tokenizer_config["add_bos_token"] = True
            tokenizer_config["add_eos_token"] = False

            with open(path_to_tokenizer_config, "w") as file_object:
                json.dump(tokenizer_config, file_object, indent=2)
        logger.info(f"Model saved locally to {config.fused_model_local_path}")
    else:
        logger.info(f"Fused model will not be saved locally. Fused model localpath: {config.fused_model_local_path}.")

    if config.push_to_hub or (not config.push_to_hub and config.hub_model_id is not None):
        logger.info(f"Pushing model to the hub {config.hub_model_id}")
        if config.hub_model_id is not None:
            tokenizer.push_to_hub(
                repo_id=config.hub_model_id,
                private=config.hub_private_repo,
                safe_serialization=config.save_safetensors,
            )
            model.push_to_hub(
                repo_id=config.hub_model_id,
                private=config.hub_private_repo,
                safe_serialization=config.save_safetensors,
                max_shard_size=config.max_shard_size,
            )
            if config.push_to_hub_bos_add_bos_token:
                push_to_hub_bos_add_bos_token(repo_id=config.hub_model_id)
        else:
            raise ValueError("Fused model push to hub failed, because config.hub_model_id if None")
    else:
        logger.info(f"Fused model will not be pushed to the hub. Hub model id: {config.hub_model_id}.")

    return tokenizer, model


def push_to_hub_bos_add_bos_token(repo_id: str) -> None:
    """
    Modifies the tokenizer configuration to add a beginning-of-sentence (BOS) token and remove
    the end-of-sentence (EOS) token, then uploads the updated tokenizer configuration to the
    specified repository on the Hugging Face model hub.

    This is useful for adapting tokenizers to specific requirements for models that require
    a BOS token for correct functionality but do not need an EOS token.

    Args:
        repo_id (str):
            The repository ID on the Hugging Face model hub where the modified tokenizer
            configuration should be uploaded.

    Raises:
        OSError: If the downloaded tokenizer configuration file cannot be read or written.
        HTTPError: If the upload to the Hugging Face model hub fails.

    Example:
        >>> repo_id = 'your-username/your-model-repository'
        >>> push_to_hub_bos_add_bos_token(repo_id=repo_id)
        >>> # Tokenizer configuration in the specified repository will now include a BOS token and exclude an EOS token.
    """
    local_path = hf_hub_download(repo_id=repo_id, filename=TOKENIZER_CONFIG_FILE)

    with open(local_path) as file_object:
        tokenizer_config = json.load(file_object)

    tokenizer_config["add_bos_token"] = True
    tokenizer_config["add_eos_token"] = False

    local_tokenizer_config_file = os.path.join("./", TOKENIZER_CONFIG_FILE)

    with open(local_tokenizer_config_file, "w") as file_object:
        json.dump(tokenizer_config, file_object, indent=2)

    api = HfApi()
    api.upload_file(
        path_or_fileobj=local_tokenizer_config_file,
        path_in_repo=TOKENIZER_CONFIG_FILE,
        repo_id=repo_id,
        repo_type="model",
    )

    return None


def post_training(config: Config, tokenizer: PreTrainedTokenizer) -> None:
    """
    Handles post-training operations by pushing the tokenizer configuration to the Hugging Face model hub
    if specified in the configuration. It can also modify the tokenizer configuration on the model hub to
    add a beginning-of-sentence (BOS) token and remove an end-of-sentence (EOS) token, following a delay
    to ensure the initial push operation has completed.

    Args:
        config (Config):
            The configuration object that holds the settings for post-training operations, including
            whether to push to the hub and the relevant repository details.

        tokenizer (PreTrainedTokenizer):
            The tokenizer that may be pushed to the Hugging Face model hub.

    Raises:
        ValueError: If `config.hub_model_id` is `None` but the configuration is set to push to the hub.
    """
    if config.push_to_hub:
        if config.hub_model_id is None:
            raise ValueError("hub_model_id is None, but you want to push to HF hub")
        tokenizer.push_to_hub(repo_id=config.hub_model_id, private=config.hub_private_repo)
        if config.push_to_hub_bos_add_bos_token:
            sleep(10.0)
            push_to_hub_bos_add_bos_token(repo_id=config.hub_model_id)

    return None

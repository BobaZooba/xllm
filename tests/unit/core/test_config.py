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

import pytest
import torch

from src.xllm import enums
from src.xllm.core.config import Config
from src.xllm.core.deepspeed_configs import STAGE_2


def test_check_hub():
    config = Config(hub_model_id="llama2", push_to_hub=False)
    config.check_hub()


def test_check_none_deepspeed():
    config = Config(deepspeed_stage=None)
    config.check_deepspeed()


def test_check_not_use_flash_attention_2():
    config = Config(use_flash_attention_2=False)
    config.check_flash_attention()


def test_check_hub_push_without_repo():
    config = Config(hub_model_id=None, push_to_hub=True)
    with pytest.raises(ValueError):
        config.check_hub()


def test_tokenizer_name():
    config = Config(tokenizer_name_or_path="llama1000", model_name_or_path="llama2")
    assert config.correct_tokenizer_name_or_path == "llama1000"


def test_tokenizer_name_from_model():
    config = Config(tokenizer_name_or_path=None, model_name_or_path="llama2")
    assert config.correct_tokenizer_name_or_path == "llama2"


def test_lora_target_modules():
    config = Config(raw_lora_target_modules="q,w,e")
    assert config.lora_target_modules == ["q", "w", "e"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Running tests without GPU")
def test_dtype():
    config = Config()
    assert config.dtype == torch.float16


def test_deepspeed():
    config = Config(deepspeed_stage="2")
    assert config.deepspeed == STAGE_2


def test_check_deepspeed():
    config = Config(deepspeed_stage="0")
    config.check_deepspeed()


def test_check_deepspeed_exception():
    config = Config(deepspeed_stage="2")
    with pytest.raises(ImportError):
        config.check_deepspeed()


def test_check_flash_attention():
    config = Config(use_flash_attention_2=False)
    config.check_flash_attention()


def test_check_flash_attention_exception():
    config = Config(use_flash_attention_2=True)
    with pytest.raises(ImportError):
        config.check_flash_attention()


def test_check_auto_gptq_exception():
    config = Config()
    with pytest.raises(ImportError):
        config.check_auto_gptq()


@pytest.mark.parametrize("huggingface_hub_token", ["some_huggingface_hub_token", "another_huggingface_hub_token"])
def test_config_post_init_hf_token(huggingface_hub_token: str):
    config = Config(huggingface_hub_token=huggingface_hub_token)
    assert os.environ[enums.EnvironmentVariables.huggingface_hub_token] == huggingface_hub_token
    assert os.environ[enums.EnvironmentVariables.huggingface_hub_token] == config.huggingface_hub_token


@pytest.mark.parametrize("wandb_api_key", ["some_wandb_api_key", "another_wandb_api_key"])
@pytest.mark.parametrize("wandb_project", ["some_wandb_project", "another_wandb_project"])
@pytest.mark.parametrize("wandb_entity", [None, "some_wandb_entity", "another_wandb_entity"])
def test_config_post_init_wandb_report(wandb_api_key: str, wandb_project: str, wandb_entity: str):
    config = Config(
        report_to_wandb=True,
        wandb_api_key=wandb_api_key,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
    )

    assert os.environ[enums.EnvironmentVariables.wandb_api_key] == wandb_api_key
    assert os.environ[enums.EnvironmentVariables.wandb_api_key] == config.wandb_api_key

    assert os.environ[enums.EnvironmentVariables.wandb_project] == wandb_project
    assert os.environ[enums.EnvironmentVariables.wandb_project] == config.wandb_project

    if wandb_entity is not None:
        assert os.environ[enums.EnvironmentVariables.wandb_entity] == wandb_entity
        assert os.environ[enums.EnvironmentVariables.wandb_entity] == config.wandb_entity


@pytest.mark.parametrize("master_port", [0, 8080, 9999])
def test_config_apply_deepspeed_single_gpu(master_port: int):
    config = Config(single_gpu=True, master_port=master_port)
    config.apply_deepspeed_single_gpu()
    assert os.environ[enums.EnvironmentVariables.master_address] == "localhost"
    assert os.environ[enums.EnvironmentVariables.master_port] == str(master_port)
    assert os.environ[enums.EnvironmentVariables.rank] == "0"
    assert os.environ[enums.EnvironmentVariables.local_rank] == "0"
    assert os.environ[enums.EnvironmentVariables.world_size] == "1"

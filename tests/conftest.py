import json
import os

import pytest
from _pytest.tmpdir import TempPathFactory
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoTokenizer,
    FalconConfig,
    FalconForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizer,
    TrainingArguments,
)

from src.xllm import enums
from src.xllm.collators.lm import LMCollator
from src.xllm.core.config import Config
from src.xllm.datasets.soda import SodaDataset
from tests.helpers.constants import (
    FALCON_TOKENIZER_DIR,
    LLAMA_TOKENIZER_DIR,
    LORA_FOR_LLAMA_DEFAULT_TARGET_MODULES,
)
from tests.helpers.dummy_data import DATA, SODA_DATASET


@pytest.fixture(scope="session")
def llama_tokenizer() -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_TOKENIZER_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.fixture(scope="session")
def llama_model_config(llama_tokenizer: PreTrainedTokenizer) -> LlamaConfig:
    config = LlamaConfig(
        vocab_size=len(llama_tokenizer),
        hidden_size=8,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=32,
    )
    return config


@pytest.fixture(scope="session")
def llama_model(llama_model_config: LlamaConfig) -> LlamaForCausalLM:
    model = LlamaForCausalLM(config=llama_model_config)
    return model


@pytest.fixture(scope="session")
def llama_lora_config() -> LoraConfig:
    lora_config = LoraConfig(
        r=2,
        target_modules=LORA_FOR_LLAMA_DEFAULT_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
        lora_alpha=8,
        lora_dropout=0.1,
    )
    return lora_config


@pytest.fixture(scope="session")
def llama_lora_model(llama_model: LlamaForCausalLM, llama_lora_config: LoraConfig) -> PeftModel:
    llama_model = get_peft_model(model=llama_model, peft_config=llama_lora_config)
    return llama_model


@pytest.fixture(scope="session")
def falcon_tokenizer() -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(FALCON_TOKENIZER_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.fixture(scope="session")
def falcon_model_config(falcon_tokenizer: PreTrainedTokenizer) -> FalconConfig:
    config = FalconConfig(
        vocab_size=len(falcon_tokenizer),
        hidden_size=8,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=32,
    )
    return config


@pytest.fixture(scope="session")
def falcon_model(falcon_model_config: FalconConfig) -> FalconForCausalLM:
    model = FalconForCausalLM(config=falcon_model_config)
    return model


@pytest.fixture(scope="session")
def soda_dataset() -> SodaDataset:
    dataset = SodaDataset(data=SODA_DATASET)
    return dataset


@pytest.fixture(scope="session")
def llama_lm_collator(llama_tokenizer: PreTrainedTokenizer) -> LMCollator:
    collator = LMCollator(tokenizer=llama_tokenizer, max_length=32)
    return collator


@pytest.fixture(scope="session")
def training_arguments() -> TrainingArguments:
    arguments = TrainingArguments(
        output_dir="./outputs/",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        warmup_steps=50,
        learning_rate=2e-4,
        max_steps=500,
        num_train_epochs=1,
        weight_decay=0.001,
        max_grad_norm=1.0,
        label_smoothing_factor=0.1,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=1,
        hub_strategy="checkpoint",
        push_to_hub=False,
        save_safetensors=True,
        remove_unused_columns=False,
        log_level=enums.LogLevel.info,
        disable_tqdm=False,
    )
    return arguments


@pytest.fixture(scope="session")
def config() -> Config:
    hf_config = Config(deepspeed_stage=0)
    return hf_config


@pytest.fixture(scope="session")
def path_to_train_dummy_data(tmp_path_factory: TempPathFactory) -> str:
    path = tmp_path_factory.mktemp("tmp") / "train.jsonl"
    with open(path, "w") as file_object:
        for sample in DATA:
            file_object.write(json.dumps(sample) + "\n")
    return os.path.abspath(path)


@pytest.fixture(scope="session")
def path_to_train_prepared_dummy_data(tmp_path_factory: TempPathFactory) -> str:
    path = tmp_path_factory.mktemp("tmp") / "prepared_train.jsonl"
    with open(path, "w") as file_object:
        for sample in SODA_DATASET:
            file_object.write(json.dumps(sample) + "\n")
    return os.path.abspath(path)


@pytest.fixture(scope="session")
def path_to_empty_train_dummy_data(tmp_path_factory: TempPathFactory) -> str:
    path = tmp_path_factory.mktemp("tmp") / "empty_train.jsonl"
    return os.path.abspath(path)


@pytest.fixture(scope="session")
def path_to_eval_dummy_data(tmp_path_factory: TempPathFactory) -> str:
    path = tmp_path_factory.mktemp("tmp") / "eval.jsonl"
    with open(path, "w") as file_object:
        for sample in DATA:
            file_object.write(json.dumps(sample) + "\n")
    return os.path.abspath(path)


@pytest.fixture(scope="session")
def path_to_empty_eval_dummy_data(tmp_path_factory: TempPathFactory) -> str:
    path = tmp_path_factory.mktemp("tmp") / "empty_eval.jsonl"
    return os.path.abspath(path)


@pytest.fixture(scope="session")
def path_to_fused_model_local_path(tmp_path_factory: TempPathFactory) -> str:
    path = tmp_path_factory.mktemp("tmp") / "fused_model/"
    return os.path.abspath(path)


@pytest.fixture(scope="session")
def path_to_download_result(tmp_path_factory: TempPathFactory) -> str:
    path = tmp_path_factory.mktemp("tmp") / "data.jsonl"
    return os.path.abspath(path)

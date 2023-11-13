import pytest
from peft import PeftModel
from pytest import MonkeyPatch
from torch import Tensor
from transformers import (
    BitsAndBytesConfig,
    GPTQConfig,
    PreTrainedTokenizer,
    TrainingArguments,
)

from src.xllm.collators.lm import LMCollator
from src.xllm.collators.registry import collators_registry
from src.xllm.core.config import Config
from src.xllm.core.dependencies import (
    build_collator,
    build_dataset,
    build_model,
    build_quantization_config,
    build_tokenizer,
    build_trainer,
    build_training_arguments,
)
from src.xllm.datasets.registry import datasets_registry
from src.xllm.datasets.soda import SodaDataset
from src.xllm.trainers.registry import trainers_registry
from tests.helpers.constants import LLAMA_TOKENIZER_DIR
from tests.helpers.dummy_data import DATA, DummyDataset
from tests.helpers.patches import patch_from_pretrained_auto_causal_lm


def test_build_training_arguments(config: Config):
    arguments = build_training_arguments(config=config)
    assert arguments.per_device_train_batch_size == config.per_device_train_batch_size
    assert arguments.deepspeed is None


def test_build_dataset_train(path_to_train_dummy_data: str):
    datasets_registry.add(key="dummy", value=DummyDataset)
    config = Config(dataset_key="dummy", train_local_path_to_data=path_to_train_dummy_data)
    dataset = build_dataset(config=config, is_train=True)
    assert dataset[0] is not None


def test_build_dataset_eval(path_to_train_dummy_data: str):
    datasets_registry.add(key="dummy1", value=DummyDataset)
    config = Config(dataset_key="dummy1", eval_local_path_to_data=path_to_train_dummy_data)
    dataset = build_dataset(config=config, is_train=False)
    assert dataset[0] is not None


def test_build_dataset_eval_none(path_to_train_dummy_data: str):
    datasets_registry.add(key="dummy2", value=DummyDataset)
    config = Config(
        dataset_key="dummy2",
        train_local_path_to_data=path_to_train_dummy_data,
        eval_local_path_to_data=None,
    )
    dataset = build_dataset(config=config, is_train=False)
    assert dataset is None


def test_build_dataset_exception(path_to_train_dummy_data: str):
    datasets_registry.add(key="exc", value=Config)
    config = Config(dataset_key="exc", train_local_path_to_data=path_to_train_dummy_data)
    with pytest.raises(ValueError):
        build_dataset(config=config, is_train=True)


def test_build_tokenizer():
    config = Config(tokenizer_name_or_path=LLAMA_TOKENIZER_DIR)
    tokenizer = build_tokenizer(config=config)
    tokenizer("hello")


def test_build_tokenizer_use_fast():
    config = Config(tokenizer_name_or_path=LLAMA_TOKENIZER_DIR)
    tokenizer = build_tokenizer(config=config, use_fast=False)
    tokenizer("hello")


def test_build_tokenizer_padding_size():
    config = Config(tokenizer_name_or_path=LLAMA_TOKENIZER_DIR, tokenizer_padding_side="right")
    tokenizer = build_tokenizer(config=config)
    tokenizer("hello")


def test_build_collator(config: Config, llama_tokenizer: PreTrainedTokenizer):
    collator = build_collator(config=config, tokenizer=llama_tokenizer)
    batch = collator(DATA)
    for value in batch.values():
        assert isinstance(value, Tensor)


def test_build_collator_exception(llama_tokenizer: PreTrainedTokenizer):
    collators_registry.add(key="exc", value=Config)
    config = Config(collator_key="exc")
    with pytest.raises(ValueError):
        _ = build_collator(config=config, tokenizer=llama_tokenizer)


def test_build_quantization_config_bnb():
    config = Config(load_in_8bit=True)
    quantization_config = build_quantization_config(config=config)
    assert isinstance(quantization_config, BitsAndBytesConfig)
    assert quantization_config.load_in_8bit


def test_build_quantization_config_gptq():
    config = Config(gptq_bits=4, gptq_group_size=128, from_gptq=True)
    quantization_config = build_quantization_config(config=config)
    assert isinstance(quantization_config, GPTQConfig)
    assert quantization_config.bits == 4
    assert quantization_config.group_size == 128


def test_build_quantization_config_none():
    config = Config(from_gptq=False, load_in_4bit=False, load_in_8bit=False)
    quantization_config = build_quantization_config(config=config)
    assert quantization_config is None


@pytest.mark.parametrize("apply_lora", [False, True])
def test_build_model(monkeypatch: MonkeyPatch, apply_lora: bool):
    config = Config(apply_lora=apply_lora)
    with patch_from_pretrained_auto_causal_lm(monkeypatch=monkeypatch):
        _ = build_model(
            config=config,
            quantization_config=None,
        )


def test_build_model_bnb_after_init(monkeypatch: MonkeyPatch):
    config = Config(bnb_quantize_after_model_init=True)
    with patch_from_pretrained_auto_causal_lm(monkeypatch=monkeypatch):
        _ = build_model(
            config=config,
            quantization_config=None,
        )


def test_build_trainer(
    config: Config,
    training_arguments: TrainingArguments,
    llama_lora_model: PeftModel,
    soda_dataset: SodaDataset,
    llama_lm_collator: LMCollator,
):
    trainer = build_trainer(
        config=config,
        pad_token_id=2,
        training_arguments=training_arguments,
        model=llama_lora_model,
        train_dataset=soda_dataset,
        collator=llama_lm_collator,
        eval_dataset=None,
    )
    dataloader = trainer.get_train_dataloader()
    batch = next(iter(dataloader))
    loss = trainer.compute_loss(model=llama_lora_model, inputs=batch, return_outputs=False)
    assert isinstance(loss, Tensor)
    assert len(loss.size()) == 0


def test_build_trainer_exception(
    training_arguments: TrainingArguments,
    llama_lora_model: PeftModel,
    soda_dataset: SodaDataset,
    llama_lm_collator: LMCollator,
):
    trainers_registry.add(key="exc", value=Config)
    config = Config(trainer_key="exc")
    with pytest.raises(ValueError):
        _ = build_trainer(
            config=config,
            pad_token_id=2,
            training_arguments=training_arguments,
            model=llama_lora_model,
            train_dataset=soda_dataset,
            collator=llama_lm_collator,
            eval_dataset=None,
        )

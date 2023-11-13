import os

from pytest import MonkeyPatch

from src.xllm.core.config import Config
from src.xllm.experiments.base import Experiment
from tests.helpers.constants import LLAMA_TOKENIZER_DIR
from tests.helpers.patches import patch_from_pretrained_auto_causal_lm, patch_trainer_train


def test_base_experiment_init(monkeypatch: MonkeyPatch, path_to_train_dummy_data: str):
    config = Config(
        push_to_hub=False,
        deepspeed_stage=0,
        train_local_path_to_data=path_to_train_dummy_data,
        tokenizer_name_or_path=LLAMA_TOKENIZER_DIR,
    )
    with patch_from_pretrained_auto_causal_lm(monkeypatch=monkeypatch):
        Experiment(config=config)


def test_base_experiment_train(monkeypatch: MonkeyPatch, path_to_train_prepared_dummy_data: str, path_to_outputs: str):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    config = Config(
        push_to_hub=False,
        deepspeed_stage=0,
        train_local_path_to_data=path_to_train_prepared_dummy_data,
        report_to_wandb=False,
        save_total_limit=0,
        max_steps=2,
        tokenizer_name_or_path=LLAMA_TOKENIZER_DIR,
        output_dir=path_to_outputs,
    )

    with patch_from_pretrained_auto_causal_lm(monkeypatch=monkeypatch):
        experiment = Experiment(config=config)
        with patch_trainer_train(monkeypatch=monkeypatch):
            experiment.build()
            experiment.run()

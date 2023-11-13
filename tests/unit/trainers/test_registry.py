from peft import PeftModel
from transformers import TrainingArguments

from src.xllm import enums
from src.xllm.collators.lm import LMCollator
from src.xllm.core.config import Config
from src.xllm.datasets.soda import SodaDataset
from src.xllm.trainers.lm import LMTrainer
from src.xllm.trainers.registry import trainers_registry


def test_get_lm_trainer(
    config: Config,
    llama_lora_model: PeftModel,
    training_arguments: TrainingArguments,
    llama_lm_collator: LMCollator,
    soda_dataset: SodaDataset,
    path_to_outputs: str,
):
    trainer_cls = trainers_registry.get(key=enums.Trainers.lm)
    trainer = trainer_cls(
        config=config,
        model=llama_lora_model,
        args=training_arguments,
        data_collator=llama_lm_collator,
        train_dataset=soda_dataset,
        ignore_index=0,
        eval_dataset=None,
    )
    assert isinstance(trainer, LMTrainer)

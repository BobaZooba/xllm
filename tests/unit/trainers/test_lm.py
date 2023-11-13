from peft import PeftModel
from torch import Tensor
from transformers import TrainingArguments

from src.xllm.collators.lm import LMCollator
from src.xllm.core.config import Config
from src.xllm.datasets.soda import SodaDataset
from src.xllm.trainers.lm import LMTrainer


def test_lm_trainer(
    config: Config,
    llama_lora_model: PeftModel,
    training_arguments: TrainingArguments,
    llama_lm_collator: LMCollator,
    soda_dataset: SodaDataset,
):
    trainer = LMTrainer(
        config=config,
        model=llama_lora_model,
        args=training_arguments,
        data_collator=llama_lm_collator,
        train_dataset=soda_dataset,
        ignore_index=0,
        eval_dataset=None,
    )
    dataloader = trainer.get_train_dataloader()
    batch = next(iter(dataloader))
    loss = trainer.compute_loss(model=llama_lora_model, inputs=batch, return_outputs=False)
    assert isinstance(loss, Tensor)
    assert len(loss.size()) == 0

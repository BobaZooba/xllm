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

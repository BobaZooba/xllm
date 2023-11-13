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

from typing import Dict, Optional, Tuple, Union

from peft import PeftModel  # type: ignore
from torch import Tensor, nn
from transformers import PreTrainedModel, Trainer, TrainingArguments

from .. import enums
from ..collators.base import BaseCollator
from ..core.config import Config
from ..datasets.base import BaseDataset


class LMTrainer(Trainer):
    def __init__(
        self,
        config: Config,
        model: Union[PreTrainedModel, PeftModel],
        args: TrainingArguments,
        data_collator: BaseCollator,
        train_dataset: BaseDataset,
        ignore_index: int,
        eval_dataset: Optional[BaseDataset] = None,
    ):
        self.config = config

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        self.ignore_index = ignore_index

        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=self.args.label_smoothing_factor,
            ignore_index=self.ignore_index,
        )

    def compute_loss(
        self,
        model: Union[PreTrainedModel, PeftModel],
        inputs: Dict[str, Tensor],
        return_outputs: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        labels = inputs.pop(enums.Transformers.labels)

        outputs = model(**inputs)

        # NOTE from BobaZooba: just copy from hf code. did not go deeper
        #
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        loss = self.criterion(outputs.logits.reshape(-1, outputs.logits.size(-1)), labels.reshape(-1))

        return (loss, outputs) if return_outputs else loss

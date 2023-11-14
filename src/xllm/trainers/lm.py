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
    """
    The `LMTrainer` class is a specialized trainer for language modeling tasks, extending the functionality
    of the `Trainer` class from the Transformers library. It integrates with the `Config` object and custom datasets
    to provide a tailored training experience, accommodating the nuances of language model training.

    This trainer is designed to work with both `PreTrainedModel` and `PeftModel`, facilitating training with different
    types of language models, including those that support parameter-efficient fine-tuning techniques like LoRA.

    Key Features and Methods:
    - `__init__`: Sets up the trainer with the necessary components for training language models, including the model,
        datasets, data collator, and the training arguments. It also initializes the loss criterion with support for
        label smoothing and ignoring specific indices during loss calculation.
    - `compute_loss`: Overrides the default loss computation to suit the language modeling tasks. It accepts model
        inputs and labels, computes the cross-entropy loss, and optionally returns model outputs for analysis.

    Attributes:
    - `config` (`Config`): Configuration object containing all experiment settings that influence the training process.
    - `model` (`Union[PreTrainedModel, PeftModel]`): The model that will be trained.
    - `args` (`TrainingArguments`): Arguments specifying training and evaluation parameters.
    - `data_collator` (`BaseCollator`): The data collator to format batches of data for model input.
    - `train_dataset` (`BaseDataset`): Dataset containing training data samples.
    - `ignore_index` (`int`): The index that is ignored during loss computation.
    - `eval_dataset` (`Optional[BaseDataset]`): Optional dataset for evaluation.
    - `criterion` (`nn.Module`): The loss function used to compute the model's loss during training.
    """

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
        """
        Initializes an LMTrainer object for training and evaluating language models, with specific considerations
        for handling loss calculation and providing necessary datasets and collators.

        Args:
            config (`Config`):
                A configuration object containing all necessary parameters for the trainer such as model, dataset,
                and training-specific settings.
            model (`Union[PreTrainedModel, PeftModel]`):
                The model that will be trained. Can be any model that inherits from `PreTrainedModel` or `PeftModel`.
            args (`TrainingArguments`):
                The training arguments specifying the training and evaluation details, such as number of epochs,
                batch size, learning rate schedule, etc.
            data_collator (`BaseCollator`):
                The data collator that prepares and formats the batches of data that will be fed to the model
                during training.
            train_dataset (`BaseDataset`):
                The dataset object containing the training data samples.
            ignore_index (`int`):
                Specifies a target value that is ignored during loss computation, allowing certain tokens
                (like padding) to not influence the gradient.
            eval_dataset (`Optional[BaseDataset]`, defaults to `None`):
                The dataset object containing the evaluation data samples. If `None`, evaluation will not be
                performed during training.

        The LMTrainer extends the `Trainer` class from the Transformers library, adding a custom loss computation
        method adapted to the specifics of the training configuration provided in the `Config` object.
        The class takes in a mixed-precision or full-precision model along with training arguments, a data collator
        to format the dataset, and other utilities required for efficient and effective model training.

        Upon initialization, LMTrainer also sets up a criterion for the loss, with consideration for label smoothing
        and ignoring specific index values (like padding) during loss calculation.

        This specialized trainer class includes functionality tailored for language modeling tasks and should be
        used when working with models that perform tasks like text generation or language understanding, where
        specific handling of data and loss computation is required.
        """
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
        """
        Calculates the loss for a batch of inputs using the provided language model and labels.

        This method overrides the default `compute_loss` method from the `Trainer` class to accommodate
        custom loss computation that may be specific to language models or the particular training configuration
        used in an LMTrainer instance.

        Args:
            model (`Union[PreTrainedModel, PeftModel]`):
                The language model that is being trained. This can be any model compatible with the Trainer interface
                and designed for language-related tasks.
            inputs (`Dict[str, Tensor]`):
                A dictionary of tensors containing the batch of inputs for the model. This may include items like
                input IDs and attention masks, depending on the model's requirements.
            return_outputs (`bool`, defaults to `False`):
                A flag determining whether to return the model's outputs along with the loss. If `True`, the
                method returns a tuple of the loss and the model outputs.

        Returns:
            `Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]`: If `return_outputs` is `False`, returns the
            computed loss as a single tensor. If `True`, returns a tuple consisting of the computed loss tensor
            and a dictionary of the model's outputs.

        The `compute_loss` method performs the following steps:

        - Extracts the labels from the input dictionary using a key defined in `enums.Transformers.labels`.
        - Feeds the inputs without the labels into the model to obtain logits (unscaled probabilities).
        - Reshapes the logits and labels as necessary, ensuring they align for loss computation.
        - Calculates the cross-entropy loss using the criterion set during LMTrainer initialization,
            considering any label smoothing and the specified `ignore_index` for the loss.
        - Fixes state support for the past tensor if it exists, as is required mainly by autoregressive models
            (like GPT-2).

        Additionally, if `return_outputs` is `True`, the method bundles the calculated loss with all outputs
        from the model, allowing for further processing or analysis outside this method.
        """

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

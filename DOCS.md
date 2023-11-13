from xllm.types import RawSamplefrom typing import List

# 🦖 X—LLM Documentation

План


# Steps

Using `X—LLM` to train a model is simple and involves these few steps:

1. `Prepare` — Get the data and the model ready by downloading and preparing them. Saves data locally
   to `config.train_local_path_to_data` and `config.eval_local_path_to_data` if you are using eval dataset
2. `Train` — Use the data prepared in the previous step to train the model
3. `Fuse` — If you used LoRA during the training, fuse LoRA
4. `Quantize` — Make your model take less memory by quantizing it

## Prepare

At this step, the following occurs:

- Downloading and preparing data
- Downloading the model

We've separated this as a distinct step for a few reasons, chief among them being: if you're utilizing distributed training across several GPUs - for instance via DeepSpeed - you would otherwise end up redundantly downloading the dataset and model on each individual GPU, even though you only need to do it once.

### Results of this step
- Prepared data (training and optionally eval), which are stored locally on the machine at the corresponding paths: `train_local_path_to_data` и `eval_local_path_to_data`
- A model downloaded to the local cache (typical model download from the Hugging Face Hub)

## Train

At this step, model training occurs, which is controlled through `Config`. More information on how to better create a config can be learned here:
- [Important config fields for different steps](https://github.com/BobaZooba/xllm#how-config-controls-xllm)
- [How do I choose the methods for training?](https://github.com/BobaZooba/xllm/blob/main/DOCS.md#how-do-i-choose-the-methods-for-training)
- [Detailed description of all config fields](https://github.com/BobaZooba/xllm/blob/main/DOCS.md#detailed-config-explanation)

### Data pipeline

What happens to the data? One step of training is described below

- Each sample is transformed into a dictionary with specified keys in the get_sample method of the Dataset
- A batch is assembled, and the batch goes through a collator, which is responsible for `tokenization`, `label preparation`, in other words, for **preparing the input data for the model**. The collator knows which keys it needs to accept as input and how to process each key
- The batch is processed by the `compute_loss` method of the `Trainer`, and this method returns the `loss`. A `backward` operation is performed based on the `loss`

### Results of this step

- Model artifacts like local checkpoints and/or Huggingface Hub checkpoints
- _[Optional]_ Training artifacts in W&B like loss curve

## Fuse

This step is only necessary if you trained the model using the `LoRA` adapter and want to obtain the ready for inference and fused model.

### Results of this step

- Fused model locally and/or in Hugging Face Hub

## Quantize

In this step, post-training quantization of the model occurs using [auto-gptq](https://github.com/PanQiWei/AutoGPTQ). For this, you will need to install `auto-gptq`, for example, like this:

```bash
pip install xllm[train]
```

This method allows for a significant reduction in the size of the model and a slight increase in speed at the cost of a small decrease in accuracy. For example, the Mistral model is reduced from 15GB to 4.3GB.

### Results of this step
- Quantized model locally and/or in Hugging Face Hub

# How to add CLI tools to your project

It is advisable to add `xllm` CLI functions to your project to simplify all the steps above.

Also, please check:
- [Demo project](https://github.com/BobaZooba/xllm-demo): here's a step-by-step example of how to use `X—LLM` and fit
  it
  into your own project
- [Template project](https://github.com/BobaZooba/xllm-template): here's a template, a kickoff point you can use for
  your project

### Project structure

```txt
my_project
  cli
    __init__.py
    prepare.py
    train.py
    fuse.py
    quantize.py
  core
    __init__.py
    dataset.py
    collator.py
    trainer.py
    experiment.py
  __init__.py
```

### Files

`prepare.py`

```python
from xllm.cli import cli_run_prepare

if __name__ == "__main__":
    cli_run_prepare()
```

`train.py`

```python
from xllm.cli import cli_run_train

if __name__ == "__main__":
    cli_run_train()
```

`fuse.py`

```python
from xllm.cli import cli_run_fuse

if __name__ == "__main__":
    cli_run_fuse()
```

`quantize.py`

```python
from xllm.cli import cli_run_quantize

if __name__ == "__main__":
    cli_run_quantize()
```

### Run

More detailed examples here: [Production soluton](https://github.com/BobaZooba/xllm#production-solution-)

```bash
python my_project/cli/prepare.py --model_name_or_path mistralai/Mistral-7B-v0.1
python my_project/cli/train.py --model_name_or_path mistralai/Mistral-7B-v0.1
python my_project/cli/fuse.py --model_name_or_path mistralai/Mistral-7B-v0.1
python my_project/cli/quantize.py --model_name_or_path mistralai/Mistral-7B-v0.1
```

You also could run train via DeepSpeed (if you have multiple GPUs on your machine):
```bash
deepspeed --num_gpus=8 my_project/cli/train.py --model_name_or_path
```

### Registry

You can implement `datasets`, `collators`, `trainers`, and `experiments`. Here, you will learn how to make the above CLI functions aware of the new components.

Available registries:
```python
from xllm.datasets import datasets_registry
from xllm.collators import collators_registry
from xllm.trainers import trainers_registry
from xllm.experiments import experiments_registry
```

Add the new dataset to the registry:
```python
from xllm.datasets import datasets_registry

datasets_registry.add(key="my_new_dataset", value=MyDataset)
```

This code should be added before the invocation of CLI functions. For example:

`my_project/core/registry.py`

```python
from xllm.datasets import datasets_registry

from dataset import MyDataset

def registry_components():
    datasets_registry.add(key="my_new_dataset", value=MyDataset)
```

`my_project/cli/prepare.py`

```python
from xllm.cli import cli_run_prepare

from my_project.core.registry import registry_components

if __name__ == "__main__":
    registry_components()
    cli_run_prepare()
```

`my_project/cli/train.py`

```python

from xllm.cli import cli_run_train

from my_project.core.registry import registry_components

if __name__ == "__main__":
    registry_components()
    cli_run_train()
```

#### Run

```bash
python my_project/cli/prepare.py --dataset_key my_new_dataset
python my_project/cli/train.py --dataset_key my_new_dataset
```

You could extend registy with different components:

`registry.py`

```python
from xllm.datasets import datasets_registry
from xllm.collators import collators_registry
from xllm.trainers import trainers_registry
from xllm.experiments import experiments_registry

from my_project.core.dataset import MyDataset
from my_project.core.collator import MyCollator
from my_project.core.trainer import MyTrainer
from my_project.core.experiment import MyExperiment

def registry_components():
    datasets_registry.add(key="my_new_dataset", value=MyDataset)
    collators_registry.add(key="my_new_collator", value=MyCollator)
    trainers_registry.add(key="my_new_trainer", value=MyTrainer)
    experiments_registry.add(key="my_new_experiment", value=MyExperiment)
```

# Customization

You have the flexibility to tweak many aspects of your model's training: data, how data is processed, trainer, config,
how the model is loaded, what happens before and after training, and so much more.

We've got ready-to-use components for every part of the `xllm` pipeline. You can entirely switch out some components
like the dataset, collator, trainer, and experiment.
For some components like experiment and config, you have the option to just build on what's already there.

## How to implement dataset

Dataset is the most basic component in `xllm`. With this component we can describe the logic of how and from where the data should be loaded, how they should be preprocessed and in what form they should be presented further throughout the training pipeline.

Each dataset should be inherited from `xllm.datasets.base.BaseDataset`

It can be imported like this:

```python
from xllm.datasets import BaseDataset
```

In each new dataset, we must implement two methods:
- get_data
- get_sample


Let's start implementing new dataset
```python
from typing import Optional, Tuple, List

from xllm import Config
from xllm.datasets import BaseDataset
from xllm.types import RawSample


class MyDataset(BaseDataset):

    def get_data(cls, config: Config) -> Optional[Tuple[List[RawSample], Optional[List[RawSample]]]]:
        ...

    def get_sample(self, index: int) -> RawSample:
        ...
```

### `get_data`

The input values of this method are cls from `MyDataset` and `Config`. By default, the default config from `xllm` is used: `from xllm import Config`

The `get_data` method should return a tuple of two elements: **training data** and **eval data**. These data should be lists of type `RawSample`. `RawSample`, in turn, is simply an alias for the following type: `Dict[str, Union[str, int, float, List[str]]]`. In other words, each sample in the data should be a dictionary, where the keys are strings, and the values can be `str`, `int`, `float`, or `list of str`.

The structure of `RawSample` can be arbitrary, with arbitrary keys. Later in the `get_sample` step, we will be transforming this dictionary into the required construction.

Eval data can be `None`, while training data must definitely be a list of `RawSample` elements. We can start training without eval data, but we cannot start it without training data.

In the type annotation for the output data of this method, it is indicated that the output can be simply `None`. This is not recommended behavior, but it is possible if you do not need to download and preprocess data at all.

#### Why download data?
Typically, we use rented machines for training. This involves a number of necessary steps, such as downloading a dataset. The xllm library suggests implementing this as a separate step to facilitate project deployment in such variable conditions.

Training data can be shuffled if specified in the config by the shuffle key. By default, shuffling occurs.

#### When the method is called and what is the result
The method is called during the prepare step and saves the resulting datasets into the corresponding paths of the config: `train_local_path_to_data` and `eval_local_path_to_data`. These data will later be loaded in subsequent steps, such as train.

#### If in Config eval_local_path_to_data is set to None
In this case, if `add_eval_to_train_if_no_path` is indicated in the config, then these data will be added to the training data.

Also, if the size of evaluation data is greater than the value in the config `max_eval_samples`, then eval data will be truncated to this value, and the remaining evaluation data will be added to the training data.

### `get_sample`

This method converts data from the previous step into data for working with `xllm`. In `xllm`, we always use a component such as Collator. It is responsible for tokenization, preparing labels for the model, that is, to gather samples from `Dataset` and transform them into a batch for the model.

#### What should a sample look like?

We mentioned earlier that the `Collator` takes samples of a certain structure as input. In its simplest form, this structure looks like this:

```python
{
    "text_parts": [
        "Hello!",
        "My name is Boris"
    ]
}
```

Thus, each sample is a dictionary with the key `text_parts`, and the value is a `list of strings`. Ultimately, in the `Collator`, the list of texts is converted into text using a specified separator. In `CompletionCollator`, we calculate `loss` (train the model) only for the last text in the list, that is, the last utterance (of the assistant) in the dialogue. Also, a list of texts allows for easier customization of the dialogue and the use of different separators for these texts.

In the `Dataset`, there is a parameter called `must_have_keys`. Each sample is checked before being fed into the `Collator` to ensure it contains all `must_have_keys`; otherwise, an exception is thrown. You can expand this list in the init method.

#### Change both `Collator` and `Dataset`

We can complicate our data, for example, for PPO training. In such a case, we need to implement both `Dataset` and `Collator` (and even `Trainer`) for it right away. The implementation of your own collator is described below.

### Registry

The last important detail is the `datasets_registry`. We have implemented a new dataset and now want to use it while training through the command line. How do we do that? It is necessary to add the new dataset to the `datasets_registry`.

Please check this: [Registry](https://github.com/BobaZooba/xllm/blob/main/DOCS.md#registry)

## How to implement collator

It was mentioned earlier that typically `Collator` receives a list of elements like this:

```python
{
    "text_parts": [
        "Hello!",
        "My name is Boris"
    ]
}
```

We can change the structure of the sample by implementing a new dataset, and we can also change the logic of processing this sample by implementing a new collator. 

The main task of the collator is to convert a list of samples into a batch that can be input to the model. That is, to turn texts into a tokenized batch and create targets.

Every collator must be inherited from `BaseCollator` and the method `parse_batch` must be implemented.

```python
from typing import List

from xllm.types import RawSample, Batch
from xllm.collators import BaseCollator


class MyCollator(BaseCollator):
    
    def parse_batch(self, raw_batch: List[RawSample]) -> Batch:
        ...
```

This method receives a list of samples that are output by the `Dataset`. Please read about the implementation of the dataset above if you have not done so already.

Your task is to write the logic of how to process the list in order to eventually obtain a `Batch`. A `Batch` is a dictionary where the key is a string, and the value is a `PyTorch Tensor`.

Similarly to the dataset, the collator also has `must_have_keys`, which you can expand in the init method.

### `LMCollator`

This collator is needed for simple language modeling. It compiles the lists of texts from each example into a single text by concatenating them using a separator.

### `CompletionCollator`

This collator is needed for cases when we want to calculate the loss only for the last text in the list. For example, these are instances of interacting with an assistant. We don't want to train the model on how the user speaks. We don't need the model to be able to imitate the user, so we construct the dataset in such a way that at the end of the list of texts (dialogue), there is a phrase by the assistant. Essentially, we will be training the model to generate these completions by the assistant.

### Registry

You need to register your `Collator` in order to use it in `xllm` CLI functions.

Please check this: [Registry](https://github.com/BobaZooba/xllm/blob/main/DOCS.md#registry)

## How to implement trainer

For users of the `transformers` library, this is the most familiar component. In it, we describe different logic within training. Typically, implementing a new component may be necessary to realize custom logic in the `compute_loss` method.

Every `trainer` must be inherited from `transformers.Trainer`.

```python
from typing import Dict, Tuple, Union

from torch import Tensor
from peft import PeftModel
from transformers import Trainer, PreTrainedModel

class MyTrainer(Trainer):

    def compute_loss(
        self,
        model: Union[PreTrainedModel, PeftModel],
        inputs: Dict[str, Tensor],
        return_outputs: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        ...
```

### `LMTrainer`

This is a trainer for language modeling training, but it also works properly with the `CompletionCollator` because the `CompletionCollator` forms targets in such a way that `loss` is ultimately calculated only for the last text in the list.

### Registry

You need to register your `Trainer` in order to use it in `xllm` CLI functions.

Please check this: [Registry](https://github.com/BobaZooba/xllm/blob/main/DOCS.md#registry)

## How to implement experiment

The experiment acts as an aggregator of all logic during training. It dictates how to load the model, what needs to be done before or after a certain step, for example, before initializing the `tokenizer`.

Every `experiment` must be inherited from `xllm.experiments.base.Experiment`.

The experiment does not have any abstract methods that must be implemented, but there are many methods that you can implement. And of course, you can override methods. In the future, it is planned to add callbacks.

Methods:
- before_checks
- after_checks
- before_training_arguments_build
- after_training_arguments_build
- after_training_arguments_build
- build_train_dataset_additional_kwargs (must return dict)
- after_train_dataset_build
- before_eval_dataset_build
- build_eval_dataset_additional_kwargs (must return dict)
- after_eval_dataset_build
- before_tokenizer_build
- after_tokenizer_build
- before_collator_build
- build_collator_additional_kwargs (must return dict)
- after_collator_build
- before_quantization_config_build
- after_quantization_config_build
- before_model_build
- after_model_build
- before_bnb_quantization
- after_bnb_quantization
- before_lora_apply
- after_lora_apply
- before_stabilize_training
- after_stabilize_training
- before_trainer_build
- build_trainer_additional_kwargs (must return dict)
- after_trainer_build
- before_train
- after_fuse
- after_train
- at_beginning
- at_end

```python
from xllm.experiments import Experiment

class MyExperiment(Experiment):

    def before_lora_apply(self) -> None:
        # do whatever you want
        ...
```

### Registry

You need to register your `Experiment` in order to use it in `xllm` CLI functions.

Please check this: [Registry](https://github.com/BobaZooba/xllm/blob/main/DOCS.md#registry)

## How to extend config

Details here: [How to extend config](https://github.com/BobaZooba/xllm/blob/main/DOCS.md#how-to-extend-config)

# Config

The `X—LLM` library uses a single config setup for all steps like preparing, training and the other steps. It's
designed in a way that
lets you easily understand the available features and what you can adjust. `Config` has control almost over every
single part of each step. Thanks to the config, you can pick your dataset, set your collator, manage the type of
quantization during training, decide if you want to use LoRA, if you need to push a checkpoint to the `HuggingFace Hub`,
and a
lot more.

Config path: `src.xllm.core.config.Config`

Or

```python
from xllm import Config
```

The most important `Config` keys are specified here: [How Config controls xllm](https://github.com/BobaZooba/xllm#how-config-controls-xllm)

You can initialize `Config` by specifying the necessary values, but also, if you are using command-line tools, you can pass values through the command line.

### Init `Config`

```python
from xllm import Config

config = Config(
  model_name_or_path="mistralai/Mistral-7B-v0.1",
  apply_lora=True,
  stabilize=True,
)

# Do whatever you want using xllm (for example train the model)
```

### Pass values through CLI

1. Write for example a train script 

`train.py`

  ```python
from xllm import Config
from xllm.cli import cli_run_train

if __name__ == '__main__':
    ...  # initializing your dataset or registering
    cli_run_train(config_cls=Config)
  ```

2. Run the script above using CLI and provide `Config` key values

  ```bash
python train.py \
    --model_name_or_path mistralai/Mistral-7B-v0.1 \
    --apply_lora True \
    --stabilize True \
  ```

### Detailed Config explanation

<details>
  <summary>Expand</summary>

| Key                             | Default value             | Type           | Entity             | Comment                                                                                                                                                                                                   |
|---------------------------------|---------------------------|----------------|--------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `experiment_key`                  | base                      | str            | general            | Experiment class key                                                                                                                                                                                      |
| `save_safetensors`                | True                      | bool           | general            | Safe serialization                                                                                                                                                                                        |
| `max_shard_size`                  | 10GB                      | str            | general            | max_shard_size for the model pushing to the HuggingFace Hub                                                                                                                                               |
| `local_rank`                      | 0                         | int            | general            | Local rank for logging and saving. Works only in distributed training                                                                                                                                     |
| `use_gradient_checkpointing`      | False                     | bool           | general            | If True, use gradient checkpointing to save memory at the expense of slower backward pass                                                                                                                 |
| `trainer_key`                     | lm                        | str            | general            | Key of the trainer for loading from trainers_registry                                                                                                                                                     |
| `force_fp32`                      | False                     | bool           | general            | Force using fp32 when model loading                                                                                                                                                                       |
| `force_fp16`                      | False                     | bool           | general            | Force using fp16 when model loading                                                                                                                                                                       |
| `from_gptq`                       | False                     | bool           | general            | If you loadining GPTQ quantized model                                                                                                                                                                     |
| `huggingface_hub_token`           | None                      | Optional[str]  | general            | HuggingFace Hub token. You can also set this key using .env file                                                                                                                                          |
| `deepspeed_stage`                 | 0                         | int            | general            | Predifined DeepSpeed stage                                                                                                                                                                                |
| `deepspeed_config_path`           | None                      | Optional[int]  | general            | Path to DeepSpeed config                                                                                                                                                                                  |
| `fsdp_strategy`                   |                           | str            | general            | FSDP strategy                                                                                                                                                                                             |
| `fsdp_offload`                    | True                      | True           | general            | Offload weights when using FSDP                                                                                                                                                                           |
| `seed`                            | 42                        | int            | general            | Seed value for random operations                                                                                                                                                                          |
| `stabilize`                       | False                     | bool           | general            | Stabilize the model. Convert some weights to fp32, some to fp16/bf16                                                                                                                                      |
| `path_to_env_file`                | ./.env                    | Optional[str]  | general            | Custom path to .env file                                                                                                                                                                                  |
| `prepare_dataset`                 | True                      | bool           | general            | Prepare the dataset. Works only at "prepare" stage                                                                                                                                                        |
| `lora_hub_model_id`               | None                      | Optional[str]  | fuse               | Fusing LoRA. The name of the LoRA model at the hub for fusing. Example: BobaZooba/Shurale                                                                                                                 |
| `lora_model_local_path`           | None                      | Optional[str]  | fuse               | Fusing LoRA. Local path to the LoRA model                                                                                                                                                                 |
| `fused_model_local_path`          | None                      | Optional[str]  | fuse               | Local path to fused model. Useful if you want to quantize model after fusing on the same machine                                                                                                          |
| `fuse_after_training`             | False                     | bool           | fuse               | Fuse or not model after training                                                                                                                                                                          |
| `quantization_dataset_id`         | None                      | Optional[str]  | gptq quantization  | Dataset id for GPTQ quantization. You can install either the idi dataset, or use any dataset                                                                                                              |
| `quantization_max_samples`        | 1024                      | int            | gptq quantization  | Max samples for GPTQ quantization if you use custom dataset                                                                                                                                               |
| `quantized_model_path`            | ./quantized_model/        | str            | gptq quantization  | Path to GPTQ quantized model                                                                                                                                                                              |
| `quantized_hub_model_id`          | None                      | Optional[str]  | gptq quantization  | The name of the model at the hub for GPTQ quantization. Example: BobaZooba/Shurale-GPTQ                                                                                                                   |
| `quantized_hub_private_repo`      | True                      | bool           | gptq quantization  | Private repository for GPTQ quantization model or not                                                                                                                                                     |
| `dataset_key`                     | soda                      | str            | dataset            | Key of the dataset for loading from datasets_registry                                                                                                                                                     |
| `train_local_path_to_data`        | ./train.jsonl             | str            | dataset            | The path to the local training data file                                                                                                                                                                  |
| `eval_local_path_to_data`         | None                      | Optional[str]  | dataset            | The path to the local eval data file                                                                                                                                                                      |
| `shuffle`                         | True                      | bool           | dataset            | Shuffle training data                                                                                                                                                                                     |
| `max_eval_samples`                | 1_000                     | int            | dataset            | Maximum number of examples for evaluation                                                                                                                                                                 |
| `add_eval_to_train_if_no_path`    | False                     | bool           | dataset            | Add evaluation data to training data if their number is greater than max_eval_samples                                                                                                                     |
| `tokenizer_name_or_path`          | None                      | Optional[str]  | tokenizer          | Tokenizer name or path. If the value is not set, then it will be taken from the model_name_or_path                                                                                                        |
| `tokenizer_use_fast`              | None                      | Optional[bool] | tokenizer          | Use fast flag for the tokenizer                                                                                                                                                                           |
| `tokenizer_padding_side`          | None                      | Optional[str]  | tokenizer          | Padding side of the collator: None, right or left                                                                                                                                                         |
| `collator_key`                    | lm                        | str            | collator           | Key of the collator for loading from collators_registry                                                                                                                                                   |
| `max_length`                      | 2048                      | int            | collator           | Max sequence length of the model                                                                                                                                                                          |
| `model_name_or_path`              | mistralai/Mistral-7B-v0.1 | str            | model              | Model name or path. It could be from HuggingFace or locally                                                                                                                                               |
| `push_to_hub_bos_add_bos_token`   | False                     | bool           | model              | Upload to the hub tokenization config with add_bos_token equals to True. Might be helpful for TGI                                                                                                         |
| `use_flash_attention_2`           | False                     | bool           | model              | Use or not flash attention 2. Requires 1) CUDA >= 11.6; 2) install flash-attn 3) compatible model                                                                                                         |
| `trust_remote_code`               | False                     | bool           | model              | Trust remote code from HuggingFace                                                                                                                                                                        |
| `device_map`                      | None                      | None           | model              | Device map for loading the model                                                                                                                                                                          |
| `prepare_model_for_kbit_training` | True                      | bool           | model              | Prepare or not for kbit training                                                                                                                                                                          |
| `load_in_8bit`                    | False                     | bool           | bitsandbytes       | Load the model in 8 bit using bitsandbytes                                                                                                                                                                |
| `load_in_4bit`                    | False                     | bool           | bitsandbytes       | Load the model in 4 bit using bitsandbytes                                                                                                                                                                |
| `llm_int8_threshold`              | 6.0                       | float          | bitsandbytes       | Threshold for outlier detection                                                                                                                                                                           |
| `llm_int8_has_fp16_weight`        | True                      | bool           | bitsandbytes       | LLM has weights in fp16                                                                                                                                                                                   |
| `bnb_4bit_use_double_quant`       | True                      | bool           | bitsandbytes       | Double quantization. This will enable a second quantization after the first one to save an additional 0.4 bits per parameter                                                                              |
| `bnb_4bit_quant_type`             | nf4                       | str            | bitsandbytes       | Quantization type for 4 bit                                                                                                                                                                               |
| `bnb_quantize_after_model_init`   | False                     | bool           | bitsandbytes       | If False, quantization will be at model init                                                                                                                                                              |
| `gptq_bits`                       | 4                         | int            | gptq               | Bits for GPTQ quantization                                                                                                                                                                                |
| `gptq_group_size`                 | 128                       | int            | gptq               | Group size for GPTQ quantization                                                                                                                                                                          |
| `gptq_disable_exllama`            | True                      | bool           | gptq               | Disable ExLlama kernels for GPTQ quantization                                                                                                                                                             |
| `apply_lora`                      | False                     | bool           | lora               | Apply LoRA to the model or not                                                                                                                                                                            |
| `lora_rank`                       | 8                         | int            | lora               | LoRA rank value. LoRA matrices W_A x R and R x W_B, where R is LoRA rank                                                                                                                                  |
| `lora_alpha`                      | 32                        | int            | lora               | LoRA alpha value. The resulting LoRA matrix will be multiplied by this value                                                                                                                              |
| `lora_dropout`                    | 0.1                       | float          | lora               | LoRA dropout value                                                                                                                                                                                        |
| `raw_lora_target_modules`         | all                       | str            | lora               | Names of modules to apply LoRA. A comma-separated string, for example: "k,q,v". When setting the value "all", LoRA will be applied to all linear layers, except for the input embeddings and the lm_head. |
| `output_dir`                      | ./outputs/                | str            | training arguments | The path to the directory where the artifacts will be saved                                                                                                                                               |
| `per_device_train_batch_size`     | 2                         | int            | training arguments | Batch size on each GPU                                                                                                                                                                                    |
| `do_eval`                         | False                     | bool           | training arguments | Run eval or not                                                                                                                                                                                           |
| `per_device_eval_batch_size`      | None                      | Optional[int]  | training arguments | Batch size on each GPU for evaluation. If None per_device_eval_batch_size equals to per_device_train_batch_size                                                                                           |
| `gradient_accumulation_steps`     | 1                         | int            | training arguments | Number of steps to accumulate gradients                                                                                                                                                                   |
| `eval_accumulation_steps`         | None                      | Optional[int]  | training arguments | Number of steps to accumulate gradients at evaluation. If None eval_accumulation_steps equals to gradient_accumulation_steps                                                                              |
| `eval_delay`                      | 0                         | int            | training arguments | Number of epochs or steps to wait for before the first evaluation can be performed, depending on the evaluation_strategy                                                                                  |
| `eval_steps`                      | 1_000                     | Optional[int]  | training arguments | Number of update steps between two evaluations                                                                                                                                                            |
| `warmup_steps`                    | 1_000                     | int            | training arguments | Number of steps to warm up                                                                                                                                                                                |
| `max_steps`                       | None                      | Optional[int]  | training arguments | Maximum number of training steps                                                                                                                                                                          |
| `num_train_epochs`                | 1                         | int            | training arguments | Number of training epochs                                                                                                                                                                                 |
| `learning_rate`                   | 2e-4                      | float          | training arguments | Learning rate value                                                                                                                                                                                       |
| `max_grad_norm`                   | 1.0                       | float          | training arguments | Clip grad value                                                                                                                                                                                           |
| `weight_decay`                    | 0.001                     | float          | training arguments | Weight decay value                                                                                                                                                                                        |
| `label_smoothing_factor`          | 0.0                       | float          | training arguments | Label smoothing value                                                                                                                                                                                     |
| `logging_steps`                   | 10                        | int            | training arguments | Number of steps between logging                                                                                                                                                                           |
| `save_steps`                      | 100                       | int            | training arguments | The number of training steps between saving the checkpoint and uploading to the hub                                                                                                                       |
| `save_total_limit`                | 1                         | int            | training arguments | The number of checkpoints that are saved locally                                                                                                                                                          |
| `optim`                           | paged_adamw_8bit          | Optional[str]  | training arguments | Optimizer name. It will be overwritten if you use deepspeed                                                                                                                                               |
| `push_to_hub`                     | False                     | bool           | training arguments | Upload the model to the hub. The model will be uploaded to the hub every save_steps. If LoRA is used, then LoRA's weights will be loaded onto the hub                                                     |
| `hub_model_id`                    | None                      | Optional[str]  | training arguments | The name of the model at the hub. Example: BobaZooba/Shurale                                                                                                                                              |
| `hub_private_repo`                | True                      | bool           | training arguments | Private repository or not                                                                                                                                                                                 |
| `report_to_wandb`                 | False                     | bool           | wandb              | Report or not to Weight & Biases                                                                                                                                                                          |
| `wandb_api_key`                   | None                      | Optional[str]  | wandb              | Weight & Biases API key. You can also set this key using .env file                                                                                                                                        |
| `wandb_project`                   | None                      | Optional[str]  | wandb              | Weight & Biases project name                                                                                                                                                                              |
| `wandb_entity`                    | None                      | Optional[str]  | wandb              | Weight & Biases entity name (user or company)                                                                                                                                                             |

</details>

## How to extend config

You may need to extend the config. To do this, you will need to inherit from the default config and write your own logic or add fields.

`xllm_demo/core/config.py`

```python
from dataclasses import dataclass, field

from xllm import Config

@dataclass
class DemoXLLMConfig(Config):
    text_field: str = field(default="chosen", metadata={
        "help": "Field for Antropic RLHF dataset",
    })
```

Next, you will need to add this class to the xllm CLI function calls.

`xllm_demo/cli/train.py`

```python
from xllm.cli import cli_run_train

from xllm_demo.core.config import DemoXLLMConfig

if __name__ == "__main__":
    cli_run_train(config_cls=DemoXLLMConfig)
```

# How do I choose the methods for training?

- It is recommended to use at least the ``stabilize`` feature and `use_flash_attention_2` (if your GPU allows it).
- Another incredibly effective method is LoRA (`apply_lora`). It allows for a tremendous reduction in training costs and, moreover, helps very effectively combat catastrophic forgetting.
- Then, I advise using `load_in_4bit` and `prepare_model_for_kbit_training` together. This also significantly reduces memory consumption.
- Lastly, I would recommend apply `use_gradient_checkpointing`. This method also greatly reduces memory consumption, but at the expense of slowing down training.
- If your project allows, I suggest enabling `push_to_hub` and `hub_private_repo`, also specifying the model name in `hub_model_id` and `save_steps`. Example: "BobaZooba/SupaDupaLlama-7B-LoRA". During training, every checkpoint of your model will be saved in the HuggingFace Hub. If you specified `apply_lora`, then only the LoRA weights will be saved, which you can later easily fuse with the main model, for example, using `xllm`.
- I also recommend using `report_to_wandb`, also specifying `wandb_project` (the project name in W&B) and `wandb_entity` (user or organization name in W&B).
- Note that for `push_to_hub`, you need to log in to the HuggingFace Hub beforehand or specify the token (`HUGGING_FACE_HUB_TOKEN`) in the .env file. Similarly, when using `report_to_wandb`, you will need to log in to W&B. You can either specify the token (`WANDB_API_KEY`) in the .env file or you will be prompted to enter the token on the command line.

## Multiple GPUs training with DeepSpeed

`DeepSpeed` is needed for training models on `multiple GPUs`. `DeepSpeed` allows you
to `efficiently manage the resources of several GPUs during training`. For example, you
can `distribute the gradients and the state of the optimizer to several GPUs`, rather than storing a complete set of
gradients and the state of the optimizer on each GPU. Starting training using `DeepSpeed` can only happen from
the `command line`.

`train.py`

  ```python
from xllm import Config
from xllm.datasets import GeneralDataset
from xllm.cli import cli_run_train

if __name__ == '__main__':
    train_data = ["Hello!"] * 100
    train_dataset = GeneralDataset.from_list(data=train_data)
    cli_run_train(config_cls=Config, train_dataset=train_dataset)
  ```

Run train (in the `num_gpus` parameter, specify as many GPUs as you have)

  ```bash
  deepspeed --num_gpus=8 train.py --deepspeed_stage 2
  ```

You also can pass other parameters

  ```bash
deepspeed --num_gpus=8 train.py \
    --deepspeed_stage 2 \
    --apply_lora True \
    --stabilize True \
    --use_gradient_checkpointing True
  ```

# Special details

#### DeepSpeed, LoRA and push_to_hub

Currently, when using DeepSpeed, sending a checkpoint to the Hugging Face Hub also results in sending a checkpoint of the entire model in DeepSpeed format. This behavior is planned to be corrected in the future.

#### DeepSpeed stage 3 and bitsandbytes

Currently, it is not possible to use DeepSpeed Stage 3 and bitsandbytes together. This behavior is planned to be corrected in the future.

# FAQ

**Q:** How to understand that my GPU is suitable for Flash Attention 2 (`use_flash_attention_2`)?  
**A:** The simplest and most cost-effective way is to just try running it with this function and without it. For more details, you can refer to this source: https://github.com/Dao-AILab/flash-attention

**Q:** Can I use `bitsandbytes int4`, `bitsandbytes int8` and `gptq model` at once?  
**A:** You can choose either `bitsandbytes int4` or `bitsandbytes int8` only. Training through the gptq model is not recommended, but it is supported. Your model must already be converted to the GPTQ format.

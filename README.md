# 🦖 X—LLM: Cutting Edge & Easy LLM Finetuning

<div align="center">

[![Build](https://github.com/BobaZooba/xllm/actions/workflows/build.yaml/badge.svg?branch=main)](https://github.com/BobaZooba/xllm/actions/workflows/build.yaml)
[![Github: License](https://img.shields.io/github/license/BobaZooba/xllm.svg?color=63C462&style=flat-square)](https://github.com/BobaZooba/xllm/blob/main/LICENSE)
[![Github: Release](https://img.shields.io/github/v/release/BobaZooba/xllm.svg?color=63C462&style=flat-square)](https://github.com/BobaZooba/xllm/releases)

[![PyPI - Version](https://img.shields.io/pypi/v/xllm.svg?logo=pypi&label=PyPI&logoColor=gold&color=63C462&style=flat-square)](https://pypi.org/project/xllm/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/xllm.svg?color=63C462&label=Downloads&logo=pypi&logoColor=gold&style=flat-square)](https://pypi.org/project/xllm/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/xllm?logo=python&label=Python&logoColor=gold&color=63C462&style=flat-square)](https://pypi.org/project/xllm/)

[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=gold&color=63C462&style=flat-square)](https://github.com/modelfront/predictor/blob/master/.pre-commit-config.yaml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?logoColor=gold&color=63C462&style=flat-square)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json?logoColor=gold&color=63C462&style=flat-square)](https://github.com/astral-sh/ruff)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![codecov](https://codecov.io/gh/BobaZooba/xllm/graph/badge.svg?token=ZOBMDXVW4B)](https://codecov.io/gh/BobaZooba/xllm)

[![](https://dcbadge.vercel.app/api/server/n5QNPUSm)](https://discord.gg/n5QNPUSm)

Cutting Edge & Easy LLM Finetuning using the most advanced methods (QLoRA, DeepSpeed, GPTQ, Flash Attention 2, FSDP,
etc)

Developed
by [Boris Zubarev](https://t.me/BobaZooba) | [CV](https://docs.google.com/document/d/1BhFvIHQ1mpm81P-n2A-lhNac-U2wOGc6F2uS9gKvk88/edit?usp=sharing) | [LinkedIn](https://www.linkedin.com/in/boriszubarev/) | [bobazooba@gmail.com](mailto:bobazooba@gmail.com)

</div>

# Why you should use X—LLM 🪄

Are you using **Large Language Models (LLMs)** for your work and want to train them more efficiently with advanced
methods? Wish to focus on the data and improvements rather than repetitive and time-consuming coding for LLM training?

**X—LLM** is your solution. It's a user-friendly library that streamlines training optimization, so you can **focus on
enhancing your models and data**. Equipped with **cutting-edge training techniques**, X—LLM is engineered for efficiency
by engineers who understand your needs.

**X—LLM** is ideal whether you're **gearing up for production** or need a **fast prototyping tool**.

## Features

- Hassle-free training for Large Language Models
- Seamless integration of new data and data processing
- Effortless expansion of the library
- Speed up your training, while simultaneously reducing model sizes
- Each checkpoint is saved to the 🤗 HuggingFace Hub
- Easy-to-use integration with your existing project
- Customize almost any part of your training with ease
- Track your training progress using `W&B`
- Supported many 🤗 Transformers models
  like `Yi-34B`, `Mistal AI`, `Llama 2`, `Zephyr`, `OpenChat`, `Falcon`, `Phi`, `Qwen`, `MPT` and many more
- Benefit from cutting-edge advancements in LLM training optimization
  - QLoRA and fusing
  - Flash Attention 2
  - Gradient checkpointing
  - bitsandbytes
  - GPTQ (including post-training quantization)
  - DeepSpeed
  - FSDP
  - And many more

# Quickstart 🦖

### Installation

`X—LLM` is tested on Python 3.8+, PyTorch 2.0.1+ and CUDA 11.8.

```sh
pip install xllm
```

Version which include `deepspeed`, `flash-attn` and `auto-gptq`:

```sh
pip install xllm[train]
```

Default `xllm` version recommended for local development, `xllm[train]` recommended for training.

#### Training recommended environment

CUDA version: `11.8`
Docker: `huggingface/transformers-pytorch-gpu:latest`

## Fast prototyping ⚡

```python
from xllm import Config
from xllm.datasets import GeneralDataset
from xllm.experiments import Experiment

# Init Config which controls the internal logic of xllm
# QLoRA example
config = Config(
  model_name_or_path="HuggingFaceH4/zephyr-7b-beta",
  apply_lora=True,
  load_in_4bit=True,
)

# Prepare the data
train_data = ["Hello!"] * 100
train_dataset = GeneralDataset.from_list(data=train_data)

# Build Experiment from Config: init tokenizer and model, apply LoRA and so on
experiment = Experiment(config=config, train_dataset=train_dataset)
experiment.build()

# Run Experiment (training)
experiment.run()

# # [Optional] Fuse LoRA layers
# experiment.fuse_lora()

# [Optional] Or push LoRA weights to HuggingFace Hub
experiment.push_to_hub(repo_id="YOUR_NAME/MODEL_NAME")
```

### How `Config` controls `xllm`

[More about config](https://github.com/BobaZooba/xllm/blob/main/GUIDE.md#config)

<details>
  <summary>LoRA</summary>

#### Simple

  ```python
  config = Config(
    model_name_or_path="openchat/openchat_3.5",
    apply_lora=True,
)
  ```

#### Advanced

  ```python
  config = Config(
    model_name_or_path="openchat/openchat_3.5",
    apply_lora=True,
    lora_rank=8,
    lora_alpha=32,
    lora_dropout=0.05,
    raw_lora_target_modules="all",
    # Names of modules to apply LoRA. A comma-separated string, for example: "k,q,v" or "all".
)
  ```

</details>

<details>
  <summary>QLoRA</summary>

To train the `QLoRA` model, we need to load the backbone model using `bitsandbytes` library and int4 (or int8) weights.

#### Simple

  ```python
  config = Config(
    model_name_or_path="01-ai/Yi-34B",
    apply_lora=True,
    load_in_4bit=True,
    prepare_model_for_kbit_training=True,
)
  ```

#### Advanced

  ```python
  config = Config(
    model_name_or_path="01-ai/Yi-34B",
    apply_lora=True,
    load_in_4bit=True,
    prepare_model_for_kbit_training=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
  ```

</details>

<details>
  <summary>Push checkpoints to the HuggingFace Hub</summary>

Before that, you must log in to `Huggingface Hub` or add an `API Token` to the environment variables.

  ```python
  config = Config(
    model_name_or_path="HuggingFaceH4/zephyr-7b-beta",
    push_to_hub=True,
    hub_private_repo=True,
    hub_model_id="BobaZooba/AntModel-7B-XLLM-Demo-LoRA",
    save_steps=25,
)
  ```

- Checkpoints will be saved locally and in Huggingface Hub each `save_steps`
- If you train a model with `LoRA`, then only `LoRA` weights will be saved

</details>

<details>
  <summary>Report to W&B</summary>

Before that, you must log in to `W&B` or add an `API Token` to the environment variables.

  ```python
  config = Config(
    model_name_or_path="HuggingFaceH4/zephyr-7b-beta",
    report_to_wandb=True,
    wandb_project="xllm-demo",
    wandb_entity="bobazooba",
)
  ```

</details>

<details>
  <summary>Gradient checkpointing</summary>

This will help to use `less GPU memory` during training, that is, you will be able to learn more than without this
technique. The disadvantages of this technique is slowing down the forward step, that is, `slowing down training`.

You will be training larger models (for example 7B in colab), but at the expense of training speed.

  ```python
  config = Config(
    model_name_or_path="HuggingFaceH4/zephyr-7b-beta",
    use_gradient_checkpointing=True,
)
  ```

</details>

<details>
  <summary>Flash Attention 2</summary>

This speeds up training and GPU memory consumption, but it does not work with all models and GPUs. You also need to
install `flash-attn` for this. This can be done using:

`pip install xllm[train]`

  ```python
  config = Config(
    model_name_or_path="meta-llama/Llama-2-7b-hf",
    use_flash_attention_2=True,
)
  ```

</details>

</details>

<details>
  <summary><b>Recommended setup</b></summary>

#### Recommendations

- Incredibly effective method is LoRA (`apply_lora`). It allows for a tremendous reduction in training costs
  and, moreover, helps very effectively combat catastrophic forgetting.
- Then, I advise using `load_in_4bit` and `prepare_model_for_kbit_training` together. This also significantly reduces
  memory consumption.
- Lastly, I would recommend apply `use_gradient_checkpointing`. This method also greatly reduces memory consumption, but
  at the expense of slowing down training.
- If your project allows, I suggest enabling `push_to_hub` and `hub_private_repo`, also specifying the model name
  in `hub_model_id` and `save_steps`. Example: "BobaZooba/SupaDupaLlama-7B-LoRA". During training, every checkpoint of
  your model will be saved in the HuggingFace Hub. If you specified `apply_lora`, then only the LoRA weights will be
  saved, which you can later easily fuse with the main model, for example, using `xllm`.
- If your GPU allows it add `use_flash_attention_2`
- I also recommend using `report_to_wandb`, also specifying `wandb_project` (the project name in W&B)
  and `wandb_entity` (user or organization name in W&B).
- Note that for `push_to_hub`, you need to log in to the HuggingFace Hub beforehand or specify the
  token (`HUGGING_FACE_HUB_TOKEN`) in the .env file. Similarly, when using `report_to_wandb`, you will need to log in to
  W&B. You can either specify the token (`WANDB_API_KEY`) in the .env file or you will be prompted to enter the token on
  the command line.

#### Features

- QLoRA
- Gradient checkpointing
- Flash Attention 2
- Stabilize training
- Push checkpoints to HuggingFace Hub
- W&B report

  ```python
  config = Config(
    model_name_or_path="meta-llama/Llama-2-7b-hf",
    tokenizer_padding_side="right",  # good for llama2
  
    warmup_steps=1000,
    max_steps=10000,
    logging_steps=1,
    save_steps=1000,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    max_length=2048,
  
    stabilize=True,
    use_flash_attention_2=True,
  
    apply_lora=True,
  
    load_in_4bit=True,
    prepare_model_for_kbit_training=True,
  
    use_gradient_checkpointing=True,

    push_to_hub=False,
    hub_private_repo=True,
    hub_model_id="BobaZooba/SupaDupaLlama-7B-LoRA",

    report_to_wandb=False,
    wandb_project="xllm-demo",
    wandb_entity="bobazooba",
  )
  ```

</details>

<details>
  <summary>Fuse</summary>

This operation is only for models with a LoRA adapter.

You can explicitly specify to fuse the model after training.

  ```python
  config = Config(
    model_name_or_path="HuggingFaceH4/zephyr-7b-beta",
    apply_lora=True,
    fuse_after_training=True,
)
  ```

Even when you are using QLoRa

  ```python
  config = Config(
    model_name_or_path="HuggingFaceH4/zephyr-7b-beta",
    apply_lora=True,
    load_in_4bit=True,
    prepare_model_for_kbit_training=True,
    fuse_after_training=True,
)
  ```

Or you can fuse the model yourself after training.

  ```python
  experiment.fuse_lora()
  ```

</details>

<details>
  <summary>DeepSpeed</summary>

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

  ```sh
  deepspeed --num_gpus=8 train.py --deepspeed_stage 2
  ```

You also can pass other parameters

  ```sh
deepspeed --num_gpus=8 train.py \
    --deepspeed_stage 2 \
    --apply_lora True \
    --stabilize True \
    --use_gradient_checkpointing True
  ```

</details>

### Notebooks

| Name                                      | Comment                                                                            | Link                                                                                                                                                                   |
|-------------------------------------------|------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| X—LLM Prototyping                         | In this notebook you will learn the basics of the library                          | [![xllm_prototyping](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1zsNmJFns1PKZy5VE5p5nsQL-mZF7SwHf?usp=sharing) |
| Llama2 & Mistral AI efficient fine-tuning | 7B model training in colab using QLoRA, bnb int4, gradient checkpointing and X—LLM | [![Llama2MistalAI](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1CNNB_HPhQ8g7piosdehqWlgA30xoLauP?usp=sharing)   |

## Production solution 🚀

`X—LLM` enables not only to prototype models, but also facilitates the development of production-ready solutions through
built-in capabilities and customization.

Using `X—LLM` to train a model is easy and involves these few steps:

1. `Prepare` — Get the data and the model ready by downloading and preparing them. Saves data locally
   to `config.train_local_path_to_data` and `config.eval_local_path_to_data` if you are using eval dataset
2. `Train` — Use the data prepared in the previous step to train the model
3. `Fuse` — If you used LoRA during the training, fuse LoRA
4. `Quantize` — Optimize your model's memory usage by quantizing it

Remember, these tasks in `X—LLM` start from the command line. So, when you're all set to go, launching your full project
will look something like this:

<details>
  <summary>Example how to run your project</summary>

1. Downloading and preparing data and model

  ```sh
python3 MY_PROJECT/cli/prepare.py \
    --dataset_key MY_DATASET \
    --model_name_or_path mistralai/Mistral-7B-v0.1 \
    --path_to_env_file ./.env
  ```

2. Run train using DeepSpeed on multiple GPUs

  ```sh
deepspeed --num_gpus=8 MY_PROJECT/cli/train.py \
    --use_gradient_checkpointing True \
    --deepspeed_stage 2 \
    --stabilize True \
    --model_name_or_path mistralai/Mistral-7B-v0.1 \
    --use_flash_attention_2 False \
    --load_in_4bit True \
    --apply_lora True \
    --raw_lora_target_modules all \
    --per_device_train_batch_size 8 \
    --warmup_steps 1000 \
    --save_total_limit 0 \
    --push_to_hub True \
    --hub_model_id MY_HF_HUB_NAME/LORA_MODEL_NAME \
    --hub_private_repo True \
    --report_to_wandb True \
    --path_to_env_file ./.env
  ```

3. Fuse LoRA

  ```sh
python3 MY_PROJECT/cli/fuse.py \
    --model_name_or_path mistralai/Mistral-7B-v0.1 \
    --lora_hub_model_id MY_HF_HUB_NAME/LORA_MODEL_NAME \
    --hub_model_id MY_HF_HUB_NAME/MODEL_NAME \
    --hub_private_repo True \
    --force_fp16 True \
    --fused_model_local_path ./fused_model/ \
    --path_to_env_file ./.env
  ```

4. [Optional] GPTQ quantization of the trained model with fused LoRA

  ```sh
python3 MY_PROJECT/cli/quantize.py \
    --model_name_or_path ./fused_model/ \
    --apply_lora False \
    --stabilize False \
    --quantization_max_samples 128 \
    --quantized_model_path ./quantized_model/ \
    --prepare_model_for_kbit_training False \
    --quantized_hub_model_id MY_HF_HUB_NAME/MODEL_NAME_GPTQ \
    --quantized_hub_private_repo True \
    --path_to_env_file ./.env
  ```

</details>

Right now, the `X—LLM` library lets you use only the [SODA dataset](https://huggingface.co/datasets/allenai/soda). We've
set it up this way for demo purposes, but we're planning to add more datasets soon. You'll need to figure out how to
download and handle your dataset. Simply put, you take care of your data, and `X—LLM` handles the rest. We've done it
this
way on purpose, to give you plenty of room to get creative and customize to your heart's content.

You can customize your dataset in detail, adding additional fields. All of this will enable you to implement virtually
any task in the areas of `Supervised Learning` and `Offline Reinforcement Learning`.

At the same time, you always have an easy way to submit data for language modeling.

<details>
  <summary>Example</summary>

  ```python
from xllm import Config
from xllm.datasets import GeneralDataset
from xllm.cli import cli_run_train

if __name__ == '__main__':
    train_data = ["Hello!"] * 100
    train_dataset = GeneralDataset.from_list(data=train_data)
    cli_run_train(config_cls=Config, train_dataset=train_dataset)
  ```

</details>

### Build your own project

To set up your own project using `X—LLM`, you need to do two things:

1. Implement your dataset (figure out how to download and handle it)
2. Add `X—LLM`'s command-line tools into your project

Once that's done, your project will be good to go, and you can start running the steps you need (like prepare, train,
and so on).

To get a handle on building your project with `X—LLM`, check out the materials below.

## Useful materials

- [Guide](https://github.com/BobaZooba/xllm/blob/main/GUIDE.md): here, we go into detail about everything the library can
  do
- [Demo project](https://github.com/BobaZooba/xllm-demo): here's a step-by-step example of how to use `X—LLM` and fit
  it
  into your own project
- [WeatherGPT](https://github.com/BobaZooba/wgpt): this repository features an example of how to utilize the xllm library. Included is a solution for a common type of assessment given to LLM engineers, who typically earn between $120,000 to $140,000 annually
- [Shurale](https://github.com/BobaZooba/shurale): project with the finetuned 7B Mistal model

# Config 🔧

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

## Useful materials

- [Important config fields for different steps](https://github.com/BobaZooba/xllm#how-config-controls-xllm)
- [How do I choose the methods for training?](https://github.com/BobaZooba/xllm/blob/main/GUIDE.md#how-do-i-choose-the-methods-for-training)
- [Detailed description of all config fields](https://github.com/BobaZooba/xllm/blob/main/GUIDE.md#detailed-config-explanation)

# Customization options 🛠

You have the flexibility to tweak many aspects of your model's training: data, how data is processed, trainer, config,
how the model is loaded, what happens before and after training, and so much more.

We've got ready-to-use components for every part of the `xllm` pipeline. You can entirely switch out some components
like the dataset, collator, trainer, and experiment.
For some components like experiment and config, you have the option to just build on what's already there.

### Useful materials

- [How to implement dataset](https://github.com/BobaZooba/xllm/blob/main/GUIDE.md#how-to-implement-dataset)
- [How to add CLI tools to your project](https://github.com/BobaZooba/xllm/blob/main/GUIDE.md#how-to-add-cli-tools-to-your-project)
- [How to implement collator](https://github.com/BobaZooba/xllm/blob/main/GUIDE.md#how-to-implement-collator)
- [How to implement trainer](https://github.com/BobaZooba/xllm/blob/main/GUIDE.md#how-to-implement-trainer)
- [How to implement experiment](https://github.com/BobaZooba/xllm/blob/main/GUIDE.md#how-to-implement-experiment)
- [How to extend config](https://github.com/BobaZooba/xllm/blob/main/GUIDE.md#how-to-extend-config)

# Projects using X—LLM 🏆

Building something cool with [X—LLM](https://github.com/BobaZooba/xllm)? Kindly reach out to me
at [bobazooba@gmail.com](mailto:bobazooba@gmail.com). I'd love to hear from you.

## Hall of Fame

Write to us so that we can add your project.

- [Shurale7B-v1](https://huggingface.co/BobaZooba/Shurale7b-v1)
- [Shurale7B-v1-GPTQ](https://huggingface.co/BobaZooba/Shurale7b-v1-GPTQ)

## Badge

Consider adding a badge to your model card.

```sh
[<img src="https://github.com/BobaZooba/xllm/blob/main/static/images/xllm-badge.png" alt="Powered by X—LLM" width="175" height="32"/>](https://github.com/BobaZooba/xllm)
```

[<img src="https://github.com/BobaZooba/xllm/blob/main/static/images/xllm-badge.png" alt="Powered by X—LLM" width="175" height="32"/>](https://github.com/BobaZooba/xllm)

# Testing 🧪

At the moment, we don't have Continuous Integration tests that utilize a GPU. However, we might develop these kinds of
tests in the future. It's important to note, though, that this would require investing time into their development, as
well as funding for machine maintenance.

# Future Work 🔮

- Add more testst
- Add RunPod deploy function
- Add [DPO](https://arxiv.org/abs/2305.18290) components
- Add Callbacks to `Experiment`
- GPU CI using RunPod
- Solve DeepSpeed + push_to_hub checkpoints bug
- Make DeepSpeed Stage 3 + bitsandbytes works correc
- Add multipacking
- Add adaptive batch size
- Fix caching in CI
- Add sequence bucketing
- Add more datasets
- Maybe add [tensor_parallel](https://github.com/BlackSamorez/tensor_parallel)

## Tale Quest

`Tale Quest` is my personal project which was built using `xllm` and `Shurale`. It's an interactive text-based game
in `Telegram` with dynamic AI characters, offering infinite scenarios

You will get into exciting journeys and complete fascinating quests. Chat
with `George Orwell`, `Tech Entrepreneur`, `Young Wizard`, `Noir Detective`, `Femme Fatale` and many more

Try it now: [https://t.me/talequestbot](https://t.me/TaleQuestBot?start=Z2g)

### Please support me here

<a href="https://www.buymeacoffee.com/talequest" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/arial-yellow.png" alt="Buy Me A Coffee" style="height: 60px !important;width: 217px !important;" ></a>

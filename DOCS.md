# Build your own project using X—LLM

# WORK IN PROGRESS

# How to implement dataset

Берешь и делаешь

# How to add CLI tools to your project

## Требования к формату выхода примеров

### 1. Download

First off, you must download the data and the model. This step also handles data preparation. You have to complete this
necessary step because it sets up how your data will be downloaded and prepared.

Before you start the training, it's crucial to get the data and the model ready. You won't be able to start training
without them since this is the point where your data gets prepped for training. We made this a separate step for good
reasons. For example, if you're training across multiple GPUs, like with DeepSpeed, you'd otherwise end up downloading
the same data and models on each GPU, when you only need to do it once.

#### 1.1 Implement your dataset

#### 1.2 Register your dataset

#### 1.3 Run the downloading

### 2. Train

### 3. Fuse

Этот шаг нужен только в том случае, если вы обучали модель с использованием LoRA, что является рекомендуемым решением.

### 4. GPTQ Quantization

Это опциональный шаг

# Config

| Key                             | Default value             | Type           | Entity             | Comment                                                                                                                                                                                                   |
|---------------------------------|---------------------------|----------------|--------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| experiment_key                  | base                      | str            | general            | Experiment class key                                                                                                                                                                                      |
| save_safetensors                | True                      | bool           | general            | Safe serialization                                                                                                                                                                                        |
| max_shard_size                  | 10GB                      | str            | general            | max_shard_size for the model pushing to the HuggingFace Hub                                                                                                                                               |
| local_rank                      | 0                         | int            | general            | Local rank for logging and saving. Works only in distributed training                                                                                                                                     |
| use_gradient_checkpointing      | False                     | bool           | general            | If True, use gradient checkpointing to save memory at the expense of slower backward pass                                                                                                                 |
| trainer_key                     | lm                        | str            | general            | Key of the trainer for loading from trainers_registry                                                                                                                                                     |
| force_fp32                      | False                     | bool           | general            | Force using fp32 when model loading                                                                                                                                                                       |
| force_fp16                      | False                     | bool           | general            | Force using fp16 when model loading                                                                                                                                                                       |
| from_gptq                       | False                     | bool           | general            | If you loadining GPTQ quantized model                                                                                                                                                                     |
| huggingface_hub_token           | None                      | Optional[str]  | general            | HuggingFace Hub token. You can also set this key using .env file                                                                                                                                          |
| deepspeed_stage                 | 0                         | int            | general            | Predifined DeepSpeed stage                                                                                                                                                                                |
| deepspeed_config_path           | None                      | Optional[int]  | general            | Path to DeepSpeed config                                                                                                                                                                                  |
| fsdp_strategy                   |                           | str            | general            | FSDP strategy                                                                                                                                                                                             |
| fsdp_offload                    | True                      | True           | general            | Offload weights when using FSDP                                                                                                                                                                           |
| seed                            | 42                        | int            | general            | Seed value for random operations                                                                                                                                                                          |
| stabilize                       | False                     | bool           | general            | Stabilize the model. Convert some weights to fp32, some to fp16/bf16                                                                                                                                      |
| path_to_env_file                | ./.env                    | Optional[str]  | general            | Custom path to .env file                                                                                                                                                                                  |
| prepare_dataset                 | True                      | bool           | general            | Prepare the dataset. Works only at "prepare" stage                                                                                                                                                        |
| lora_hub_model_id               | None                      | Optional[str]  | fuse               | Fusing LoRA. The name of the LoRA model at the hub for fusing. Example: BobaZooba/Shurale                                                                                                                 |
| lora_model_local_path           | None                      | Optional[str]  | fuse               | Fusing LoRA. Local path to the LoRA model                                                                                                                                                                 |
| fused_model_local_path          | None                      | Optional[str]  | fuse               | Local path to fused model. Useful if you want to quantize model after fusing on the same machine                                                                                                          |
| fuse_after_training             | False                     | bool           | fuse               | Fuse or not model after training                                                                                                                                                                          |
| quantization_dataset_id         | None                      | Optional[str]  | gptq quantization  | Dataset id for GPTQ quantization. You can install either the idi dataset, or use any dataset                                                                                                              |
| quantization_max_samples        | 1024                      | int            | gptq quantization  | Max samples for GPTQ quantization if you use custom dataset                                                                                                                                               |
| quantized_model_path            | ./quantized_model/        | str            | gptq quantization  | Path to GPTQ quantized model                                                                                                                                                                              |
| quantized_hub_model_id          | None                      | Optional[str]  | gptq quantization  | The name of the model at the hub for GPTQ quantization. Example: BobaZooba/Shurale-GPTQ                                                                                                                   |
| quantized_hub_private_repo      | True                      | bool           | gptq quantization  | Private repository for GPTQ quantization model or not                                                                                                                                                     |
| dataset_key                     | soda                      | str            | dataset            | Key of the dataset for loading from datasets_registry                                                                                                                                                     |
| train_local_path_to_data        | ./train.jsonl             | str            | dataset            | The path to the local training data file                                                                                                                                                                  |
| eval_local_path_to_data         | None                      | Optional[str]  | dataset            | The path to the local eval data file                                                                                                                                                                      |
| shuffle                         | True                      | bool           | dataset            | Shuffle training data                                                                                                                                                                                     |
| max_eval_samples                | 1_000                     | int            | dataset            | Maximum number of examples for evaluation                                                                                                                                                                 |
| add_eval_to_train_if_no_path    | False                     | bool           | dataset            | Add evaluation data to training data if their number is greater than max_eval_samples                                                                                                                     |
| tokenizer_name_or_path          | None                      | Optional[str]  | tokenizer          | Tokenizer name or path. If the value is not set, then it will be taken from the model_name_or_path                                                                                                        |
| tokenizer_use_fast              | None                      | Optional[bool] | tokenizer          | Use fast flag for the tokenizer                                                                                                                                                                           |
| tokenizer_padding_side          | None                      | Optional[str]  | tokenizer          | Padding side of the collator: None, right or left                                                                                                                                                         |
| collator_key                    | lm                        | str            | collator           | Key of the collator for loading from collators_registry                                                                                                                                                   |
| max_length                      | 2048                      | int            | collator           | Max sequence length of the model                                                                                                                                                                          |
| model_name_or_path              | mistralai/Mistral-7B-v0.1 | str            | model              | Model name or path. It could be from HuggingFace or locally                                                                                                                                               |
| push_to_hub_bos_add_bos_token   | False                     | bool           | model              | Upload to the hub tokenization config with add_bos_token equals to True. Might be helpful for TGI                                                                                                         |
| use_flash_attention_2           | False                     | bool           | model              | Use or not flash attention 2. Requires 1) CUDA >= 11.6; 2) install flash-attn 3) compatible model                                                                                                         |
| trust_remote_code               | False                     | bool           | model              | Trust remote code from HuggingFace                                                                                                                                                                        |
| device_map                      | None                      | None           | model              | Device map for loading the model                                                                                                                                                                          |
| prepare_model_for_kbit_training | True                      | bool           | model              | Prepare or not for kbit training                                                                                                                                                                          |
| load_in_8bit                    | False                     | bool           | bitsandbytes       | Load the model in 8 bit using bitsandbytes                                                                                                                                                                |
| load_in_4bit                    | False                     | bool           | bitsandbytes       | Load the model in 4 bit using bitsandbytes                                                                                                                                                                |
| llm_int8_threshold              | 6.0                       | float          | bitsandbytes       | Threshold for outlier detection                                                                                                                                                                           |
| llm_int8_has_fp16_weight        | True                      | bool           | bitsandbytes       | LLM has weights in fp16                                                                                                                                                                                   |
| bnb_4bit_use_double_quant       | True                      | bool           | bitsandbytes       | Double quantization. This will enable a second quantization after the first one to save an additional 0.4 bits per parameter                                                                              |
| bnb_4bit_quant_type             | nf4                       | str            | bitsandbytes       | Quantization type for 4 bit                                                                                                                                                                               |
| bnb_quantize_after_model_init   | False                     | bool           | bitsandbytes       | If False, quantization will be at model init                                                                                                                                                              |
| gptq_bits                       | 4                         | int            | gptq               | Bits for GPTQ quantization                                                                                                                                                                                |
| gptq_group_size                 | 128                       | int            | gptq               | Group size for GPTQ quantization                                                                                                                                                                          |
| gptq_disable_exllama            | True                      | bool           | gptq               | Disable ExLlama kernels for GPTQ quantization                                                                                                                                                             |
| apply_lora                      | False                     | bool           | lora               | Apply LoRA to the model or not                                                                                                                                                                            |
| lora_rank                       | 8                         | int            | lora               | LoRA rank value. LoRA matrices W_A x R and R x W_B, where R is LoRA rank                                                                                                                                  |
| lora_alpha                      | 32                        | int            | lora               | LoRA alpha value. The resulting LoRA matrix will be multiplied by this value                                                                                                                              |
| lora_dropout                    | 0.1                       | float          | lora               | LoRA dropout value                                                                                                                                                                                        |
| raw_lora_target_modules         | all                       | str            | lora               | Names of modules to apply LoRA. A comma-separated string, for example: "k,q,v". When setting the value "all", LoRA will be applied to all linear layers, except for the input embeddings and the lm_head. |
| output_dir                      | ./outputs/                | str            | training arguments | The path to the directory where the artifacts will be saved                                                                                                                                               |
| per_device_train_batch_size     | 2                         | int            | training arguments | Batch size on each GPU                                                                                                                                                                                    |
| do_eval                         | False                     | bool           | training arguments | Run eval or not                                                                                                                                                                                           |
| per_device_eval_batch_size      | None                      | Optional[int]  | training arguments | Batch size on each GPU for evaluation. If None per_device_eval_batch_size equals to per_device_train_batch_size                                                                                           |
| gradient_accumulation_steps     | 1                         | int            | training arguments | Number of steps to accumulate gradients                                                                                                                                                                   |
| eval_accumulation_steps         | None                      | Optional[int]  | training arguments | Number of steps to accumulate gradients at evaluation. If None eval_accumulation_steps equals to gradient_accumulation_steps                                                                              |
| eval_delay                      | 0                         | int            | training arguments | Number of epochs or steps to wait for before the first evaluation can be performed, depending on the evaluation_strategy                                                                                  |
| eval_steps                      | 1_000                     | Optional[int]  | training arguments | Number of update steps between two evaluations                                                                                                                                                            |
| warmup_steps                    | 1_000                     | int            | training arguments | Number of steps to warm up                                                                                                                                                                                |
| max_steps                       | None                      | Optional[int]  | training arguments | Maximum number of training steps                                                                                                                                                                          |
| num_train_epochs                | 1                         | int            | training arguments | Number of training epochs                                                                                                                                                                                 |
| learning_rate                   | 2e-4                      | float          | training arguments | Learning rate value                                                                                                                                                                                       |
| max_grad_norm                   | 1.0                       | float          | training arguments | Clip grad value                                                                                                                                                                                           |
| weight_decay                    | 0.001                     | float          | training arguments | Weight decay value                                                                                                                                                                                        |
| label_smoothing_factor          | 0.0                       | float          | training arguments | Label smoothing value                                                                                                                                                                                     |
| logging_steps                   | 10                        | int            | training arguments | Number of steps between logging                                                                                                                                                                           |
| save_steps                      | 100                       | int            | training arguments | The number of training steps between saving the checkpoint and uploading to the hub                                                                                                                       |
| save_total_limit                | 1                         | int            | training arguments | The number of checkpoints that are saved locally                                                                                                                                                          |
| optim                           | paged_adamw_8bit          | Optional[str]  | training arguments | Optimizer name. It will be overwritten if you use deepspeed                                                                                                                                               |
| push_to_hub                     | False                     | bool           | training arguments | Upload the model to the hub. The model will be uploaded to the hub every save_steps. If LoRA is used, then LoRA's weights will be loaded onto the hub                                                     |
| hub_model_id                    | None                      | Optional[str]  | training arguments | The name of the model at the hub. Example: BobaZooba/Shurale                                                                                                                                              |
| hub_private_repo                | True                      | bool           | training arguments | Private repository or not                                                                                                                                                                                 |
| report_to_wandb                 | False                     | bool           | wandb              | Report or not to Weight & Biases                                                                                                                                                                          |
| wandb_api_key                   | None                      | Optional[str]  | wandb              | Weight & Biases API key. You can also set this key using .env file                                                                                                                                        |
| wandb_project                   | None                      | Optional[str]  | wandb              | Weight & Biases project name                                                                                                                                                                              |
| wandb_entity                    | None                      | Optional[str]  | wandb              | Weight & Biases entity name (user or company)                                                                                                                                                             |

## Important config keys for different steps

## How do I choose the methods for training?

### Reduce the size of the model during training

### Single GPU

### Multiple GPUs

# Дополнительные комментарии

## FAQ

# DeepSpeed and push_to_hub

Не можете использовать сразу int4, int8, gptq модель

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

# Config, все поля

## Important config keys for different steps

## How do I choose the methods for training?

### Reduce the size of the model during training

### Single GPU

### Multiple GPUs

# Дополнительные комментарии

## FAQ

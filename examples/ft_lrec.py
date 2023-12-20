import os
import sys
sys.path.append(os.path.abspath('.'))

# from src.xllm.cli.train import cli_run_train
from src.xllm.experiments import Experiment
from src.xllm.core.config import Config
from src.xllm.datasets import GeneralDataset
from datasets import load_dataset


def prepare_data(dataset):
    data = list()

    for sample in dataset:
        data.append({
            "text": (
            f"Réponds à la question suivante en t'appuyant exclusivement sur le document fourni:"
            f" {sample['question']} documents: {sample['title']} {' '.join(sample['documents'])}"
            f"target: {sample['output']}"
            )
            })
    return data


if __name__ == "__main__":

    dataset = load_dataset("LsTam/cquae_lrec")
    train_dataset = GeneralDataset(data=prepare_data(dataset['train']), separator="target: ")
    eval_dataset = GeneralDataset(data=prepare_data(dataset['eval']), separator="target: ")

    config = Config(
        collator_key="lm",
        model_name_or_path="mistralai/Mistral-7B-Instruct-v0.1",
        use_gradient_checkpointing=True,
        stabilize=True,
        use_flash_attention_2=True,
        load_in_4bit=True, # change to 8bit after first tests
        prepare_model_for_kbit_training=True,
        apply_lora=True,
        # one step is one batch
        warmup_steps=5,
        num_train_epochs=2,
        max_steps=600,
        logging_steps=1,
        save_steps=25,
        save_total_limit=3,

        per_device_train_batch_size=2,
        gradient_accumulation_steps=32,
        max_length=1024, #2048,
        # device_map={'':0},

        # tokenizer_padding_side="right",  # good for llama2

        push_to_hub=True,
        hub_private_repo=True,
        hub_model_id="LsTam/mistral-xllm-7B-LoRA",

        # W&B
        report_to_wandb=False,
        # wandb_project="xllm-demo",
        # wandb_entity="mistral-xllm-2",
    )

    experiment = Experiment(config=config, train_dataset=train_dataset, eval_dataset=eval_dataset)

    experiment.build()

    experiment.run()

    # cli_run_train(config_cls=Config, train_dataset=train_dataset, eval_dataset=eval_dataset)
import os
import sys
sys.path.append(os.path.abspath('.'))

# from src.xllm.cli.train import cli_run_train
from src.xllm.experiments import Experiment
from src.xllm.core.config import Config
from src.xllm.datasets import GeneralDataset
from src.xllm.collators.completion import CompletionCollator

# from transformers import AutoTokenizer
# from datasets import load_dataset


# def prepare_data(dataset):
#     data = list()

#     for sample in dataset:
#         data.append({
#             "text": (
#             f"[INST] Réponds à la question suivante en t'appuyant exclusivement sur le document fourni:"
#             f" {sample['question']} documents: {' '.join([doc['source_text'] for doc in sample['context']])} [/INST]"
#             f"123target: {sample['answer'].strip()} </s>"
#             )
#             })
#     return data


if __name__ == "__main__":

    # dataset = load_dataset("ProfessorBob/cquae_v2", token='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')
    # train_dataset = GeneralDataset(data=prepare_data(dataset['train']), separator="123target: ")
    # eval_dataset = GeneralDataset(data=prepare_data(dataset['eval']), separator="123target: ")



    # tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
    # tok.pad_token = tok.eos_token
    # tok.padding_side = 'right'
    # mistral_collator = CompletionCollator(
    #     tokenizer=tok,
    #     max_length=1024,
    #     separator=''
    # )

    config = Config(
        # Data
        train_local_path_to_data = "/home/louis/data_llm/cqua_v2_train.json", # Local path to the training data file.
        eval_local_path_to_data = "/home/louis/data_llm/cqua_v2_eval.json", # Local path to the evaluation data file.
        collator_key="completion",
        tokenizer_padding_side = 'right',
        tokenizer_name_or_path = "mistralai/Mistral-7B-Instruct-v0.1",
        dataset_key = "input_output",

        output_dir = "/home/louis/run_name",

        model_name_or_path="mistralai/Mistral-7B-Instruct-v0.1",
        use_gradient_checkpointing=True,
        stabilize=True,
        use_flash_attention_2=True,
        load_in_4bit=True, # cahnge to 8 after testing
        prepare_model_for_kbit_training=True,
        apply_lora=True,
        # one step is one batch
        warmup_steps=5,
        num_train_epochs=1,
        # max_steps=250, # if specify don't care about number of epochs
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
        hub_model_id="LsTam/mistral-xllm-7B-LoRA-cquae_v2",

        # W&B
        # TODO make w&b offline
        report_to_wandb=False,
        # wandb_project="xllm-demo",
        # wandb_entity="mistral-xllm-2",
    )

    experiment = Experiment(config=config) #, train_dataset=train_dataset, eval_dataset=eval_dataset, collator=mistral_collator)

    experiment.build()

    experiment.run()

    # cli_run_train(config_cls=Config, train_dataset=train_dataset, eval_dataset=eval_dataset)
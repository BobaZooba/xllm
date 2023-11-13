import datasets
from tqdm import tqdm
from xllm import Config
from xllm.datasets import GeneralDataset
from xllm.experiments import Experiment


def run():
    rlhf_dataset = datasets.load_dataset("Anthropic/hh-rlhf")

    parsed_data = dict()

    for split in ["train", "test"]:
        parsed_data[split] = list()

        for sample in tqdm(rlhf_dataset[split], desc=f"Parsing {split}"):
            text_parts = sample["chosen"].split("\n\n")[1:]

            parsed_data[split].append(text_parts)

        train = parsed_data["train"]
        evaluation = parsed_data["test"]

    train_dataset = GeneralDataset.from_list(data=train)
    eval_dataset = GeneralDataset.from_list(data=evaluation)

    config = Config(model_name_or_path="facebook/opt-350m")

    experiment = Experiment(config=config, train_dataset=train_dataset, eval_dataset=eval_dataset)

    experiment.build()

    experiment.run()

    experiment.push_to_hub(repo_id="YOUR_NAME/MODEL_NAME")

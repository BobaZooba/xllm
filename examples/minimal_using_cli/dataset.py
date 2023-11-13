from typing import Dict, List, Optional, Tuple

import datasets
from tqdm import tqdm

from src.xllm import enums
from src.xllm.core.config import Config
from src.xllm.datasets.base import BaseDataset
from src.xllm.types import RawSample


class AntropicDataset(BaseDataset):
    _HF_DATASET_ID = "Anthropic/hh-rlhf"

    @classmethod
    def get_data(cls, config: Config) -> Tuple[List[RawSample], Optional[List[RawSample]]]:
        rlhf_dataset = datasets.load_dataset(cls._HF_DATASET_ID)

        parsed_data: Dict[str, List[RawSample]] = dict()

        for split in ["train", "test"]:
            parsed_data[split] = list()

            for sample in tqdm(rlhf_dataset[split], desc=f"Parsing {split}"):
                text_parts = sample["chosen"].split("\n\n")[1:]

                parsed_data[split].append(text_parts)

        train = parsed_data["train"]
        evaluation = parsed_data["test"]

        return train, evaluation

    def get_sample(self, index: int) -> RawSample:
        sample = {enums.General.text_parts: self.data[index]}
        return sample

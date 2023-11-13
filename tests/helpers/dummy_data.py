from typing import List, Optional, Tuple

from src.xllm import enums
from src.xllm.core.config import Config
from src.xllm.datasets.base import BaseDataset
from src.xllm.datasets.soda import SodaDataset
from src.xllm.types import RawSample

DATA = [
    {
        enums.General.text_parts: [
            "Person 1: Hello",
            "Person 2: It's me",
            "Person 1: I was wondering",
        ]
    },
    {
        enums.General.text_parts: [
            "You are a sith lord",
            "Kenobi: Hello there",
            "General Grievous: General Kenobi",
        ]
    },
]

SODA_DATASET = [
    {
        SodaDataset.HEADER_KEY: "This is dialog",
        SodaDataset.DIALOG_KEY: ["Hello", "It's me", "I was wondering"],
    },
    {
        SodaDataset.HEADER_KEY: "This is another dialog",
        SodaDataset.DIALOG_KEY: ["Sup", "Hello", "It's me", "I was wondering", "Please buy some coins"],
    },
]

RAW_SODA_DATASET = {
    "train": [
        {
            "narrative": "You are a sith lord",
            "dialogue": ["Hello there", "General Kenobi"],
            "speakers": ["Kenobi", "General Grievous"],
        }
    ],
    "validation": [
        {
            "narrative": "You are a sith lord",
            "dialogue": ["Hello there", "General Kenobi"],
            "speakers": ["Kenobi", "General Grievous"],
        }
    ],
    "test": [
        {
            "narrative": "You are a sith lord",
            "dialogue": ["Hello there", "General Kenobi"],
            "speakers": ["Kenobi", "General Grievous"],
        }
    ],
}


class DummyDataset(BaseDataset):
    @classmethod
    def get_data(cls, config: Config) -> Tuple[List[RawSample], Optional[List[RawSample]]]:
        return DATA, None

    def get_sample(self, index: int) -> RawSample:
        return self.data[index]

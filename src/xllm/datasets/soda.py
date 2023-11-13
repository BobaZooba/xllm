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

from typing import Dict, List, Optional, Tuple

import datasets
import numpy as np
from tqdm import tqdm

from .. import enums
from ..core.config import Config
from ..datasets.base import BaseDataset
from ..types import RawSample


class SodaDataset(BaseDataset):
    HEADER_KEY = "header"
    DIALOG_KEY = "dialog"

    _HF_DATASET_ID = "allenai/soda"

    def __init__(self, data: List[RawSample], header_drop_probability: float = 0.05):
        super().__init__(data=data)
        self.header_drop_probability = header_drop_probability

    @classmethod
    def get_data(cls, config: Config) -> Optional[Tuple[List[RawSample], Optional[List[RawSample]]]]:
        soda_dataset = datasets.load_dataset(cls._HF_DATASET_ID)

        parsed_data: Dict[str, List[RawSample]] = dict()

        known_indices = set()

        for split in ["train", "test"]:
            parsed_data[split] = list()

            for sample in tqdm(soda_dataset[split], desc=f"Parsing SODA {split}"):
                index = sample.get("original_index")

                if index in known_indices:
                    continue

                parsed_sample = {
                    cls.HEADER_KEY: sample.get("narrative"),
                    cls.DIALOG_KEY: [
                        f"{speaker}: {phrase}"
                        for speaker, phrase in zip(sample.get("speakers"), sample.get("dialogue"))
                    ],
                }

                parsed_data[split].append(parsed_sample)
                known_indices.add(index)

        train = parsed_data["train"]
        valid = parsed_data["test"]

        return train, valid

    def get_sample(self, index: int) -> RawSample:
        sample = self.data[index]

        dialog = sample[self.DIALOG_KEY]

        phrases = list()

        if not isinstance(dialog, list):
            raise ValueError(f"{self.DIALOG_KEY} of sample is not a list: {type(dialog)}")

        for phrase in dialog:
            if isinstance(phrase, str):
                phrases.append(phrase)

        if self.HEADER_KEY in sample:
            header = sample[self.HEADER_KEY]

            is_drop_header = np.random.rand() <= self.header_drop_probability

            if not is_drop_header and isinstance(header, str):
                phrases.insert(0, header)

        sample = {enums.General.text_parts: [phrase.replace("\n", " ").replace("\r", " ") for phrase in phrases]}

        return sample

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

import json
import os
import random
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple

from loguru import logger
from torch.utils.data import Dataset

from ..core.config import Config
from ..core.constants import DATASETS_MUST_HAVE_KEYS
from ..types import RawSample
from ..utils.miscellaneous import have_missing_keys


class BaseDataset(Dataset[RawSample], ABC):
    def __init__(self, data: List[RawSample], must_have_keys: List[str] = DATASETS_MUST_HAVE_KEYS):
        super().__init__()
        self.data = data
        self.must_have_keys = must_have_keys

    @classmethod
    def prepare(cls, config: Config) -> None:
        raw_data = cls.get_data(config=config)

        if raw_data is None:
            raise ValueError("Method get_data returned None")
        else:
            train_data, eval_data = raw_data

        if config.eval_local_path_to_data is None and eval_data is not None:
            logger.warning("eval_local_path_to_data is None, but eval_data is not None")
            if config.add_eval_to_train_if_no_path:
                train_data += eval_data
                logger.info("Add eval data to train")

        if eval_data is not None and config.eval_local_path_to_data is not None:
            if len(eval_data) > config.max_eval_samples and config.max_eval_samples > 0:
                train_data += eval_data[config.max_eval_samples :]
                eval_data = eval_data[: config.max_eval_samples]
                logger.info(f"Eval data size truncated to {config.max_eval_samples}")
            else:
                logger.info(f"Eval data size: {len(eval_data)}")

            with open(config.eval_local_path_to_data, mode="w") as file_object:
                for raw_sample in eval_data:
                    file_object.write(json.dumps(raw_sample) + "\n")
        else:
            logger.warning("eval data or eval_local_path_to_data is None")

        if config.shuffle:
            random.shuffle(train_data)
            logger.info("Train data shuffled")

        with open(config.train_local_path_to_data, mode="w") as file_object:
            for raw_sample in train_data:
                file_object.write(json.dumps(raw_sample) + "\n")

        logger.info(f"Train data size: {len(train_data)}")

        return None

    @classmethod
    def load(cls, path_to_data: str, **kwargs: Any) -> "BaseDataset":
        data = list()

        if not os.path.isfile(path_to_data):
            raise FileNotFoundError(f"File {path_to_data} not found. Probably you should run .prepare before")

        with open(path_to_data) as file_object:
            for line in file_object:
                sample = json.loads(line)
                data.append(sample)

        dataset = cls(data=data)

        return dataset

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> RawSample:
        sample = self.get_sample(index=index)
        flag, difference = have_missing_keys(data=sample, must_have_keys=self.must_have_keys)
        if flag:
            raise ValueError(f"RawSample from {self.__class__.__name__} must have {difference} keys")
        return sample

    @classmethod
    @abstractmethod
    def get_data(cls, config: Config) -> Optional[Tuple[List[RawSample], Optional[List[RawSample]]]]:
        raise NotImplementedError

    @abstractmethod
    def get_sample(self, index: int) -> RawSample:
        raise NotImplementedError

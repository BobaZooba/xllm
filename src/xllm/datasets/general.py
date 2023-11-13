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

from typing import List, Optional, Tuple

from .. import enums
from ..core.config import Config
from ..datasets.base import BaseDataset
from ..types import RawSample
from ..utils.logger import dist_logger


class GeneralDataset(BaseDataset):
    def __init__(
        self,
        data: List[RawSample],
        sample_field: str = enums.General.default_sample_field,
        separator: Optional[str] = None,
    ):
        super().__init__(data=data)

        self.sample_field = sample_field
        self.separator = separator

    @classmethod
    def get_data(cls, config: Config) -> Optional[Tuple[List[RawSample], Optional[List[RawSample]]]]:
        dist_logger.warning(
            "This is a special type of dataset in which it is not supposed to get_data anything. "
            "You must pass the data here through __init__ or use from_list, "
            "or through the path in config.train_local_path_to_data and config.eval_local_path_to_data (optional)"
        )
        return None

    @classmethod
    def from_list(
        cls,
        data: List[str],
        sample_field: str = enums.General.default_sample_field,
        separator: Optional[str] = None,
    ) -> "GeneralDataset":
        prepared_data: List[RawSample] = [{sample_field: text} for text in data]
        dataset = cls(data=prepared_data, sample_field=sample_field, separator=separator)
        return dataset

    def get_sample(self, index: int) -> RawSample:
        text = self.data[index][self.sample_field]

        assert isinstance(text, str)

        if self.separator is not None:
            text_parts: List[str] = text.split(self.separator)
        else:
            text_parts = [text]

        sample: RawSample = {enums.General.text_parts: text_parts}

        return sample

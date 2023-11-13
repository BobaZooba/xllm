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

from .. import enums
from ..datasets.general import GeneralDataset
from ..datasets.soda import SodaDataset
from ..utils.registry import Registry

datasets_registry = Registry(name=enums.Registry.datasets)

datasets_registry.add(key=enums.Datasets.default, value=GeneralDataset)
datasets_registry.add(key=enums.Datasets.general, value=GeneralDataset)
datasets_registry.add(key=enums.Datasets.soda, value=SodaDataset)

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

from ..core.config import Config
from ..quantization.quantizer import Quantizer


def quantize(config: Config) -> Quantizer:
    """
    Orchestrates the model quantization process using the configuration parameters provided.

    This function facilitates the reduction of model size and potentially improves inference speed by quantizing the
    model weights according to the given configuration. It sets up the required environment, performs the quantization,
    and saves the resulting smaller model.

    Args:
        config (`Config`):
            The configuration object that contains settings and parameters influencing the quantization process,
            such as model and tokenizer paths, quantization bits, dataset for calibration, and other related settings.

    Returns:
        Quantizer:
            An instance of the `Quantizer` class that has conducted the quantization process and holds the resulting
            quantized model.

    The function executes the following steps:
    - Initializes a new instance of `Quantizer` with the provided configuration.
    - Calls the `build` method on the `Quantizer` to construct or set up the tokenizer, model, and dataset needed for
      quantization.
    - Invokes the `quantize` method to perform the actual quantization process on the model weights.
    - Calls the `save` method to save the resulting quantized model and push it to the Hugging Face model hub if
      configured.

    Example usage:
        ```python
        from some_module.config import Config

        # Assuming we have a predefined Config object for quantization.
        config = Config(...)
        quantizer_instance = quantize(config=config)

        # The quantizer_instance now contains the quantized model ready to use or to be further saved or analyzed.
        ```

    Note:
        Quantization is a critical step for deploying models to environments with resource constraints or specific
        performance targets. This function streamlines the quantization process and ensures that the quantized model
        is accessible through the `Quantizer` instance returned.
    """
    quantizer = Quantizer(config=config)
    quantizer.build()
    quantizer.quantize()
    quantizer.save()

    return quantizer

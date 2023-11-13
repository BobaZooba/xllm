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

from typing import Any, Dict, Union

STAGE_1 = {
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1,
    },
    "bf16": {"enabled": "auto"},
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto",
        },
    },
    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto",
        },
    },
    "zero_optimization": {
        "stage": 1,
        "offload_optimizer": {"device": "cpu", "pin_memory": False},
        "allgather_partitions": True,
        "allgather_bucket_size": 200000000.0,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 200000000.0,
        "contiguous_gradients": True,
        "round_robin_gradients": True,
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": False,
}

STAGE_2 = {
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1,
    },
    "bf16": {"enabled": "auto"},
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto",
        },
    },
    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "total_num_steps": "auto",
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto",
        },
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "cpu", "pin_memory": False},
        "allgather_partitions": True,
        "allgather_bucket_size": 200000000.0,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 200000000.0,
        "contiguous_gradients": True,
        "round_robin_gradients": True,
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": False,
}

STAGE_3 = {
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1,
    },
    "bf16": {"enabled": "auto"},
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto",
        },
    },
    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "total_num_steps": "auto",
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto",
        },
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "cpu", "pin_memory": False},
        "offload_param": {"device": "cpu", "pin_memory": False},
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": 1000000000.0,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1000000000.0,
        "stage3_max_reuse_distance": 1000000000.0,
        "gather_16bit_weights_on_model_save": True,
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": False,
}

DS_CONFIG_MAPPER: Dict[Union[str, int], Dict[str, Any]] = {
    1: STAGE_1,
    "1": STAGE_1,
    "stage_1": STAGE_1,
    2: STAGE_2,
    "2": STAGE_2,
    "stage_2": STAGE_2,
    3: STAGE_3,
    "3": STAGE_3,
    "stage_3": STAGE_3,
}

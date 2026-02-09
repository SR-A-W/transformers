# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
"""Qwen2 Path-Locked MoE (PL-MoE) model."""

from .configuration_qwen2_pl_moe import Qwen2PlMoeConfig
from .modeling_qwen2_pl_moe import (
    Qwen2PlMoeExpertMLP,
    Qwen2PlMoeForCausalLM,
    Qwen2PlMoeMLP,
    Qwen2PlMoeModel,
    Qwen2PlMoeDecoderLayer,
    Qwen2PlMoePreTrainedModel,
)

__all__ = [
    "Qwen2PlMoeConfig",
    "Qwen2PlMoeExpertMLP",
    "Qwen2PlMoeForCausalLM",
    "Qwen2PlMoeMLP",
    "Qwen2PlMoeModel",
    "Qwen2PlMoeDecoderLayer",
    "Qwen2PlMoePreTrainedModel",
]

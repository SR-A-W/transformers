# coding=utf-8
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
"""Qwen3 Path-Locked MoE (PL-MoE) model."""

from .configuration_qwen3_pl_moe import Qwen3PlMoeConfig
from .modeling_qwen3_pl_moe import (
    Qwen3PlMoeExpertMLP,
    Qwen3PlMoeForCausalLM,
    Qwen3PlMoeMLP,
    Qwen3PlMoeModel,
    Qwen3PlMoeDecoderLayer,
    Qwen3PlMoePreTrainedModel,
)

__all__ = [
    "Qwen3PlMoeConfig",
    "Qwen3PlMoeExpertMLP",
    "Qwen3PlMoeForCausalLM",
    "Qwen3PlMoeMLP",
    "Qwen3PlMoeModel",
    "Qwen3PlMoeDecoderLayer",
    "Qwen3PlMoePreTrainedModel",
]

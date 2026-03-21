# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
"""LLaMA Path-Locked MoE (PL-MoE) model."""

from .configuration_llama_pl_moe import LlamaPlMoeConfig
from .modeling_llama_pl_moe import (
    LlamaPlMoeExpertMLP,
    LlamaPlMoeForCausalLM,
    LlamaPlMoeMLP,
    LlamaPlMoeModel,
    LlamaPlMoeDecoderLayer,
    LlamaPlMoePreTrainedModel,
)

__all__ = [
    "LlamaPlMoeConfig",
    "LlamaPlMoeExpertMLP",
    "LlamaPlMoeForCausalLM",
    "LlamaPlMoeMLP",
    "LlamaPlMoeModel",
    "LlamaPlMoeDecoderLayer",
    "LlamaPlMoePreTrainedModel",
]

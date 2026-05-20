# coding=utf-8
# Copyright 2025 the PL-MoE team. All rights reserved.
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
"""PyTorch Phi-4-mini Path-Locked MoE (PL-MoE) model.

序列级路由：根据 control token（/think 或 /no_think）在整条序列上固定使用 expert 0 或 expert 1。
无 gate 网络，路由由控制 token 硬决定。

路由规则：
- expert 0 = no_think 模式
- expert 1 = think 模式
- 默认走 expert 0（no_think），因为初始化阶段共享层来自源模型，与 expert 0 兼容

基座模型：Phi-4-mini (model_type: phi3)
- MLP: gate_up_proj (融合 gate+up) + down_proj，SiLU 激活
- Attention: qkv_proj (融合 Q/K/V) + o_proj，无 bias
- 支持 partial_rotary_factor (RoPE 仅旋转部分 head_dim)
- 有 residual dropout (resid_pdrop)
"""

from typing import Optional

import torch
from torch import nn

from ...cache_utils import Cache, DynamicCache
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_outputs import BaseModelOutputWithPast
from ...processing_utils import Unpack
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...utils import TransformersKwargs
from ...utils.generic import check_model_inputs
from ..phi3.modeling_phi3 import (
    Phi3DecoderLayer,
    Phi3ForCausalLM,
    Phi3MLP,
    Phi3Model,
)
from .configuration_phi4_mini_pl_moe import Phi4MiniPlMoeConfig


class Phi4MiniPlMoeExpertMLP(Phi3MLP):
    """单个 MLP 专家（与 Phi3MLP 结构完全一致），用于 PL-MoE 的双专家容器。

    Phi3MLP 结构：
    - gate_up_proj: Linear(hidden_size, 2 * intermediate_size, bias=False)
    - down_proj: Linear(intermediate_size, hidden_size, bias=False)
    - forward: gate, up = gate_up_proj(x).chunk(2); down_proj(up * act_fn(gate))
    """
    pass


class Phi4MiniPlMoeMLP(nn.Module):
    """Path-Locked 双专家 MLP 容器。

    包含两个独立的 MLP 专家：
    - experts[0]：no_think 专家
    - experts[1]：think 专家
    根据 routing_index 选择使用哪个专家，整条序列一致。
    """

    def __init__(self, config: Phi4MiniPlMoeConfig):
        super().__init__()
        self.experts = nn.ModuleList([Phi4MiniPlMoeExpertMLP(config) for _ in range(2)])

    def forward(self, hidden_states: torch.Tensor, routing_index: int = 0) -> torch.Tensor:
        return self.experts[routing_index](hidden_states)


def _determine_routing_index(input_ids: torch.LongTensor, config: Phi4MiniPlMoeConfig) -> int:
    """根据 input_ids 中的控制 token 确定路由索引。

    路由决策逻辑：
    1. 若 think_token_id 和 no_think_token_id 均未配置 → 返回 default_routing_index
    2. 扫描 input_ids，找到两种控制 token 各自最后出现的位置
    3. 以最后出现的控制 token 为准
    4. 若均未找到 → 返回 default_routing_index

    Args:
        input_ids: 输入 token 序列，shape [batch_size, seq_len]
        config: PL-MoE 配置

    Returns:
        int: 0（no_think / expert 0）或 1（think / expert 1）
    """
    think_id = config.think_token_id
    no_think_id = config.no_think_token_id
    default_idx = getattr(config, "default_routing_index", 0)

    # 未配置任何控制 token → 直接返回默认值
    if think_id is None and no_think_id is None:
        return default_idx

    # 在整个 batch 的 input_ids 中查找各控制 token 最后出现的位置
    last_think_pos = -1
    last_no_think_pos = -1

    if think_id is not None:
        think_mask = (input_ids == think_id)
        if think_mask.any():
            last_think_pos = think_mask.nonzero()[-1, -1].item()

    if no_think_id is not None:
        no_think_mask = (input_ids == no_think_id)
        if no_think_mask.any():
            last_no_think_pos = no_think_mask.nonzero()[-1, -1].item()

    # 两种控制 token 均未在 input_ids 中找到 → 使用默认值
    if last_think_pos == -1 and last_no_think_pos == -1:
        return default_idx

    # 以最后出现的控制 token 为准
    if last_think_pos > last_no_think_pos:
        return 1  # think 模式
    else:
        return 0  # no_think 模式


class Phi4MiniPlMoeDecoderLayer(Phi3DecoderLayer):
    """Decoder 层：Attention 与 Phi3 一致，MLP 替换为 Phi4MiniPlMoeMLP，forward 透传 routing_index。

    Phi3 Attention 特点（共享层，不涉及 expert 分离）：
    - 融合 qkv_proj（Q/K/V 合并为一个 Linear）
    - 无 attention bias
    - 支持 partial_rotary_factor
    - 有 resid_attn_dropout 和 resid_mlp_dropout
    """

    def __init__(self, config: Phi4MiniPlMoeConfig, layer_idx: int):
        super().__init__(config=config, layer_idx=layer_idx)
        # 将父类创建的 Phi3MLP 替换为 PL-MoE 双专家容器
        self.mlp = Phi4MiniPlMoeMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        routing_index: int = 0,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention（共享层，融合 qkv_proj）
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + self.resid_attn_dropout(hidden_states)

        # Fully Connected（路由到指定 expert）
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, routing_index=routing_index)
        hidden_states = residual + self.resid_mlp_dropout(hidden_states)
        return hidden_states


class Phi4MiniPlMoeModel(Phi3Model):
    """骨干模型：在 forward 开头确定 routing_index，并透传给每一层的 MLP。

    routing_index 的来源（按优先级）：
    1. kwargs 中显式传入的 routing_index
    2. 根据 input_ids 中的控制 token 自动计算（训练阶段）
    3. 兜底使用 config.default_routing_index（默认 = 0 / no_think）

    与 Phi3Model 的区别：
    - layers 使用 Phi4MiniPlMoeDecoderLayer（双专家 MLP）
    - forward 增加 routing_index 检测和透传
    - causal mask 使用 Phi3 的简单模式（非 Qwen3 的 causal_mask_mapping 模式）
    """

    def __init__(self, config: Phi4MiniPlMoeConfig):
        super().__init__(config)
        # 将父类创建的 Phi3DecoderLayer 替换为 Phi4MiniPlMoeDecoderLayer
        self.layers = nn.ModuleList(
            [Phi4MiniPlMoeDecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )

    @check_model_inputs
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        # ===== 路由决策 =====
        routing_index = kwargs.pop("routing_index", None)

        if routing_index is None and input_ids is not None:
            routing_index = _determine_routing_index(input_ids, self.config)

        if routing_index is None:
            routing_index = getattr(self.config, "default_routing_index", 0)

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Phi3 的 causal mask 模式：根据是否有 sliding_window 选择 mask 函数
        mask_function = create_causal_mask if self.config.sliding_window is None else create_sliding_window_causal_mask
        causal_mask = mask_function(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                routing_index=routing_index,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class Phi4MiniPlMoeForCausalLM(Phi3ForCausalLM):
    """Causal LM 顶层：使用 Phi4MiniPlMoeModel 作为 backbone。

    重写 prepare_inputs_for_generation 以在 generate 阶段正确传递路由决策。
    """

    def __init__(self, config: Phi4MiniPlMoeConfig):
        super().__init__(config)
        self.model = Phi4MiniPlMoeModel(config)
        self._cached_routing_index = None

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        routing_index=None,
        **kwargs,
    ):
        """重写以在 generate 阶段正确传递路由决策。

        路由优先级：
        1. 用户通过 generate(routing_index=N) 显式传入
        2. 从 input_ids 中的控制 token 自动检测
        3. config.default_routing_index
        """
        if routing_index is not None:
            self._cached_routing_index = routing_index
        else:
            self._cached_routing_index = _determine_routing_index(input_ids, self.config)

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

        model_inputs["routing_index"] = self._cached_routing_index
        return model_inputs


__all__ = [
    "Phi4MiniPlMoeExpertMLP",
    "Phi4MiniPlMoeMLP",
    "Phi4MiniPlMoeDecoderLayer",
    "Phi4MiniPlMoeModel",
    "Phi4MiniPlMoeForCausalLM",
]

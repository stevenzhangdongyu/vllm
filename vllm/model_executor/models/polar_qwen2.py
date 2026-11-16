"""Experimental heterogeneous layer-path executor for Qwen2 on vLLM V1."""

from __future__ import annotations

import copy
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from torch import nn

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed import get_pp_group
from vllm.forward_context import get_forward_context, override_forward_context
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.qwen2 import (
    Qwen2Attention,
    Qwen2ForCausalLM,
    Qwen2Model,
)
from vllm.model_executor.models.utils import PPMissingLayer, maybe_prefix
from vllm.v1.worker.polar_batch_ops import (
    execute_layer_batches,
    request_token_indices,
    slice_flash_attention_metadata,
)


@contextmanager
def _attention_subbatch_context(forward_context, slot_name: str, query_offsets, request_indices):
    if not isinstance(forward_context.attn_metadata, dict):
        raise RuntimeError("PoLar executor requires eager non-microbatched attention metadata")
    full_metadata = forward_context.attn_metadata.get(slot_name)
    if full_metadata is None:
        raise KeyError(f"vLLM did not build attention metadata for KV slot {slot_name}")
    metadata = slice_flash_attention_metadata(
        full_metadata, query_offsets, request_indices
    )
    slot_mapping = forward_context.slot_mapping.get(slot_name)
    if slot_mapping is None:
        raise KeyError(f"vLLM did not build slot mapping for KV slot {slot_name}")
    token_indices = request_token_indices(
        query_offsets, request_indices, slot_mapping.device
    )
    sub_context = copy.copy(forward_context)
    sub_context.attn_metadata = {slot_name: metadata}
    sub_context.slot_mapping = {
        slot_name: slot_mapping.index_select(0, token_indices)
    }
    with override_forward_context(sub_context):
        yield


class PolarQwen2Model(Qwen2Model):
    """Runs paths with independent execution KV or physical-layer overwrite KV."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        if get_pp_group().world_size != 1:
            raise ValueError("Experimental PoLar Qwen2 executor requires pipeline parallelism 1")
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.polar_depth = self.config.num_hidden_layers
        self.polar_cache_mode = os.environ.get(
            "VLLM_POLAR_KV_CACHE_MODE", "execution"
        ).lower()
        if self.polar_cache_mode not in {"layer", "execution"}:
            raise ValueError("VLLM_POLAR_KV_CACHE_MODE must be layer or execution")
        manifest_path = os.environ.get("VLLM_POLAR_PATH_MANIFEST")
        if not manifest_path:
            raise ValueError("Set VLLM_POLAR_PATH_MANIFEST before loading PoLar Qwen2")
        with Path(manifest_path).open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        entries = manifest.get("entries") if isinstance(manifest, dict) else None
        if not isinstance(entries, dict) or not entries:
            raise ValueError("PoLar path manifest must contain non-empty entries")
        paths = [entry.get("path") for entry in entries.values()]
        if any(not isinstance(path, list) or not path for path in paths):
            raise ValueError("Every manifest entry must contain a non-empty path")
        if any(
            type(layer_idx) is not int or not 0 <= layer_idx < self.polar_depth
            for path in paths
            for layer_idx in path
        ):
            raise ValueError("Manifest path contains a layer outside model depth")
        self.polar_max_visits = max(map(len, paths))
        self.polar_extra_kv_slots = nn.ModuleList()
        source_attention = self.layers[0].self_attn
        slot_count = (self.polar_max_visits if self.polar_cache_mode == "execution"
                      else self.polar_depth)
        for slot_idx in range(self.polar_depth, slot_count):
            slot = Qwen2Attention(
                hidden_size=source_attention.hidden_size,
                num_heads=source_attention.total_num_heads,
                num_kv_heads=source_attention.total_num_kv_heads,
                max_position=self.config.max_position_embeddings,
                cache_config=vllm_config.cache_config,
                quant_config=vllm_config.quant_config,
                rope_parameters=self.config.rope_parameters,
                prefix=f"{prefix}.layers.{slot_idx}.self_attn",
                dual_chunk_attention_config=getattr(
                    self.config, "dual_chunk_attention_config", None
                ),
                qk_norm=source_attention.qk_norm,
                rms_norm_eps=self.config.rms_norm_eps,
                attn_type=source_attention.attn.attn_type,
            )
            slot.qkv_proj = source_attention.qkv_proj
            slot.o_proj = source_attention.o_proj
            slot.rotary_emb = source_attention.rotary_emb
            if source_attention.qk_norm:
                slot.q_norm = source_attention.q_norm
                slot.k_norm = source_attention.k_norm
            self.polar_extra_kv_slots.append(slot)

        self.polar_kv_slot_names = [
            layer.self_attn.attn.layer_name for layer in self.layers
        ] + [slot.attn.layer_name for slot in self.polar_extra_kv_slots]
        for layer in self.layers:
            attention = layer.self_attn.attn
            backend_name = attention.attn_backend.get_name()
            if backend_name != "FLASH_ATTN":
                raise ValueError(
                    "Experimental PoLar Qwen2 executor currently requires "
                    f"FLASH_ATTN, got {backend_name}"
                )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
    ):
        if intermediate_tensors is not None or get_pp_group().world_size != 1:
            raise ValueError("PoLar dynamic layer paths require pipeline parallelism 1")
        forward_context = get_forward_context()
        additional = forward_context.additional_kwargs
        paths = additional.get("polar_paths")
        if paths is None:
            return super().forward(input_ids, positions, intermediate_tensors, inputs_embeds)
        if forward_context.cudagraph_runtime_mode != CUDAGraphMode.NONE:
            raise RuntimeError("Start the PoLar path executor with --enforce-eager")
        if forward_context.ubatch_slices is not None:
            raise RuntimeError("PoLar dynamic paths do not support microbatching yet")

        query_offsets = additional["polar_query_start_loc"]
        request_ids = additional["polar_request_ids"]
        if len(paths) != len(request_ids) or len(query_offsets) != len(paths) + 1:
            raise ValueError("PoLar request paths do not match the active token batch")
        for request_idx, path in enumerate(paths):
            if not path or len(path) > self.polar_max_visits:
                raise ValueError(
                    f"request {request_ids[request_idx]} path length must be in [1, "
                    f"{self.polar_max_visits}] for execution-index KV slots"
                )

        hidden_states = (
            inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
        )
        residual = None
        batches = additional["polar_layer_batches"]

        def run_physical_layer(
            layer_idx: int,
            execution_idx: int,
            request_indices: tuple[int, ...],
            selected_hidden: torch.Tensor,
            selected_residual: torch.Tensor | None,
        ):
            physical_layer = self.layers[layer_idx]
            physical_attention = physical_layer.self_attn.attn
            slot_idx = execution_idx if self.polar_cache_mode == "execution" else layer_idx
            slot_name = self.polar_kv_slot_names[slot_idx]
            old_layer_name = physical_attention.layer_name
            token_indices = request_token_indices(
                query_offsets, request_indices, positions.device
            )
            selected_positions = positions.index_select(0, token_indices)
            physical_attention.layer_name = slot_name
            try:
                with _attention_subbatch_context(
                    forward_context, slot_name, query_offsets, request_indices
                ):
                    return physical_layer(
                        selected_positions, selected_hidden, selected_residual
                    )
            finally:
                physical_attention.layer_name = old_layer_name

        hidden_states, residual = execute_layer_batches(
            hidden_states,
            residual,
            query_offsets,
            batches,
            run_physical_layer,
            cache_mode=self.polar_cache_mode,
        )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class PolarQwen2ForCausalLM(Qwen2ForCausalLM):
    """Qwen2ForCausalLM registered as ``PolarQwen2ForCausalLM`` in model configs."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config.get_text_config()
        self.config = config
        self.quant_config = vllm_config.quant_config
        self.model = PolarQwen2Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=self.quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
        **kwargs: Any,
    ):
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

"""Eager token gather/scatter primitives used by the PoLar layer executor."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import torch

from vllm.v1.worker.polar_layer_scheduler import LayerBatch


def request_token_indices(
    query_start_loc: Sequence[int], request_indices: Sequence[int], device: torch.device
) -> torch.Tensor:
    """Create packed-token indices for selected requests in stable request order."""
    if len(query_start_loc) < 2:
        raise ValueError("query_start_loc must contain at least one request")
    indices: list[int] = []
    num_requests = len(query_start_loc) - 1
    for request_idx in request_indices:
        if not 0 <= request_idx < num_requests:
            raise IndexError(f"request index {request_idx} is outside the batch")
        start, end = int(query_start_loc[request_idx]), int(query_start_loc[request_idx + 1])
        if start < 0 or end < start:
            raise ValueError("query_start_loc must be non-negative and non-decreasing")
        indices.extend(range(start, end))
    return torch.tensor(indices, dtype=torch.long, device=device)


def gather_request_tokens(
    packed: torch.Tensor,
    query_start_loc: Sequence[int],
    request_indices: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather selected request token ranges and return data plus source indices."""
    indices = request_token_indices(query_start_loc, request_indices, packed.device)
    if len(query_start_loc) and int(query_start_loc[-1]) > packed.shape[0]:
        raise ValueError("query_start_loc points beyond the packed token tensor")
    return packed.index_select(0, indices), indices


def scatter_request_tokens_(
    packed: torch.Tensor, values: torch.Tensor, token_indices: torch.Tensor
) -> torch.Tensor:
    """Scatter a gathered sub-batch back to its original packed token positions."""
    if values.shape[0] != token_indices.numel():
        raise ValueError("values and token_indices must have matching token counts")
    if values.shape[1:] != packed.shape[1:]:
        raise ValueError("values and packed tensor must have matching trailing shapes")
    packed.index_copy_(0, token_indices, values)
    return packed


def slice_flash_attention_metadata(
    metadata,
    query_start_loc: Sequence[int],
    request_indices: Sequence[int],
):
    """Slice vLLM FlashAttention metadata for a gathered request sub-batch.

    The initial PoLar executor deliberately supports eager text-only full
    attention without cascade, DCP, or CUDA-graph scheduler metadata.
    """
    if metadata.__class__.__name__ != "FlashAttentionMetadata":
        raise TypeError(
            "PoLar dynamic sub-batches currently require FlashAttentionMetadata; "
            f"got {metadata.__class__.__name__}"
        )
    if metadata.use_cascade:
        raise ValueError("cascade attention is not supported by PoLar sub-batches")
    if metadata.max_dcp_context_kv_len not in (None, 0):
        raise ValueError("decode-context parallel attention is not supported")
    if metadata.scheduler_metadata is not None:
        raise ValueError(
            "FlashAttention AOT scheduler metadata cannot be safely sliced; "
            "disable AOT scheduling or use the supported eager backend"
        )
    if metadata.mm_prefix_range_tensor is not None:
        raise ValueError("multimodal prefix attention is not supported")
    if metadata.rswa_prefix_lens is not None:
        raise ValueError("reference sliding-window attention is not supported")

    q_lens = [
        int(query_start_loc[index + 1]) - int(query_start_loc[index])
        for index in request_indices
    ]
    if any(query_len <= 0 for query_len in q_lens):
        raise ValueError("selected requests must have at least one scheduled token")
    token_indices = request_token_indices(
        query_start_loc, request_indices, metadata.slot_mapping.device
    )
    req_indices = torch.tensor(
        request_indices, dtype=torch.long, device=metadata.seq_lens.device
    )
    selected_seq_lens = metadata.seq_lens.index_select(0, req_indices)
    selected_block_table = metadata.block_table.index_select(0, req_indices)
    new_query_start_loc = torch.tensor(
        [0, *torch.tensor(q_lens).cumsum(0).tolist()],
        dtype=metadata.query_start_loc.dtype,
        device=metadata.query_start_loc.device,
    )

    return replace(
        metadata,
        num_actual_tokens=sum(q_lens),
        max_query_len=max(q_lens),
        query_start_loc=new_query_start_loc,
        max_seq_len=int(selected_seq_lens.max().item()),
        seq_lens=selected_seq_lens,
        block_table=selected_block_table,
        slot_mapping=metadata.slot_mapping.index_select(0, token_indices),
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        max_dcp_context_kv_len=None,
        dcp_context_kv_lens=None,
        num_decode_reqs=0,
        num_prefill_reqs=0,
        num_decode_tokens=0,
        num_prefill_tokens=0,
        scheduler_metadata=None,
        prefix_scheduler_metadata=None,
        rswa_prefix_lens=(
            metadata.rswa_prefix_lens.index_select(0, req_indices)
            if metadata.rswa_prefix_lens is not None
            else None
        ),
    )


def execute_layer_batches(
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    query_start_loc: Sequence[int],
    batches: Sequence[LayerBatch],
    layer_forward,
    cache_mode: str = "execution",
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Execute a layer plan over packed request tokens using an eager callback.

    The callback receives physical layer, representative execution index,
    request indices, hidden, and residual. Execution mode splits by position;
    layer mode merges positions but keeps first visits separate for residuals.
    """
    if cache_mode not in {"layer", "execution"}:
        raise ValueError("cache_mode must be layer or execution")
    for batch in batches:
        groups: dict[tuple[int, bool], list[tuple[int, int]]] = {}
        for request_idx, execution_idx in zip(
            batch.request_indices, batch.execution_indices
        ):
            slot_idx = execution_idx if cache_mode == "execution" else batch.layer_idx
            groups.setdefault((slot_idx, execution_idx == 0), []).append(
                (request_idx, execution_idx)
            )

        for members in groups.values():
            execution_idx = members[0][1]
            request_indices = [member[0] for member in members]
            selected_hidden, token_indices = gather_request_tokens(
                hidden_states, query_start_loc, request_indices
            )
            selected_residual = None
            if execution_idx > 0 and residual is None:
                raise ValueError(
                    "a request cannot execute a later path position before its first layer"
                )
            if residual is not None and execution_idx > 0:
                selected_residual, _ = gather_request_tokens(
                    residual, query_start_loc, request_indices
                )
            next_hidden, next_residual = layer_forward(
                batch.layer_idx,
                execution_idx,
                tuple(request_indices),
                selected_hidden,
                selected_residual,
            )
            if next_hidden.shape != selected_hidden.shape:
                raise ValueError("layer callback changed the selected hidden shape")
            if next_residual is None:
                raise ValueError("layer callback must return a residual tensor")
            hidden_states = scatter_request_tokens_(
                hidden_states, next_hidden, token_indices
            )
            if residual is None:
                residual = torch.zeros_like(hidden_states)
            residual = scatter_request_tokens_(residual, next_residual, token_indices)
    return hidden_states, residual

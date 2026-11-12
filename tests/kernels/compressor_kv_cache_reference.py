# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Readable PyTorch reference for the DeepSeek-V4 KV compressor core.

This module deliberately favors clarity over speed. It mirrors the reference
path used by ``test_compressor_kv_cache.py`` from state-cache gather through
RMSNorm and GPT-J-style RoPE. It does not implement the learned projections
that produce ``kv``/``score``, quantization, or paged-cache insertion.
"""

from __future__ import annotations

import torch


def compressor_reference(
    state_cache: torch.Tensor,
    block_table: torch.Tensor,
    positions: torch.Tensor,
    rms_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    compress_ratio: int,
    overlap: int = 0,
    rms_eps: float = 1e-6,
) -> torch.Tensor:
    """Compute compressed, normalized, and rotary-encoded KV rows.

    Args:
        state_cache: Single-request paged state, shaped
            ``[num_pages, page_size, 2 * state_width]``. The first half stores
            KV vectors and the second half stores per-dimension scores.
        block_table: Physical page IDs for this request, shaped ``[1, pages]``.
        positions: Input-token positions at which compressed rows are emitted.
            The compressor emits only positions satisfying
            ``(position + 1) % compress_ratio == 0``.
        rms_weight: RMSNorm scale, shaped ``[head_dim]``.
        cos_sin_cache: RoPE table shaped ``[max_position, rope_dim]``. The first
            half of the last dimension stores cosine values and the second half
            stores sine values for interleaved pairs.
        compress_ratio: Number of new input positions combined into one row.
        overlap: Number of additional previous compression windows to include.
            The tested DeepSeek-V4 configurations use 0 or 1.
        rms_eps: Small constant added to the mean-square value before rsqrt.

    Returns:
        Tensor shaped ``[len(positions), head_dim]`` in the state-cache dtype.
        Positions that are not compression boundaries produce zero rows, as
        production kernels leave their output/cache slots untouched.

    Mathematical reference (per output position ``p`` and feature ``d``):

        S = (1 + overlap) * compress_ratio
        x_i,d = KV state gathered from input position ``p - S + 1 + i``
        a_i,d = score state gathered from the same position
        alpha_i,d = exp(a_i,d) / sum_j exp(a_j,d)
        c_d = sum_i alpha_i,d * x_i,d

    The softmax is over source positions ``i`` independently for every feature
    ``d``. Next, with ``H = head_dim`` and learned scale ``gamma``:

        r = 1 / sqrt((sum_d c_d^2) / H + eps)
        z_d = c_d * r * gamma_d

    Finally, GPT-J RoPE rotates adjacent pairs in the last ``rope_dim`` features.
    This is a transparent correctness oracle, not a production implementation.
    """
    if state_cache.ndim != 3:
        raise ValueError("state_cache must have shape [num_pages, page_size, width]")
    if block_table.ndim != 2 or block_table.shape[0] != 1:
        raise ValueError("This simple reference supports one request: block_table[0]")
    if state_cache.shape[-1] % 2:
        raise ValueError("state_cache final dimension must contain KV and score halves")
    if compress_ratio <= 0 or overlap < 0:
        raise ValueError("compress_ratio must be positive and overlap non-negative")

    input_dtype = state_cache.dtype
    device = state_cache.device
    head_dim = rms_weight.numel()
    rope_dim = cos_sin_cache.shape[-1]
    state_width = state_cache.shape[-1] // 2
    page_size = state_cache.shape[1]

    if state_width < head_dim:
        raise ValueError("state_width must be at least head_dim")
    if rope_dim > head_dim or rope_dim % 2:
        raise ValueError("rope_dim must be even and no greater than head_dim")
    if cos_sin_cache.shape[0] == 0:
        raise ValueError("cos_sin_cache must contain at least one position")

    # Store the calculation in FP32 as the fused kernel/reference does for
    # reductions. Casting back happens once at the end, matching the cache row.
    state_fp32 = state_cache.float()
    rms_weight_fp32 = rms_weight.float()
    rope_cache_fp32 = cos_sin_cache.float()
    result = torch.zeros(
        (positions.numel(), head_dim), dtype=torch.float32, device=device
    )

    # The loop is intentionally explicit: one output row corresponds to one
    # compression boundary. A fused GPU kernel maps this work to a CTA/program.
    for output_idx, position_tensor in enumerate(positions):
        position = int(position_tensor.item())

        # Compressor outputs are aligned to complete compression windows.
        # Non-boundary tokens do not represent a new compressed KV row.
        if (position + 1) % compress_ratio != 0:
            continue

        # With overlap=0, gather the most recent compression window.
        # With overlap=1, gather two adjacent windows (the tested CR=4 case).
        gather_count = (1 + overlap) * compress_ratio
        source_positions = torch.arange(
            position - gather_count + 1,
            position + 1,
            dtype=torch.long,
            device=device,
        )

        # Negative positions at the beginning of a sequence are masked. Clamp
        # only to make page-table indexing legal; their values are excluded
        # below by assigning score=-inf and KV=0.
        valid = source_positions >= 0
        safe_positions = source_positions.clamp(min=0)
        logical_pages = torch.div(safe_positions, page_size, rounding_mode="floor")
        page_ids = block_table[0, logical_pages].long()
        offsets = safe_positions % page_size

        # Resolve logical sequence positions through the paged block table:
        # logical position -> logical page + in-page offset -> physical page.
        rows = state_fp32[page_ids, offsets]

        # Normal layout: [KV[0:H], score[0:H]]. For overlap, the state writer
        # stores the two windows in separate feature halves:
        #   window 0: KV in first half, score in first score half
        #   window 1: KV in second half, score in second score half.
        if overlap:
            if overlap != 1:
                raise ValueError(
                    "The packed overlap layout currently supports overlap=1"
                )
            first = rows[:compress_ratio]
            second = rows[compress_ratio:]
            kv = torch.cat(
                [first[:, :head_dim], second[:, head_dim : 2 * head_dim]], dim=0
            )
            score = torch.cat(
                [
                    first[:, state_width : state_width + head_dim],
                    second[:, state_width + head_dim : state_width + 2 * head_dim],
                ],
                dim=0,
            )
        else:
            kv = rows[:, :head_dim]
            score = rows[:, state_width : state_width + head_dim]

        # Mask invalid left-padding before softmax. Setting KV to zero as well
        # makes the weighted sum safe even if a caller inspects intermediate
        # values while debugging.
        score = score.masked_fill(~valid[:, None], float("-inf"))
        kv = kv.masked_fill(~valid[:, None], 0.0)

        # Equation: alpha[i,d] = softmax_i(score[i,d]).
        # This differs from one scalar weight per token: every feature has its
        # own attention distribution across the source window.
        alpha = torch.softmax(score, dim=0)

        # Equation: c[d] = sum_i alpha[i,d] * kv[i,d].
        compressed = torch.sum(alpha * kv, dim=0)

        # RMSNorm uses the root mean square (not mean-centered variance):
        # r = rsqrt(mean_d(c[d]^2) + eps), z[d] = c[d] * r * gamma[d].
        inv_rms = torch.rsqrt(compressed.square().mean() + rms_eps)
        normalized = compressed * inv_rms * rms_weight_fp32

        # RoPE position follows the implementation: round the source token
        # position down to its compression-window boundary.
        compressed_position = (position // compress_ratio) * compress_ratio
        if compressed_position >= rope_cache_fp32.shape[0]:
            raise ValueError(
                f"RoPE position {compressed_position} exceeds cos_sin_cache"
            )
        cos, sin = rope_cache_fp32[compressed_position].chunk(2)

        # Keep the leading NoPE channels unchanged. Rotate only the final
        # rope_dim channels using GPT-J's interleaved (even, odd) convention:
        #   y_even = x_even*cos - x_odd*sin
        #   y_odd  = x_odd*cos + x_even*sin
        nope_dim = head_dim - rope_dim
        nope = normalized[:nope_dim]
        rope = normalized[nope_dim:]
        even = rope[0::2]
        odd = rope[1::2]
        rotated_rope = torch.stack(
            (even * cos - odd * sin, odd * cos + even * sin), dim=-1
        ).reshape(rope_dim)

        result[output_idx] = torch.cat((nope, rotated_rope))

    return result.to(input_dtype)

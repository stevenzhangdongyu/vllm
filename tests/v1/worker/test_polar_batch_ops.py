import pytest
import torch

from vllm.v1.worker.polar_batch_ops import (
    execute_layer_batches,
    gather_request_tokens,
    scatter_request_tokens_,
)


def test_gather_scatter_variable_query_lengths():
    packed = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    selected, indices = gather_request_tokens(packed, (0, 2, 3, 6), (0, 2))
    assert indices.tolist() == [0, 1, 3, 4, 5]
    assert torch.equal(selected, packed[indices])

    updated = selected + 100
    scatter_request_tokens_(packed, updated, indices)
    assert torch.equal(packed[indices], updated)
    assert torch.equal(packed[2], torch.tensor([4.0, 5.0]))


def test_invalid_request_index_is_rejected():
    with pytest.raises(IndexError):
        gather_request_tokens(torch.zeros(1, 2), (0, 1), (1,))


def test_out_of_range_offsets_are_rejected():
    with pytest.raises(ValueError, match="beyond"):
        gather_request_tokens(torch.zeros(1, 2), (0, 2), (0,))


def test_layer_plan_executes_and_scatters_in_path_order():
    from vllm.v1.worker.polar_layer_scheduler import dynamic_layer_schedule

    hidden = torch.zeros((2, 1))
    calls = []

    def layer_forward(layer_idx, execution_idx, request_indices, selected_hidden, residual):
        calls.append((layer_idx, execution_idx, selected_hidden.shape[0]))
        if residual is None:
            residual = selected_hidden.clone()
        return selected_hidden + layer_idx + 1, residual

    result, residual = execute_layer_batches(
        hidden,
        None,
        (0, 1, 2),
        dynamic_layer_schedule([[0, 1], [0, 0]], depth=2),
        layer_forward,
    )
    assert calls == [(0, 0, 2), (0, 1, 1), (1, 1, 1)]
    assert torch.equal(result[:, 0], torch.tensor([3.0, 2.0]))
    assert residual is not None


@pytest.mark.parametrize("cache_mode,expected_calls", [("execution", 5), ("layer", 4)])
def test_cache_modes_preserve_loop_order_and_residuals(cache_mode, expected_calls):
    """Mixed-depth visits merge only when their KV and residual semantics agree."""
    from vllm.v1.worker.polar_layer_scheduler import dynamic_layer_schedule

    calls = []
    cache = {}

    def forward(layer, position, requests, hidden, residual):
        calls.append((layer, position, requests))
        slot = layer if cache_mode == "layer" else position
        for index, request in enumerate(requests):
            cache[request, slot] = hidden[index].clone()
        residual = hidden.clone() if residual is None else residual + 1
        return hidden + layer + 1, residual

    result, residual = execute_layer_batches(
        torch.zeros(2, 1), None, (0, 1, 2),
        dynamic_layer_schedule([[0, 1, 0, 2], [0, 2]], 3),
        forward, cache_mode=cache_mode,
    )
    assert len(calls) == expected_calls
    assert torch.equal(result[:, 0], torch.tensor([7., 4.]))
    assert torch.equal(residual[:, 0], torch.tensor([3., 1.]))
    if cache_mode == "layer":
        assert cache[0, 0].item() == 3  # Repeat replaces the original value.
        assert calls[-1][2] == (0, 1)
    else:
        assert cache[0, 0].item() == 0
        assert cache[0, 2].item() == 3

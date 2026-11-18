"""Deterministic layer-batch planning for heterogeneous PoLar paths."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

from vllm.v1.worker.polar_profile import polar_profile


@dataclass(frozen=True)
class LayerBatch:
    layer_idx: int
    request_indices: tuple[int, ...]
    execution_indices: tuple[int, ...]


def validate_paths(paths: Sequence[Sequence[int]], depth: int) -> list[tuple[int, ...]]:
    if not paths:
        raise ValueError("at least one path is required")
    result = []
    for request_idx, path in enumerate(paths):
        if not path:
            raise ValueError(f"path {request_idx} must not be empty")
        if any(type(layer) is not int or not 0 <= layer < depth for layer in path):
            raise ValueError(f"path {request_idx} contains an invalid physical layer")
        result.append(tuple(path))
    return result


@polar_profile("polar/plan/serial")
def serial_schedule(paths: Sequence[Sequence[int]], depth: int) -> list[LayerBatch]:
    normalized = validate_paths(paths, depth)
    return [
        LayerBatch(layer, (request_idx,), (execution_idx,))
        for request_idx, path in enumerate(normalized)
        for execution_idx, layer in enumerate(path)
    ]


@polar_profile("polar/plan/identical_path")
def identical_path_schedule(
    paths: Sequence[Sequence[int]], depth: int
) -> list[LayerBatch]:
    normalized = validate_paths(paths, depth)
    groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for request_idx, path in enumerate(normalized):
        groups[path].append(request_idx)
    return [
        LayerBatch(layer, tuple(request_indices), (execution_idx,) * len(request_indices))
        for path, request_indices in groups.items()
        for execution_idx, layer in enumerate(path)
    ]


@polar_profile("polar/plan/dynamic_layer")
def dynamic_layer_schedule(
    paths: Sequence[Sequence[int]], depth: int
) -> list[LayerBatch]:
    """Batch requests ready at the same physical layer, then advance them."""
    normalized = validate_paths(paths, depth)
    positions = [0] * len(normalized)
    batches = []
    while True:
        ready: dict[int, list[int]] = defaultdict(list)
        for request_idx, path in enumerate(normalized):
            if positions[request_idx] < len(path):
                ready[path[positions[request_idx]]].append(request_idx)
        if not ready:
            break
        layer = min(ready, key=lambda candidate: (-len(ready[candidate]), candidate))
        request_indices = tuple(ready[layer])
        execution_indices = tuple(positions[index] for index in request_indices)
        batches.append(LayerBatch(layer, request_indices, execution_indices))
        for request_idx in request_indices:
            positions[request_idx] += 1
    return batches

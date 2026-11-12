import pytest

from vllm.v1.worker.polar_layer_scheduler import (
    LayerBatch,
    dynamic_layer_schedule,
    identical_path_schedule,
    serial_schedule,
)


def test_serial_is_one_request_per_layer_call():
    batches = serial_schedule([[0, 1], [0]], depth=2)
    assert [(batch.layer_idx, batch.request_indices) for batch in batches] == [
        (0, (0,)),
        (1, (0,)),
        (0, (1,)),
    ]


def test_identical_paths_batch_full_matches():
    batches = identical_path_schedule([[0, 1], [0, 1], [1, 0]], depth=2)
    assert batches[0].request_indices == (0, 1)
    assert batches[0].execution_indices == (0, 0)


def test_dynamic_schedule_batches_current_layer_and_tracks_visits():
    batches = dynamic_layer_schedule([[0, 1], [0, 0]], depth=2)
    assert batches == [
        LayerBatch(0, (0, 1), (0, 0)),
        LayerBatch(0, (1,), (1,)),
        LayerBatch(1, (0,), (1,)),
    ]


def test_invalid_layer_is_rejected():
    with pytest.raises(ValueError, match="invalid physical layer"):
        dynamic_layer_schedule([[2]], depth=2)

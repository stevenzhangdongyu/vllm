import json

import pytest

from vllm.v1.worker.polar_path_context import (
    build_request_context,
    load_path_manifest,
    prompt_token_ids_key,
)


class RequestState:
    def __init__(self, prompt_token_ids):
        self.prompt_token_ids = prompt_token_ids


def test_load_manifest_and_resolve_batch_context(tmp_path):
    token_ids = [1, 7, 3]
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "entries": {
                    prompt_token_ids_key(token_ids): {
                        "request_id": "request-a",
                        "path": [0, 2, 2],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    entries = load_path_manifest(manifest_file)
    offsets = [0, 3]
    cache = {}
    context = build_request_context(
        ["request-a"],
        {"request-a": RequestState(token_ids)},
        offsets,
        entries,
        cache,
    )
    assert context == {
        "polar_request_ids": ("request-a",),
        "polar_paths": ((0, 2, 2),),
        "polar_query_start_loc": offsets[:2],
    }


def test_unknown_prompt_is_rejected():
    with pytest.raises(KeyError, match="request-a"):
        build_request_context(
            ["request-a"],
            {"request-a": RequestState([8, 9])},
            [0, 2],
            {},
            {},
        )


def test_empty_batch_has_no_context():
    assert build_request_context([], {}, [0], {}, {}) is None


def test_unscheduled_requests_are_removed_without_changing_packed_offsets():
    token_ids = [4, 5]
    entries = {
        prompt_token_ids_key(token_ids): {"request_id": "b", "path": [1, 2]}
    }
    context = build_request_context(
        ["a", "b", "c"],
        {"a": RequestState([8]), "b": RequestState(token_ids), "c": RequestState([9])},
        [0, 0, 2, 2],
        entries,
        {},
    )
    assert context["polar_request_ids"] == ("b",)
    assert context["polar_paths"] == ((1, 2),)
    assert context["polar_query_start_loc"] == (0, 2)

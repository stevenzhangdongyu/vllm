"""Per-request PoLar path routing context for the V1 GPU model runner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def prompt_token_ids_key(token_ids: list[int]) -> str:
    payload = ",".join(str(int(token_id)) for token_id in token_ids).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def load_path_manifest(filename: str | Path) -> dict[str, Any]:
    with Path(filename).open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict):
        raise ValueError(f"Invalid PoLar path manifest: {filename}")
    entries = manifest.get("entries")
    if manifest.get("schema_version") != 1 or not isinstance(entries, dict):
        raise ValueError(f"Invalid PoLar path manifest: {filename}")
    return entries


def build_request_context(
    request_ids: list[str],
    requests: dict[str, Any],
    query_start_loc: Any,
    manifest_entries: dict[str, Any],
    path_cache: dict[str, tuple[tuple[int, ...], tuple[int, ...]]],
) -> dict[str, Any] | None:
    """Resolve current batch paths and provide token offsets to the model."""
    if not request_ids:
        return None

    active_ids = set(request_ids)
    for request_id in tuple(path_cache):
        if request_id not in active_ids:
            del path_cache[request_id]

    offsets = tuple(
        int(offset) for offset in query_start_loc[: len(request_ids) + 1]
    )
    scheduled_indices = [
        index
        for index in range(len(request_ids))
        if offsets[index + 1] > offsets[index]
    ]
    paths: list[tuple[int, ...]] = []
    scheduled_request_ids: list[str] = []
    for request_idx in scheduled_indices:
        request_id = request_ids[request_idx]
        request = requests.get(request_id)
        if request is None or request.prompt_token_ids is None:
            raise ValueError(
                f"PoLar dynamic routing requires prompt_token_ids for {request_id}"
            )
        token_ids = tuple(request.prompt_token_ids)
        cached = path_cache.get(request_id)
        if cached is not None and cached[0] == token_ids:
            path = cached[1]
        else:
            entry = manifest_entries.get(prompt_token_ids_key(list(token_ids)))
            if entry is None:
                raise KeyError(
                    f"No PoLar path manifest entry for request {request_id}"
                )
            raw_path = entry.get("path")
            if (
                not isinstance(raw_path, list)
                or not raw_path
                or any(type(layer) is not int or layer < 0 for layer in raw_path)
            ):
                raise ValueError(f"Invalid path for request {request_id}")
            path = tuple(raw_path)
            path_cache[request_id] = (token_ids, path)
        paths.append(path)
        scheduled_request_ids.append(request_id)

    return {
        "polar_request_ids": tuple(scheduled_request_ids),
        "polar_paths": tuple(paths),
        "polar_query_start_loc": tuple(
            offsets[index] for index in scheduled_indices
        ) + ((offsets[scheduled_indices[-1] + 1],) if scheduled_indices else ()),
    }

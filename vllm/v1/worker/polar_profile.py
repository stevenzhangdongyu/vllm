"""Optional CPU/CUDA attribution ranges for the experimental PoLar executor."""

import os
from contextlib import nullcontext
from functools import wraps

ENABLED = os.environ.get("VLLM_POLAR_PROFILE") == "1"


def polar_range(name):
    if not ENABLED:
        return nullcontext()
    from torch.profiler import record_function

    return record_function(name)


def polar_profile(name):
    def decorate(function):
        if not ENABLED:
            return function

        @wraps(function)
        def wrapped(*args, **kwargs):
            with polar_range(name):
                return function(*args, **kwargs)

        return wrapped

    return decorate

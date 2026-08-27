from __future__ import annotations

import gc
import sys
from typing import Any


# dghs-imgutils 0.19.0 keeps ONNX Runtime sessions in thread-safe LRU
# caches. The interactive workflow intentionally runs several pipeline steps
# in one process, so those sessions must be dropped before a host GPU lease
# checks whether the device is idle.
_IMGUTILS_SESSION_CACHES = (
    ("imgutils.tagging.wd14", "_get_wd14_model"),
    ("imgutils.metrics.ccip", "_open_feat_model"),
    ("imgutils.metrics.ccip", "_open_metric_model"),
)


def release_inprocess_gpu_resources() -> tuple[str, ...]:
    """Release loaded optional-backend GPU caches without importing them."""

    cleared: list[str] = []
    for module_name, function_name in _IMGUTILS_SESSION_CACHES:
        module = sys.modules.get(module_name)
        function: Any = getattr(module, function_name, None) if module is not None else None
        cache_clear = getattr(function, "cache_clear", None)
        if callable(cache_clear):
            cache_clear()
            cleared.append(f"{module_name}.{function_name}")

    # Clearing an LRU drops its references immediately, but a collection here
    # also finalizes any cycles holding an ONNX InferenceSession or tensor.
    gc.collect()

    torch = sys.modules.get("torch")
    cuda = getattr(torch, "cuda", None) if torch is not None else None
    is_initialized = getattr(cuda, "is_initialized", None)
    empty_cache = getattr(cuda, "empty_cache", None)
    if callable(is_initialized) and is_initialized() and callable(empty_cache):
        empty_cache()
        cleared.append("torch.cuda")

    return tuple(cleared)

from __future__ import annotations

import sys
from types import SimpleNamespace

from pipeline import gpu_resources


class _CachedSessionFactory:
    def __init__(self) -> None:
        self.clear_calls = 0

    def cache_clear(self) -> None:
        self.clear_calls += 1


def test_release_inprocess_gpu_resources_clears_loaded_sessions(monkeypatch) -> None:
    wd14 = _CachedSessionFactory()
    ccip_features = _CachedSessionFactory()
    ccip_metrics = _CachedSessionFactory()
    torch_empty_calls: list[bool] = []
    collect_calls: list[bool] = []

    monkeypatch.setitem(
        sys.modules,
        "imgutils.tagging.wd14",
        SimpleNamespace(_get_wd14_model=wd14),
    )
    monkeypatch.setitem(
        sys.modules,
        "imgutils.metrics.ccip",
        SimpleNamespace(
            _open_feat_model=ccip_features,
            _open_metric_model=ccip_metrics,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                is_initialized=lambda: True,
                empty_cache=lambda: torch_empty_calls.append(True),
            )
        ),
    )
    monkeypatch.setattr(gpu_resources.gc, "collect", lambda: collect_calls.append(True))

    cleared = gpu_resources.release_inprocess_gpu_resources()

    assert cleared == (
        "imgutils.tagging.wd14._get_wd14_model",
        "imgutils.metrics.ccip._open_feat_model",
        "imgutils.metrics.ccip._open_metric_model",
        "torch.cuda",
    )
    assert wd14.clear_calls == 1
    assert ccip_features.clear_calls == 1
    assert ccip_metrics.clear_calls == 1
    assert torch_empty_calls == [True]
    assert collect_calls == [True]


def test_release_inprocess_gpu_resources_does_not_import_optional_backends(monkeypatch) -> None:
    for module_name, _ in gpu_resources._IMGUTILS_SESSION_CACHES:
        monkeypatch.delitem(sys.modules, module_name, raising=False)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    collect_calls: list[bool] = []
    monkeypatch.setattr(gpu_resources.gc, "collect", lambda: collect_calls.append(True))

    assert gpu_resources.release_inprocess_gpu_resources() == ()
    assert collect_calls == [True]

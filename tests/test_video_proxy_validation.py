from __future__ import annotations

import pytest

from pipeline.models import PipelineError
from pipeline.video_source import VideoProxy


def test_custom_proxy_rejects_missing_host() -> None:
    with pytest.raises(PipelineError):
        VideoProxy(mode="custom", url="http://")


def test_custom_proxy_rejects_non_numeric_port_as_user_error() -> None:
    with pytest.raises((PipelineError, ValueError)):
        VideoProxy(mode="custom", url="http://127.0.0.1:not-a-port")

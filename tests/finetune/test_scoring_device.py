from __future__ import annotations

import pytest
from settings_factory import SettingsFactory

from physverify.app import finetune as campaign
from physverify.core.errors import UsageError


class _Probe:
    def __init__(self, available: bool) -> None:
        self._available = available

    def available(self) -> bool:
        return self._available


@pytest.fixture
def no_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(campaign, "TorchCudaProbe", lambda: _Probe(False))


@pytest.mark.usefixtures("no_cuda")
def test_scoring_accepts_cpu_without_a_cuda_device(make_settings: SettingsFactory) -> None:
    campaign.require_scoring_device(make_settings("runtime.device=cpu"))


@pytest.mark.usefixtures("no_cuda")
@pytest.mark.parametrize("device", ["cuda", "cuda:1"])
def test_scoring_on_cuda_needs_a_present_device(
    make_settings: SettingsFactory, device: str
) -> None:
    with pytest.raises(UsageError, match="needs a CUDA device"):
        campaign.require_scoring_device(make_settings(f"runtime.device={device}"))


def test_scoring_on_cuda_passes_with_a_device(
    make_settings: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(campaign, "TorchCudaProbe", lambda: _Probe(True))
    campaign.require_scoring_device(make_settings("runtime.device=cuda"))


@pytest.mark.usefixtures("no_cuda")
def test_generation_still_needs_cuda(make_settings: SettingsFactory) -> None:
    with pytest.raises(UsageError, match="--device cuda"):
        campaign.require_cuda(make_settings("runtime.device=cpu"))

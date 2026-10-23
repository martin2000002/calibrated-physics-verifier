"""Calibration fixtures: the committed settings, and calibration_cases.py on the import path."""

import sys
from pathlib import Path

import pytest
from settings_factory import SettingsFactory

from physverify.config.models import Settings

# The calibration tests share the builders of calibration_cases.py; importlib mode does not add
# this directory to sys.path by itself.
HERE = str(Path(__file__).resolve().parent)
if HERE not in sys.path:
    sys.path.insert(0, HERE)


@pytest.fixture
def settings(make_settings: SettingsFactory) -> Settings:
    return make_settings()

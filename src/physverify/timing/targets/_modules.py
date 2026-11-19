"""Whether an optional library is installed, without importing it."""

import importlib.util


def module_found(name: str) -> bool:
    return importlib.util.find_spec(name) is not None

"""Scene family contract: samples a ``SceneSpec`` from a seeded generator."""

from __future__ import annotations

from typing import ClassVar, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel

from physverify.config.plugin_context import PluginKind
from physverify.schema.scene import SceneSpec


@runtime_checkable
class SceneFamily(Protocol):
    """Registered by ``name``; instantiated without arguments."""

    name: ClassVar[str]
    Params: ClassVar[type[BaseModel]]

    def sample(self, rng: np.random.Generator, params: BaseModel) -> SceneSpec: ...


FAMILIES = PluginKind[type[SceneFamily]]("families")

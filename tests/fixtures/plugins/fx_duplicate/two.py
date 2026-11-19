from __future__ import annotations


class Second:
    name = "same"

    def run(self) -> str:
        return "second"


PLUGINS = (Second,)

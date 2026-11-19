from __future__ import annotations


class First:
    name = "same"

    def run(self) -> str:
        return "first"


PLUGINS = (First,)

from __future__ import annotations


class Beta:
    name = "beta"

    def run(self) -> str:
        return "beta"


class BetaTwo:
    name = "beta_two"

    def run(self) -> str:
        return "beta_two"


PLUGINS = (Beta, BetaTwo)

from __future__ import annotations

import argparse


class ModuleCLIUsageError(ValueError):
    """Argument error that module CLIs can serialize as JSON."""


class ModuleArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ModuleCLIUsageError(message)


__all__ = ["ModuleArgumentParser", "ModuleCLIUsageError"]

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ModuleResult:
    """Small, stable result envelope used by customer-facing module tests."""

    module: str
    ok: bool
    output: Any = None
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    error_type: str | None = None

    @classmethod
    def success(
        cls,
        module: str,
        output: Any = None,
        *,
        metrics: dict[str, Any] | None = None,
        artifacts: dict[str, str] | None = None,
    ) -> ModuleResult:
        return cls(
            module=module,
            ok=True,
            output=output,
            metrics=dict(metrics or {}),
            artifacts=dict(artifacts or {}),
        )

    @classmethod
    def failure(cls, module: str, error: Exception | str) -> ModuleResult:
        return cls(
            module=module,
            ok=False,
            error=str(error),
            error_type=type(error).__name__ if isinstance(error, Exception) else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = ["ModuleResult"]

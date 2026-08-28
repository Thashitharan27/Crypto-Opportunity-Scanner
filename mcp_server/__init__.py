"""Read-only MCP access to Crypto Strategy Lab backtest reports."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .server import BacktestReports

__all__ = ["BacktestReports"]


def __getattr__(name: str) -> Any:
    """Load public server types lazily so ``python -m mcp_server.server`` is clean."""
    if name == "BacktestReports":
        from .server import BacktestReports

        return BacktestReports
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

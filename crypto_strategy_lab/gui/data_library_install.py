"""Final GUI composition for the compact Data Library workspace."""
from __future__ import annotations

from .data_library_workspace import DataLibraryWorkspace


def apply_data_library_workspace(window) -> None:
    """Replace duplicate coverage views while preserving readiness plumbing."""
    if getattr(window, "data_library_workspace", None) is not None:
        return
    required = (
        "_page",
        "_replace_page",
        "library_table",
        "coverage",
        "quality",
        "quality_table",
        "resolution",
        "refresh_data_library",
    )
    if not all(hasattr(window, name) for name in required):
        return

    workspace = DataLibraryWorkspace(window)
    page = window._page("Data Library", workspace)
    window._replace_page(7, page)
    window.data_library_workspace = workspace
    window.refresh_data_library()

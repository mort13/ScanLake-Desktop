"""Every module imports (catches syntax and import errors in code no other test touches)."""

from __future__ import annotations

import importlib
import importlib.util
import pkgutil

import pytest

PACKAGES = [
    "scanlake_core", "scanlake_inference", "scanlake_io", "scanlake_desktop",
    "scanlake_models", "scanlake_training",
]  # fmt: skip
NEEDS_TORCH = ("scanlake_models", "scanlake_training", "scanlake_inference.torch_backend")


def _modules() -> list[str]:
    names = []
    for package in PACKAGES:
        spec = importlib.util.find_spec(package)
        assert spec is not None and spec.submodule_search_locations is not None
        names.append(package)
        for info in pkgutil.walk_packages(spec.submodule_search_locations, f"{package}."):
            names.append(info.name)
    return names


@pytest.mark.parametrize("name", _modules())
def test_module_imports(name: str) -> None:
    if name.startswith(NEEDS_TORCH) and importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch not installed")
    importlib.import_module(name)

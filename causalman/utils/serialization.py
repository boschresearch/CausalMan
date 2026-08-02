"""Compatibility helpers for CausalMan's bundled pickle files."""

import pickle
from typing import Any, BinaryIO


_LEGACY_MODULES = {
    "fcm": "causalman.fcm",
    "node": "causalman.node",
}
_LEGACY_PACKAGE_PREFIXES = {
    "FCM_Definitions": "causalman.FCM_Definitions",
    "line_structure": "causalman.line_structure",
    "utils": "causalman.utils",
}


def _current_module_name(module: str) -> str:
    """Map module names stored before the package rename to their current names."""
    if module in _LEGACY_MODULES:
        return _LEGACY_MODULES[module]

    for legacy_prefix, current_prefix in _LEGACY_PACKAGE_PREFIXES.items():
        if module == legacy_prefix or module.startswith(f"{legacy_prefix}."):
            return f"{current_prefix}{module[len(legacy_prefix):]}"

    return module


class _CausalManUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        return super().find_class(_current_module_name(module), name)


def load_pickle(file_obj: BinaryIO) -> Any:
    """Load a trusted CausalMan pickle, including files made before packaging."""
    return _CausalManUnpickler(file_obj).load()

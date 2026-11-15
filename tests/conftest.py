"""Test configuration: register lightweight dummy stubs only when GPU/git-only packages are absent."""
from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _ensure_stub_module(fullname: str, attrs: dict[str, Any] | None = None) -> None:
    parts = fullname.split(".")
    try:
        exists = importlib.util.find_spec(fullname) is not None
    except ModuleNotFoundError:
        exists = False
    if exists or fullname in sys.modules:
        return
    for i in range(1, len(parts) + 1):
        subname = ".".join(parts[:i])
        if subname not in sys.modules:
            is_pkg = i < len(parts)
            mod = types.ModuleType(subname)
            mod.__spec__ = importlib.machinery.ModuleSpec(subname, loader=None, is_package=is_pkg)
            if is_pkg:
                mod.__path__ = []
            sys.modules[subname] = mod
            if i > 1:
                parent = sys.modules[".".join(parts[: i - 1])]
                setattr(parent, parts[i - 1], mod)
    if attrs:
        target = sys.modules[fullname]
        for key, value in attrs.items():
            setattr(target, key, value)


_ensure_stub_module("tqdm", {"tqdm": lambda it=None, **kw: it})
_ensure_stub_module("geocalib", {"GeoCalib": object})
_ensure_stub_module("pi3.models.pi3", {"Pi3": object})
_ensure_stub_module("pi3.utils.geometry", {"homogenize_points": lambda x: x, "se3_inverse": lambda x: x})
_ensure_stub_module("dinov3.hub.backbones", {"dinov3_vitb16": object})
_ensure_stub_module("huggingface_hub", {"hf_hub_download": object, "snapshot_download": object})
_ensure_stub_module(
    "peft",
    {
        "LoraConfig": object,
        "PeftModel": object,
        "get_peft_model": object,
        "get_peft_model_state_dict": object,
        "set_peft_model_state_dict": object,
    },
)
_ensure_stub_module("safetensors.torch", {"load_file": object})
_ensure_stub_module("viser", {"ViserServer": object})
_ensure_stub_module("viser.transforms", {"SO3": object})

from __future__ import annotations

import importlib.util
import platform
import sys


MODULES = [
    "torch",
    "transformers",
    "openvino",
    "huggingface_hub",
    "safetensors",
    "optimum",
    "nncf",
]


def module_version(name: str) -> str:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return "not installed"
    module = __import__(name)
    return getattr(module, "__version__", "installed")


def main() -> int:
    print(f"python: {sys.version.split()[0]}")
    print(f"platform: {platform.platform()}")
    for name in MODULES:
        print(f"{name}: {module_version(name)}")

    if sys.version_info[:2] != (3, 12):
        print("warning: upstream README was tested on Python 3.12; this environment may still work for tooling.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

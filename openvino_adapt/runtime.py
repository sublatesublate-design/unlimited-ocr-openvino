from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from .artifacts import ArtifactSet


@dataclass(frozen=True)
class ComponentDevices:
    embed: str
    vision: str
    prefill: str
    decode: str

    def as_dict(self) -> dict[str, str]:
        return {
            "embed": self.embed,
            "vision": self.vision,
            "prefill": self.prefill,
            "decode": self.decode,
        }


@dataclass(frozen=True)
class CompiledArtifacts:
    embed: object
    vision: object
    prefill: object
    decode: object
    devices: ComponentDevices


def add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="CPU", help="Default OpenVINO device for all components.")
    parser.add_argument("--embed-device", default="", help="OpenVINO device for token embeddings. Defaults to --device.")
    parser.add_argument("--vision-device", default="", help="OpenVINO device for vision tokens. Defaults to --device.")
    parser.add_argument("--prefill-device", default="", help="OpenVINO device for decoder prefill. Defaults to --device.")
    parser.add_argument("--decode-device", default="", help="OpenVINO device for one-token decoder. Defaults to --device.")
    parser.add_argument("--cache-dir", default="", help="Optional OpenVINO model/kernel cache directory.")


def component_devices(args: argparse.Namespace) -> ComponentDevices:
    default = args.device
    return ComponentDevices(
        embed=args.embed_device or default,
        vision=args.vision_device or default,
        prefill=args.prefill_device or default,
        decode=args.decode_device or default,
    )


def make_core(cache_dir: str | Path = ""):
    import openvino as ov

    core = ov.Core()
    if cache_dir:
        path = Path(cache_dir)
        path.mkdir(parents=True, exist_ok=True)
        core.set_property({"CACHE_DIR": str(path)})
    return core


def compile_artifacts(core, artifacts: ArtifactSet, devices: ComponentDevices) -> CompiledArtifacts:
    return CompiledArtifacts(
        embed=core.compile_model(artifacts.embed_tokens, devices.embed),
        vision=core.compile_model(artifacts.vision_tokens, devices.vision),
        prefill=core.compile_model(artifacts.prefill_model, devices.prefill),
        decode=core.compile_model(artifacts.decode_model, devices.decode),
        devices=devices,
    )

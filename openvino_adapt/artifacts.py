from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re


@dataclass(frozen=True)
class ArtifactSet:
    base_model_dir: Path
    prefill_model: Path
    decode_model: Path
    tokenizer_dir: Path

    @classmethod
    def from_paths(
        cls,
        base_model_dir: str | Path = "openvino_models/unlimited_ocr",
        prefill_model: str | Path = "openvino_models/unlimited_ocr_kv_dense_prefill277/decoder_prefill_kv.xml",
        decode_model: str | Path = "openvino_models/unlimited_ocr_kv_dense/decoder_decode_one.xml",
        tokenizer_dir: str | Path = "models/Unlimited-OCR",
    ) -> "ArtifactSet":
        return cls(
            base_model_dir=Path(base_model_dir),
            prefill_model=Path(prefill_model),
            decode_model=Path(decode_model),
            tokenizer_dir=Path(tokenizer_dir),
        )

    @property
    def embed_tokens(self) -> Path:
        return self.base_model_dir / "embed_tokens.xml"

    @property
    def vision_tokens(self) -> Path:
        return self.base_model_dir / "vision_tokens.xml"

    def require_files(self) -> None:
        missing = [
            path
            for path in [
                self.embed_tokens,
                self.vision_tokens,
                self.prefill_model,
                self.decode_model,
                self.tokenizer_dir / "tokenizer.json",
            ]
            if not path.exists()
        ]
        if missing:
            joined = "\n".join(f"  - {path}" for path in missing)
            raise FileNotFoundError(f"Missing OpenVINO adaptation artifacts:\n{joined}")


def static_dim(model_path: Path, input_index: int, dim_index: int) -> int | None:
    import openvino as ov

    model_path = Path(model_path)
    model = ov.Core().read_model(model_path)
    dim = model.inputs[input_index].partial_shape[dim_index]
    return int(dim.get_length()) if dim.is_static else None


def prefill_seq_len(model_path: Path) -> int | None:
    model_path = Path(model_path)
    meta = model_metadata(model_path)
    if "seq_len" in meta:
        return int(meta["seq_len"])
    static = static_dim(model_path, 0, 1)
    if static is not None:
        return static
    match = re.search(r"prefill(\d+)", str(model_path))
    return int(match.group(1)) if match else None


def decode_prior_len(model_path: Path) -> int | None:
    model_path = Path(model_path)
    meta = model_metadata(model_path)
    if "past_len" in meta:
        return int(meta["past_len"])
    static = static_dim(model_path, 3, 2)
    if static is not None:
        return static
    match = re.search(r"past(\d+)", str(model_path))
    return int(match.group(1)) if match else None


def model_metadata(model_path: Path) -> dict:
    model_path = Path(model_path)
    meta_path = model_path.with_suffix(".json")
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def validate_prompt_and_decode_lengths(
    artifacts: ArtifactSet,
    prompt_len: int,
    ring_window: int,
) -> None:
    expected_prefill = prefill_seq_len(artifacts.prefill_model)
    if expected_prefill is not None and expected_prefill != prompt_len:
        raise ValueError(
            f"Prefill graph expects sequence length {expected_prefill}, but this prompt needs {prompt_len}. "
            "Export a matching graph, for example:\n"
            f"python -m openvino_adapt.export_openvino --model models/Unlimited-OCR "
            f"--component decoder_prefill_kv --seq-len {prompt_len} --allow-weight-download "
            f"--output-dir openvino_models/unlimited_ocr_kv_dense_prefill{prompt_len}"
        )

    expected_prior = decode_prior_len(artifacts.decode_model)
    needed_prior = prompt_len + ring_window - 1
    if expected_prior is not None and expected_prior != needed_prior:
        raise ValueError(
            f"Decode graph expects prior KV length {expected_prior}, but prompt_len={prompt_len} "
            f"and ring_window={ring_window} need {needed_prior}. Export a matching graph, for example:\n"
            f"python -m openvino_adapt.export_openvino --model models/Unlimited-OCR "
            f"--component decoder_decode_one --past-len {needed_prior} --allow-weight-download "
            f"--output-dir openvino_models/unlimited_ocr_kv_dense_past{needed_prior}"
        )

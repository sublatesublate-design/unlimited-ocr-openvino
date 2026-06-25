from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume uploading OpenVINO artifact files to Hugging Face.")
    parser.add_argument("--repo-id", default="sublatesublate-design/unlimited-ocr-openvino")
    parser.add_argument("--repo-type", default="model")
    parser.add_argument("--local-dir", default="openvino_models/sparse_decode_past677_mixed_fp4")
    parser.add_argument("--only", choices=("all", "bins", "small-bins", "large-bins"), default="bins")
    parser.add_argument("--max-files", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--sleep", type=float, default=10.0)
    parser.add_argument("--state-json", default="outputs_openvino_hf_upload/hf_resume_state.json")
    return parser.parse_args()


def should_include(path: Path, root: Path, mode: str) -> bool:
    if mode == "all":
        return path.is_file()
    if path.suffix.lower() != ".bin":
        return False
    if mode == "bins":
        return True
    large = path.stat().st_size > 20 * 1024 * 1024
    return large if mode == "large-bins" else not large


def main() -> int:
    args = parse_args()
    root = Path(args.local_dir)
    api = HfApi()
    remote_files = set(api.list_repo_files(args.repo_id, repo_type=args.repo_type))

    local_files = [
        path
        for path in sorted(root.rglob("*"))
        if should_include(path, root, args.only)
    ]
    missing = [(path, path.relative_to(root).as_posix()) for path in local_files if path.relative_to(root).as_posix() not in remote_files]
    if args.max_files:
        missing = missing[: args.max_files]

    state_path = Path(args.state_json)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "repo_id": args.repo_id,
        "local_dir": str(root),
        "mode": args.only,
        "started_missing": len(missing),
        "uploaded": [],
        "failed": [],
    }
    print(json.dumps({"missing": len(missing), "mode": args.only}, ensure_ascii=False))

    for index, (local_path, remote_path) in enumerate(missing, 1):
        ok = False
        last_error = ""
        for attempt in range(1, args.retries + 1):
            try:
                print(f"[{index}/{len(missing)}] upload {remote_path} attempt {attempt}")
                api.upload_file(
                    path_or_fileobj=str(local_path),
                    path_in_repo=remote_path,
                    repo_id=args.repo_id,
                    repo_type=args.repo_type,
                    commit_message=f"Upload {remote_path}",
                )
                ok = True
                break
            except Exception as exc:  # noqa: BLE001
                last_error = repr(exc)
                print(f"  failed: {last_error}")
                time.sleep(args.sleep * attempt)
        if ok:
            state["uploaded"].append(remote_path)
        else:
            state["failed"].append({"path": remote_path, "error": last_error})
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"uploaded": len(state["uploaded"]), "failed": len(state["failed"])}, ensure_ascii=False))
    return 0 if not state["failed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

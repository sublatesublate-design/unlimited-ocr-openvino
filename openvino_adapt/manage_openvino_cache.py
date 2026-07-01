from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect or prune a controlled OpenVINO cache directory.")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--max-gb", type=float, default=0.0, help="Prune oldest entries until cache is below this size. 0 only reports.")
    parser.add_argument("--clear", action="store_true", help="Delete all contents inside --cache-dir.")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def entry_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def cache_entries(cache_dir: Path) -> list[dict]:
    entries = []
    if not cache_dir.exists():
        return entries
    for path in cache_dir.iterdir():
        try:
            stat = path.stat()
            size = entry_size(path)
        except FileNotFoundError:
            continue
        entries.append(
            {
                "path": path,
                "name": path.name,
                "size_bytes": size,
                "mtime": stat.st_mtime,
                "is_dir": path.is_dir(),
            }
        )
    return entries


def clear_contents(cache_dir: Path) -> None:
    if not cache_dir.exists():
        return
    for path in cache_dir.iterdir():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def prune(cache_dir: Path, max_bytes: int) -> list[dict]:
    removed = []
    skipped = []
    entries = cache_entries(cache_dir)
    total = sum(item["size_bytes"] for item in entries)
    for item in sorted(entries, key=lambda entry: entry["mtime"]):
        if total <= max_bytes:
            break
        path = item["path"]
        try:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
        except PermissionError as exc:
            skipped.append(
                {
                    "name": item["name"],
                    "size_bytes": item["size_bytes"],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        total -= item["size_bytes"]
        removed.append({key: value for key, value in item.items() if key != "path"})
    return removed, skipped


def main() -> int:
    args = parse_args()
    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.clear:
        clear_contents(cache_dir)

    removed = []
    skipped = []
    if args.max_gb > 0:
        removed, skipped = prune(cache_dir, int(args.max_gb * 1024**3))

    entries = cache_entries(cache_dir)
    total_bytes = sum(item["size_bytes"] for item in entries)
    payload = {
        "cache_dir": str(cache_dir),
        "total_bytes": total_bytes,
        "total_gb": round(total_bytes / 1024**3, 4),
        "entries": len(entries),
        "removed": removed,
        "skipped": skipped,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"cache_dir: {payload['cache_dir']}")
        print(f"entries: {payload['entries']}")
        print(f"total_gb: {payload['total_gb']}")
        if removed:
            print(f"removed_entries: {len(removed)}")
        if skipped:
            print(f"skipped_locked_entries: {len(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

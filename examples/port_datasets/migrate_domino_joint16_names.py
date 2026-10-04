#!/usr/bin/env python
"""Migrate generated DOMINO joint16 datasets back to original camera/robot names."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pyarrow.parquet as pq


RENAMES = {
    "observation.images.head": "observation.images.head_camera",
    "observation.images.left_wrist": "observation.images.left_camera",
    "observation.images.right_wrist": "observation.images.right_camera",
}


def rename_text(value: str) -> str:
    for old, new in RENAMES.items():
        value = value.replace(old, new)
    return value


def rename_json_keys(value):
    if isinstance(value, dict):
        return {rename_text(key): rename_json_keys(item) for key, item in value.items()}
    if isinstance(value, list):
        return [rename_json_keys(item) for item in value]
    return value


def migrate_json(path: Path, *, set_robot_type: bool = False) -> None:
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    payload = rename_json_keys(payload)
    if set_robot_type:
        payload["robot_type"] = "franka-panda"
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4 if path.name == "info.json" else 2, ensure_ascii=False)
        file.write("\n")
    os.replace(temporary, path)


def migrate_parquet(path: Path) -> None:
    table = pq.read_table(path)
    names = [rename_text(name) for name in table.column_names]
    if names == table.column_names:
        return
    table = table.rename_columns(names)
    temporary = path.with_suffix(".tmp.parquet")
    pq.write_table(table, temporary)
    os.replace(temporary, path)


def migrate(root: Path) -> None:
    info = root / "meta" / "info.json"
    if not info.is_file():
        raise FileNotFoundError(info)
    migrate_json(info, set_robot_type=True)
    stats = root / "meta" / "stats.json"
    if stats.is_file():
        migrate_json(stats)
    for path in root.rglob("*.parquet"):
        migrate_parquet(path)

    videos = root / "videos"
    for old, new in RENAMES.items():
        old_path = videos / old
        new_path = videos / new
        if old_path.exists():
            if new_path.exists():
                raise FileExistsError(f"Both old and new video directories exist: {old_path}, {new_path}")
            old_path.rename(new_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    args = parser.parse_args()
    for root in args.roots:
        migrate(root.resolve())
        print(f"Migrated {root}")


if __name__ == "__main__":
    main()

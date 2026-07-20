#!/usr/bin/env python

import argparse
import json
import shutil
from pathlib import Path

import jsonlines
import pandas as pd
from datasets import Dataset

from lerobot.datasets import aggregate_stats
from lerobot.datasets.io_utils import (
    cast_stats_to_numpy,
    get_parquet_file_size_in_mb,
    get_parquet_num_frames,
    load_info,
    write_episodes,
    write_info,
    write_stats,
    write_tasks,
)
from lerobot.datasets.utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DATA_FILE_SIZE_IN_MB,
    DEFAULT_DATA_PATH,
    DEFAULT_VIDEO_FILE_SIZE_IN_MB,
    DEFAULT_VIDEO_PATH,
    LEGACY_EPISODES_PATH,
    LEGACY_EPISODES_STATS_PATH,
    LEGACY_TASKS_PATH,
    update_chunk_file_indices,
)
from lerobot.datasets.video_utils import concatenate_video_files
from lerobot.utils.utils import flatten_dict


def load_jsonlines(path: Path) -> list[dict]:
    with jsonlines.open(path, "r") as reader:
        return list(reader)


def sort_file_index(path: Path) -> int:
    return int(path.stem.split("-")[-1])


def convert_info(root: Path, new_root: Path) -> None:
    info = load_info(root)
    info["codebase_version"] = "v3.0"
    info.pop("total_chunks", None)
    info.pop("total_videos", None)
    info["data_files_size_in_mb"] = DEFAULT_DATA_FILE_SIZE_IN_MB
    info["video_files_size_in_mb"] = DEFAULT_VIDEO_FILE_SIZE_IN_MB
    info["data_path"] = DEFAULT_DATA_PATH
    info["video_path"] = DEFAULT_VIDEO_PATH if info.get("video_path") is not None else None
    info["fps"] = int(info["fps"])
    for feature in info["features"].values():
        if feature["dtype"] != "video":
            feature["fps"] = info["fps"]
    write_info(info, new_root)


def convert_tasks(root: Path, new_root: Path) -> None:
    tasks = load_jsonlines(root / LEGACY_TASKS_PATH)
    tasks = sorted(tasks, key=lambda item: item["task_index"])
    df = pd.DataFrame(
        {"task_index": [item["task_index"] for item in tasks]},
        index=pd.Index([item["task"] for item in tasks], name="task"),
    )
    write_tasks(df, new_root)


def convert_data(root: Path, new_root: Path) -> list[dict]:
    data_paths = sorted((root / "data").glob("chunk-*/*.parquet"), key=sort_file_index)
    chunk_idx = 0
    file_idx = 0
    size_mb = 0.0
    num_frames = 0
    paths_to_cat = []
    episodes_metadata = []

    def flush(paths: list[Path], out_chunk_idx: int, out_file_idx: int) -> None:
        if not paths:
            return
        df = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
        out_path = new_root / DEFAULT_DATA_PATH.format(chunk_index=out_chunk_idx, file_index=out_file_idx)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)

    for episode_index, data_path in enumerate(data_paths):
        ep_size_mb = get_parquet_file_size_in_mb(data_path)
        ep_num_frames = get_parquet_num_frames(data_path)
        if size_mb + ep_size_mb >= DEFAULT_DATA_FILE_SIZE_IN_MB and paths_to_cat:
            flush(paths_to_cat, chunk_idx, file_idx)
            chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, DEFAULT_CHUNK_SIZE)
            size_mb = 0.0
            paths_to_cat = []

        episodes_metadata.append(
            {
                "episode_index": episode_index,
                "data/chunk_index": chunk_idx,
                "data/file_index": file_idx,
                "dataset_from_index": num_frames,
                "dataset_to_index": num_frames + ep_num_frames,
            }
        )
        paths_to_cat.append(data_path)
        size_mb += ep_size_mb
        num_frames += ep_num_frames

    flush(paths_to_cat, chunk_idx, file_idx)
    return episodes_metadata


def get_video_keys(root: Path) -> list[str]:
    info = load_info(root)
    return sorted(key for key, ft in info["features"].items() if ft["dtype"] == "video")


def convert_videos(root: Path, new_root: Path, episodes: list[dict]) -> list[dict]:
    fps = int(load_info(root)["fps"])
    all_camera_metadata = []
    for video_key in get_video_keys(root):
        paths = sorted((root / "videos" / video_key).glob("chunk-*/*.mp4"), key=sort_file_index)
        if len(paths) != len(episodes):
            raise ValueError(f"{video_key} has {len(paths)} videos but metadata has {len(episodes)} episodes")

        chunk_idx = 0
        file_idx = 0
        size_mb = 0.0
        timestamp = 0.0
        paths_to_cat = []
        camera_metadata = []

        def flush(paths_to_flush: list[Path], out_chunk_idx: int, out_file_idx: int) -> None:
            if not paths_to_flush:
                return
            out_path = new_root / DEFAULT_VIDEO_PATH.format(
                video_key=video_key, chunk_index=out_chunk_idx, file_index=out_file_idx
            )
            concatenate_video_files(paths_to_flush, out_path)

        for episode_index, (episode, video_path) in enumerate(zip(episodes, paths, strict=True)):
            ep_size_mb = video_path.stat().st_size / (1024 * 1024)
            if size_mb + ep_size_mb >= DEFAULT_VIDEO_FILE_SIZE_IN_MB and paths_to_cat:
                flush(paths_to_cat, chunk_idx, file_idx)
                chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, DEFAULT_CHUNK_SIZE)
                size_mb = 0.0
                paths_to_cat = []

            duration = episode["length"] / fps
            camera_metadata.append(
                {
                    "episode_index": episode_index,
                    f"videos/{video_key}/chunk_index": chunk_idx,
                    f"videos/{video_key}/file_index": file_idx,
                    f"videos/{video_key}/from_timestamp": timestamp,
                    f"videos/{video_key}/to_timestamp": timestamp + duration,
                }
            )
            timestamp += duration
            size_mb += ep_size_mb
            paths_to_cat.append(video_path)

        flush(paths_to_cat, chunk_idx, file_idx)
        all_camera_metadata.append(camera_metadata)

    episodes_video_metadata = []
    for episode_index in range(len(episodes)):
        merged = {"episode_index": episode_index}
        for camera_metadata in all_camera_metadata:
            item = camera_metadata[episode_index]
            if item["episode_index"] != episode_index:
                raise ValueError("Video metadata episode ordering mismatch")
            merged.update(item)
        episodes_video_metadata.append(merged)
    return episodes_video_metadata


def convert_episodes_metadata(
    root: Path,
    new_root: Path,
    data_metadata: list[dict],
    video_metadata: list[dict],
) -> None:
    episodes = sorted(load_jsonlines(root / LEGACY_EPISODES_PATH), key=lambda item: item["episode_index"])
    stats_rows = sorted(load_jsonlines(root / LEGACY_EPISODES_STATS_PATH), key=lambda item: item["episode_index"])
    stats_by_episode = {
        item["episode_index"]: cast_stats_to_numpy(item["stats"])
        for item in stats_rows
    }

    rows = []
    for episode_index, episode in enumerate(episodes):
        if episode["episode_index"] != episode_index:
            raise ValueError(f"Expected episode_index={episode_index}, got {episode['episode_index']}")
        if data_metadata[episode_index]["episode_index"] != episode_index:
            raise ValueError("Data metadata episode ordering mismatch")
        if video_metadata[episode_index]["episode_index"] != episode_index:
            raise ValueError("Video metadata episode ordering mismatch")

        row = {**data_metadata[episode_index], **video_metadata[episode_index], **episode}
        row.update(flatten_dict({"stats": stats_by_episode[episode_index]}))
        row["meta/episodes/chunk_index"] = 0
        row["meta/episodes/file_index"] = 0
        rows.append(row)

    write_episodes(Dataset.from_list(rows), new_root)
    write_stats(aggregate_stats(list(stats_by_episode.values())), new_root)


def validate_new_dataset(new_root: Path) -> None:
    required = [
        new_root / "meta" / "info.json",
        new_root / "meta" / "stats.json",
        new_root / "meta" / "tasks.parquet",
        new_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing converted files: {missing}")


def convert(root: Path) -> None:
    old_root = root.parent / f"{root.name}_old"
    new_root = root.parent / f"{root.name}_v30"
    if old_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing backup: {old_root}")
    if new_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing temporary output: {new_root}")

    convert_info(root, new_root)
    convert_tasks(root, new_root)
    data_metadata = convert_data(root, new_root)
    episodes = sorted(load_jsonlines(root / LEGACY_EPISODES_PATH), key=lambda item: item["episode_index"])
    video_metadata = convert_videos(root, new_root, episodes)
    convert_episodes_metadata(root, new_root, data_metadata, video_metadata)
    validate_new_dataset(new_root)

    shutil.move(str(root), str(old_root))
    shutil.move(str(new_root), str(root))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    convert(args.root)


if __name__ == "__main__":
    main()

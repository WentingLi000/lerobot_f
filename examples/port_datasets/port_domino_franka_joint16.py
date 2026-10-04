#!/usr/bin/env python

"""Convert DOMINO dual-Franka demonstrations to a LeRobot v3 joint-space dataset.

The output contract is deliberately explicit:

    observation.state[t] = joint_action/vector[t]
    action[t]            = joint_action/vector[t + 1]

Both vectors contain ``left_arm(7), left_gripper(1), right_arm(7),
right_gripper(1)``.  End-effector poses and quaternion conversions are not used.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import random
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from lerobot.datasets import LeRobotDataset


CAMERAS = {
    "head_camera": "head_camera",
    "left_camera": "left_camera",
    "right_camera": "right_camera",
}
JOINT_NAMES = [
    *(f"left_joint_{index}" for index in range(1, 8)),
    "left_gripper",
    *(f"right_joint_{index}" for index in range(1, 8)),
    "right_gripper",
]


@dataclass(frozen=True)
class Episode:
    hdf5_path: Path
    instruction_path: Path
    source_name: str
    episode_number: int


@dataclass(frozen=True)
class ManifestEntry:
    output_episode_index: int
    source_name: str
    source_episode_number: int
    source_hdf5: str
    condition: str
    level: str
    active_arm: str
    raw_frames: int
    converted_frames: int
    instruction_split: str
    instruction_index: int
    instruction: str


def natural_key(path: Path) -> list[str | int]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(path))]


def discover_episodes(raw_root: Path) -> list[Episode]:
    episodes = []
    for hdf5_path in sorted(raw_root.glob("**/data/episode*.hdf5"), key=natural_key):
        match = re.fullmatch(r"episode(\d+)\.hdf5", hdf5_path.name)
        if match is None:
            continue
        dataset_dir = hdf5_path.parent.parent
        instruction_path = dataset_dir / "instructions" / f"{hdf5_path.stem}.json"
        if not instruction_path.is_file():
            raise FileNotFoundError(f"Missing instruction file: {instruction_path}")
        episodes.append(
            Episode(hdf5_path, instruction_path, dataset_dir.name, int(match.group(1)))
        )
    return episodes


def decode_rgb(value: object) -> np.ndarray:
    encoded = bytes(value).rstrip(b"\x00")
    if not encoded:
        raise ValueError("Encountered an empty encoded RGB frame")
    with Image.open(io.BytesIO(encoded)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def load_instruction(path: Path, split: str, seed: int) -> tuple[str, int]:
    with path.open(encoding="utf-8") as file:
        candidates = json.load(file).get(split)
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"{path} has no non-empty {split!r} instruction list")
    index = random.Random(seed).randrange(len(candidates))
    instruction = candidates[index]
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(f"Invalid instruction at index {index} in {path}")
    return instruction.strip(), index


def split_source_name(source_name: str) -> tuple[str, str]:
    condition_match = re.search(r"_(clean|randomized)_", source_name)
    level_match = re.search(r"_(level[123])$", source_name)
    return (
        condition_match.group(1) if condition_match else "unknown",
        level_match.group(1) if level_match else "unknown",
    )


def validate_joint_vector(file: h5py.File, path: Path) -> np.ndarray:
    required = [
        "joint_action/vector",
        "joint_action/left_arm",
        "joint_action/left_gripper",
        "joint_action/right_arm",
        "joint_action/right_gripper",
    ]
    missing = [key for key in required if key not in file]
    if missing:
        raise KeyError(f"Missing {missing} in {path}")
    vector = np.asarray(file["joint_action/vector"][:], dtype=np.float32)
    expected = np.concatenate(
        [
            file["joint_action/left_arm"][:],
            file["joint_action/left_gripper"][:][:, None],
            file["joint_action/right_arm"][:],
            file["joint_action/right_gripper"][:][:, None],
        ],
        axis=1,
    ).astype(np.float32)
    if vector.ndim != 2 or vector.shape[1] != 16:
        raise ValueError(f"Expected joint vector (T, 16), got {vector.shape} in {path}")
    if not np.array_equal(vector, expected):
        error = float(np.max(np.abs(vector - expected)))
        raise ValueError(f"joint_action/vector concatenation mismatch ({error=}) in {path}")
    if len(vector) < 2:
        raise ValueError(f"Episode needs at least two frames: {path}")
    return vector


def inspect_features(first_episode: Episode) -> dict[str, dict]:
    with h5py.File(first_episode.hdf5_path, "r") as file:
        vector = validate_joint_vector(file, first_episode.hdf5_path)
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (16,),
                "names": JOINT_NAMES,
            },
            "action": {"dtype": "float32", "shape": (16,), "names": JOINT_NAMES},
        }
        for output_name, raw_name in CAMERAS.items():
            key = f"observation/{raw_name}/rgb"
            if key not in file or len(file[key]) != len(vector):
                raise ValueError(f"Missing or misaligned camera {key} in {first_episode.hdf5_path}")
            height, width, channels = decode_rgb(file[key][0]).shape
            features[f"observation.images.{output_name}"] = {
                "dtype": "video",
                "shape": (height, width, channels),
                "names": ["height", "width", "channel"],
            }
    return features


def detect_active_arm(vector: np.ndarray, tolerance: float = 1e-5) -> str:
    left_motion = float(np.max(np.abs(vector[:, :7] - vector[0, :7])))
    right_motion = float(np.max(np.abs(vector[:, 8:15] - vector[0, 8:15])))
    left_active = left_motion > tolerance
    right_active = right_motion > tolerance
    if left_active and not right_active:
        return "left"
    if right_active and not left_active:
        return "right"
    if left_active and right_active:
        return "both"
    return "none"


def convert_episode(
    dataset: LeRobotDataset,
    episode: Episode,
    output_episode_index: int,
    instruction_split: str,
    instruction_seed: int,
) -> ManifestEntry:
    instruction, instruction_index = load_instruction(
        episode.instruction_path,
        instruction_split,
        instruction_seed + output_episode_index,
    )
    with h5py.File(episode.hdf5_path, "r") as file:
        vector = validate_joint_vector(file, episode.hdf5_path)
        for raw_name in CAMERAS.values():
            key = f"observation/{raw_name}/rgb"
            if key not in file or len(file[key]) != len(vector):
                raise ValueError(f"Missing or misaligned camera {key} in {episode.hdf5_path}")

        for index in range(len(vector) - 1):
            frame = {
                "observation.state": vector[index],
                "action": vector[index + 1],
                "task": instruction,
            }
            for output_name, raw_name in CAMERAS.items():
                frame[f"observation.images.{output_name}"] = decode_rgb(
                    file[f"observation/{raw_name}/rgb"][index]
                )
            dataset.add_frame(frame)

    dataset.save_episode()
    condition, level = split_source_name(episode.source_name)
    return ManifestEntry(
        output_episode_index=output_episode_index,
        source_name=episode.source_name,
        source_episode_number=episode.episode_number,
        source_hdf5=str(episode.hdf5_path),
        condition=condition,
        level=level,
        active_arm=detect_active_arm(vector),
        raw_frames=len(vector),
        converted_frames=len(vector) - 1,
        instruction_split=instruction_split,
        instruction_index=instruction_index,
        instruction=instruction,
    )


def write_manifest(output_root: Path, entries: list[ManifestEntry], fps: int) -> None:
    metadata = {
        "format": "DOMINO dual-Franka joint16 to LeRobot v3",
        "fps": fps,
        "control_space": "joint",
        "action_type": "absolute_joint_position",
        "temporal_alignment": "observation.state[t] -> action[t+1]",
        "joint_order": JOINT_NAMES,
        "n_obs_steps_note": "PI0.5 uses one current observation; all sequential frames remain stored",
    }
    with (output_root / "conversion_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)
        file.write("\n")
    with (output_root / "conversion_manifest.jsonl").open("w", encoding="utf-8") as file:
        for entry in entries:
            file.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")


def validate_metadata(output_root: Path, expected_episodes: int, expected_frames: int) -> None:
    with (output_root / "meta" / "info.json").open(encoding="utf-8") as file:
        info = json.load(file)
    actual = (int(info["total_episodes"]), int(info["total_frames"]), int(info["fps"]))
    expected = (expected_episodes, expected_frames, 30)
    if actual != expected:
        raise RuntimeError(f"Output metadata mismatch: {actual=} {expected=}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/domino_adjust_bottle_franka_joint16")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=30, choices=(30,))
    parser.add_argument("--instruction-split", choices=("seen", "unseen"), default="seen")
    parser.add_argument("--instruction-seed", type=int, default=42)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raw_root = args.raw_root.resolve()
    output_root = args.output_root.resolve()
    episodes = discover_episodes(raw_root)
    if not episodes:
        raise FileNotFoundError(f"No episodes found below {raw_root}")
    if args.max_episodes is not None:
        if args.max_episodes <= 0:
            raise ValueError("--max-episodes must be positive")
        episodes = episodes[: args.max_episodes]
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output_root}; pass --overwrite")
        logging.warning("Removing requested output directory: %s", output_root)
        shutil.rmtree(output_root)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=output_root,
        robot_type="franka-panda",
        fps=args.fps,
        features=inspect_features(episodes[0]),
        use_videos=True,
        vcodec="h264",
        streaming_encoding=True,
        encoder_queue_maxsize=120,
    )
    manifest = []
    try:
        for output_index, episode in enumerate(episodes):
            entry = convert_episode(
                dataset,
                episode,
                output_index,
                args.instruction_split,
                args.instruction_seed,
            )
            manifest.append(entry)
            logging.info(
                "[%d/%d] %s episode%d: %d frames, active=%s",
                output_index + 1,
                len(episodes),
                episode.source_name,
                episode.episode_number,
                entry.converted_frames,
                entry.active_arm,
            )
    finally:
        dataset.finalize()

    total_frames = sum(entry.converted_frames for entry in manifest)
    write_manifest(output_root, manifest, args.fps)
    validate_metadata(output_root, len(episodes), total_frames)
    logging.info("Complete: %d episodes, %d frames at %s", len(episodes), total_frames, output_root)
    if args.push_to_hub:
        dataset.push_to_hub(tags=["DOMINO", "RoboTwin", "LeRobot", "Franka", "joint-space"])


if __name__ == "__main__":
    main()

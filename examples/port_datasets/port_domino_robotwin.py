#!/usr/bin/env python

"""Convert downloaded DOMINO/RoboTwin HDF5 demonstrations to LeRobot v3.

This converter targets the DOMINO archive layout::

    <dataset>/data/episodeN.hdf5
    <dataset>/instructions/episodeN.json

For PI0.5 compatibility it uses the selected arm's end-effector pose as a
6-dimensional state (XYZ + XYZ Euler angles) and pose plus gripper as a
7-dimensional action. JPEG-compressed camera frames are decoded directly from
the HDF5 file; the separately supplied preview MP4 is not used.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from lerobot.datasets import LeRobotDataset


CAMERA_NAMES = ("head_camera", "left_camera", "right_camera")
STATE_NAMES = ("x", "y", "z", "roll", "pitch", "yaw")
ACTION_NAMES = (*STATE_NAMES, "gripper")


@dataclass(frozen=True)
class Episode:
    hdf5_path: Path
    instruction_path: Path
    source_name: str
    episode_number: int


def _natural_key(path: Path) -> list[str | int]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(path))]


def discover_episodes(raw_root: Path) -> list[Episode]:
    episodes: list[Episode] = []
    for hdf5_path in sorted(raw_root.glob("**/data/episode*.hdf5"), key=_natural_key):
        match = re.fullmatch(r"episode(\d+)\.hdf5", hdf5_path.name)
        if match is None:
            continue
        dataset_dir = hdf5_path.parent.parent
        instruction_path = dataset_dir / "instructions" / f"{hdf5_path.stem}.json"
        if not instruction_path.is_file():
            raise FileNotFoundError(f"Missing instruction file for {hdf5_path}: {instruction_path}")
        episodes.append(
            Episode(
                hdf5_path=hdf5_path,
                instruction_path=instruction_path,
                source_name=dataset_dir.name,
                episode_number=int(match.group(1)),
            )
        )
    return episodes


def decode_rgb(value: object) -> np.ndarray:
    """Decode one fixed-width HDF5 byte string containing a JPEG/PNG frame."""
    encoded = bytes(value).rstrip(b"\x00")
    if not encoded:
        raise ValueError("Encountered an empty encoded RGB frame")
    with Image.open(io.BytesIO(encoded)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def quaternion_to_euler_xyz(quaternion: np.ndarray, order: str) -> np.ndarray:
    """Convert a quaternion to intrinsic XYZ Euler angles in radians."""
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape != (4,):
        raise ValueError(f"Expected a quaternion with shape (4,), got {quaternion.shape}")
    if order == "wxyz":
        w, x, y, z = quaternion
    elif order == "xyzw":
        x, y, z, w = quaternion
    else:
        raise ValueError(f"Unsupported quaternion order: {order}")

    norm = np.linalg.norm([w, x, y, z])
    if norm < 1e-8:
        raise ValueError("Encountered a zero-length quaternion")
    w, x, y, z = np.asarray([w, x, y, z]) / norm

    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sin_pitch)
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.asarray([roll, pitch, yaw], dtype=np.float32)


def pose_to_xyz_euler(pose: np.ndarray, quaternion_order: str) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (7,):
        raise ValueError(f"Expected end pose with shape (7,), got {pose.shape}")
    return np.concatenate(
        [pose[:3].astype(np.float32), quaternion_to_euler_xyz(pose[3:], quaternion_order)]
    )


def choose_instruction(path: Path, split: str, selection_index: int) -> str:
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    candidates = payload.get(split)
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"{path} does not contain a non-empty instruction list named {split!r}")
    instruction = candidates[selection_index % len(candidates)]
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(f"Invalid instruction in {path}")
    return instruction.strip()


def inspect_schema(first_episode: Episode, arm: str, quaternion_order: str) -> tuple[dict, int]:
    with h5py.File(first_episode.hdf5_path, "r") as file:
        pose_key = f"endpose/{arm}_endpose"
        gripper_key = f"endpose/{arm}_gripper"
        if pose_key not in file or gripper_key not in file:
            raise KeyError(f"Missing {pose_key!r} or {gripper_key!r} in {first_episode.hdf5_path}")
        frame_count = len(file[pose_key])
        pose_to_xyz_euler(file[pose_key][0], quaternion_order)

        features: dict[str, dict] = {
            "observation.state": {
                "dtype": "float32",
                "shape": (6,),
                "names": list(STATE_NAMES),
            },
            "action": {
                "dtype": "float32",
                "shape": (7,),
                "names": list(ACTION_NAMES),
            },
        }
        for camera_name in CAMERA_NAMES:
            rgb_key = f"observation/{camera_name}/rgb"
            if rgb_key not in file:
                raise KeyError(f"Missing {rgb_key!r} in {first_episode.hdf5_path}")
            if len(file[rgb_key]) != frame_count:
                raise ValueError(f"Frame-count mismatch for {rgb_key} in {first_episode.hdf5_path}")
            image = decode_rgb(file[rgb_key][0])
            height, width, channels = image.shape
            features[f"observation.images.{camera_name}"] = {
                "dtype": "video",
                "shape": (height, width, channels),
                "names": ["height", "width", "channel"],
            }
    return features, frame_count


def convert_episode(
    output: LeRobotDataset,
    episode: Episode,
    output_episode_index: int,
    arm: str,
    quaternion_order: str,
    instruction_split: str,
) -> int:
    instruction = choose_instruction(episode.instruction_path, instruction_split, output_episode_index)
    pose_key = f"endpose/{arm}_endpose"
    gripper_key = f"endpose/{arm}_gripper"

    with h5py.File(episode.hdf5_path, "r") as file:
        poses = file[pose_key]
        grippers = file[gripper_key]
        frame_count = len(poses)
        if len(grippers) != frame_count:
            raise ValueError(f"Pose/gripper length mismatch in {episode.hdf5_path}")
        for camera_name in CAMERA_NAMES:
            if len(file[f"observation/{camera_name}/rgb"]) != frame_count:
                raise ValueError(f"Camera length mismatch for {camera_name} in {episode.hdf5_path}")

        for frame_index in range(frame_count):
            state = pose_to_xyz_euler(poses[frame_index], quaternion_order)
            gripper = np.asarray([grippers[frame_index]], dtype=np.float32)
            frame = {
                "observation.state": state,
                "action": np.concatenate([state, gripper]).astype(np.float32),
                "task": instruction,
            }
            for camera_name in CAMERA_NAMES:
                frame[f"observation.images.{camera_name}"] = decode_rgb(
                    file[f"observation/{camera_name}/rgb"][frame_index]
                )
            output.add_frame(frame)

    output.save_episode()
    return frame_count


def validate_output(repo_id: str, output_root: Path, expected_episodes: int) -> None:
    """Validate finalized metadata without initializing the HF datasets cache."""
    info_path = output_root / "meta" / "info.json"
    if not info_path.is_file():
        raise RuntimeError(f"Converted dataset is missing {info_path}")
    with info_path.open(encoding="utf-8") as file:
        info = json.load(file)
    total_episodes = int(info.get("total_episodes", 0))
    total_frames = int(info.get("total_frames", 0))
    if total_episodes != expected_episodes:
        raise RuntimeError(
            f"Expected {expected_episodes} output episodes, found {total_episodes}"
        )
    if total_frames <= 0:
        raise RuntimeError("Converted dataset contains no frames")
    logging.info(
        "Validated %s: %d episodes, %d frames at %s",
        repo_id,
        total_episodes,
        total_frames,
        output_root,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True, help="Directory containing the six archives")
    parser.add_argument("--repo-id", default="local/domino_adjust_bottle_franka_mixed")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--arm", choices=("left", "right"), default="right")
    parser.add_argument(
        "--quaternion-order",
        choices=("wxyz", "xyzw"),
        default="wxyz",
        help="Order of the final four values in endpose (default follows the existing RoboTwin converter)",
    )
    parser.add_argument("--instruction-split", choices=("seen", "unseen"), default="seen")
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Convert only the first N episodes; use 1 for a smoke test",
    )
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
        raise FileNotFoundError(f"No **/data/episode*.hdf5 files found under {raw_root}")
    if args.max_episodes is not None:
        if args.max_episodes <= 0:
            raise ValueError("--max-episodes must be positive")
        episodes = episodes[: args.max_episodes]

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output_root}. Pass --overwrite to replace it.")
        logging.warning("Removing existing output directory: %s", output_root)
        shutil.rmtree(output_root)

    features, first_episode_frames = inspect_schema(
        episodes[0], args.arm, args.quaternion_order
    )
    logging.info(
        "Discovered %d episodes; first episode has %d frames; writing to %s",
        len(episodes),
        first_episode_frames,
        output_root,
    )
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=output_root,
        robot_type="franka-panda",
        fps=args.fps,
        features=features,
        use_videos=True,
        vcodec="h264",
        streaming_encoding=True,
        encoder_queue_maxsize=120,
    )

    total_frames = 0
    try:
        for output_episode_index, episode in enumerate(episodes):
            frames = convert_episode(
                dataset,
                episode,
                output_episode_index,
                args.arm,
                args.quaternion_order,
                args.instruction_split,
            )
            total_frames += frames
            logging.info(
                "[%d/%d] %s/episode%d: %d frames",
                output_episode_index + 1,
                len(episodes),
                episode.source_name,
                episode.episode_number,
                frames,
            )
    finally:
        dataset.finalize()

    validate_output(args.repo_id, output_root, len(episodes))
    logging.info("Conversion complete: %d frames", total_frames)
    if args.push_to_hub:
        dataset.push_to_hub(tags=["DOMINO", "RoboTwin", "LeRobot", "dynamic-manipulation"])


if __name__ == "__main__":
    main()

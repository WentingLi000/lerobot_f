#!/usr/bin/env python

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import matplotlib

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from analysis.pi05_attention_logger import PI05AttentionLogger
from analysis.pi05_feature_scores_to_txt import (
    build_layer_tables,
    build_overall_table,
    load_rows,
    write_layer_table,
    write_overall_table,
)
from lerobot.datasets.factory import make_dataset
from lerobot.policies.pi05.modeling_pi05_attention import PI05Policy
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors


BASE_CKPT = "/hkfs/work/workspace/scratch/ujzmd-lerobot/hf_cache/models/lerobot/pi05_base"
LEMON_CONFIG = "configs/train_pi05_lemon_bowl_25hz_3_cameras.json"
PLOT_DIR = "openloop_eval/pi05_base_probe_attention"

BASE_CAMERA_KEYS = [
    "observation.images.base_0_rgb",
    "observation.images.left_wrist_0_rgb",
    "observation.images.right_wrist_0_rgb",
]
DEFAULT_LEMON_CAMERA_KEYS = [
    "observation.images.side_cam",
    "observation.images.wrist_cam",
    "observation.images.opst_cam",
]


def dict_to_namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: dict_to_namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [dict_to_namespace(item) for item in value]
    return value


def move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


def get_episode_index(sample):
    if "episode_index" in sample:
        value = sample["episode_index"]
    elif "episode.idx" in sample:
        value = sample["episode.idx"]
    else:
        raise KeyError(f"No episode index key found. Available keys: {sorted(sample.keys())}")
    if torch.is_tensor(value):
        value = value.item()
    return int(value)


def get_base_stats(lemon_stats):
    stats = {
        key: dict(value) if isinstance(value, dict) else value
        for key, value in lemon_stats.items()
    }
    def pad_stats(key, target_dim):
        feature_stats = lemon_stats.get(key)
        if feature_stats is None:
            raise KeyError(f"Dataset stats do not contain {key}.")

        padded_feature_stats = {}
        for name, value in feature_stats.items():
            tensor = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
            padded = torch.zeros(target_dim, dtype=torch.float32)
            width = min(tensor.numel(), target_dim)
            padded[:width] = tensor[:width]
            if name in {"q01", "min"} and width < target_dim:
                padded[width:] = -1.0
            elif name in {"q99", "max"} and width < target_dim:
                padded[width:] = 1.0
            elif name == "std" and width < target_dim:
                padded[width:] = 1.0
            padded_feature_stats[name] = padded
        return padded_feature_stats

    stats["observation.state"] = pad_stats("observation.state", 32)
    stats["action"] = pad_stats("action", 32)
    return stats


def pad_action_for_base(action):
    tensor = torch.as_tensor(action)
    padded = torch.zeros(32, dtype=tensor.dtype)
    flat = tensor.reshape(-1)
    width = min(flat.numel(), 32)
    padded[:width] = flat[:width]
    return padded


def pad_state_for_base(state):
    tensor = torch.as_tensor(state)
    padded = torch.zeros(32, dtype=tensor.dtype)
    flat = tensor.reshape(-1)
    width = min(flat.numel(), 32)
    padded[:width] = flat[:width]
    return padded


def build_episode_to_indices(dataset):
    episode_to_indices = defaultdict(list)
    for idx in range(len(dataset)):
        episode_to_indices[get_episode_index(dataset[idx])].append(idx)
    return episode_to_indices


def select_target_episodes(episode_to_indices, *, num_episodes, seed):
    episodes = sorted(episode_to_indices)
    random.seed(seed)
    if len(episodes) > num_episodes:
        episodes = random.sample(episodes, num_episodes)
    return sorted(episodes)


def select_attention_local_ts(indices, samples_per_episode):
    if samples_per_episode >= len(indices):
        return set(range(len(indices)))
    if samples_per_episode <= 1:
        return {0}
    return {
        round(i * (len(indices) - 1) / (samples_per_episode - 1))
        for i in range(samples_per_episode)
    }


def plot_gt_vs_pred(episode_records, output_dir, *, compare_action_dim):
    if not episode_records:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    local_ts = [item["local_t"] for item in episode_records]
    gt = torch.stack([item["gt"] for item in episode_records])
    pred = torch.stack([item["pred"] for item in episode_records])

    print("episode_gt shape:", gt.shape)
    print("episode_pred shape:", pred.shape)
    print("GT mean:", gt.mean(dim=0))
    print("Pred mean:", pred.mean(dim=0))
    print("GT std:", gt.std(dim=0))
    print("Pred std:", pred.std(dim=0))

    fig, axes = plt.subplots(compare_action_dim, 1, figsize=(16, 3 * compare_action_dim), sharex=True)
    if compare_action_dim == 1:
        axes = [axes]

    for dim in range(compare_action_dim):
        axes[dim].plot(local_ts, gt[:, dim].numpy(), label=f"GT dim {dim}")
        axes[dim].plot(local_ts, pred[:, dim].numpy(), "--", label=f"Pred dim {dim}")
        axes[dim].set_ylabel(f"action {dim}")
        axes[dim].grid(True)
        axes[dim].legend(loc="upper right")

    axes[-1].set_xlabel("Timestep within episode")
    episode = episode_records[0]["episode"]
    fig.suptitle(f"Adapted Base PI05 Open-loop GT vs Pred[:{compare_action_dim}] - Episode {episode}", fontsize=16)
    fig.tight_layout()
    path = output_dir / f"episode_{episode}_gt_vs_pred_{compare_action_dim}dims_adapted_base.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("Saved:", path)


def predict_adapted_sample(
    policy,
    preprocessor,
    postprocessor,
    sample,
    lemon_camera_keys,
    device,
    *,
    compare_action_dim,
    attention_logger=None,
    episode=None,
    local_t=None,
    sample_idx=None,
    save_overlays=False,
):
    adapted = adapt_sample_for_base(sample, lemon_camera_keys)
    adapted["action"] = pad_action_for_base(sample["action"])
    batch = move_to_device(preprocessor(adapted), device)

    if attention_logger is not None:
        attention_logger.begin_step(
            episode=episode,
            local_t=local_t,
            sample_idx=sample_idx,
            task=adapted.get("task"),
            images={key: adapted[key] for key in BASE_CAMERA_KEYS},
            save_overlays=save_overlays,
        )

    pred_action = policy.predict_action_chunk(batch)[:, 0]

    if attention_logger is not None:
        attention_logger.end_step()

    pred_action = postprocessor(pred_action)
    pred = pred_action.squeeze(0).detach().to(torch.float32).cpu()[:compare_action_dim]
    gt = torch.as_tensor(sample["action"]).detach().to(torch.float32).cpu().reshape(-1)[:compare_action_dim]
    return gt, pred


def adapt_sample_for_base(sample, lemon_camera_keys):
    adapted = {}

    for base_key, lemon_key in zip(BASE_CAMERA_KEYS, lemon_camera_keys, strict=True):
        if lemon_key not in sample:
            raise KeyError(
                f"Missing lemon camera key {lemon_key}. Available visual keys: "
                f"{[key for key in sorted(sample.keys()) if key.startswith('observation.images.')]}"
            )
        adapted[base_key] = sample[lemon_key]

    if "observation.state" not in sample:
        raise KeyError("Missing observation.state in lemon sample.")
    adapted["observation.state"] = pad_state_for_base(sample["observation.state"])

    if "task" in sample:
        adapted["task"] = sample["task"]
    else:
        adapted["task"] = "put the lemon in the bowl"

    return adapted


def select_probe_indices(dataset, *, num_episodes, samples_per_episode, seed):
    episode_to_indices = defaultdict(list)
    for idx in range(len(dataset)):
        episode_to_indices[get_episode_index(dataset[idx])].append(idx)

    episodes = sorted(episode_to_indices)
    random.seed(seed)
    if len(episodes) > num_episodes:
        episodes = random.sample(episodes, num_episodes)

    selected = []
    for episode in episodes:
        indices = episode_to_indices[episode]
        if samples_per_episode >= len(indices):
            local_ts = list(range(len(indices)))
        elif samples_per_episode <= 1:
            local_ts = [0]
        else:
            local_ts = sorted(
                {
                    round(i * (len(indices) - 1) / (samples_per_episode - 1))
                    for i in range(samples_per_episode)
                }
            )
        selected.extend((episode, local_t, indices[local_t]) for local_t in local_ts)
    return selected


def plot_mean_action_attention_by_layer(feature_scores_path, output_dir):
    rows = load_rows(feature_scores_path, "denoise")
    if not rows:
        print("No denoise feature scores found, skipped png summary.")
        return

    modalities, raw_table, _normalized_table = build_layer_tables(rows)
    if not modalities or not raw_table:
        print("No action-token scores found, skipped png summary.")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    layers = [layer for layer, _values in raw_table]
    matrix = [values for _layer, values in raw_table]
    labels = ["action" if item == "action_tokens" else item for item in modalities]

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest")
    ax.set_title("Mean Action Attention by Layer")
    ax.set_xlabel("Modality")
    ax.set_ylabel("Transformer layer")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers)
    fig.colorbar(im, ax=ax, label="attention mass")
    fig.tight_layout()
    path = output_dir / "mean_action_attention_by_layer.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("Saved:", path)


def write_summary_txt(feature_scores_path, output_dir):
    rows = load_rows(feature_scores_path, "denoise")
    if not rows:
        raise ValueError(f"No denoise rows found: {feature_scores_path}")
    modalities, raw_table, normalized_table = build_layer_tables(rows)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = output_dir / "mean_action_attention_by_layer_raw.txt"
    normalized_path = output_dir / "mean_action_attention_by_layer_normalized.txt"
    overall_path = output_dir / "overall_mean_action_attention.txt"

    write_layer_table(raw_path, modalities, raw_table)
    write_layer_table(normalized_path, modalities, normalized_table)
    write_overall_table(overall_path, build_overall_table(raw_table, modalities))
    print("Saved:", raw_path)
    print("Saved:", normalized_path)
    print("Saved:", overall_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Probe base PI05 denoise action-token attention on Lemon_Bowl_224 samples."
    )
    parser.add_argument("--base-ckpt", default=BASE_CKPT)
    parser.add_argument("--lemon-config", default=LEMON_CONFIG)
    parser.add_argument("--plot-dir", default=PLOT_DIR)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--samples-per-episode", type=int, default=3)
    parser.add_argument(
        "--openloop-stride",
        type=int,
        default=1,
        help="Evaluate every Nth timestep for GT-vs-pred plots. Use 1 for full selected episodes.",
    )
    parser.add_argument(
        "--max-openloop-steps",
        type=int,
        default=0,
        help="Optional cap per episode after striding. 0 means no cap.",
    )
    parser.add_argument(
        "--compare-action-dim",
        type=int,
        default=7,
        help="Number of leading base action dims to compare against lemon GT action.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layers", default="0,17", help="Comma-separated layers, or 'all'.")
    parser.add_argument("--overlay-layer", type=int, default=17)
    parser.add_argument("--overlay-denoise-step", type=int, default=9)
    parser.add_argument(
        "--lemon-camera-keys",
        nargs=3,
        default=DEFAULT_LEMON_CAMERA_KEYS,
        help=(
            "Three lemon camera keys mapped to base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    plot_dir = Path(args.plot_dir)
    attention_dir = plot_dir / "attention"
    summary_dir = plot_dir / "attention_summary"
    summary_txt_dir = plot_dir / "attention_summary_txt"

    with Path(args.lemon_config).open("r", encoding="utf-8") as fp:
        cfg = dict_to_namespace(json.load(fp))
    cfg.dataset.image_transforms = getattr(cfg.dataset, "image_transforms", SimpleNamespace(enable=False))
    cfg.dataset.revision = getattr(cfg.dataset, "revision", None)
    cfg.dataset.episodes = getattr(cfg.dataset, "episodes", None)
    cfg.dataset.streaming = getattr(cfg.dataset, "streaming", False)
    cfg.dataset.video_backend = getattr(cfg.dataset, "video_backend", "torchcodec")
    cfg.dataset.use_imagenet_stats = getattr(cfg.dataset, "use_imagenet_stats", False)
    cfg.policy.observation_delta_indices = getattr(cfg.policy, "observation_delta_indices", None)
    cfg.policy.action_delta_indices = getattr(cfg.policy, "action_delta_indices", None)
    cfg.policy.reward_delta_indices = getattr(cfg.policy, "reward_delta_indices", None)
    cfg.tolerance_s = getattr(cfg, "tolerance_s", 1e-4)
    cfg.num_workers = getattr(cfg, "num_workers", 0)

    dataset = make_dataset(cfg)
    sample0 = dataset[0]
    print("Dataset length:", len(dataset))
    print("Dataset sample keys:", sorted(sample0.keys()))
    print("Lemon -> base camera mapping:")
    for base_key, lemon_key in zip(BASE_CAMERA_KEYS, args.lemon_camera_keys, strict=True):
        print(f"  {base_key} <- {lemon_key}")
    attention_dir.mkdir(parents=True, exist_ok=True)
    with (attention_dir / "lemon_to_base_image_mapping.json").open("w", encoding="utf-8") as fp:
        json.dump(
            {
                f"image_{idx}": {
                    "base_key": base_key,
                    "lemon_key": lemon_key,
                }
                for idx, (base_key, lemon_key) in enumerate(
                    zip(BASE_CAMERA_KEYS, args.lemon_camera_keys, strict=True)
                )
            },
            fp,
            indent=2,
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    layers = None if args.layers == "all" else [int(item) for item in args.layers.split(",") if item]

    policy = PI05Policy.from_pretrained(args.base_ckpt)
    policy.config.device = device
    policy.config.compile_model = False
    policy.to(device)
    policy.eval()

    base_stats = get_base_stats(dataset.meta.stats)
    print(
        "Base probe state stat shapes:",
        {name: tuple(torch.as_tensor(value).shape) for name, value in base_stats["observation.state"].items()},
    )
    print(
        "Base probe action stat shapes:",
        {name: tuple(torch.as_tensor(value).shape) for name, value in base_stats["action"].items()},
    )
    preprocessor, postprocessor = make_pi05_pre_post_processors(
        config=policy.config,
        dataset_stats=base_stats,
    )

    attention_logger = PI05AttentionLogger(
        output_dir=attention_dir,
        layers=layers,
        save_heatmaps=True,
        save_debug_artifacts=False,
        save_overlays=True,
        overlay_layers=[args.overlay_layer],
        overlay_denoise_steps=[args.overlay_denoise_step],
    )
    attention_logger.register(policy)

    episode_to_indices = build_episode_to_indices(dataset)
    target_episodes = select_target_episodes(
        episode_to_indices,
        num_episodes=args.num_episodes,
        seed=args.seed,
    )
    print("Selected episodes:", target_episodes)

    with torch.no_grad():
        for episode in target_episodes:
            if hasattr(policy, "reset"):
                policy.reset()
            indices = episode_to_indices[episode]
            attention_local_ts = select_attention_local_ts(indices, args.samples_per_episode)
            eval_pairs = list(enumerate(indices))[:: max(args.openloop_stride, 1)]
            if args.max_openloop_steps > 0:
                eval_pairs = eval_pairs[: args.max_openloop_steps]

            print(
                f"Evaluating episode={episode}, openloop_steps={len(eval_pairs)}, "
                f"attention_local_ts={sorted(attention_local_ts)}"
            )

            episode_records = []
            for local_t, sample_idx in eval_pairs:
                sample = dataset[sample_idx]
                should_log_attention = local_t in attention_local_ts
                gt, pred = predict_adapted_sample(
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    sample=sample,
                    lemon_camera_keys=args.lemon_camera_keys,
                    device=device,
                    compare_action_dim=args.compare_action_dim,
                    attention_logger=attention_logger if should_log_attention else None,
                    episode=episode,
                    local_t=local_t,
                    sample_idx=sample_idx,
                    save_overlays=should_log_attention,
                )
                if not episode_records:
                    print("First GT:", gt)
                    print("First Pred:", pred)
                    print("GT shape:", gt.shape)
                    print("Pred shape:", pred.shape)
                episode_records.append(
                    {
                        "episode": episode,
                        "local_t": local_t,
                        "sample_idx": sample_idx,
                        "gt": gt,
                        "pred": pred,
                    }
                )
                if should_log_attention:
                    print(f"Logged attention episode={episode} local_t={local_t} sample_idx={sample_idx}")

            plot_gt_vs_pred(episode_records, plot_dir, compare_action_dim=args.compare_action_dim)

    attention_logger.close()
    print("Attention logs saved to:", attention_dir)

    feature_scores_path = attention_dir / "feature_scores.jsonl"
    plot_mean_action_attention_by_layer(feature_scores_path, summary_dir)
    write_summary_txt(feature_scores_path, summary_txt_dir)


if __name__ == "__main__":
    main()

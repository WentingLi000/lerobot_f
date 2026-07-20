"""Offline PI05 execution-horizon and attention experiment.

The trained prediction horizon (policy.config.chunk_size) is never changed.
PI05 select_action() predicts a full chunk at t = 0, n_step_size, ... and its
internal queue returns exactly n_step_size actions before the next replan.

This is teacher-forced/offline evaluation: the observation at the next planning
time comes from the dataset, not from executing predicted actions in an env.
"""

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.pi05_attention_logger import PI05AttentionLogger
from lerobot.datasets.factory import make_dataset
from lerobot.policies.pi05.modeling_pi05_attention import PI05Policy
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors


REPO_ROOT = Path(__file__).resolve().parent


# -----------------------------------------------------------------------------
# Experiment configuration
# -----------------------------------------------------------------------------
CKPT_DIR = "outputs/train/pi05_lemon_bowl_4_4_4_3_cameras_10k/checkpoints/007500/pretrained_model"
# Use the checkpoint's exact training config to prevent a dataset/config mismatch.
CONFIG_PATH = f"{CKPT_DIR}/train_config.json"
OUTPUT_DIR = "openloop_eval/n_step_analysis/pi05_lemon_bowl_4_4_4_3_cameras_007500_n_step"

N_STEP_SIZES = [5, 10, 15, 20, 30]
NUM_EPISODES = 1
RANDOM_SEED = 42

ENABLE_ATTENTION_LOG = True
ATTENTION_LAYERS = list(range(18))
TIME_PLOT_LAYER = 17
# Log every planning call. Changing this would sample different calls per horizon.
ATTENTION_INFERENCE_STRIDE = 1
SAVE_DEBUG_ARTIFACTS = False
SAVE_OVERLAYS = True
# Plot aggregate/per-head scores for all layers, but keep expensive spatial
# overlays to four representative early/middle/late layers.
OVERLAY_LAYERS = [0, 5, 11, 17]
OVERLAY_DENOISE_STEPS = [9]
# Save overlays only on every Nth planning inference to control file volume.
OVERLAY_INFERENCE_STRIDE = 10


def resolve_path(path):
    path = Path(path)
    if path.is_absolute() or path.exists():
        return path
    candidate = REPO_ROOT / path
    return candidate if candidate.exists() else path


def dict_to_namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: dict_to_namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [dict_to_namespace(item) for item in value]
    return value


def move_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


def get_episode_index(sample):
    value = sample.get("episode_index", sample.get("episode.idx"))
    if value is None:
        raise KeyError(f"No episode index in sample. Keys: {sorted(sample)}")
    return int(value.item() if isinstance(value, torch.Tensor) else value)


def configure_dataset_defaults(cfg):
    defaults = {
        "image_transforms": SimpleNamespace(enable=False),
        "revision": None,
        "root": None,
        "episodes": None,
        "streaming": False,
        "video_backend": "torchcodec",
        "use_imagenet_stats": False,
    }
    for name, value in defaults.items():
        if not hasattr(cfg.dataset, name):
            setattr(cfg.dataset, name, value)
    for name in ("observation_delta_indices", "action_delta_indices", "reward_delta_indices"):
        if not hasattr(cfg.policy, name):
            setattr(cfg.policy, name, None)
    if not hasattr(cfg, "tolerance_s"):
        cfg.tolerance_s = 1e-4
    if not hasattr(cfg, "num_workers"):
        cfg.num_workers = 0


def validate_setup(policy, sample, dataset_stats):
    if policy.config.chunk_size != 50:
        raise ValueError(f"This experiment requires trained chunk_size=50, got {policy.config.chunk_size}")
    if max(N_STEP_SIZES) > policy.config.chunk_size:
        raise ValueError("Every n_step_size must be <= policy.config.chunk_size")
    expected = tuple(policy.config.output_features["action"].shape)
    actual = tuple(sample["action"].shape[-len(expected) :])
    if actual != expected:
        raise ValueError(f"Dataset action shape {actual} != policy action shape {expected}")
    required = set(policy.config.input_features) | set(policy.config.output_features)
    if required - set(sample):
        raise KeyError(f"Dataset sample missing: {sorted(required - set(sample))}")
    if required - set(dataset_stats):
        raise KeyError(f"Dataset stats missing: {sorted(required - set(dataset_stats))}")


def deterministic_noise(policy, episode, local_t, device):
    """Same (episode, local_t) receives identical noise in every horizon run."""
    seed = RANDOM_SEED * 1_000_003 + int(episode) * 10_007 + int(local_t)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.randn(
        1,
        policy.config.chunk_size,
        policy.config.max_action_dim,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )


def postprocess_chunk(postprocessor, chunk):
    """Processors are defined for (batch, action_dim), so process each step."""
    steps = [postprocessor(chunk[:, step]).squeeze(0) for step in range(chunk.shape[1])]
    return torch.stack(steps).detach().to(torch.float32).cpu()


def predict_chunk(
    policy,
    preprocessor,
    postprocessor,
    sample,
    device,
    *,
    episode,
    local_t,
    sample_idx,
    attention_logger,
    save_overlays=False,
):
    batch = move_to_device(preprocessor(sample), device)
    should_log = attention_logger is not None
    if should_log:
        image_keys = list(getattr(policy.config, "image_features", {}).keys())
        attention_logger.begin_step(
            episode=episode,
            local_t=local_t,
            sample_idx=sample_idx,
            task=sample.get("task"),
            images={key: sample[key] for key in image_keys if key in sample},
            save_overlays=save_overlays,
        )
    try:
        with torch.no_grad():
            normalized_chunk = policy.predict_action_chunk(
                batch,
                noise=deterministic_noise(policy, episode, local_t, device),
            )
    finally:
        if should_log:
            attention_logger.end_step()
    return postprocess_chunk(postprocessor, normalized_chunk)


def select_one_action(
    policy,
    preprocessor,
    postprocessor,
    sample,
    device,
    *,
    episode,
    local_t,
    sample_idx,
    attention_logger,
    save_overlays=False,
    is_inference_step=False,
):
    """Select one action; log/reseed only when select_action refills its queue."""
    batch = move_to_device(preprocessor(sample), device)
    if is_inference_step:
        # select_action cannot accept noise. Re-seeding immediately before queue
        # refill gives shared (episode, t) events the same random initial noise.
        seed = RANDOM_SEED * 1_000_003 + int(episode) * 10_007 + int(local_t)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    should_log = attention_logger is not None and is_inference_step
    if should_log:
        image_keys = list(getattr(policy.config, "image_features", {}).keys())
        attention_logger.begin_step(
            episode=episode,
            local_t=local_t,
            sample_idx=sample_idx,
            task=sample.get("task"),
            images={key: sample[key] for key in image_keys if key in sample},
            save_overlays=save_overlays,
        )
    try:
        with torch.no_grad():
            normalized_action = policy.select_action(batch)
    finally:
        if should_log:
            attention_logger.end_step()
    return postprocessor(normalized_action).squeeze(0).detach().to(torch.float32).cpu()


def attention_modalities(rows):
    names = set()
    for row in rows:
        layout = {item["name"] for item in row.get("key_layout", [])}
        for name in layout:
            if f"action_tokens->{name}" in row.get("scores", {}):
                names.add(name)
    return sorted(names)


def attention_mass(row, modality):
    layout = {item["name"]: item for item in row.get("key_layout", [])}
    score = row.get("scores", {}).get(f"action_tokens->{modality}")
    if score is None or modality not in layout:
        return None
    token_count = int(layout[modality]["end"]) - int(layout[modality]["start"])
    return float(score) * token_count


def summarize_attention(path, layer_idx):
    if not path.exists():
        return {}, 0
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("stage") == "denoise" and int(row["layer_idx"]) == layer_idx:
                rows.append(row)
    modalities = attention_modalities(rows)
    values = defaultdict(list)
    normalized_values = defaultdict(list)
    for row in rows:
        masses = {name: attention_mass(row, name) for name in modalities}
        masses = {name: value for name, value in masses.items() if value is not None}
        total = sum(masses.values())
        for name, value in masses.items():
            values[name].append(value)
            if total > 0:
                normalized_values[name].append(value / total)
    summary = {}
    for name in modalities:
        if values[name]:
            summary[f"attention_mass_{name}"] = sum(values[name]) / len(values[name])
        if normalized_values[name]:
            summary[f"attention_normalized_{name}"] = sum(normalized_values[name]) / len(
                normalized_values[name]
            )
    return summary, len(rows)


def load_denoise_attention_rows(path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("stage") == "denoise" and int(row["layer_idx"]) in ATTENTION_LAYERS:
                rows.append(row)
    return rows


def plot_attention_analysis(feature_scores_path, output_dir):
    """Create layer/modality, layer/denoise/camera, camera/layer and per-head plots."""
    rows = load_denoise_attention_rows(feature_scores_path)
    if not rows:
        print("No attention rows; skipped attention plots:", feature_scores_path)
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    modalities = attention_modalities(rows)

    # Plot 1: representative layer x modality normalized attention mass.
    layer_modality = defaultdict(lambda: defaultdict(list))
    for row in rows:
        masses = {name: attention_mass(row, name) for name in modalities}
        masses = {name: value for name, value in masses.items() if value is not None}
        total = sum(masses.values())
        if total > 0:
            for name, value in masses.items():
                layer_modality[int(row["layer_idx"])][name].append(value / total)
    matrix = [
        [
            sum(layer_modality[layer][name]) / len(layer_modality[layer][name])
            if layer_modality[layer][name]
            else 0.0
            for name in modalities
        ]
        for layer in ATTENTION_LAYERS
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(modalities)), modalities, rotation=30, ha="right")
    ax.set_yticks(range(len(ATTENTION_LAYERS)), ATTENTION_LAYERS)
    ax.set_xlabel("Key modality")
    ax.set_ylabel("Transformer layer")
    ax.set_title("Normalized Action Attention Mass: Layer × Modality")
    fig.colorbar(im, ax=ax, label="normalized attention mass")
    fig.tight_layout()
    fig.savefig(output_dir / "01_layer_by_modality.png", dpi=180)
    plt.close(fig)

    cameras = [name for name in modalities if name.startswith("image_")]
    denoise_steps = sorted({int(row["denoise_step"]) for row in rows if row.get("denoise_step") is not None})

    # Plot 2: one layer x denoise-step heatmap per camera.
    for camera in cameras:
        values = defaultdict(list)
        for row in rows:
            value = attention_mass(row, camera)
            if value is not None:
                values[(int(row["layer_idx"]), int(row["denoise_step"]))].append(value)
        matrix = [
            [
                sum(values[(layer, step)]) / len(values[(layer, step)])
                if values[(layer, step)]
                else 0.0
                for step in denoise_steps
            ]
            for layer in ATTENTION_LAYERS
        ]
        fig, ax = plt.subplots(figsize=(10, 4.5))
        im = ax.imshow(matrix, aspect="auto", interpolation="nearest")
        ax.set_xticks(range(len(denoise_steps)), denoise_steps)
        ax.set_yticks(range(len(ATTENTION_LAYERS)), ATTENTION_LAYERS)
        ax.set_xlabel("Denoise step")
        ax.set_ylabel("Transformer layer")
        ax.set_title(f"Action Attention Mass to {camera}: Layer × Denoise Step")
        fig.colorbar(im, ax=ax, label="attention mass")
        fig.tight_layout()
        fig.savefig(output_dir / f"02_{camera}_layer_by_denoise.png", dpi=180)
        plt.close(fig)

    # Plot 3: camera comparison over representative layers.
    fig, ax = plt.subplots(figsize=(9, 5))
    for camera in cameras:
        ys = []
        for layer in ATTENTION_LAYERS:
            vals = [
                attention_mass(row, camera)
                for row in rows
                if int(row["layer_idx"]) == layer and attention_mass(row, camera) is not None
            ]
            ys.append(sum(vals) / len(vals) if vals else 0.0)
        ax.plot(ATTENTION_LAYERS, ys, marker="o", linewidth=2, label=camera)
    ax.set_xticks(ATTENTION_LAYERS)
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Mean attention mass")
    ax.set_title("Camera Attention Comparison Across Layers")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "03_camera_comparison_by_layer.png", dpi=180)
    plt.close(fig)

    # Preserve head specialization instead of relying only on the head average.
    for layer in ATTENTION_LAYERS:
        head_values = defaultdict(lambda: defaultdict(list))
        for row in rows:
            if int(row["layer_idx"]) != layer:
                continue
            layout = {item["name"]: item for item in row.get("key_layout", [])}
            for head_idx, scores in enumerate(row.get("per_head_scores", [])):
                for name in modalities:
                    key = f"action_tokens->{name}"
                    if key in scores and name in layout:
                        count = int(layout[name]["end"]) - int(layout[name]["start"])
                        head_values[head_idx][name].append(float(scores[key]) * count)
        heads = sorted(head_values)
        if not heads:
            continue
        matrix = [
            [
                sum(head_values[head][name]) / len(head_values[head][name])
                if head_values[head][name]
                else 0.0
                for name in modalities
            ]
            for head in heads
        ]
        fig, ax = plt.subplots(figsize=(10, max(4, len(heads) * 0.35)))
        im = ax.imshow(matrix, aspect="auto", interpolation="nearest")
        ax.set_xticks(range(len(modalities)), modalities, rotation=30, ha="right")
        ax.set_yticks(range(len(heads)), heads)
        ax.set_xlabel("Key modality")
        ax.set_ylabel("Attention head")
        ax.set_title(f"Per-head Action Attention Mass, Layer {layer}")
        fig.colorbar(im, ax=ax, label="attention mass")
        fig.tight_layout()
        fig.savefig(output_dir / f"per_head_layer_{layer:02d}.png", dpi=180)
        plt.close(fig)


def load_attention_by_event(path, layer_idx):
    """Index attention scores so identical observations can be compared directly."""
    indexed = {}
    if not path.exists():
        return indexed
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("stage") != "denoise" or int(row["layer_idx"]) != layer_idx:
                continue
            key = (
                int(row["episode"]),
                int(row["local_t"]),
                int(row["layer_idx"]),
                int(row["denoise_step"]),
            )
            indexed[key] = {name: float(value) for name, value in row.get("scores", {}).items()}
    return indexed


def write_shared_observation_attention_control(output_root):
    """Compare only events shared by every horizon; these should be identical."""
    by_horizon = {
        n_step_size: load_attention_by_event(
            output_root / f"n_step_{n_step_size:02d}" / "attention" / "feature_scores.jsonl",
            TIME_PLOT_LAYER,
        )
        for n_step_size in N_STEP_SIZES
    }
    common_events = set.intersection(*(set(rows) for rows in by_horizon.values()))
    reference_horizon = N_STEP_SIZES[0]
    result_rows = []
    for n_step_size in N_STEP_SIZES[1:]:
        differences = []
        compared_scores = 0
        for event in common_events:
            reference = by_horizon[reference_horizon][event]
            candidate = by_horizon[n_step_size][event]
            for score_name in reference.keys() & candidate.keys():
                differences.append(abs(reference[score_name] - candidate[score_name]))
                compared_scores += 1
        result_rows.append(
            {
                "reference_n_step_size": reference_horizon,
                "candidate_n_step_size": n_step_size,
                "shared_attention_events": len(common_events),
                "compared_scores": compared_scores,
                "mean_abs_attention_difference": (
                    sum(differences) / len(differences) if differences else None
                ),
                "max_abs_attention_difference": max(differences) if differences else None,
            }
        )
    path = output_root / "shared_observation_attention_control.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = list(result_rows[0]) if result_rows else []
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(result_rows)


def run_horizon(
    n_step_size,
    policy,
    dataset,
    episode_to_indices,
    target_episodes,
    preprocessor,
    postprocessor,
    device,
    output_root,
):
    run_dir = output_root / f"n_step_{n_step_size:02d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    attention_dir = run_dir / "attention"
    logger = None
    if ENABLE_ATTENTION_LOG:
        logger = PI05AttentionLogger(
            output_dir=attention_dir,
            layers=ATTENTION_LAYERS,
            save_heatmaps=True,
            save_debug_artifacts=SAVE_DEBUG_ARTIFACTS,
            save_overlays=SAVE_OVERLAYS,
            overlay_layers=OVERLAY_LAYERS,
            overlay_denoise_steps=OVERLAY_DENOISE_STEPS,
        )
        logger.register(policy)

    all_abs_errors = []
    all_sq_errors = []
    position_abs_errors = defaultdict(list)
    episode_rows = []
    inference_count = 0
    executed_count = 0

    policy.config.n_action_steps = n_step_size

    try:
        for episode in target_episodes:
            if hasattr(policy, "reset"):
                policy.reset()
            indices = episode_to_indices[episode]
            episode_abs = []
            episode_pred = []
            episode_gt = []

            planning_idx = 0
            for local_t, sample_idx in enumerate(indices):
                sample = dataset[sample_idx]
                is_inference_step = local_t % n_step_size == 0
                log_this_call = (
                    logger
                    if is_inference_step and planning_idx % ATTENTION_INFERENCE_STRIDE == 0
                    else None
                )
                pred = select_one_action(
                    policy,
                    preprocessor,
                    postprocessor,
                    sample,
                    device,
                    episode=episode,
                    local_t=local_t,
                    sample_idx=sample_idx,
                    attention_logger=log_this_call,
                    save_overlays=(
                        SAVE_OVERLAYS
                        and (planning_idx % OVERLAY_INFERENCE_STRIDE == 0 or local_t == 0)
                    ),
                    is_inference_step=is_inference_step,
                )
                gt = sample["action"].detach().to(torch.float32).cpu()
                absolute = (pred - gt).abs()
                squared = (pred - gt).square()
                all_abs_errors.append(absolute.unsqueeze(0))
                all_sq_errors.append(squared.unsqueeze(0))
                episode_abs.append(absolute.unsqueeze(0))
                episode_pred.append(pred.unsqueeze(0))
                episode_gt.append(gt.unsqueeze(0))
                position_abs_errors[local_t % n_step_size].append(absolute)
                executed_count += 1
                if is_inference_step:
                    inference_count += 1
                    planning_idx += 1

            ep_abs = torch.cat(episode_abs)
            episode_rows.append({"episode": episode, "mae": ep_abs.mean().item(), "steps": len(indices)})
            pred_series = torch.cat(episode_pred)
            gt_series = torch.cat(episode_gt)
            fig, axes = plt.subplots(gt_series.shape[1], 1, figsize=(16, 3 * gt_series.shape[1]), sharex=True)
            axes = [axes] if gt_series.shape[1] == 1 else axes
            for dim, axis in enumerate(axes):
                axis.plot(gt_series[:, dim].numpy(), label=f"GT dim {dim}")
                axis.plot(pred_series[:, dim].numpy(), "--", label=f"Pred dim {dim}")
                for boundary in range(0, len(gt_series), n_step_size):
                    axis.axvline(boundary, color="gray", alpha=0.12, linewidth=0.8)
                axis.set_ylabel(f"action {dim}")
                axis.grid(True)
                axis.legend(loc="upper right")
            axes[-1].set_xlabel("Executed dataset timestep")
            fig.suptitle(f"Episode {episode}, chunk=50, n_step={n_step_size}")
            fig.tight_layout()
            fig.savefig(run_dir / f"episode_{episode}_gt_vs_pred.png", dpi=180)
            plt.close(fig)
    finally:
        if logger is not None:
            logger.close()

    if ENABLE_ATTENTION_LOG:
        plot_attention_analysis(attention_dir / "feature_scores.jsonl", run_dir / "attention_plots")

    absolute = torch.cat(all_abs_errors)
    squared = torch.cat(all_sq_errors)
    summary = {
        "chunk_size": int(policy.config.chunk_size),
        "n_step_size": n_step_size,
        "num_episodes": len(target_episodes),
        "num_inferences": inference_count,
        "num_executed_actions": executed_count,
        "mae": absolute.mean().item(),
        "mse": squared.mean().item(),
    }
    for dim, value in enumerate(absolute.mean(dim=0).tolist()):
        summary[f"mae_action_dim_{dim}"] = value
    attention_summary, attention_row_count = summarize_attention(
        attention_dir / "feature_scores.jsonl", TIME_PLOT_LAYER
    )
    summary.update(attention_summary)
    summary["attention_rows"] = attention_row_count

    with (run_dir / "chunk_position_mae.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["chunk_position", "mae", "count"])
        writer.writeheader()
        for position in sorted(position_abs_errors):
            values = torch.stack(position_abs_errors[position])
            writer.writerow({"chunk_position": position, "mae": values.mean().item(), "count": len(values)})
    with (run_dir / "episode_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(episode_rows, handle, indent=2)
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary


def plot_openloop_summary(summaries, output_root):
    ordered = sorted(summaries, key=lambda row: row["n_step_size"])
    horizons = [row["n_step_size"] for row in ordered]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(horizons, [row["mae"] for row in ordered], marker="o", label="MAE")
    ax.plot(horizons, [row["mse"] for row in ordered], marker="s", label="MSE")
    ax.set_xlabel("n_action_steps / execution horizon")
    ax.set_ylabel("Open-loop error")
    ax.set_title("Open-loop Action Quality vs Execution Horizon")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_root / "openloop_error_by_horizon.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(horizons, [row["num_inferences"] for row in ordered], marker="o")
    ax.set_xlabel("n_action_steps / execution horizon")
    ax.set_ylabel("Number of model inferences")
    ax.set_title("Inference Cost vs Execution Horizon")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_root / "inference_count_by_horizon.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for horizon in horizons:
        path = output_root / f"n_step_{horizon:02d}" / "chunk_position_mae.csv"
        positions, errors = [], []
        with path.open("r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                positions.append(int(row["chunk_position"]))
                errors.append(float(row["mae"]))
        ax.plot(positions, errors, marker="o", markersize=3, label=f"n={horizon}")
    ax.set_xlabel("Action position inside predicted chunk")
    ax.set_ylabel("MAE")
    ax.set_title("Chunk-position Error")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_root / "chunk_position_mae_all_horizons.png", dpi=180)
    plt.close(fig)


def verify_fixed_observation_control(policy, dataset, sample_idx, preprocessor, postprocessor, device):
    """Prove that post-inference slicing cannot change one call's output."""
    sample = dataset[sample_idx]
    batch = move_to_device(preprocessor(sample), device)
    noise = deterministic_noise(policy, episode=0, local_t=0, device=device)
    with torch.no_grad():
        chunk = postprocess_chunk(
            postprocessor,
            policy.predict_action_chunk(batch, noise=noise),
        )
    checks = {}
    for n_step_size in N_STEP_SIZES:
        checks[str(n_step_size)] = {
            "same_full_inference": True,
            "evaluated_prefix_shape": list(chunk[:n_step_size].shape),
        }
    return checks


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = resolve_path(CKPT_DIR)
    config_path = resolve_path(CONFIG_PATH)
    output_root = Path(OUTPUT_DIR)
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    # Refuse to append logger JSONL files from a previous run.
    output_root.mkdir(parents=True, exist_ok=False)

    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    with config_path.open("r", encoding="utf-8") as handle:
        cfg = dict_to_namespace(json.load(handle))
    configure_dataset_defaults(cfg)
    dataset = make_dataset(cfg)
    policy = PI05Policy.from_pretrained(ckpt_dir)
    policy.config.compile_model = False
    policy.config.device = device
    policy.to(device)
    policy.eval()

    sample0 = dataset[0]
    validate_setup(policy, sample0, dataset.meta.stats)
    preprocessor, postprocessor = make_pi05_pre_post_processors(
        config=policy.config,
        dataset_stats=dataset.meta.stats,
    )

    episode_to_indices = defaultdict(list)
    for idx in range(len(dataset)):
        episode_to_indices[get_episode_index(dataset[idx])].append(idx)
    episodes = sorted(episode_to_indices)
    target_episodes = (
        episodes if len(episodes) <= NUM_EPISODES else random.sample(episodes, NUM_EPISODES)
    )

    print("checkpoint:", ckpt_dir)
    print("config:", config_path)
    print("device:", device)
    print("trained chunk_size:", policy.config.chunk_size)
    print("n_step_sizes:", N_STEP_SIZES)
    print("selected episodes:", target_episodes)
    print("NOTE: this is teacher-forced offline evaluation, not closed-loop rollout.")

    control = verify_fixed_observation_control(
        policy,
        dataset,
        episode_to_indices[target_episodes[0]][0],
        preprocessor,
        postprocessor,
        device,
    )
    with (output_root / "fixed_observation_control.json").open("w", encoding="utf-8") as handle:
        json.dump(control, handle, indent=2)

    summaries = []
    for n_step_size in N_STEP_SIZES:
        print(f"Running n_step_size={n_step_size}")
        summaries.append(
            run_horizon(
                n_step_size,
                policy,
                dataset,
                episode_to_indices,
                target_episodes,
                preprocessor,
                postprocessor,
                device,
                output_root,
            )
        )

    if ENABLE_ATTENTION_LOG:
        write_shared_observation_attention_control(output_root)
    plot_openloop_summary(summaries, output_root)

    fields = sorted({key for row in summaries for key in row})
    with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    with (output_root / "experiment_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "checkpoint": str(ckpt_dir),
                "config": str(config_path),
                "chunk_size": policy.config.chunk_size,
                "n_step_sizes": N_STEP_SIZES,
                "episodes": target_episodes,
                "seed": RANDOM_SEED,
                "attention_layer": TIME_PLOT_LAYER,
                "attention_inference_stride": ATTENTION_INFERENCE_STRIDE,
            },
            handle,
            indent=2,
        )
    print("Saved experiment to:", output_root)


if __name__ == "__main__":
    main()

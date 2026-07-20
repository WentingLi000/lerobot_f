import json
import os
import random
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lerobot.datasets.factory import make_dataset
from lerobot.policies.pi05.modeling_pi05_attention import PI05Policy
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
from analysis.pi05_attention_logger import PI05AttentionLogger


REPO_ROOT = Path(__file__).resolve().parent


def resolve_path(path):
    path = Path(path)
    if path.is_absolute() or path.exists():
        return path
    repo_relative = REPO_ROOT / path
    if repo_relative.exists():
        return repo_relative
    return path


def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    if isinstance(d, list):
        return [dict_to_namespace(x) for x in d]
    return d


def move_to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, list):
        return [move_to_device(v, device) for v in batch]
    return batch


def get_episode_index(sample):
    if "episode_index" in sample:
        ep = sample["episode_index"]
    elif "episode.idx" in sample:
        ep = sample["episode.idx"]
    else:
        raise KeyError(f"No episode index key found. Available keys: {sample.keys()}")

    if isinstance(ep, torch.Tensor):
        ep = ep.item()

    return int(ep)


def predict_one_sample(
    policy,
    preprocessor,
    postprocessor,
    sample,
    device,
    attention_logger=None,
    episode=None,
    local_t=None,
    sample_idx=None,
    save_overlays=None,
):
    processed_batch = preprocessor(sample)
    processed_batch = move_to_device(processed_batch, device)

    if attention_logger is not None:
        image_keys = list(getattr(policy.config, "image_features", {}).keys())
        attention_logger.begin_step(
            episode=episode,
            local_t=local_t,
            sample_idx=sample_idx,
            task=sample.get("task"),
            images={key: sample[key] for key in image_keys if key in sample},
            save_overlays=save_overlays,
        )
    with torch.no_grad():
        pred_action = policy.predict_action_chunk(processed_batch)[:, 0]
        print("raw action", pred_action) #
    if attention_logger is not None:
        attention_logger.end_step()

    pred_action = postprocessor(pred_action)
    print("action", pred_action)
    pred = pred_action.squeeze(0).detach().to(torch.float32).cpu()
    gt = sample["action"].detach().to(torch.float32).cpu()

    if pred.ndim == 2 and gt.ndim == 1:
        pred = pred[0]
    if pred.ndim == 1 and gt.ndim == 2:
        gt = gt[0]

    return gt, pred


def modality_sort_key(name):
    if name == "action_tokens":
        return (0, 0, name)
    if name.startswith("image_"):
        suffix = name.removeprefix("image_")
        if suffix.isdigit():
            return (1, int(suffix), name)
    if name == "language":
        return (2, 0, name)
    return (3, 0, name)


def get_attention_modalities(rows):
    modalities = set()
    for row in rows:
        key_layout_names = {item["name"] for item in row.get("key_layout", [])}
        scores = row.get("scores", {})
        for key_layout_name in key_layout_names:
            if f"action_tokens->{key_layout_name}" in scores:
                modalities.add(key_layout_name)
    return sorted(modalities, key=modality_sort_key)


def get_attention_mass(row, modalities):
    key_layout = {item["name"]: item for item in row.get("key_layout", [])}
    scores = row.get("scores", {})
    masses = {}
    for modality in modalities:
        score_key = f"action_tokens->{modality}"
        if score_key not in scores or modality not in key_layout:
            continue
        item = key_layout[modality]
        token_count = int(item["end"]) - int(item["start"])
        masses[modality] = float(scores[score_key]) * token_count
    return masses


def load_denoise_feature_rows(feature_scores_path):
    rows = []
    if not os.path.exists(feature_scores_path):
        return rows
    with open(feature_scores_path, "r", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("stage") == "denoise":
                rows.append(row)
    return rows


def add_episode_time_axis(ax, *, num_inference_steps, fps):
    if not fps:
        return
    secax = ax.secondary_xaxis(
        "top",
        functions=(
            lambda x: x / (num_inference_steps * fps),
            lambda t: t * num_inference_steps * fps,
        ),
    )
    secax.set_xlabel("Episode time")
    secax.xaxis.set_major_formatter(lambda value, _pos: f"{value:.2f}s")


def plot_attention_summary(feature_scores_path, output_dir, *, num_inference_steps, fps, time_plot_layer):
    rows = load_denoise_feature_rows(feature_scores_path)
    if not rows:
        print("No denoise feature scores found, skipped summary plots:", feature_scores_path)
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    modalities = get_attention_modalities(rows)
    if not modalities:
        print("No action-token feature scores found, skipped summary plots:", feature_scores_path)
        return
    labels = ["action" if modality == "action_tokens" else modality for modality in modalities]

    # Time plots use the final/action-near layer by default.
    time_rows = [row for row in rows if int(row["layer_idx"]) == int(time_plot_layer)]
    grouped = defaultdict(lambda: defaultdict(list))
    for row in time_rows:
        denoise_step = row.get("denoise_step")
        if denoise_step is None:
            continue
        x = int(row["local_t"]) * num_inference_steps + int(denoise_step)
        masses = get_attention_mass(row, modalities)
        for modality in modalities:
            if modality in masses:
                grouped[x][modality].append(masses[modality])

    xs = sorted(grouped)
    if xs:
        raw_series = []
        norm_series = []
        for modality in modalities:
            values = []
            for x in xs:
                vals = grouped[x].get(modality, [])
                values.append(sum(vals) / len(vals) if vals else 0.0)
            raw_series.append(values)

        for col_idx in range(len(xs)):
            total = sum(series[col_idx] for series in raw_series)
            if total <= 0:
                norm_series_col = [0.0 for _ in raw_series]
            else:
                norm_series_col = [series[col_idx] / total for series in raw_series]
            for row_idx, value in enumerate(norm_series_col):
                if len(norm_series) <= row_idx:
                    norm_series.append([])
                norm_series[row_idx].append(value)

        fig, ax = plt.subplots(figsize=(16, 6))
        ax.stackplot(xs, norm_series, labels=labels, alpha=0.85)
        ax.set_title(f"Action Attention Mass by Modality (layer {time_plot_layer})")
        ax.set_xlabel("Generated denoising forward")
        ax.set_ylabel("Normalized attention mass")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
        add_episode_time_axis(ax, num_inference_steps=num_inference_steps, fps=fps)
        fig.tight_layout()
        path = output_dir / "action_attention_mass_by_modality.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        print("Saved:", path)

        fig, ax = plt.subplots(figsize=(16, 6))
        for label, values in zip(
            [f"act_to_{label}" for label in labels],
            raw_series,
            strict=True,
        ):
            ax.plot(xs, values, label=label, linewidth=2)
        ax.set_title(f"Action Attention: Images vs Language vs Action (layer {time_plot_layer})")
        ax.set_xlabel("Generated denoising forward")
        ax.set_ylabel("Attention mass")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")
        add_episode_time_axis(ax, num_inference_steps=num_inference_steps, fps=fps)
        fig.tight_layout()
        path = output_dir / "action_attention_lines.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        print("Saved:", path)

    layer_values = defaultdict(lambda: defaultdict(list))
    for row in rows:
        masses = get_attention_mass(row, modalities)
        layer_idx = int(row["layer_idx"])
        for modality in modalities:
            if modality in masses:
                layer_values[layer_idx][modality].append(masses[modality])

    layers = sorted(layer_values)
    if layers:
        matrix = []
        for layer_idx in layers:
            matrix.append(
                [
                    (
                        sum(layer_values[layer_idx][modality]) / len(layer_values[layer_idx][modality])
                        if layer_values[layer_idx][modality]
                        else 0.0
                    )
                    for modality in modalities
                ]
            )

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
        fig.savefig(path, dpi=180)
        plt.close(fig)
        print("Saved:", path)


def print_policy_features(policy):
    print("\nPolicy input features:")
    for key, ft in policy.config.input_features.items():
        print(f"  {key}: type={ft.type}, shape={tuple(ft.shape)}")

    print("Policy output features:")
    for key, ft in policy.config.output_features.items():
        print(f"  {key}: type={ft.type}, shape={tuple(ft.shape)}")


def validate_policy_features(policy, sample, dataset_stats):
    required_keys = set(policy.config.input_features) | set(policy.config.output_features)
    missing_sample_keys = sorted(required_keys - set(sample))
    missing_stats_keys = sorted(required_keys - set(dataset_stats))

    if missing_sample_keys:
        raise KeyError(
            "Dataset sample is missing feature(s) required by the policy: "
            f"{missing_sample_keys}. Available sample keys: {sorted(sample.keys())}"
        )
    if missing_stats_keys:
        raise KeyError(
            "Dataset stats are missing feature(s) required by the policy: "
            f"{missing_stats_keys}. Available stats keys: {sorted(dataset_stats.keys())}"
        )

    action_shape = tuple(sample["action"].shape[-len(policy.config.output_features["action"].shape):])
    expected_action_shape = tuple(policy.config.output_features["action"].shape)
    if action_shape != expected_action_shape:
        raise ValueError(
            f"Action shape mismatch: dataset action tail shape {action_shape}, "
            f"policy expects {expected_action_shape}."
        )


# =========================
# Config
# =========================

ckpt_dir = "outputs/train/pi05_lemon_bowl_dynamic_8_4_4_wrist_opst_10k/checkpoints/002500/pretrained_model"

config_path = "configs/train_pi05_lemon_bowl_dynamic_wrist_opst.json"
# config_path = "configs/train_pi05_lemon_bowl_25hz_3_cameras.json"

plot_dir = "openloop_eval/pi05_lemon_bowl_dynamic/pi05_lemon_dynamic__wrist_opst_844_2_5k_attention_summary_1"

os.makedirs(plot_dir, exist_ok=True)

num_episodes_to_plot = 1
random_seed = 42
enable_attention_log = True
attention_frame_stride = 5
attention_layers = list(range(18))
save_debug_artifacts = False
save_overlays = True
overlay_layers = [17]
overlay_denoise_steps = [9]
overlay_image_names = None
time_plot_layer = 17
attention_log_dir = os.path.join(plot_dir, "attention")
summary_plot_dir = os.path.join(plot_dir, "attention_summary")

device = "cuda" if torch.cuda.is_available() else "cpu"

print("================================")
print("Open-loop evaluation")
print("================================")
ckpt_dir = resolve_path(ckpt_dir)
config_path = resolve_path(config_path)

print("checkpoint:", ckpt_dir)
print("config:", config_path)
print("device:", device)


with open(config_path, "r") as f:
    cfg = dict_to_namespace(json.load(f))


# defaults needed by make_dataset
if not hasattr(cfg.dataset, "image_transforms"):
    cfg.dataset.image_transforms = SimpleNamespace(enable=False)
if not hasattr(cfg.dataset, "revision"):
    cfg.dataset.revision = None
if not hasattr(cfg.dataset, "root"):
    cfg.dataset.root = None
if not hasattr(cfg.dataset, "episodes"):
    cfg.dataset.episodes = None
if not hasattr(cfg.dataset, "streaming"):
    cfg.dataset.streaming = False
if not hasattr(cfg.dataset, "video_backend"):
    cfg.dataset.video_backend = "torchcodec"
if not hasattr(cfg.dataset, "use_imagenet_stats"):
    cfg.dataset.use_imagenet_stats = False
if not hasattr(cfg.policy, "observation_delta_indices"):
    cfg.policy.observation_delta_indices = None
if not hasattr(cfg.policy, "action_delta_indices"):
    cfg.policy.action_delta_indices = None
if not hasattr(cfg.policy, "reward_delta_indices"):
    cfg.policy.reward_delta_indices = None
if not hasattr(cfg, "tolerance_s"):
    cfg.tolerance_s = 1e-4
if not hasattr(cfg, "num_workers"):
    cfg.num_workers = 0


dataset = make_dataset(cfg)


print("\n================================")
print("Dataset info")
print("================================")
print("dataset length:", len(dataset))

sample0 = dataset[0]
print("sample keys:", sample0.keys())
print("action shape:", sample0["action"].shape)
print("dataset features:", sorted(dataset.meta.features.keys()))

if hasattr(dataset.meta, "fps"):
    print("dataset fps:", dataset.meta.fps)


print("\n================================")
print("Loading policy")
print("================================")

policy = PI05Policy.from_pretrained(ckpt_dir)
policy.config.compile_model = False
policy.config.device = device
policy.to(device)
policy.eval()
attention_logger = None
if enable_attention_log:
    attention_logger = PI05AttentionLogger(
        output_dir=attention_log_dir,
        layers=attention_layers,
        save_heatmaps=True,
        save_debug_artifacts=save_debug_artifacts,
        save_overlays=save_overlays,
        overlay_layers=overlay_layers,
        overlay_denoise_steps=overlay_denoise_steps,
        overlay_image_names=overlay_image_names,
    )
    attention_logger.register(policy)
    print("Attention logging enabled:", attention_log_dir)

print("Loaded policy from:", ckpt_dir)
print_policy_features(policy)
image_feature_names = list(policy.config.image_features.keys())
print("\nImage token mapping:")
for image_idx, image_key in enumerate(image_feature_names):
    print(f"  image_{image_idx}: {image_key}")
validate_policy_features(policy, sample0, dataset.meta.stats)

preprocessor, postprocessor = make_pi05_pre_post_processors(
    config=policy.config,
    dataset_stats=dataset.meta.stats,
)


# =========================
# Collect episode indices
# =========================

episode_to_indices = {}

for idx in range(len(dataset)):
    sample = dataset[idx]
    ep = get_episode_index(sample)
    episode_to_indices.setdefault(ep, []).append(idx)

all_episodes = sorted(episode_to_indices.keys())

print("\n================================")
print("Episode info")
print("================================")
print("num episodes:", len(all_episodes))
print("first episodes:", all_episodes[:10])

random.seed(random_seed)

if len(all_episodes) <= num_episodes_to_plot:
    target_episodes = all_episodes
else:
    target_episodes = random.sample(all_episodes, num_episodes_to_plot)

print("selected episodes:", target_episodes)


# =========================
# Evaluate selected episodes
# =========================

for ep in target_episodes:
    print("\n================================")
    print(f"Evaluating episode {ep}")
    print("================================")

    # Important: reset only once at the beginning of each episode
    if hasattr(policy, "reset"):
        policy.reset()
        print("Policy reset once at episode start.")

    episode_gt = []
    episode_pred = []

    indices = episode_to_indices[ep]
    overlay_local_ts = {0, len(indices) // 2, len(indices) - 1}
    print("Attention frame stride:", attention_frame_stride)
    print("Overlay local_t values:", sorted(overlay_local_ts))

    for local_t, idx in enumerate(indices):
        sample = dataset[idx]
        should_log_attention = (
            attention_logger is not None
            and (local_t % attention_frame_stride == 0 or local_t in overlay_local_ts)
        )

        gt, pred = predict_one_sample(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            sample=sample,
            device=device,
            attention_logger=attention_logger if should_log_attention else None,
            episode=ep,
            local_t=local_t,
            sample_idx=idx,
            save_overlays=local_t in overlay_local_ts,
        )

        if local_t == 0:
            print("First GT:")
            print(gt)
            print("First Pred:")
            print(pred)
            print("GT shape:", gt.shape)
            print("Pred shape:", pred.shape)

        if pred.shape != gt.shape:
            print("WARNING: pred / gt shape mismatch")
            print("idx:", idx)
            print("gt shape:", gt.shape)
            print("pred shape:", pred.shape)
            continue

        episode_gt.append(gt)
        episode_pred.append(pred)

    episode_gt = torch.stack(episode_gt)
    episode_pred = torch.stack(episode_pred)

    print("episode_gt shape:", episode_gt.shape)
    print("episode_pred shape:", episode_pred.shape)

    print("GT mean:", episode_gt.mean(dim=0))
    print("Pred mean:", episode_pred.mean(dim=0))
    print("GT std:", episode_gt.std(dim=0))
    print("Pred std:", episode_pred.std(dim=0))

    action_dim = episode_gt.shape[1]

    fig, axes = plt.subplots(action_dim, 1, figsize=(16, 3 * action_dim), sharex=True)

    if action_dim == 1:
        axes = [axes]

    for d in range(action_dim):
        axes[d].plot(episode_gt[:, d].numpy(), label=f"GT dim {d}")
        axes[d].plot(episode_pred[:, d].numpy(), "--", label=f"Pred dim {d}")
        axes[d].set_ylabel(f"action {d}")
        axes[d].grid(True)
        axes[d].legend(loc="upper right")

    axes[-1].set_xlabel("Timestep within episode")
    fig.suptitle(f"Open-loop GT vs Pred - Episode {ep}", fontsize=16)
    plt.tight_layout()

    save_path = os.path.join(
        plot_dir,
        f"episode_{ep}_gt_vs_pred_8dims_reset_once.png"
    )
    plt.savefig(save_path, dpi=200)
    plt.close()

    print("Saved:", save_path)


print("\nDone.")
print("All plots saved to:", plot_dir)
if attention_logger is not None:
    attention_logger.close()
    print("Attention logs saved to:", attention_log_dir)

plot_attention_summary(
    os.path.join(attention_log_dir, "feature_scores.jsonl"),
    summary_plot_dir,
    num_inference_steps=policy.config.num_inference_steps,
    fps=getattr(dataset.meta, "fps", None),
    time_plot_layer=time_plot_layer,
)

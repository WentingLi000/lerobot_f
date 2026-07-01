import json
import os
import random
from pathlib import Path
from types import SimpleNamespace

import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lerobot.datasets.factory import make_dataset
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors


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


def predict_one_sample(policy, preprocessor, postprocessor, sample, device):
    processed_batch = preprocessor(sample)
    processed_batch = move_to_device(processed_batch, device)

    with torch.no_grad():
        pred_action = policy.predict_action_chunk(processed_batch)[:, 0]

    pred_action = postprocessor(pred_action)

    pred = pred_action.squeeze(0).detach().to(torch.float32).cpu()
    gt = sample["action"].detach().to(torch.float32).cpu()

    if pred.ndim == 2 and gt.ndim == 1:
        pred = pred[0]
    if pred.ndim == 1 and gt.ndim == 2:
        gt = gt[0]

    return gt, pred


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

# ckpt_dir = "outputs/train/pi05_lemon_bowl_224_pGbs8_20k_accu_4_t1/checkpoints/020000/pretrained_model"
ckpt_dir = "outputs/train/pi05_lemon_bowl_224_pGbs8_20k_accu_4_t1/checkpoints/002500/pretrained_model"
# ckpt_dir = "outputs/train/pi05_lemon_bowl_pGbs4_1k_t2/checkpoints/001000/pretrained_model"
# ckpt_dir = "outputs/train/pi05_lemon_bowl_pGbs8_20k_accu_4_t2/checkpoints/010000/pretrained_model"

# config_path = "configs/train_pi05_lemon_bowl_25hz_wrist_side_interactive.json"
config_path = "configs/train_pi05_lemon_bowl_25hz_wrist_side_latest.json"

# plot_dir = "openloop_eval/pi05_lemon_bowl_pGbs8_20k_accu_4_t2"
plot_dir = "openloop_eval/pi05_lemon_bowl_224_844_2_5k"

os.makedirs(plot_dir, exist_ok=True)

num_episodes_to_plot = 3
random_seed = 42

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

print("Loaded policy from:", ckpt_dir)
print_policy_features(policy)
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

    for local_t, idx in enumerate(indices):
        sample = dataset[idx]

        gt, pred = predict_one_sample(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            sample=sample,
            device=device,
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

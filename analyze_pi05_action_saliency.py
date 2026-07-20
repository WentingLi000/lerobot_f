"""Gradient×input saliency for selected PI05 action positions and x/y/z dimensions.

This is deliberately separate from open-loop evaluation. It enables gradients
through PI05 sampling without updating model parameters and uses fixed noise so
that maps differ because of the selected target, not stochastic sampling.
"""

import json
from pathlib import Path

import matplotlib
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lerobot.datasets.factory import make_dataset
from lerobot.policies.pi05.modeling_pi05_attention import PI05Policy
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from eval_openloop_attention_n_step import (
    CKPT_DIR,
    CONFIG_PATH,
    REPO_ROOT,
    configure_dataset_defaults,
    deterministic_noise,
    dict_to_namespace,
    move_to_device,
    resolve_path,
)


OUTPUT_DIR = "openloop_eval/pi05_lemon_bowl_007500_gradient_saliency"
SAMPLE_INDEX = 0
ACTION_POSITIONS = [0, 10, 20]
ACTION_DIMENSIONS = {"x": 0, "y": 1, "z": 2}
NUM_INFERENCE_STEPS = 10


def image_to_display(image):
    image = image.detach().cpu()[0]
    if image.shape[0] in (1, 3, 4):
        image = image[:3]
    else:
        image = image[..., :3].permute(2, 0, 1)
    image = image.float()
    if image.max() > 1:
        image = image / 255.0
    if image.min() < 0:
        image = (image + 1) / 2
    return image.clamp(0, 1)


def save_overlay(image, saliency, path, title):
    display = image_to_display(image)
    saliency = saliency.detach().float().cpu()
    saliency -= saliency.min()
    if saliency.max() > 0:
        saliency /= saliency.max()
    saliency = F.interpolate(
        saliency[None, None], size=display.shape[-2:], mode="bilinear", align_corners=False
    )[0, 0]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(display.permute(1, 2, 0).numpy())
    im = ax.imshow(saliency.numpy(), cmap="magma", alpha=0.48, vmin=0, vmax=1)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def differentiable_sample(policy, batch, noise):
    """Call the undecorated sample_actions implementation to retain autograd."""
    images, image_masks = policy._preprocess_images(batch)
    tokens = batch[OBS_LANGUAGE_TOKENS]
    masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
    wrapped = getattr(policy.model.sample_actions, "__wrapped__", None)
    if wrapped is None:
        raise RuntimeError("sample_actions no longer exposes __wrapped__; add a gradient-enabled analysis path")
    return wrapped(
        policy.model,
        images,
        image_masks,
        tokens,
        masks,
        noise=noise,
        num_steps=NUM_INFERENCE_STEPS,
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(OUTPUT_DIR)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=False)

    with resolve_path(CONFIG_PATH).open("r", encoding="utf-8") as handle:
        cfg = dict_to_namespace(json.load(handle))
    configure_dataset_defaults(cfg)
    dataset = make_dataset(cfg)
    policy = PI05Policy.from_pretrained(resolve_path(CKPT_DIR))
    policy.config.compile_model = False
    policy.config.device = device
    policy.to(device).eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)

    preprocessor, _ = make_pi05_pre_post_processors(
        config=policy.config, dataset_stats=dataset.meta.stats
    )
    sample = dataset[SAMPLE_INDEX]
    batch = move_to_device(preprocessor(sample), device)
    image_keys = list(policy.config.image_features.keys())
    for key in image_keys:
        # Dataset images may be uint8; gradients require a floating-point leaf.
        batch[key] = batch[key].detach().to(torch.float32).requires_grad_(True)

    noise = deterministic_noise(policy, episode=0, local_t=SAMPLE_INDEX, device=device)
    actions = differentiable_sample(policy, batch, noise)
    metadata = {
        "sample_index": SAMPLE_INDEX,
        "action_positions": ACTION_POSITIONS,
        "action_dimensions": ACTION_DIMENSIONS,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "target_space": "normalized PI05 action output before postprocessing",
        "method": "absolute gradient times input, summed over RGB channels",
    }

    saved = {}
    targets = [(position, name, dim) for position in ACTION_POSITIONS for name, dim in ACTION_DIMENSIONS.items()]
    for target_idx, (position, dim_name, dim) in enumerate(targets):
        if position >= actions.shape[1] or dim >= actions.shape[2]:
            raise ValueError(f"Invalid target action[{position}, {dim}] for shape {tuple(actions.shape)}")
        gradients = torch.autograd.grad(
            actions[0, position, dim],
            [batch[key] for key in image_keys],
            retain_graph=target_idx < len(targets) - 1,
            allow_unused=True,
        )
        for key, gradient in zip(image_keys, gradients, strict=True):
            if gradient is None:
                saliency = torch.zeros(batch[key].shape[-2:], device="cpu")
            else:
                attribution = (gradient * batch[key]).abs()[0]
                if attribution.shape[0] in (1, 3, 4):
                    saliency = attribution[:3].sum(dim=0)
                else:
                    saliency = attribution[..., :3].sum(dim=-1)
            camera = key.replace(".", "_")
            name = f"position_{position:02d}_{dim_name}_{camera}"
            save_overlay(
                batch[key],
                saliency,
                output_dir / f"{name}.png",
                f"Gradient×Input | position={position}, dim={dim_name}, camera={key}",
            )
            torch.save(saliency.detach().cpu(), output_dir / f"{name}.pt")
            saved[name] = float(saliency.sum().item())

    metadata["total_attribution_by_map"] = saved
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print("Saved gradient saliency to:", output_dir)


if __name__ == "__main__":
    main()

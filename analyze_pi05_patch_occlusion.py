"""Causal patch-occlusion maps for selected PI05 action outputs.

Each image patch is replaced by that camera's mean color, PI05 is rerun with
identical noise, and the absolute change of action[position, dimension] is used
as the occlusion score. This validates attention/saliency maps by intervention.
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

from eval_openloop_attention_n_step import (
    CKPT_DIR,
    CONFIG_PATH,
    REPO_ROOT,
    configure_dataset_defaults,
    dict_to_namespace,
    move_to_device,
    predict_chunk,
    resolve_path,
)


OUTPUT_DIR = "openloop_eval/pi05_lemon_bowl_007500_patch_occlusion"
SAMPLE_INDEX = 0
GRID_ROWS = 4
GRID_COLS = 4
ACTION_POSITIONS = [0, 10, 20]
ACTION_DIMENSIONS = {"x": 0, "y": 1, "z": 2}


def image_layout(image):
    if image.ndim != 3:
        raise ValueError(f"Expected an unbatched image, got {tuple(image.shape)}")
    if image.shape[0] in (1, 3, 4):
        return "chw", image.shape[1], image.shape[2]
    if image.shape[-1] in (1, 3, 4):
        return "hwc", image.shape[0], image.shape[1]
    raise ValueError(f"Cannot determine image layout from {tuple(image.shape)}")


def occlude_patch(image, row, col):
    result = image.clone()
    layout, height, width = image_layout(image)
    y0, y1 = row * height // GRID_ROWS, (row + 1) * height // GRID_ROWS
    x0, x1 = col * width // GRID_COLS, (col + 1) * width // GRID_COLS
    if layout == "chw":
        fill = image.to(torch.float32).mean(dim=(1, 2), keepdim=True).to(image.dtype)
        result[:, y0:y1, x0:x1] = fill
    else:
        fill = image.to(torch.float32).mean(dim=(0, 1), keepdim=True).to(image.dtype)
        result[y0:y1, x0:x1, :] = fill
    return result


def image_to_display(image):
    layout, _, _ = image_layout(image)
    display = image[:3] if layout == "chw" else image[..., :3].permute(2, 0, 1)
    display = display.detach().cpu().float()
    if display.max() > 1:
        display /= 255.0
    if display.min() < 0:
        display = (display + 1) / 2
    return display.clamp(0, 1)


def save_overlay(image, scores, path, title):
    display = image_to_display(image)
    normalized = scores.detach().cpu().float()
    normalized -= normalized.min()
    if normalized.max() > 0:
        normalized /= normalized.max()
    heatmap = F.interpolate(
        normalized[None, None], size=display.shape[-2:], mode="nearest"
    )[0, 0]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(display.permute(1, 2, 0).numpy())
    im = ax.imshow(heatmap.numpy(), cmap="magma", alpha=0.48, vmin=0, vmax=1)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="absolute action change")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


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
    preprocessor, postprocessor = make_pi05_pre_post_processors(
        config=policy.config, dataset_stats=dataset.meta.stats
    )

    sample = dataset[SAMPLE_INDEX]
    image_keys = list(policy.config.image_features.keys())
    baseline = predict_chunk(
        policy,
        preprocessor,
        postprocessor,
        sample,
        device,
        episode=0,
        local_t=SAMPLE_INDEX,
        sample_idx=SAMPLE_INDEX,
        attention_logger=None,
    )
    score_maps = {
        (key, position, name): torch.zeros(GRID_ROWS, GRID_COLS)
        for key in image_keys
        for position in ACTION_POSITIONS
        for name in ACTION_DIMENSIONS
    }

    for key in image_keys:
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                occluded_sample = dict(sample)
                occluded_sample[key] = occlude_patch(sample[key], row, col)
                occluded = predict_chunk(
                    policy,
                    preprocessor,
                    postprocessor,
                    occluded_sample,
                    device,
                    episode=0,
                    local_t=SAMPLE_INDEX,
                    sample_idx=SAMPLE_INDEX,
                    attention_logger=None,
                )
                for position in ACTION_POSITIONS:
                    for dim_name, dim in ACTION_DIMENSIONS.items():
                        score_maps[(key, position, dim_name)][row, col] = abs(
                            occluded[position, dim] - baseline[position, dim]
                        )

    totals = {}
    for (key, position, dim_name), scores in score_maps.items():
        camera = key.replace(".", "_")
        name = f"position_{position:02d}_{dim_name}_{camera}"
        save_overlay(
            sample[key],
            scores,
            output_dir / f"{name}.png",
            f"Patch occlusion | position={position}, dim={dim_name}, camera={key}",
        )
        torch.save(scores, output_dir / f"{name}.pt")
        totals[name] = float(scores.sum().item())

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "sample_index": SAMPLE_INDEX,
                "grid": [GRID_ROWS, GRID_COLS],
                "action_positions": ACTION_POSITIONS,
                "action_dimensions": ACTION_DIMENSIONS,
                "occlusion_value": "per-camera mean color",
                "score": "absolute change in postprocessed action output with fixed noise",
                "total_occlusion_score_by_map": totals,
            },
            handle,
            indent=2,
        )
    print("Saved patch occlusion maps to:", output_dir)


if __name__ == "__main__":
    main()

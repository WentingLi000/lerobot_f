from safetensors.torch import load_file
import torch

base_path = "/hkfs/work/workspace/scratch/ujzmd-lerobot/hf_cache/models/lerobot/pi05_base/model.safetensors"
ft_path = "outputs/train/你的run名/checkpoints/001000/pretrained_model/model.safetensors"

base = load_file(base_path, device="cpu")
ft = load_file(ft_path, device="cpu")

groups = {
    "action_proj": [],
    "gemma_expert": [],
    "paligemma_language": [],
    "vision_tower": [],
    "multi_modal_projector": [],
    "other": [],
}

for k in ft.keys():
    if k not in base:
        continue

    diff = (ft[k].float() - base[k].float()).abs().mean().item()

    if "action_in_proj" in k or "action_out_proj" in k:
        groups["action_proj"].append(diff)
    elif "gemma_expert" in k:
        groups["gemma_expert"].append(diff)
    elif "vision_tower" in k:
        groups["vision_tower"].append(diff)
    elif "multi_modal_projector" in k:
        groups["multi_modal_projector"].append(diff)
    elif "language_model" in k:
        groups["paligemma_language"].append(diff)
    else:
        groups["other"].append(diff)

for name, vals in groups.items():
    if len(vals) == 0:
        print(name, "no tensors")
    else:
        print(
            name,
            "num_tensors=", len(vals),
            "mean_diff=", sum(vals) / len(vals),
            "max_diff=", max(vals),
        )
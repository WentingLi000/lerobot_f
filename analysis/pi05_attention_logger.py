import json
import math
import re
from pathlib import Path

import torch
import torch.nn.functional as F

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class PI05AttentionLogger:
    """Hook-based Q/K/V logger for PI05 attention debugging."""

    _QKV_RE = re.compile(r"layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj)$")
    _SELF_ATTN_RE = re.compile(r"layers\.(\d+)\.self_attn$")

    def __init__(
        self,
        output_dir,
        layers=None,
        save_heatmaps=True,
        save_debug_artifacts=True,
        save_overlays=True,
        overlay_layers=None,
        overlay_denoise_steps=None,
        overlay_image_names=None,
    ):
        self.output_dir = Path(output_dir)
        self.layers = None if layers is None else {int(layer) for layer in layers}
        self.save_heatmaps = save_heatmaps
        self.save_debug_artifacts = save_debug_artifacts
        self.save_overlays = save_overlays
        self.overlay_layers = None if overlay_layers is None else {int(layer) for layer in overlay_layers}
        self.overlay_denoise_steps = (
            None if overlay_denoise_steps is None else {int(step) for step in overlay_denoise_steps}
        )
        self.overlay_image_names = None if overlay_image_names is None else {str(name) for name in overlay_image_names}
        self.handles = []
        self.self_attn_modules = {}
        self.current_event = None
        self.current_stage = None
        self.current_denoise_step = None
        self.current_time = None
        self.current_save_overlays = True
        self.prefix_layout = []
        self.action_token_count = None
        self.image_feature_names = []
        self.current_images = {}
        self.records = []
        self.step_index_path = self.output_dir / "events.jsonl"
        self.feature_scores_path = self.output_dir / "feature_scores.jsonl"
        self.camera_reliance_path = self.output_dir / "camera_reliance.jsonl"
        self._modeling_gemma = None
        self._original_eager_attention_forward = None

    def register(self, policy):
        self.close()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.self_attn_modules = {}
        self.image_feature_names = list(getattr(policy.config, "image_features", {}).keys())
        self._save_image_mapping()
        for name, module in policy.named_modules():
            attn_match = self._SELF_ATTN_RE.search(name)
            if attn_match is not None:
                layer_idx = int(attn_match.group(1))
                if self.layers is None or layer_idx in self.layers:
                    self.self_attn_modules[id(module)] = {
                        "module": name,
                        "layer_idx": layer_idx,
                    }

            match = self._QKV_RE.search(name)
            if match is None:
                continue
            layer_idx = int(match.group(1))
            if self.layers is not None and layer_idx not in self.layers:
                continue
            self.handles.append(module.register_forward_hook(self._make_hook(name, layer_idx, match.group(2))))
        self._patch_eager_attention_forward()
        if hasattr(policy, "model"):
            policy.model.attention_logger = self

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        if self._modeling_gemma is not None and self._original_eager_attention_forward is not None:
            self._modeling_gemma.eager_attention_forward = self._original_eager_attention_forward
        self._modeling_gemma = None
        self._original_eager_attention_forward = None

    def begin_step(self, *, episode, local_t, sample_idx, task=None, images=None, save_overlays=None):
        self.current_event = {
            "episode": int(episode),
            "local_t": int(local_t),
            "sample_idx": int(sample_idx),
            "task": _to_jsonable(task),
        }
        self.current_save_overlays = self.save_overlays if save_overlays is None else bool(save_overlays)
        self.current_images = _clone_images(images or {})
        self.records = []

    def set_stage(self, stage=None, denoise_step=None, time=None):
        self.current_stage = stage
        self.current_denoise_step = None if denoise_step is None else int(denoise_step)
        self.current_time = None if time is None else float(time)

    def set_token_layout(self, *, prefix_layout, action_token_count):
        self.prefix_layout = [
            {
                "name": str(item["name"]),
                "start": int(item["start"]),
                "end": int(item["end"]),
            }
            for item in prefix_layout
            if int(item["end"]) > int(item["start"])
        ]
        self.action_token_count = int(action_token_count)

    def end_step(self):
        if self.current_event is None:
            return

        step_dir = self.output_dir / f"episode_{self.current_event['episode']:04d}" / (
            f"step_{self.current_event['local_t']:05d}"
        )
        step_dir.mkdir(parents=True, exist_ok=True)

        tensor_path = None
        if self.save_debug_artifacts:
            tensor_path = step_dir / "qkv_records.pt"
            torch.save({"event": self.current_event, "records": self.records}, tensor_path)

        heatmap_paths = []
        if self.save_heatmaps and (self.save_debug_artifacts or self.current_save_overlays):
            heatmap_paths = self._save_heatmaps(step_dir)
        feature_scores = self._save_feature_scores()
        camera_reliance = self._save_camera_reliance(feature_scores)

        event = {
            **self.current_event,
            "image_mapping": self._image_mapping(),
            "num_qkv_records": len(self.records),
            "tensor_path": None if tensor_path is None else str(tensor_path),
            "heatmap_paths": [str(path) for path in heatmap_paths],
            "num_feature_scores": len(feature_scores),
            "num_camera_reliance": len(camera_reliance),
        }
        with self.step_index_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(event) + "\n")

        self.current_event = None
        self.set_stage()
        self.current_save_overlays = True
        self.current_images = {}
        self.records = []

    def _patch_eager_attention_forward(self):
        from transformers.models.gemma import modeling_gemma

        if self._original_eager_attention_forward is not None:
            return

        self._modeling_gemma = modeling_gemma
        self._original_eager_attention_forward = modeling_gemma.eager_attention_forward
        original_forward = self._original_eager_attention_forward

        def wrapped_eager_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            scaling,
            *args,
            **kwargs,
        ):
            self._record_real_attention(module, query, key, value, attention_mask, scaling)
            return original_forward(
                module,
                query,
                key,
                value,
                attention_mask,
                scaling,
                *args,
                **kwargs,
            )

        modeling_gemma.eager_attention_forward = wrapped_eager_attention_forward

    def _record_real_attention(self, module, query, key, value, attention_mask, scaling):
        if self.current_event is None:
            return
        module_info = self.self_attn_modules.get(id(module))
        if module_info is None:
            return
        self.records.append(
            {
                **module_info,
                "kind": "real_attention",
                "stage": self.current_stage,
                "denoise_step": self.current_denoise_step,
                "time": self.current_time,
                "scaling": _to_jsonable(scaling),
                "q_shape": tuple(query.shape),
                "k_shape": tuple(key.shape),
                "v_shape": tuple(value.shape),
                "mask_shape": None if attention_mask is None else tuple(attention_mask.shape),
                "q": query.detach().to("cpu", dtype=torch.float32),
                "k": key.detach().to("cpu", dtype=torch.float32),
                "v": value.detach().to("cpu", dtype=torch.float32),
                "attention_mask": (
                    None
                    if attention_mask is None
                    else attention_mask.detach().to("cpu", dtype=torch.float32)
                ),
            }
        )

    def _make_hook(self, module_name, layer_idx, qkv_name):
        def hook(_module, _inputs, output):
            if self.current_event is None:
                return
            tensor = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(tensor):
                return
            self.records.append(
                {
                    "module": module_name,
                    "layer_idx": layer_idx,
                    "kind": qkv_name[0],
                    "stage": self.current_stage,
                    "denoise_step": self.current_denoise_step,
                    "time": self.current_time,
                    "shape": tuple(tensor.shape),
                    "tensor": tensor.detach().to("cpu", dtype=torch.float32),
                }
            )

        return hook

    def _save_heatmaps(self, step_dir):
        paths = []
        for index, record in enumerate(self.records):
            if record["kind"] != "real_attention":
                continue
            heatmap = _real_attention_heatmap(
                record["q"],
                record["k"],
                record["attention_mask"],
                record["scaling"],
            )
            if heatmap is None:
                continue
            if self.save_debug_artifacts:
                safe_name = record["module"].replace(".", "_")
                stage = record.get("stage") or "unknown"
                denoise_step = record.get("denoise_step")
                step_suffix = "prefix" if denoise_step is None else f"denoise_{denoise_step:02d}"
                path = step_dir / f"{index:04d}_{stage}_{step_suffix}_{safe_name}_real_attention.png"
                _plot_heatmap(heatmap, path, title=f"{record['module']} {stage} {step_suffix}")
                paths.append(path)
            paths.extend(self._save_denoise_camera_overlays(record, heatmap, step_dir, index))

        if not self.save_debug_artifacts:
            return paths

        grouped = {}
        for record in self.records:
            if record["kind"] == "real_attention":
                continue
            key = record["module"].rsplit(".", 1)[0]
            grouped.setdefault(key, {})[record["kind"]] = record

        for key, group in grouped.items():
            if "q" not in group or "k" not in group:
                continue
            heatmap = _attention_heatmap(group["q"]["tensor"], group["k"]["tensor"])
            if heatmap is None:
                continue
            safe_name = key.replace(".", "_")
            path = step_dir / f"{safe_name}_qk_heatmap.png"
            _plot_heatmap(heatmap, path, title=key)
            paths.append(path)
        return paths

    def _save_feature_scores(self):
        rows = []
        for record in self.records:
            if record["kind"] != "real_attention":
                continue
            heatmap = _real_attention_heatmap(
                record["q"],
                record["k"],
                record["attention_mask"],
                record["scaling"],
            )
            if heatmap is None:
                continue
            query_layout, key_layout = self._layouts_for_record(record, heatmap)
            if not query_layout or not key_layout:
                continue
            scores = _aggregate_feature_scores(heatmap, query_layout, key_layout)
            if not scores:
                continue
            row = {
                **self.current_event,
                "module": record["module"],
                "layer_idx": record["layer_idx"],
                "stage": record.get("stage"),
                "denoise_step": record.get("denoise_step"),
                "time": record.get("time"),
                "q_len": int(heatmap.shape[0]),
                "k_len": int(heatmap.shape[1]),
                "query_layout": query_layout,
                "key_layout": key_layout,
                "scores": scores,
            }
            per_head = _real_attention_by_head(
                record["q"],
                record["k"],
                record["attention_mask"],
                record["scaling"],
            )
            if per_head is not None:
                row["per_head_scores"] = [
                    _aggregate_feature_scores(head_heatmap, query_layout, key_layout)
                    for head_heatmap in per_head
                ]
            rows.append(row)

        if rows:
            with self.feature_scores_path.open("a", encoding="utf-8") as fp:
                for row in rows:
                    fp.write(json.dumps(row) + "\n")
        return rows

    def _save_camera_reliance(self, feature_scores):
        rows = []
        for row in feature_scores:
            if row.get("stage") != "denoise":
                continue
            scores = row.get("scores", {})
            image_scores = {}
            for image_item in self._image_layout_items(row.get("key_layout", [])):
                key = f"action_tokens->{image_item['name']}"
                if key not in scores:
                    continue
                token_count = int(image_item["end"]) - int(image_item["start"])
                image_scores[image_item["name"]] = {
                    "camera_key": self._camera_key_for_image_name(image_item["name"]),
                    "mean_attention": float(scores[key]),
                    "token_count": token_count,
                    "attention_mass": float(scores[key]) * token_count,
                }
            if not image_scores:
                continue
            out_row = {
                **self.current_event,
                "module": row["module"],
                "layer_idx": row["layer_idx"],
                "denoise_step": row.get("denoise_step"),
                "time": row.get("time"),
                "image_mapping": self._image_mapping(),
                "image_scores": image_scores,
            }
            rows.append(out_row)

        if rows:
            with self.camera_reliance_path.open("a", encoding="utf-8") as fp:
                for row in rows:
                    fp.write(json.dumps(row) + "\n")
        return rows

    def _save_denoise_camera_overlays(self, record, heatmap, step_dir, record_index):
        if not self.current_save_overlays:
            return []
        if record.get("stage") != "denoise":
            return []
        if self.overlay_layers is not None and int(record["layer_idx"]) not in self.overlay_layers:
            return []
        denoise_step = record.get("denoise_step")
        if self.overlay_denoise_steps is not None:
            if denoise_step is None or int(denoise_step) not in self.overlay_denoise_steps:
                return []

        query_layout, key_layout = self._layouts_for_record(record, heatmap)
        query_item = next((item for item in query_layout if item["name"] == "action_tokens"), None)
        if query_item is None:
            return []

        q0, q1 = query_item["start"], query_item["end"]
        if q1 <= q0:
            return []

        paths = []
        step_suffix = "unknown" if denoise_step is None else f"{int(denoise_step):02d}"

        for image_item in self._image_layout_items(key_layout):
            image_name = image_item["name"]
            if self.overlay_image_names is not None and image_name not in self.overlay_image_names:
                continue
            k0, k1 = image_item["start"], image_item["end"]
            token_count = k1 - k0
            grid_size = int(math.sqrt(token_count))
            if grid_size * grid_size != token_count:
                continue

            camera_key = self._camera_key_for_image_name(image_name)
            raw_image = self.current_images.get(camera_key)
            if raw_image is None:
                continue

            patch_scores = heatmap[q0:q1, k0:k1].mean(dim=0).reshape(grid_size, grid_size)
            safe_module = record["module"].replace(".", "_")
            safe_camera = camera_key.replace(".", "_").replace("/", "_")
            out_path = step_dir / (
                f"{record_index:04d}_denoise_{step_suffix}_layer_{record['layer_idx']}_"
                f"{image_name}_{safe_camera}_action_attention_overlay.png"
            )
            _plot_patch_overlay(raw_image, patch_scores, out_path, title=f"{camera_key} {safe_module}")
            paths.append(out_path)

            tensor_path = out_path.with_suffix(".pt")
            torch.save(
                {
                    "camera_key": camera_key,
                    "image_name": image_name,
                    "layer_idx": record["layer_idx"],
                    "stage": record.get("stage"),
                    "denoise_step": denoise_step,
                    "patch_scores": patch_scores.detach().cpu(),
                },
                tensor_path,
            )
            paths.append(tensor_path)

        return paths

    def _layouts_for_record(self, record, heatmap):
        q_len = int(heatmap.shape[0])
        k_len = int(heatmap.shape[1])
        stage = record.get("stage")
        if stage == "prefix":
            prefix_layout = _clamp_layout(self.prefix_layout, q_len)
            return prefix_layout, _clamp_layout(self.prefix_layout, k_len)

        if stage == "denoise":
            action_count = self.action_token_count or q_len
            query_layout = [{"name": "action_tokens", "start": 0, "end": min(action_count, q_len)}]
            key_layout = _clamp_layout(self.prefix_layout, k_len)
            suffix_start = max(0, k_len - action_count)
            if suffix_start < k_len:
                key_layout.append({"name": "action_tokens", "start": suffix_start, "end": k_len})
            return query_layout, key_layout

        return [], []

    def _image_mapping(self):
        return {
            f"image_{idx}": key
            for idx, key in enumerate(self.image_feature_names)
        }

    def _save_image_mapping(self):
        if not self.image_feature_names:
            return
        path = self.output_dir / "image_mapping.json"
        with path.open("w", encoding="utf-8") as fp:
            json.dump(self._image_mapping(), fp, indent=2)

    def _camera_key_for_image_name(self, image_name):
        match = re.fullmatch(r"image_(\d+)", image_name)
        if match is None:
            return image_name
        image_idx = int(match.group(1))
        if 0 <= image_idx < len(self.image_feature_names):
            return self.image_feature_names[image_idx]
        return image_name

    def _image_layout_items(self, layout):
        return [item for item in layout if re.fullmatch(r"image_\d+", item["name"]) is not None]


def _real_attention_heatmap(q_tensor, k_tensor, attention_mask, scaling):
    per_head = _real_attention_by_head(q_tensor, k_tensor, attention_mask, scaling)
    if per_head is None:
        return None
    return per_head.mean(dim=0)


def _real_attention_by_head(q_tensor, k_tensor, attention_mask, scaling):
    """Return real masked attention as [num_query_heads, q_len, k_len]."""
    if q_tensor.ndim != 4 or k_tensor.ndim != 4:
        return None
    if q_tensor.shape[0] != 1 or k_tensor.shape[0] != 1:
        return None
    if k_tensor.shape[1] == 1 and q_tensor.shape[1] > 1:
        k_tensor = k_tensor.expand(-1, q_tensor.shape[1], -1, -1)
    if q_tensor.shape[1] != k_tensor.shape[1]:
        return None

    scale = float(scaling)
    scores = torch.matmul(q_tensor, k_tensor.transpose(-2, -1)) * scale
    if attention_mask is not None:
        scores = scores + attention_mask
    attn = torch.softmax(scores, dim=-1)
    return attn[0]


def _aggregate_feature_scores(heatmap, query_layout, key_layout):
    scores = {}
    for query_item in query_layout:
        q0, q1 = query_item["start"], query_item["end"]
        if q1 <= q0:
            continue
        for key_item in key_layout:
            k0, k1 = key_item["start"], key_item["end"]
            if k1 <= k0:
                continue
            value = heatmap[q0:q1, k0:k1].mean().item()
            scores[f"{query_item['name']}->{key_item['name']}"] = float(value)
    return scores


def _clamp_layout(layout, max_len):
    clamped = []
    for item in layout:
        start = max(0, min(int(item["start"]), max_len))
        end = max(0, min(int(item["end"]), max_len))
        if end > start:
            clamped.append({"name": item["name"], "start": start, "end": end})
    return clamped


def _attention_heatmap(q_tensor, k_tensor):
    if q_tensor.ndim != 3 or k_tensor.ndim != 3:
        return None
    if q_tensor.shape[0] != 1 or k_tensor.shape[0] != 1:
        return None

    q_width = q_tensor.shape[-1]
    k_width = k_tensor.shape[-1]
    if k_width <= 0 or q_width % k_width != 0:
        return None

    head_dim = k_width
    num_heads = q_width // head_dim
    q = q_tensor.reshape(1, q_tensor.shape[1], num_heads, head_dim).transpose(1, 2)
    k = k_tensor.reshape(1, k_tensor.shape[1], 1, head_dim).transpose(1, 2)
    if k.shape[1] == 1 and num_heads > 1:
        k = k.expand(-1, num_heads, -1, -1)

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
    attn = torch.softmax(scores, dim=-1)
    return attn[0].mean(dim=0)


def _plot_heatmap(heatmap, path, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(heatmap.numpy(), aspect="auto", interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("K token index")
    ax.set_ylabel("Q token index")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _clone_images(images):
    cloned = {}
    for key, value in images.items():
        if torch.is_tensor(value):
            cloned[key] = value.detach().cpu()
    return cloned


def _image_to_chw_float(image):
    image = image.detach().cpu()
    if image.ndim == 4:
        image = image[0]
    if image.ndim != 3:
        return None
    if image.shape[0] in (1, 3, 4):
        chw = image[:3].to(torch.float32)
    elif image.shape[-1] in (1, 3, 4):
        chw = image[..., :3].permute(2, 0, 1).to(torch.float32)
    else:
        return None
    if chw.max() > 1.0:
        chw = chw / 255.0
    if chw.min() < 0.0:
        chw = (chw + 1.0) / 2.0
    return chw.clamp(0.0, 1.0)


def _plot_patch_overlay(raw_image, patch_scores, path, title):
    image = _image_to_chw_float(raw_image)
    if image is None:
        return

    heatmap = patch_scores.detach().cpu().to(torch.float32)
    heatmap = heatmap - heatmap.min()
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    heatmap = F.interpolate(
        heatmap[None, None],
        size=tuple(image.shape[-2:]),
        mode="bilinear",
        align_corners=False,
    )[0, 0]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image.permute(1, 2, 0).numpy())
    im = ax.imshow(heatmap.numpy(), cmap="magma", alpha=0.45, vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=8)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _to_jsonable(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return str(value)

#!/usr/bin/env python

import argparse
import json
from collections import defaultdict
from pathlib import Path


# =========================
# Config
# =========================
#
# Edit these values, then run:
#   python analysis/pi05_feature_scores_to_txt.py
#
# Command-line arguments can still override these config values.

FEATURE_SCORES_PATH = (
    "openloop_eval/pi05_lemon_bowl_dynamic/"
    "pi05_lemon_dynamic_3_cameras_444_7_5k_attention_summary_1/"
    "attention/feature_scores.jsonl"
)
OUTPUT_DIR = None
STAGE = "denoise"  # "denoise", "prefix", or "all"

RAW_OUTPUT_NAME = "mean_action_attention_by_layer_raw.txt"
NORMALIZED_OUTPUT_NAME = "mean_action_attention_by_layer_normalized.txt"
OVERALL_OUTPUT_NAME = "overall_mean_action_attention.txt"


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


def load_rows(feature_scores_path, stage):
    rows = []
    with Path(feature_scores_path).open("r", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            row = json.loads(line)
            if stage is None or row.get("stage") == stage:
                rows.append(row)
    return rows


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


def build_layer_tables(rows):
    modalities = get_attention_modalities(rows)
    layer_values = defaultdict(lambda: defaultdict(list))

    for row in rows:
        masses = get_attention_mass(row, modalities)
        layer_idx = int(row["layer_idx"])
        for modality, value in masses.items():
            layer_values[layer_idx][modality].append(value)

    layers = sorted(layer_values)
    raw_table = []
    normalized_table = []

    for layer_idx in layers:
        raw_values = [
            (
                sum(layer_values[layer_idx][modality]) / len(layer_values[layer_idx][modality])
                if layer_values[layer_idx][modality]
                else 0.0
            )
            for modality in modalities
        ]
        total = sum(raw_values)
        normalized_values = [value / total if total > 0 else 0.0 for value in raw_values]
        raw_table.append((layer_idx, raw_values))
        normalized_table.append((layer_idx, normalized_values))

    return modalities, raw_table, normalized_table


def build_overall_table(raw_table, modalities):
    if not raw_table:
        return []

    sums = [0.0 for _ in modalities]
    for _layer_idx, values in raw_table:
        for idx, value in enumerate(values):
            sums[idx] += value

    means = [value / len(raw_table) for value in sums]
    total = sum(means)
    shares = [value / total if total > 0 else 0.0 for value in means]
    return list(zip(modalities, means, shares))


def write_layer_table(path, modalities, table):
    with Path(path).open("w", encoding="utf-8") as fp:
        fp.write("layer\t" + "\t".join(modalities) + "\n")
        for layer_idx, values in table:
            formatted = "\t".join(f"{value:.10f}" for value in values)
            fp.write(f"{layer_idx}\t{formatted}\n")


def write_overall_table(path, overall_table):
    with Path(path).open("w", encoding="utf-8") as fp:
        fp.write("modality\tmean_mass\tnormalized_share\n")
        for modality, mean_mass, normalized_share in overall_table:
            fp.write(f"{modality}\t{mean_mass:.10f}\t{normalized_share:.10f}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert PI05 feature_scores.jsonl into numeric attention summary txt files."
    )
    parser.add_argument(
        "feature_scores",
        type=Path,
        nargs="?",
        default=Path(FEATURE_SCORES_PATH),
        help="Path to attention/feature_scores.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None if OUTPUT_DIR is None else Path(OUTPUT_DIR),
        help="Directory for txt outputs. Defaults to a sibling attention_summary_txt directory.",
    )
    parser.add_argument(
        "--stage",
        default=STAGE,
        choices=("denoise", "prefix", "all"),
        help="Rows to summarize. The layer action summary usually uses denoise.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    stage = None if args.stage == "all" else args.stage
    feature_scores_path = args.feature_scores
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = feature_scores_path.parent.parent / "attention_summary_txt"

    rows = load_rows(feature_scores_path, stage)
    if not rows:
        raise ValueError(f"No rows found for stage={args.stage}: {feature_scores_path}")

    modalities, raw_table, normalized_table = build_layer_tables(rows)
    if not modalities or not raw_table:
        raise ValueError(f"No action-token attention scores found: {feature_scores_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / RAW_OUTPUT_NAME
    normalized_path = output_dir / NORMALIZED_OUTPUT_NAME
    overall_path = output_dir / OVERALL_OUTPUT_NAME

    write_layer_table(raw_path, modalities, raw_table)
    write_layer_table(normalized_path, modalities, normalized_table)
    write_overall_table(overall_path, build_overall_table(raw_table, modalities))

    print("Saved:", raw_path)
    print("Saved:", normalized_path)
    print("Saved:", overall_path)


if __name__ == "__main__":
    main()

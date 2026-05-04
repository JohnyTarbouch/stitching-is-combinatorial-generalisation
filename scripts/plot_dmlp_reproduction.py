""" 
this code plot navigation reproduction results from Ghugare training

Read runs directories containing files named train_eval_*.log, writes a CSV summary, and
creates a paper style bar plot with mean std div over (1,2,3,4,5) rand seeds.
"""

from __future__ import annotations

import argparse
import ast
import csv
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


EVAL_RE = re.compile(r"\{.*'eval/avg_reward'.*\}")
UPDATE_RE = re.compile(r"num of updates:\s*(\d+)")
SEED_RE = re.compile(r"seed(\d+)")


def parse_log(path: Path) -> list[tuple[int | None, dict[str, Any]]]:
    records: list[tuple[int | None, dict[str, Any]]] = []
    pending: dict[str, Any] | None = None

    for line in path.read_text(errors="replace").splitlines():
        match = EVAL_RE.search(line)
        if match:
            pending = ast.literal_eval(match.group(0))
            continue

        update_match = UPDATE_RE.search(line)
        if update_match and pending is not None:
            records.append((int(update_match.group(1)), pending))
            pending = None

    if pending is not None:
        records.append((None, pending))

    return records


def method_from_log(path: Path) -> str:
    name = f"{path.parent.name}/{path.name}"

    if "rvs_goal_aug" in name or "dmlp_goal_aug" in name:
        return "RvS + only goal augmentation"
    if "dt_goal_aug" in name:
        return "DT + only goal augmentation"
    if "rvs_state_aug" in name or "dmlp_state_aug" in name:
        return "RvS + state augmentation"
    if "dt_state_aug" in name:
        return "DT + state augmentation"
    if "dt_base" in name:
        return "DT"
    if "rvs_base" in name or "dmlp_base" in name or "base_full" in name or "base_seed" in name:
        return "RvS"

    # Backward-compatible labels for the earlier two-run smoke scripts.
    if "aug" in name:
        return "RvS + state augmentation"
    if "base" in name:
        return "RvS"
    return "unknown"


def dataset_from_run_dir(path: Path) -> str:
    name = path.name
    for dataset in ("antmaze-umaze-v0", "antmaze-medium-v0", "antmaze-large-v0",
                    "pointmaze-umaze-v0", "pointmaze-medium-v0", "pointmaze-large-v0"):
        safe = dataset.replace("-", "_")
        if dataset in name or safe in name:
            return dataset

    shorthand = {
        "antmaze_umaze": "antmaze-umaze-v0",
        "antmaze_medium": "antmaze-medium-v0",
        "antmaze_large": "antmaze-large-v0",
        "pointmaze_umaze": "pointmaze-umaze-v0",
        "pointmaze_medium": "pointmaze-medium-v0",
        "pointmaze_large": "pointmaze-large-v0",
    }
    for token, dataset in shorthand.items():
        if token in name:
            return dataset

    for context in path.glob("slurm_context_*.log"):
        for line in context.read_text(errors="replace").splitlines():
            if line.startswith("dataset_name:"):
                return line.split(":", 1)[1].strip()

    return path.name


def seed_from_path(path: Path) -> int:
    for piece in (path.name, path.parent.name):
        match = SEED_RE.search(piece)
        if match:
            return int(match.group(1))
    return 1


def stderr(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(var) / math.sqrt(len(values))


def short_dataset_name(dataset: str) -> str:
    if "umaze" in dataset:
        return "umaze"
    if "medium" in dataset:
        return "medium"
    if "large" in dataset:
        return "large"
    return dataset


def collect_rows(run_dirs: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        if not run_dir.exists():
            raise FileNotFoundError(run_dir)

        dataset = dataset_from_run_dir(run_dir)
        for log_path in sorted(run_dir.glob("train_eval_*.log")):
            records = parse_log(log_path)
            if not records:
                continue

            final_update, final_eval = records[-1]
            best_update, best_eval = max(
                records,
                key=lambda item: float(item[1].get("eval/avg_reward", float("-inf"))),
            )
            rows.append({
                "dataset": dataset,
                "method": method_from_log(log_path),
                "seed": seed_from_path(log_path),
                "updates_final": final_update if final_update is not None else "",
                "final_avg_reward": float(final_eval["eval/avg_reward"]),
                "final_bottom_to_top": float(final_eval.get("eval/bottom_to_top_avg_reward", float("nan"))),
                "final_top_to_bottom": float(final_eval.get("eval/top_to_bottom_avg_reward", float("nan"))),
                "updates_best": best_update if best_update is not None else "",
                "best_avg_reward": float(best_eval["eval/avg_reward"]),
                "log_path": str(log_path),
            })
    return rows


def write_csv(rows: list[dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "method",
        "seed",
        "updates_final",
        "final_avg_reward",
        "final_bottom_to_top",
        "final_top_to_bottom",
        "updates_best",
        "best_avg_reward",
        "log_path",
    ]
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_bars(rows: list[dict[str, Any]], out_png: Path, metric: str) -> None:
    method_order = [
        "RvS",
        "DT",
        "RvS + state augmentation",
        "DT + state augmentation",
        "RvS + only goal augmentation",
        "DT + only goal augmentation",
    ]
    methods = [method for method in method_order if any(row["method"] == method for row in rows)]
    datasets = sorted(
        {row["dataset"] for row in rows},
        key=lambda value: ("umaze" not in value, "medium" not in value, "large" not in value, value),
    )

    x_positions = list(range(len(datasets)))
    width = min(0.82 / max(len(methods), 1), 0.26)
    colors = {
        "RvS": "#f28e2b",
        "DT": "#4e79a7",
        "RvS + state augmentation": "#59a14f",
        "DT + state augmentation": "#8cd17d",
        "RvS + only goal augmentation": "#76b7b2",
        "DT + only goal augmentation": "#b07aa1",
    }

    fig, ax = plt.subplots(figsize=(max(7.0, 1.8 * len(datasets)), 4.0))
    for method_idx, method in enumerate(methods):
        means: list[float] = []
        errors: list[float] = []
        for dataset in datasets:
            values = [
                float(row[metric])
                for row in rows
                if row["dataset"] == dataset and row["method"] == method
            ]
            means.append(sum(values) / len(values) if values else 0.0)
            errors.append(stderr(values))

        offsets = [
            pos + (method_idx - (len(methods) - 1) / 2) * width
            for pos in x_positions
        ]
        ax.bar(
            offsets,
            means,
            width=width,
            yerr=errors,
            capsize=3,
            label=method,
            color=colors[method],
            alpha=0.95,
        )

    title_prefix = "Final" if metric == "final_avg_reward" else "Best"
    ax.set_title(f"{title_prefix} Navigation Reproduction")
    ax.set_ylabel("success rate")
    max_value = max([float(row[metric]) for row in rows] + [0.5])
    ax.set_ylim(0, min(1.0, max(0.55, max_value + 0.15)))
    ax.set_xticks(x_positions, [short_dataset_name(dataset) for dataset in datasets])
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("repro_runs/plots"))
    parser.add_argument(
        "--metric",
        choices=["final_avg_reward", "best_avg_reward"],
        default="final_avg_reward",
        help="Use final checkpoint or best checkpoint values for the bar plot.",
    )
    args = parser.parse_args()

    rows = collect_rows(args.run_dirs)
    if not rows:
        raise SystemExit("No train_eval_*.log files with eval/avg_reward were found.")

    out_csv = args.out_dir / f"dmlp_reproduction_{args.metric}.csv"
    out_png = args.out_dir / f"dmlp_reproduction_{args.metric}.png"
    write_csv(rows, out_csv)
    plot_bars(rows, out_png, args.metric)

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()

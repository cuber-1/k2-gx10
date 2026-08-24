#!/usr/bin/env python3
"""Plot isolated Q6_K scaling and representative Nsight Compute counters."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT = Path(__file__).resolve().parent.parent
DEFAULT_TIMING = PROJECT / "results/q6k-column-sweep/20260817-172654/sweep.csv"
DEFAULT_PROFILES = PROJECT / "profiles/ncu-microbenchmark"

PROFILE_DETAILS = {
    25: "q6k-stage2-bottleneck-analysis-details.txt",
    100: "q6k-stage2-bottleneck-analysis-n100-details.txt",
    512: "q6k-stage2-bottleneck-analysis-n512-details.txt",
    1024: "q6k-stage2-bottleneck-analysis-n1024-details.txt",
}

METRICS = {
    "duration_ms": r"^\s*Duration\s+ms\s+([0-9.]+)\s*$",
    "compute_pct": r"^\s*Compute \(SM\) Throughput\s+%\s+([0-9.]+)\s*$",
    "memory_pct": r"^\s*Memory Throughput\s+%\s+([0-9.]+)\s*$",
    "long_scoreboard": r"^\s*Stall Long Scoreboard\s+inst\s+([0-9.]+)\s*$",
    "lg_throttle": r"^\s*Stall LG Throttle\s+inst\s+([0-9.]+)\s*$",
    "occupancy_pct": r"^\s*Achieved Occupancy\s+%\s+([0-9.]+)\s*$",
    "cycles_issued": r"^\s*Warp Cycles Per Issued Instruction\s+cycle\s+([0-9.]+)\s*$",
    "cycles_executed": r"^\s*Warp Cycles Per Executed Instruction\s+cycle\s+([0-9.]+)\s*$",
    "active_threads": r"^\s*Avg\. Active Threads Per Warp\s+([0-9.]+)\s*$",
    "unpredicated_threads": r"^\s*Avg\. Not Predicated Off Threads Per Warp\s+([0-9.]+)\s*$",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timing-csv", type=Path, default=DEFAULT_TIMING)
    parser.add_argument("--profile-dir", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def load_timing(path: Path) -> list[dict[str, float]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as stream:
        for raw in csv.DictReader(stream):
            n = int(raw["columns"])
            median = float(raw["cuda_median_ms"])
            rows.append({
                "columns": n,
                "tile_j": int(raw["actual_j"] or raw["predicted_j"]),
                "median_ms": median,
                "min_ms": float(raw["cuda_min_ms"]),
                "max_ms": float(raw["cuda_max_ms"]),
                "columns_per_ms": n / median,
                "us_per_column": 1000.0 * median / n,
            })
    if not rows:
        raise SystemExit(f"no timing rows in {path}")
    return rows


def load_ncu(profile_dir: Path) -> list[dict[str, float]]:
    rows = []
    for n, filename in PROFILE_DETAILS.items():
        path = profile_dir / filename
        if n == 1024 and not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        row: dict[str, float] = {"columns": n}
        for name, pattern in METRICS.items():
            match = re.search(pattern, text, flags=re.MULTILINE)
            if not match:
                raise SystemExit(f"metric {name!r} not found in {path}")
            row[name] = float(match.group(1))
        rows.append(row)
    return rows


def style_axis(axis, ylabel: str, title: str) -> None:
    axis.set_ylabel(ylabel)
    axis.set_title(title, fontweight="bold")
    axis.grid(True, alpha=0.22)
    axis.spines[["top", "right"]].set_visible(False)


def save(figure, output_dir: Path, stem: str) -> None:
    figure.tight_layout()
    for extension in ("png", "svg"):
        figure.savefig(output_dir / f"{stem}.{extension}", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_scaling(rows: list[dict[str, float]], output_dir: Path) -> None:
    n = [row["columns"] for row in rows]
    median = [row["median_ms"] for row in rows]
    minimum = [row["min_ms"] for row in rows]
    maximum = [row["max_ms"] for row in rows]
    throughput = [row["columns_per_ms"] for row in rows]
    unit_cost = [row["us_per_column"] for row in rows]
    tile = [row["tile_j"] for row in rows]

    figure, axes = plt.subplots(2, 2, figsize=(15, 9))
    axis = axes[0, 0]
    axis.plot(n, median, marker="o", linewidth=2.4, color="#276FBF", label="median")
    axis.fill_between(n, minimum, maximum, color="#276FBF", alpha=0.16, label="min–max")
    linear = [median[0] * value / n[0] for value in n]
    axis.plot(n, linear, linestyle="--", color="#8A8A8A", label="linear cost from N=25")
    style_axis(axis, "CUDA time (ms)", "Total kernel time")
    axis.legend(frameon=False)

    axis = axes[0, 1]
    axis.plot(n, throughput, marker="o", linewidth=2.4, color="#17A589")
    style_axis(axis, "Columns / ms", "Effective throughput")
    axis.set_ylim(bottom=0)

    axis = axes[1, 0]
    axis.plot(n, unit_cost, marker="o", linewidth=2.4, color="#E67E22")
    style_axis(axis, "Microseconds / column", "Amortized cost per column")
    axis.set_ylim(bottom=0)

    axis = axes[1, 1]
    axis.step(n, tile, where="mid", linewidth=2.4, color="#8E44AD")
    axis.scatter(n, tile, color="#8E44AD", zorder=3)
    style_axis(axis, "Kernel tile J", "Selected specialization")
    axis.set_yticks(sorted(set(tile)))

    for axis in axes.flat:
        axis.set_xlabel("Synthetic activation columns N (token-like during prefill)")
        axis.set_xscale("log", base=2)
        axis.set_xticks(n)
        axis.set_xticklabels([str(value) for value in n], rotation=35, ha="right", fontsize=9)
    figure.suptitle("Q6_K [8192 × 28672] column scaling", fontsize=16, fontweight="bold")
    save(figure, output_dir, "q6k-kernel-scaling")


def plot_ncu(rows: list[dict[str, float]], output_dir: Path) -> None:
    n = [int(row["columns"]) for row in rows]
    labels = [f"N={value}" for value in n]
    x = list(range(len(rows)))
    width = 0.34

    figure, axes = plt.subplots(2, 2, figsize=(13, 8))
    axis = axes[0, 0]
    axis.bar([value - width / 2 for value in x], [r["compute_pct"] for r in rows], width,
             label="Compute", color="#276FBF")
    axis.bar([value + width / 2 for value in x], [r["memory_pct"] for r in rows], width,
             label="Memory", color="#17A589")
    style_axis(axis, "% of peak", "Speed-of-Light utilization")
    axis.set_ylim(0, 100)
    axis.legend(frameon=False)

    axis = axes[0, 1]
    axis.plot(x, [r["long_scoreboard"] for r in rows], marker="o", linewidth=2.4,
              label="Long scoreboard", color="#C0392B")
    axis.plot(x, [r["lg_throttle"] for r in rows], marker="o", linewidth=2.4,
              label="LG throttle", color="#E67E22")
    style_axis(axis, "Warp cycles per issued instruction", "Memory-related warp stalls")
    axis.set_ylim(bottom=0)
    axis.legend(frameon=False)

    axis = axes[1, 0]
    axis.plot(x, [r["cycles_issued"] for r in rows], marker="o", linewidth=2.4,
              label="Per issued instruction", color="#8E44AD")
    axis.plot(x, [r["cycles_executed"] for r in rows], marker="x", linewidth=1.5,
              label="Per executed instruction", color="#34495E")
    style_axis(axis, "Warp cycles", "Instruction issue cost")
    axis.set_ylim(bottom=0)
    axis.legend(frameon=False)

    axis = axes[1, 1]
    axis.plot(x, [r["occupancy_pct"] for r in rows], marker="o", linewidth=2.4,
              label="Achieved occupancy (%)", color="#D4AC0D")
    axis.plot(x, [r["active_threads"] for r in rows], marker="s", linewidth=2.0,
              label="Active threads / warp", color="#2E86C1")
    axis.plot(x, [r["unpredicated_threads"] for r in rows], marker="^", linewidth=2.0,
              label="Useful threads / warp", color="#28B463")
    style_axis(axis, "Percent or threads", "Occupancy and lane activity")
    axis.set_ylim(0, 36)
    axis.legend(frameon=False, fontsize=9)

    for axis in axes.flat:
        axis.set_xticks(x, labels)
        axis.set_xlabel("Profiled activation columns")
    figure.suptitle("Why larger Q6_K column batches use the GPU better", fontsize=16, fontweight="bold")
    save(figure, output_dir, "q6k-ncu-comparison")


def write_derived(rows: list[dict[str, float]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.timing_csv.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    timing = load_timing(args.timing_csv)
    ncu = load_ncu(args.profile_dir)
    plot_scaling(timing, output_dir)
    plot_ncu(ncu, output_dir)
    write_derived(timing, output_dir / "timing-derived.csv")
    write_derived(ncu, output_dir / "ncu-comparison.csv")
    print(f"Plots and derived CSVs: {output_dir}")


if __name__ == "__main__":
    main()

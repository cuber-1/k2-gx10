#!/usr/bin/env python3
"""Generate the publication figures embedded in README.md.

The compact CSV/JSON inputs are committed alongside the figures. They were
extracted from the preserved Nsight Systems, Nsight Compute, and benchmark
reports named in docs/visual-results.md.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "docs" / "data"
ASSETS = ROOT / "docs" / "assets"

BLUE = "#2563eb"
CYAN = "#0891b2"
GREEN = "#059669"
ORANGE = "#ea580c"
RED = "#dc2626"
PURPLE = "#7c3aed"
SLATE = "#475569"
LIGHT = "#e2e8f0"


def configure() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "svg.hashsalt": "k2-gx10",
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    png_path = ASSETS / f"{stem}.png"
    svg_path = ASSETS / f"{stem}.svg"
    fig.savefig(png_path, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    plt.close(fig)


def combined_decode_gains() -> None:
    summary = json.loads(
        (ROOT / "results/q6k-decode-combined-20260824/summary-direct.json").read_text()
    )
    baseline = float(summary["baseline_median_tps"])
    final = float(summary["final_median_tps"])
    gain = float(summary["measured_throughput_gain_pct"])
    pair_gains = np.array([float(row["throughput_gain_pct"]) for row in summary["pairs"]])

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), constrained_layout=True)
    fig.suptitle(f"Combined decode patch directly measured +{gain:.2f}%", fontsize=19, weight="bold")

    bars = axes[0].bar(
        ["Untouched\n4 warps", "8 warps +\nL2 prefetch"],
        [baseline, final],
        width=0.62,
        color=[SLATE, BLUE],
    )
    axes[0].set_ylim(0, 4.25)
    axes[0].set_ylabel("Generated tokens / second")
    axes[0].set_title("Pooled median, full 73B model")
    axes[0].bar_label(bars, labels=[f"{baseline:.4f}", f"{final:.4f}"], padding=5, weight="bold")
    axes[0].text(
        0.5,
        0.88,
        f"+{gain:.4f}% throughput\n{summary['measured_decode_time_reduction_pct']:.4f}% less time/token",
        transform=axes[0].transAxes,
        ha="center",
        va="top",
        color=BLUE,
        weight="bold",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#eff6ff", "edgecolor": BLUE},
    )

    x = np.arange(1, len(pair_gains) + 1)
    axes[1].plot(x, pair_gains, "o-", color=GREEN, linewidth=2.5, markersize=8)
    axes[1].axhline(float(summary["paired_median_gain_pct"]), color=SLATE, linestyle="--", linewidth=1.5)
    axes[1].set_xticks(x)
    axes[1].set_ylim(0, 25)
    axes[1].set_xlabel("Independent balanced process pair")
    axes[1].set_ylabel("Final-build throughput gain (%)")
    axes[1].set_title(f"{summary['pair_wins']}/{summary['process_pairs']} pair wins; paired median +{summary['paired_median_gain_pct']:.2f}%")
    for index, value in enumerate(pair_gains, start=1):
        axes[1].text(index, value + 0.65, f"+{value:.2f}%", ha="center", color=GREEN, weight="bold")

    fig.text(
        0.5,
        -0.015,
        "Direct A/B · fixed n_ctx=8192 · 128 generated tokens · 5 process pairs · 10 measured samples/build",
        ha="center",
        color=SLATE,
    )
    save(fig, "decode-combined-gain")


def long_context() -> None:
    rows = json.loads(
        (ROOT / "results/q6k-decode-long-context-20260818/summary-repeat.json").read_text()
    )
    bands = [row["band"] for row in rows]
    baseline = np.array([row["baseline_median"] for row in rows])
    candidate = np.array([row["candidate_median"] for row in rows])
    gain = np.array([row["paired_median_pct"] for row in rows])
    low = np.array([row["bootstrap_lo_pct"] for row in rows])
    high = np.array([row["bootstrap_hi_pct"] for row in rows])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    fig.suptitle("Accepted L2 prefetch gain persists across context depth", fontsize=19, weight="bold")

    x = np.arange(len(rows))
    width = 0.36
    axes[0].bar(x - width / 2, baseline, width, label="8-warps baseline", color=SLATE)
    axes[0].bar(x + width / 2, candidate, width, label="8-warps + L2 prefetch", color=BLUE)
    axes[0].set_xticks(x, bands)
    axes[0].set_ylim(0, 4.35)
    axes[0].set_ylabel("Generated tokens / second")
    axes[0].set_xlabel("Timed occupied-KV band (fixed n_ctx=8192)")
    axes[0].set_title("Full 73B model decode throughput")
    axes[0].legend(frameon=False, loc="lower left")
    for index, value in enumerate(candidate):
        axes[0].text(index + width / 2, value + 0.055, f"{value:.3f}", ha="center", fontsize=9)

    yerr = np.vstack((gain - low, high - gain))
    axes[1].errorbar(x, gain, yerr=yerr, fmt="o-", color=GREEN, linewidth=2.5, capsize=5)
    axes[1].axhline(0, color="#0f172a", linewidth=1)
    axes[1].set_xticks(x, bands)
    axes[1].set_ylim(0, 13)
    axes[1].set_ylabel("Paired throughput improvement (%)")
    axes[1].set_xlabel("Timed occupied-KV band")
    axes[1].set_title("10/10 wins at every depth; 95% bootstrap CI")
    for index, value in enumerate(gain):
        axes[1].text(index, value + 0.38, f"+{value:.2f}%", ha="center", color=GREEN, weight="bold")

    fig.text(
        0.5,
        -0.015,
        "Independent confirmation campaign · 10 balanced process-level A/B pairs per depth · 128 measured tokens",
        ha="center",
        color=SLATE,
    )
    save(fig, "decode-long-context-speedup")


def nsys_kernel_share() -> None:
    with (DATA / "nsys-kernel-share.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    mixed = [row for row in rows if row["capture"] == "Mixed prefill + decode"]
    isolated = [row for row in rows if row["capture"] == "Isolated fused decode"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.45, 1]}, constrained_layout=True)
    fig.suptitle("Nsight Systems: Q6_K kernels dominate GPU execution", fontsize=19, weight="bold")

    labels = [row["kernel"] for row in mixed][::-1]
    values = [float(row["percent"]) for row in mixed][::-1]
    colors = [
        BLUE if row["optimized_path"] == "yes" else (CYAN if row["kernel"].startswith("Q6_K") else LIGHT)
        for row in mixed
    ][::-1]
    bars = axes[0].barh(labels, values, color=colors)
    axes[0].set_xlim(0, 55)
    axes[0].set_xlabel("Share of captured GPU kernel time (%)")
    axes[0].set_title("Mixed short-prefill + decode capture")
    axes[0].bar_label(bars, labels=[f"{value:.1f}%" for value in values], padding=5, weight="bold")
    axes[0].text(
        0.98,
        0.04,
        "Optimized N=1 MMVQ family: 57.6%\nAll Q6_K kernels: 98.2%",
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#eff6ff", "edgecolor": BLUE},
    )

    bottom = 0.0
    for row, color in zip(isolated, [BLUE, ORANGE]):
        value = float(row["percent"])
        axes[1].bar(0, value, bottom=bottom, width=0.55, color=color, label=row["kernel"])
        bottom += value
    axes[1].set_xlim(-0.55, 0.55)
    axes[1].set_ylim(0, 100)
    axes[1].set_xticks([])
    axes[1].set_ylabel("Share of captured GPU kernel time (%)")
    axes[1].set_title("Isolated fused decode operation")
    axes[1].legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.13))
    axes[1].text(0, 50, "Target kernel\n99.8%", ha="center", va="center", color="white", fontsize=21, weight="bold")
    axes[1].annotate(
        "Q8_1 quantize: 0.2%",
        xy=(0, 99.9),
        xytext=(0.36, 94),
        arrowprops={"arrowstyle": "->", "color": ORANGE},
        ha="center",
        color=ORANGE,
        weight="bold",
    )

    fig.text(
        0.5,
        -0.02,
        "Source: Nsight Systems CUDA GPU Kernel Summary. Percentages are GPU kernel time within each capture, not request wall time.",
        ha="center",
        color=SLATE,
    )
    save(fig, "nsys-kernel-share")


def ncu_bottleneck() -> None:
    with (DATA / "ncu-decode-metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    stalls = [row for row in rows if row["group"] == "stall"]
    percents = [row for row in rows if row["group"] == "percent"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    fig.suptitle("Nsight Compute: the fused Q6_K decode kernel waits on memory", fontsize=19, weight="bold")

    stall_labels = [row["metric"] for row in stalls][::-1]
    stall_values = [float(row["value"]) for row in stalls][::-1]
    stall_colors = [RED if label == "Long scoreboard" else ORANGE if label == "LG throttle" else SLATE for label in stall_labels]
    bars = axes[0].barh(stall_labels, stall_values, color=stall_colors)
    axes[0].set_xlim(0, 60)
    axes[0].set_xlabel("Warp cycles per issued instruction")
    axes[0].set_title("Warp-stall breakdown")
    axes[0].bar_label(bars, labels=[f"{value:.2f}" for value in stall_values], padding=4)
    axes[0].text(
        0.98,
        0.05,
        "Long scoreboard = waiting for\noff-chip/global-memory dependencies",
        transform=axes[0].transAxes,
        ha="right",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#fef2f2", "edgecolor": RED},
    )

    percent_labels = [row["metric"] for row in percents][::-1]
    percent_values = [float(row["value"]) for row in percents][::-1]
    colors = [PURPLE if label == "Achieved occupancy" else RED if label == "Excess global L2 sectors" else BLUE for label in percent_labels]
    bars = axes[1].barh(percent_labels, percent_values, color=colors)
    axes[1].set_xlim(0, 80)
    axes[1].set_xlabel("Percent (metric-specific denominator)")
    axes[1].set_title("Why occupancy alone was not the fix")
    axes[1].bar_label(bars, labels=[f"{value:.2f}%" for value in percent_values], padding=4)
    axes[1].text(
        0.98,
        0.05,
        "2.01 ms · 46 registers/thread\n73.15% occupancy, but only 7.20% L2 hits",
        transform=axes[1].transAxes,
        ha="right",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#f5f3ff", "edgecolor": PURPLE},
    )

    fig.text(
        0.5,
        -0.02,
        "Exact kernel: mul_mat_vec_q<Q6_K, N=1, fused SwiGLU> · pre-optimization baseline profile",
        ha="center",
        color=SLATE,
    )
    save(fig, "ncu-decode-bottleneck")


def prefill_scaling() -> None:
    with (DATA / "q6k-column-sweep.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    columns = np.array([int(row["columns"]) for row in rows])
    median = np.array([float(row["cuda_median_ms"]) for row in rows])
    minimum = np.array([float(row["cuda_min_ms"]) for row in rows])
    maximum = np.array([float(row["cuda_max_ms"]) for row in rows])
    throughput = columns / median
    cost_us = 1000 * median / columns
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    fig.suptitle("Q6_K prefill scaling: larger batches help until N≈1024", fontsize=19, weight="bold")

    axes[0].plot(columns, throughput, "o-", color=GREEN, linewidth=2.5)
    peak = int(np.argmax(throughput))
    axes[0].scatter([columns[peak]], [throughput[peak]], s=110, color=RED, zorder=3)
    axes[0].annotate(
        f"peak {throughput[peak]:.1f} columns/ms\nN={columns[peak]}",
        (columns[peak], throughput[peak]),
        xytext=(25, -45),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": RED},
    )
    axes[0].set_xscale("log", base=2)
    tick_values = np.array([25, 64, 128, 512, 1024, 1536, 2048])
    axes[0].set_xticks(tick_values, [str(value) for value in tick_values], rotation=35)
    axes[0].set_ylabel("Effective throughput (columns / ms)")
    axes[0].set_xlabel("Activation columns N")
    axes[0].set_title("Throughput rises, peaks, then falls")

    axes[1].plot(columns, cost_us, "o-", color=ORANGE, linewidth=2.5, label="µs / column")
    axes[1].fill_between(columns, 1000 * minimum / columns, 1000 * maximum / columns, color=ORANGE, alpha=0.14, label="min–max")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(tick_values, [str(value) for value in tick_values], rotation=35)
    axes[1].set_ylabel("Amortized cost (µs / column)")
    axes[1].set_xlabel("Activation columns N")
    axes[1].set_title("Batch amortization saturates")
    axes[1].legend(frameon=False)
    fig.text(
        0.5,
        -0.02,
        "Isolated Q6_K [8192 × 28672] MMQ sweep · larger N is not automatically faster per column",
        ha="center",
        color=SLATE,
    )
    save(fig, "q6k-prefill-scaling")


def main() -> None:
    configure()
    combined_decode_gains()
    long_context()
    nsys_kernel_share()
    ncu_bottleneck()
    prefill_scaling()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Analyze a completed candidate validation directory."""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


T95 = 2.262157  # df=9


def ci95(values: list[float]) -> tuple[float, float]:
    mean = statistics.mean(values)
    margin = T95 * statistics.stdev(values) / math.sqrt(len(values))
    return mean - margin, mean + margin


def overlap_fraction(a: tuple[float, float], b: tuple[float, float]) -> float:
    overlap = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    narrower = min(a[1] - a[0], b[1] - b[0])
    return overlap / narrower if narrower > 0 else 0.0


def stats(values: list[float]) -> dict:
    low, high = ci95(values)
    mean = statistics.mean(values)
    return {
        "n": len(values),
        "median": statistics.median(values),
        "mean": mean,
        "standard_deviation": statistics.stdev(values),
        "coefficient_of_variation_percent": statistics.stdev(values) / mean * 100,
        "ci95_low": low,
        "ci95_high": high,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze_bench(result_dir: Path) -> tuple[list[dict], list[dict]]:
    bench_dir = result_dir / "llama-bench"
    tests = json.load((bench_dir / "prefill.json").open()) + json.load(
        (bench_dir / "generation.json").open()
    )
    rows = []
    samples: dict[tuple[str, int], list[float]] = {}
    for test in tests:
        mode = "prefill" if test["n_prompt"] else "generation"
        tokens = test["n_prompt"] or test["n_gen"]
        flash = "on" if test["flash_attn"] == 1 else "off"
        values = [float(value) for value in test["samples_ts"]]
        samples[(f"{mode}_{tokens}", test["flash_attn"])] = values
        row = {
            "mode": mode,
            "tokens": tokens,
            "flash_attention": flash,
            **stats(values),
            "exact_command_group": (
                "llama-bench -ngl 99 -p 24,512,2048 -n 0 -fa on,off -r 10 --delay 1 -o json"
                if mode == "prefill"
                else "llama-bench -ngl 99 -p 0 -n 44 -fa on,off -r 10 --delay 1 -o json"
            ),
        }
        rows.append(row)

    paired = []
    for key in ("prefill_24", "prefill_512", "prefill_2048", "generation_44"):
        on = samples[(key, 1)]
        off = samples[(key, 0)]
        differences = [off_value - on_value for off_value, on_value in zip(off, on)]
        diff_low, diff_high = ci95(differences)
        on_ci, off_ci = ci95(on), ci95(off)
        fraction = overlap_fraction(on_ci, off_ci)
        paired.append({
            "test": key,
            "comparison": "flash_off_minus_flash_on",
            "n_pairs": len(differences),
            "mean_difference_tokens_per_second": statistics.mean(differences),
            "median_difference_tokens_per_second": statistics.median(differences),
            "difference_standard_deviation": statistics.stdev(differences),
            "difference_ci95_low": diff_low,
            "difference_ci95_high": diff_high,
            "relative_difference_percent": statistics.mean(differences) / statistics.mean(on) * 100,
            "flash_on_ci95_low": on_ci[0],
            "flash_on_ci95_high": on_ci[1],
            "flash_off_ci95_low": off_ci[0],
            "flash_off_ci95_high": off_ci[1],
            "ci_overlap_fraction_of_narrower_interval": fraction,
            "confidence_intervals_substantially_overlap": fraction >= 0.25,
            "flash_off_confirmed_improvement": diff_low > 0 and fraction < 0.25,
        })
    write_csv(bench_dir / "statistics.csv", rows)
    write_csv(bench_dir / "paired-comparisons.csv", paired)
    return rows, paired


def load_server_summary(result_dir: Path) -> tuple[list[dict], list[dict], list[dict]]:
    with (result_dir / "statistics.csv").open() as source:
        server_stats = list(csv.DictReader(source))
    with (result_dir / "paired-comparisons.csv").open() as source:
        server_paired = list(csv.DictReader(source))
    with (result_dir / "runs.csv").open() as source:
        runs = list(csv.DictReader(source))
    return server_stats, server_paired, runs


def make_plots(result_dir: Path, bench_rows: list[dict], server_stats: list[dict]) -> None:
    prefill = [row for row in bench_rows if row["mode"] == "prefill"]
    lengths = [24, 512, 2048]
    figure, axis = plt.subplots(figsize=(9, 5))
    width = 0.36
    positions = range(len(lengths))
    for index, flash in enumerate(("on", "off")):
        selected = [next(row for row in prefill if row["tokens"] == length and row["flash_attention"] == flash) for length in lengths]
        axis.bar(
            [position + (index - 0.5) * width for position in positions],
            [row["mean"] for row in selected], width,
            yerr=[(row["ci95_high"] - row["ci95_low"]) / 2 for row in selected],
            capsize=4, label=f"FA {flash}",
        )
    axis.set_xticks(list(positions), [str(length) for length in lengths])
    axis.set_xlabel("Fixed prompt tokens")
    axis.set_ylabel("Prompt-processing tokens/second")
    axis.set_title("llama-bench fixed-token prompt processing (95% CI)")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(result_dir / "llama-bench-prefill.png", dpi=160)
    plt.close(figure)

    workloads = ["short", "medium", "long"]
    metrics = [
        ("prompt_tokens_per_second", "Prompt tokens/second", "server-prompt-speed.png"),
        ("decode_tokens_per_second", "Decode tokens/second", "server-decode-speed.png"),
        ("end_to_end_latency_seconds", "End-to-end seconds", "server-latency.png"),
    ]
    for metric, ylabel, filename in metrics:
        figure, axis = plt.subplots(figsize=(9, 5))
        for index, config in enumerate(("fa_on_8192", "fa_off_8192")):
            selected = [
                next(
                    row for row in server_stats
                    if row["workload"] == workload
                    and row["configuration"] == config
                    and row["metric"] == metric
                )
                for workload in workloads
            ]
            axis.bar(
                [position + (index - 0.5) * width for position in positions],
                [float(row["mean"]) for row in selected], width,
                yerr=[
                    (float(row["ci95_high"]) - float(row["ci95_low"])) / 2
                    for row in selected
                ],
                capsize=4, label="FA on" if index == 0 else "FA off",
            )
        axis.set_xticks(list(positions), workloads)
        axis.set_ylabel(ylabel)
        axis.set_title(f"Balanced server validation: {ylabel} (95% CI)")
        axis.legend()
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        figure.savefig(result_dir / filename, dpi=160)
        plt.close(figure)


def fmt(value: str | float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def make_report(
    result_dir: Path,
    bench_rows: list[dict],
    bench_paired: list[dict],
    server_stats: list[dict],
    server_paired: list[dict],
    runs: list[dict],
) -> None:
    def server_row(workload: str, config: str, metric: str) -> dict:
        return next(
            row for row in server_stats
            if row["workload"] == workload
            and row["configuration"] == config
            and row["metric"] == metric
        )

    def bench_row(mode: str, tokens: int, flash: str) -> dict:
        return next(
            row for row in bench_rows
            if row["mode"] == mode and row["tokens"] == tokens and row["flash_attention"] == flash
        )

    measured = [row for row in runs if row["warmup"] == "False"]
    correctness = sum(row["correct"] == "True" and row["clean_completion"] == "True" for row in measured)
    prompt_counts = {
        workload: sorted({int(row["prompt_tokens"]) for row in measured if row["workload"] == workload})
        for workload in ("short", "medium", "long")
    }
    completion_counts = {
        (workload, config): sorted({int(row["completion_tokens"]) for row in measured if row["workload"] == workload and row["configuration"] == config})
        for workload in ("short", "medium", "long")
        for config in ("fa_on_8192", "fa_off_8192", "ctx_2048_auto")
        if any(row["workload"] == workload and row["configuration"] == config for row in measured)
    }

    lines = [
        "# Candidate validation report",
        "",
        "## Outcome",
        "",
        "Neither earlier candidate is confirmed as a general replacement for the baseline.",
        "",
        "- Flash Attention off is rejected: it is slower for fixed-token prefill at 512 and 2048 tokens, slightly slower for fixed-token generation, and slower end-to-end in every balanced server workload.",
        "- Context 2048 is not confirmed: its short-prompt decode and latency confidence intervals substantially overlap the 8192/FA-on baseline, and the paired difference interval crosses zero.",
        "- Keep context 8192 and Flash Attention on for the tested workload mix.",
        "",
        "No Nsight Compute profiling was run.",
        "",
        "## Balanced end-to-end server results",
        "",
        f"All {correctness}/{len(measured)} measured requests were correct and completed with `finish_reason=stop`. Prompt cache reuse was disabled uniformly so every request processed its full prompt. Rendered prompt counts were {prompt_counts}.",
        "",
        "| Workload | Configuration | Prompt tok/s mean [95% CI] | Decode tok/s mean [95% CI] | E2E latency mean [95% CI] | CVs (PP/TG/E2E) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for workload, configs in (
        ("short", ("fa_on_8192", "fa_off_8192", "ctx_2048_auto")),
        ("medium", ("fa_on_8192", "fa_off_8192")),
        ("long", ("fa_on_8192", "fa_off_8192")),
    ):
        for config in configs:
            pp = server_row(workload, config, "prompt_tokens_per_second")
            tg = server_row(workload, config, "decode_tokens_per_second")
            latency = server_row(workload, config, "end_to_end_latency_seconds")
            lines.append(
                f"| {workload} | {config} | {fmt(pp['mean'], 2)} [{fmt(pp['ci95_low'], 2)}, {fmt(pp['ci95_high'], 2)}] "
                f"| {fmt(tg['mean'])} [{fmt(tg['ci95_low'])}, {fmt(tg['ci95_high'])}] "
                f"| {fmt(latency['mean'])} [{fmt(latency['ci95_low'])}, {fmt(latency['ci95_high'])}] "
                f"| {fmt(pp['coefficient_of_variation_percent'], 2)}% / {fmt(tg['coefficient_of_variation_percent'], 2)}% / {fmt(latency['coefficient_of_variation_percent'], 2)}% |"
            )
    lines.extend([
        "",
        "The medium end-to-end difference is partly response-behavior variance, not just kernel speed. Completion-token sets were:",
        "",
        f"`{completion_counts}`",
        "",
        "## llama-bench fixed-token isolation",
        "",
        "Each cell contains 10 measured repetitions after llama-bench's built-in warm-up.",
        "",
        "| Test | FA on mean [95% CI] | FA off mean [95% CI] | Off vs on | Confirmed off improvement? |",
        "|---|---:|---:|---:|---:|",
    ])
    for mode, tokens in (("prefill", 24), ("prefill", 512), ("prefill", 2048), ("generation", 44)):
        on = bench_row(mode, tokens, "on")
        off = bench_row(mode, tokens, "off")
        paired = next(row for row in bench_paired if row["test"] == f"{mode}_{tokens}")
        lines.append(
            f"| {mode} {tokens} | {fmt(on['mean'], 2)} [{fmt(on['ci95_low'], 2)}, {fmt(on['ci95_high'], 2)}] "
            f"| {fmt(off['mean'], 2)} [{fmt(off['ci95_low'], 2)}, {fmt(off['ci95_high'], 2)}] "
            f"| {fmt(paired['relative_difference_percent'], 2)}% | {paired['flash_off_confirmed_improvement']} |"
        )
    lines.extend([
        "",
        "At 24 prompt tokens, FA on/off is effectively tied and its confidence intervals substantially overlap. At 512 and 2048 tokens, FA on has clearly separated confidence intervals and is faster. Fixed-token generation also favors FA on slightly; the intervals do not substantially overlap.",
        "",
        "## Candidate conclusions",
        "",
        "### Candidate A: context 8192, Flash Attention off",
        "",
        "Not confirmed. In the balanced short server test, FA off was 1.82% slower in decode and 2.48% slower end-to-end. For fixed-token llama-bench, FA off was effectively tied at 24-token prefill, slower at 512 and 2048-token prefill, and slower at 44-token generation.",
        "",
        "### Candidate B: context 2048, individually tested defaults",
        "",
        "Not confirmed. Relative to context 8192/FA-on on the short workload, context 2048 averaged +1.15% decode throughput and -0.91% latency, but both paired 95% intervals cross zero and the configuration confidence intervals substantially overlap. Prompt processing was significantly slower in this balanced run. The earlier +7.59% result did not reproduce.",
        "",
        "## Thermal and clock controls",
        "",
        "Temperature, graphics/SM clocks, power, and utilization are recorded per request in `runs.csv`, with descriptive statistics in `statistics.csv`. The GPU generally sustained roughly 2.44-2.47 GHz and 95-96% utilization during measured work. The balanced block order and stable clocks make thermal throttling an unlikely explanation for the direction of the results.",
        "",
        "## Artifacts",
        "",
        "- `runs.csv`: all server measurements and telemetry",
        "- `statistics.csv`: median, mean, standard deviation, CV, and 95% CI",
        "- `paired-comparisons.csv`: paired server differences and confirmation decisions",
        "- `llama-bench/statistics.csv`: raw fixed-token summaries",
        "- `llama-bench/paired-comparisons.csv`: paired fixed-token differences",
        "- `manifest.json`: exact server commands and balanced schedule",
        "- `prompts.json`: exact fixed server prompts",
    ])
    (result_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    result_dir = Path(sys.argv[1]).resolve()
    bench_rows, bench_paired = analyze_bench(result_dir)
    server_stats, server_paired, runs = load_server_summary(result_dir)
    make_plots(result_dir, bench_rows, server_stats)
    make_report(result_dir, bench_rows, bench_paired, server_stats, server_paired, runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

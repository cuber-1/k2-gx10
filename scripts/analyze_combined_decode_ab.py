#!/usr/bin/env python3
"""Summarize the direct four-warp versus final combined decode experiment."""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path


def main() -> None:
    result_dir = Path(sys.argv[1])
    records: list[dict[str, object]] = []

    for path in sorted((result_dir / "raw").glob("pair-*.json")):
        _, pair_text, variant = path.stem.split("-")
        rows = json.loads(path.read_text())
        if len(rows) != 1:
            raise SystemExit(f"expected one benchmark row in {path}, found {len(rows)}")
        row = rows[0]
        samples = [float(value) for value in row["samples_ts"]]
        if len(samples) != 2:
            raise SystemExit(f"expected two measured samples in {path}, found {len(samples)}")
        records.append(
            {
                "pair": int(pair_text),
                "variant": variant,
                "median_tps": statistics.median(samples),
                "samples_tps": samples,
                "avg_tps": float(row["avg_ts"]),
            }
        )

    if len(records) != 10:
        raise SystemExit(f"expected 10 process results, found {len(records)}")

    variants = {str(record["variant"]) for record in records}
    if variants != {"baseline", "final"}:
        raise SystemExit(f"unexpected variants: {sorted(variants)}")

    paired: list[dict[str, float | int]] = []
    for pair in range(1, 6):
        pair_records = {str(record["variant"]): record for record in records if record["pair"] == pair}
        if set(pair_records) != {"baseline", "final"}:
            raise SystemExit(f"pair {pair} is incomplete")
        baseline = float(pair_records["baseline"]["median_tps"])
        final = float(pair_records["final"]["median_tps"])
        paired.append(
            {
                "pair": pair,
                "baseline_median_tps": baseline,
                "final_median_tps": final,
                "throughput_gain_pct": 100.0 * (final / baseline - 1.0),
                "decode_time_reduction_pct": 100.0 * (1.0 - baseline / final),
            }
        )

    baseline_samples = [
        sample
        for record in records
        if record["variant"] == "baseline"
        for sample in record["samples_tps"]
    ]
    final_samples = [
        sample
        for record in records
        if record["variant"] == "final"
        for sample in record["samples_tps"]
    ]
    baseline_median = statistics.median(baseline_samples)
    final_median = statistics.median(final_samples)
    gain = 100.0 * (final_median / baseline_median - 1.0)
    time_reduction = 100.0 * (1.0 - baseline_median / final_median)

    summary = {
        "comparison": "untouched 4-warps upstream vs 8-warps + L2 prefetch",
        "workload": "full 73B Q6_K model, n_ctx=8192, depth=0, 128 generated tokens",
        "process_pairs": 5,
        "samples_per_build": 10,
        "baseline_median_tps": baseline_median,
        "final_median_tps": final_median,
        "measured_throughput_gain_pct": gain,
        "measured_decode_time_reduction_pct": time_reduction,
        "pair_wins": sum(float(row["throughput_gain_pct"]) > 0 for row in paired),
        "paired_median_gain_pct": statistics.median(float(row["throughput_gain_pct"]) for row in paired),
        "paired_min_gain_pct": min(float(row["throughput_gain_pct"]) for row in paired),
        "paired_max_gain_pct": max(float(row["throughput_gain_pct"]) for row in paired),
        "pairs": paired,
    }
    (result_dir / "summary-direct.json").write_text(json.dumps(summary, indent=2) + "\n")

    with (result_dir / "paired-invocations-samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["pair", "variant", "sample", "tokens_per_second"])
        for record in records:
            for index, value in enumerate(record["samples_tps"], start=1):
                writer.writerow([record["pair"], record["variant"], index, f"{float(value):.6f}"])

    with (result_dir / "paired-invocations-direct.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(paired)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

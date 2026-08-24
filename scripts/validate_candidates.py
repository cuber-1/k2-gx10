#!/usr/bin/env python3
"""Balanced validation of K2 context-size and Flash Attention candidates."""

from __future__ import annotations

import csv
import json
import math
import os
import shlex
import signal
import socket
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_DIR / "results" / "candidate-validation"
SERVER = Path(os.environ.get("LLAMA_SERVER", Path.home() / "llama.cpp/build/bin/llama-server"))
MODEL = Path(os.environ.get("K2_MODEL", "K2-Think-V2-Q6_K-00001-of-00004.gguf"))
HOST = "127.0.0.1"
PORT = 30000
BASE_URL = f"http://{HOST}:{PORT}"
REASONING_BUDGET = 256
MAX_TOKENS = 320
T_CRITICAL_95 = 2.262157  # two-sided Student t, df=9 (10 measured repetitions)
REASONING_BUDGET_MESSAGE = (
    "\nThe reasoning budget is exhausted. Give only the requested final answer.\n"
)
SHORT_PROMPT = "What is 2+2? Reply with only the answer."
FILLER_PREFIX = (
    "Ignore the neutral filler tokens below. After reading them, answer the final "
    "question with only its answer.\n"
)
FILLER_SUFFIX = "\nFinal question: What is 2+2? Reply with only the answer."


@dataclass(frozen=True)
class Config:
    name: str
    context_size: int
    flash_attention: str


CONFIGS = {
    "fa_on_8192": Config("fa_on_8192", 8192, "on"),
    "fa_off_8192": Config("fa_off_8192", 8192, "off"),
    "ctx_2048_auto": Config("ctx_2048_auto", 2048, "auto"),
}

# Balanced block order. Every block starts a new server and receives one warm-up.
SCHEDULE = [
    ("short", "fa_on_8192", 5),
    ("short", "fa_off_8192", 5),
    ("short", "ctx_2048_auto", 5),
    ("short", "ctx_2048_auto", 5),
    ("short", "fa_off_8192", 5),
    ("short", "fa_on_8192", 5),
    ("medium", "fa_on_8192", 5),
    ("medium", "fa_off_8192", 10),
    ("medium", "fa_on_8192", 5),
    ("long", "fa_off_8192", 5),
    ("long", "fa_on_8192", 10),
    ("long", "fa_off_8192", 5),
]


def port_is_occupied() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex((HOST, PORT)) == 0


def command_for(config: Config) -> list[str]:
    return [
        str(SERVER),
        "--model", str(MODEL),
        "--host", HOST,
        "--port", str(PORT),
        "--n-gpu-layers", "99",
        "--ctx-size", str(config.context_size),
        "--parallel", "1",
        "--no-warmup",
        "--no-cache-prompt",
        "--flash-attn", config.flash_attention,
        "--temp", "1",
        "--top-p", "1",
        "--top-k", "0",
        "--min-p", "0",
        "--reasoning-format", "deepseek",
        "--reasoning-budget", str(REASONING_BUDGET),
        "--reasoning-budget-message", REASONING_BUDGET_MESSAGE,
    ]


def exact_command(config: Config) -> str:
    return shlex.join(command_for(config))


def post(path: str, body: dict, timeout: float | None = None) -> dict:
    request = urllib.request.Request(
        BASE_URL + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout or MAX_TOKENS * 2 + 60) as response:
        return json.load(response)


def request_body(prompt: str) -> dict:
    return {
        "model": "K2 Think V2",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "seed": 42,
        "reasoning_format": "deepseek",
        "reasoning_budget_tokens": REASONING_BUDGET,
        "reasoning_budget_message": REASONING_BUDGET_MESSAGE,
    }


def wait_until_ready(process: subprocess.Popen, timeout: float = 120.0) -> float:
    started = time.monotonic()
    deadline = started + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited during load with status {process.returncode}")
        try:
            with urllib.request.urlopen(f"{BASE_URL}/health", timeout=1):
                return time.monotonic() - started
        except urllib.error.URLError:
            time.sleep(0.25)
    raise TimeoutError("server was not ready within 120 seconds")


def stop_server(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def sample_gpu() -> dict:
    fields = [
        "timestamp", "temperature.gpu", "clocks.gr", "clocks.sm",
        "power.draw", "utilization.gpu",
    ]
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=" + ",".join(fields),
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    values = [part.strip() for part in result.stdout.strip().split(",")]
    sample = {"telemetry_available": result.returncode == 0 and len(values) == len(fields)}
    if not sample["telemetry_available"]:
        sample["telemetry_error"] = result.stderr.strip() or result.stdout.strip()
        return sample
    sample["timestamp"] = values[0]
    for key, value in zip(fields[1:], values[1:]):
        normalized = key.replace(".", "_")
        try:
            sample[normalized] = float(value)
        except ValueError:
            sample[normalized] = None
    return sample


def rendered_token_count(prompt: str) -> int:
    body = request_body(prompt)
    rendered = post("/apply-template", body)
    tokens = post(
        "/tokenize",
        {"content": rendered["prompt"], "add_special": False, "parse_special": True},
    )["tokens"]
    return len(tokens)


def calibrate_prompt(target_tokens: int) -> tuple[str, int]:
    candidates = []
    low, high = 0, target_tokens * 2
    while low <= high:
        repeats = (low + high) // 2
        prompt = FILLER_PREFIX + (" datum" * repeats) + FILLER_SUFFIX
        count = rendered_token_count(prompt)
        candidates.append((abs(count - target_tokens), prompt, count))
        if count < target_tokens:
            low = repeats + 1
        elif count > target_tokens:
            high = repeats - 1
        else:
            break
    _, prompt, count = min(candidates, key=lambda item: item[0])
    return prompt, count


def run_request(
    workload: str,
    prompt: str,
    config: Config,
    block_index: int,
    block_run: int,
    config_run: int,
    warmup: bool,
    model_load_seconds: float,
    raw_dir: Path,
) -> dict:
    body = request_body(prompt)
    before = sample_gpu()
    started = time.monotonic()
    result = post("/v1/chat/completions", body)
    wall_seconds = time.monotonic() - started
    after = sample_gpu()
    choice = result["choices"][0]
    message = choice["message"]
    usage = result.get("usage") or {}
    timings = result.get("timings") or {}
    content = (message.get("content") or "").strip()
    record = {
        "workload": workload,
        "configuration": config.name,
        "context_size": config.context_size,
        "flash_attention": config.flash_attention,
        "block_index": block_index,
        "block_run": block_run,
        "config_run": config_run,
        "warmup": warmup,
        "prompt_tokens_per_second": timings.get("prompt_per_second"),
        "decode_tokens_per_second": timings.get("predicted_per_second"),
        "server_total_latency_seconds": (
            timings.get("prompt_ms", 0) + timings.get("predicted_ms", 0)
        ) / 1000,
        "end_to_end_latency_seconds": wall_seconds,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "finish_reason": choice.get("finish_reason"),
        "correct": content == "4",
        "clean_completion": choice.get("finish_reason") == "stop" and bool(content),
        "content": message.get("content"),
        "model_load_seconds": model_load_seconds,
        "exact_command": exact_command(config),
        "telemetry_before": before,
        "telemetry_after": after,
        "response": result,
    }
    label = "warmup" if warmup else f"measured-{config_run:02d}"
    path = raw_dir / f"block-{block_index:02d}-{label}.json"
    with path.open("w") as output:
        json.dump(record, output, indent=2)
        output.write("\n")
    return record


RUN_FIELDS = [
    "workload", "configuration", "context_size", "flash_attention", "block_index",
    "block_run", "config_run", "warmup", "prompt_tokens_per_second",
    "decode_tokens_per_second", "server_total_latency_seconds",
    "end_to_end_latency_seconds", "prompt_tokens", "completion_tokens", "total_tokens",
    "finish_reason", "correct", "clean_completion", "content", "model_load_seconds",
    "temp_before_c", "temp_after_c", "graphics_clock_before_mhz",
    "graphics_clock_after_mhz", "sm_clock_before_mhz", "sm_clock_after_mhz",
    "power_before_w", "power_after_w", "gpu_util_before_percent",
    "gpu_util_after_percent", "exact_command",
]


def flattened(record: dict) -> dict:
    row = dict(record)
    before, after = record["telemetry_before"], record["telemetry_after"]
    row.update({
        "temp_before_c": before.get("temperature_gpu"),
        "temp_after_c": after.get("temperature_gpu"),
        "graphics_clock_before_mhz": before.get("clocks_gr"),
        "graphics_clock_after_mhz": after.get("clocks_gr"),
        "sm_clock_before_mhz": before.get("clocks_sm"),
        "sm_clock_after_mhz": after.get("clocks_sm"),
        "power_before_w": before.get("power_draw"),
        "power_after_w": after.get("power_draw"),
        "gpu_util_before_percent": before.get("utilization_gpu"),
        "gpu_util_after_percent": after.get("utilization_gpu"),
    })
    return row


def write_runs(records: list[dict], path: Path) -> None:
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=RUN_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flattened(record) for record in records)


def ci95(values: list[float]) -> tuple[float, float]:
    mean = statistics.mean(values)
    if len(values) < 2:
        return mean, mean
    margin = T_CRITICAL_95 * statistics.stdev(values) / math.sqrt(len(values))
    return mean - margin, mean + margin


def descriptive_stats(records: list[dict]) -> list[dict]:
    rows = []
    measured = [record for record in records if not record["warmup"]]
    groups = sorted({(record["workload"], record["configuration"]) for record in measured})
    metrics = [
        "prompt_tokens_per_second",
        "decode_tokens_per_second",
        "server_total_latency_seconds",
        "end_to_end_latency_seconds",
    ]
    telemetry = [
        "temp_after_c", "graphics_clock_after_mhz", "sm_clock_after_mhz",
        "power_after_w", "gpu_util_after_percent",
    ]
    flat = [flattened(record) for record in measured]
    for workload, configuration in groups:
        group = [
            record for record in flat
            if record["workload"] == workload and record["configuration"] == configuration
        ]
        for metric in [*metrics, *telemetry]:
            values = [float(record[metric]) for record in group if record.get(metric) is not None]
            if not values:
                continue
            mean = statistics.mean(values)
            low, high = ci95(values)
            rows.append({
                "workload": workload,
                "configuration": configuration,
                "metric": metric,
                "n": len(values),
                "median": statistics.median(values),
                "mean": mean,
                "standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
                "coefficient_of_variation_percent": (
                    statistics.stdev(values) / mean * 100 if len(values) > 1 and mean else 0.0
                ),
                "ci95_low": low,
                "ci95_high": high,
            })
    return rows


STAT_FIELDS = [
    "workload", "configuration", "metric", "n", "median", "mean",
    "standard_deviation", "coefficient_of_variation_percent", "ci95_low", "ci95_high",
]


def write_csv(rows: list[dict], fields: list[str], path: Path) -> None:
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def intervals_substantially_overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    overlap = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    narrower_width = min(a[1] - a[0], b[1] - b[0])
    return overlap > 0 and (narrower_width <= 0 or overlap / narrower_width >= 0.25)


def paired_stats(records: list[dict], stats_rows: list[dict]) -> list[dict]:
    measured = [record for record in records if not record["warmup"]]
    comparisons = [
        ("short", "fa_off_8192", "fa_on_8192"),
        ("short", "ctx_2048_auto", "fa_on_8192"),
        ("medium", "fa_off_8192", "fa_on_8192"),
        ("long", "fa_off_8192", "fa_on_8192"),
    ]
    metrics = [
        ("prompt_tokens_per_second", True),
        ("decode_tokens_per_second", True),
        ("server_total_latency_seconds", False),
        ("end_to_end_latency_seconds", False),
    ]

    def metric_ci(workload: str, config: str, metric: str) -> tuple[float, float]:
        row = next(
            item for item in stats_rows
            if item["workload"] == workload
            and item["configuration"] == config
            and item["metric"] == metric
        )
        return row["ci95_low"], row["ci95_high"]

    rows = []
    for workload, candidate, baseline in comparisons:
        candidate_records = sorted(
            [r for r in measured if r["workload"] == workload and r["configuration"] == candidate],
            key=lambda record: record["config_run"],
        )
        baseline_records = sorted(
            [r for r in measured if r["workload"] == workload and r["configuration"] == baseline],
            key=lambda record: record["config_run"],
        )
        if len(candidate_records) != 10 or len(baseline_records) != 10:
            continue
        for metric, higher_is_better in metrics:
            differences = [
                float(candidate_record[metric]) - float(baseline_record[metric])
                for candidate_record, baseline_record in zip(candidate_records, baseline_records)
            ]
            mean_difference = statistics.mean(differences)
            diff_low, diff_high = ci95(differences)
            baseline_mean = statistics.mean(float(record[metric]) for record in baseline_records)
            candidate_ci = metric_ci(workload, candidate, metric)
            baseline_ci = metric_ci(workload, baseline, metric)
            substantial_overlap = intervals_substantially_overlap(candidate_ci, baseline_ci)
            favorable_difference = diff_low > 0 if higher_is_better else diff_high < 0
            rows.append({
                "workload": workload,
                "candidate": candidate,
                "baseline": baseline,
                "metric": metric,
                "n_pairs": len(differences),
                "mean_paired_difference": mean_difference,
                "median_paired_difference": statistics.median(differences),
                "paired_difference_stddev": statistics.stdev(differences),
                "paired_difference_ci95_low": diff_low,
                "paired_difference_ci95_high": diff_high,
                "relative_difference_percent": mean_difference / baseline_mean * 100,
                "candidate_ci95_low": candidate_ci[0],
                "candidate_ci95_high": candidate_ci[1],
                "baseline_ci95_low": baseline_ci[0],
                "baseline_ci95_high": baseline_ci[1],
                "confidence_intervals_substantially_overlap": substantial_overlap,
                "confirmed_improvement": favorable_difference and not substantial_overlap,
            })
    return rows


PAIRED_FIELDS = [
    "workload", "candidate", "baseline", "metric", "n_pairs",
    "mean_paired_difference", "median_paired_difference", "paired_difference_stddev",
    "paired_difference_ci95_low", "paired_difference_ci95_high",
    "relative_difference_percent", "candidate_ci95_low", "candidate_ci95_high",
    "baseline_ci95_low", "baseline_ci95_high",
    "confidence_intervals_substantially_overlap", "confirmed_improvement",
]


def main() -> int:
    if port_is_occupied():
        raise SystemExit(f"port {PORT} is occupied; refusing to start validation")
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    result_dir = RESULTS_ROOT / f"validation-{run_id}"
    raw_dir = result_dir / "raw"
    logs_dir = result_dir / "server-logs"
    raw_dir.mkdir(parents=True)
    logs_dir.mkdir()

    records: list[dict] = []
    prompts: dict[str, str] | None = None
    prompt_counts: dict[str, int] | None = None
    config_run_counts: dict[tuple[str, str], int] = {}
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "reasoning_budget": REASONING_BUDGET,
        "max_tokens": MAX_TOKENS,
        "sampling": {"temperature": 0, "seed": 42},
        "prompt_cache": "disabled uniformly to measure full prompt processing",
        "schedule": SCHEDULE,
        "configurations": {
            name: {
                "context_size": config.context_size,
                "flash_attention": config.flash_attention,
                "exact_command": exact_command(config),
            }
            for name, config in CONFIGS.items()
        },
        "telemetry_query": (
            "timestamp,temperature.gpu,clocks.gr,clocks.sm,power.draw,utilization.gpu"
        ),
    }

    try:
        for block_index, (workload, config_name, repetitions) in enumerate(SCHEDULE, start=1):
            config = CONFIGS[config_name]
            if port_is_occupied():
                raise RuntimeError(f"port {PORT} occupied before block {block_index}")
            log_path = logs_dir / f"block-{block_index:02d}-{workload}-{config_name}.log"
            with log_path.open("w") as server_log:
                process = subprocess.Popen(
                    command_for(config),
                    cwd=PROJECT_DIR,
                    stdout=server_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    load_seconds = wait_until_ready(process)
                    if prompts is None:
                        medium_prompt, medium_count = calibrate_prompt(512)
                        long_prompt, long_count = calibrate_prompt(2048)
                        prompts = {
                            "short": SHORT_PROMPT,
                            "medium": medium_prompt,
                            "long": long_prompt,
                        }
                        prompt_counts = {
                            "short": rendered_token_count(SHORT_PROMPT),
                            "medium": medium_count,
                            "long": long_count,
                        }
                        manifest["rendered_prompt_token_counts"] = prompt_counts
                        with (result_dir / "prompts.json").open("w") as output:
                            json.dump(prompts, output, indent=2)
                            output.write("\n")
                    print(
                        f"block {block_index:02d}/{len(SCHEDULE)} {workload} {config_name}: "
                        f"load={load_seconds:.3f}s target_prompt_tokens={prompt_counts[workload]}"
                    )
                    key = (workload, config_name)
                    config_run_counts.setdefault(key, 0)
                    warmup_record = run_request(
                        workload, prompts[workload], config, block_index, 0,
                        config_run_counts[key], True, load_seconds, raw_dir,
                    )
                    records.append(warmup_record)
                    if not warmup_record["correct"] or not warmup_record["clean_completion"]:
                        raise RuntimeError(f"warm-up failed in block {block_index}")
                    for block_run in range(1, repetitions + 1):
                        config_run_counts[key] += 1
                        record = run_request(
                            workload, prompts[workload], config, block_index, block_run,
                            config_run_counts[key], False, load_seconds, raw_dir,
                        )
                        records.append(record)
                        write_runs(records, result_dir / "runs.csv")
                        print(
                            f"  run {record['config_run']:02d}: "
                            f"pp={record['prompt_tokens_per_second']:.2f} tok/s "
                            f"tg={record['decode_tokens_per_second']:.3f} tok/s "
                            f"e2e={record['end_to_end_latency_seconds']:.3f}s"
                        )
                        if not record["correct"] or not record["clean_completion"]:
                            raise RuntimeError(
                                f"correctness/clean completion failed in block {block_index}"
                            )
                finally:
                    stop_server(process)
    finally:
        write_runs(records, result_dir / "runs.csv")
        manifest["completed_records"] = len(records)
        with (result_dir / "manifest.json").open("w") as output:
            json.dump(manifest, output, indent=2)
            output.write("\n")

    stats_rows = descriptive_stats(records)
    paired_rows = paired_stats(records, stats_rows)
    write_csv(stats_rows, STAT_FIELDS, result_dir / "statistics.csv")
    write_csv(paired_rows, PAIRED_FIELDS, result_dir / "paired-comparisons.csv")
    print(f"validation complete: {result_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

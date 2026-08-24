#!/usr/bin/env python3
"""Controlled one-factor-at-a-time K2 llama-server parameter sweep."""

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
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_DIR / "results" / "parameter-sweeps"
BASELINE_DATASET = PROJECT_DIR / "results" / "repeat-256-clean"
SERVER = Path(os.environ.get("LLAMA_SERVER", Path.home() / "llama.cpp/build/bin/llama-server"))
MODEL = Path(os.environ.get("K2_MODEL", "K2-Think-V2-Q6_K-00001-of-00004.gguf"))
HOST = "127.0.0.1"
PORT = 30000
BASE_URL = f"http://{HOST}:{PORT}"
PROMPT = "What is 2+2? Reply with only the answer."
REASONING_BUDGET = 256
MAX_TOKENS = 320
MEASURED_REPETITIONS = 3
REASONING_BUDGET_MESSAGE = (
    "\nThe reasoning budget is exhausted. Give only the requested final answer.\n"
)


@dataclass(frozen=True)
class Config:
    name: str
    variable: str
    value: str
    args: tuple[str, ...] = field(default_factory=tuple)
    env: tuple[tuple[str, str], ...] = field(default_factory=tuple)


INDIVIDUAL_CONFIGS = [
    Config("baseline", "baseline", "exact run-k2-server defaults"),
    Config("flash_on", "flash_attention", "on", ("--flash-attn", "on")),
    Config("flash_off", "flash_attention", "off", ("--flash-attn", "off")),
    Config("ctx_2048", "context_size", "2048", ("--ctx-size", "2048")),
    Config("ctx_4096", "context_size", "4096", ("--ctx-size", "4096")),
    Config("batch_512", "batch_size", "512", ("--batch-size", "512")),
    Config("batch_1024", "batch_size", "1024", ("--batch-size", "1024")),
    Config("ubatch_128", "microbatch_size", "128", ("--ubatch-size", "128")),
    Config("ubatch_256", "microbatch_size", "256", ("--ubatch-size", "256")),
    Config(
        "cuda_graph_off",
        "cuda_graph",
        "disabled",
        env=(("GGML_CUDA_DISABLE_GRAPHS", "1"),),
    ),
]


def base_command() -> list[str]:
    return [
        str(SERVER),
        "--model", str(MODEL),
        "--host", HOST,
        "--port", str(PORT),
        "--n-gpu-layers", "99",
        "--ctx-size", "8192",
        "--parallel", "1",
        "--no-warmup",
        "--temp", "1",
        "--top-p", "1",
        "--top-k", "0",
        "--min-p", "0",
        "--reasoning-format", "deepseek",
        "--reasoning-budget", str(REASONING_BUDGET),
        "--reasoning-budget-message", REASONING_BUDGET_MESSAGE,
    ]


def command_for(config: Config) -> list[str]:
    command = base_command()
    replacements = {config.args[i]: config.args[i + 1] for i in range(0, len(config.args), 2)}
    for flag, value in replacements.items():
        if flag in command:
            command[command.index(flag) + 1] = value
        else:
            command.extend([flag, value])
    return command


def exact_command(config: Config) -> str:
    prefix = ["env", *(f"{key}={value}" for key, value in config.env)] if config.env else []
    return shlex.join([*prefix, *command_for(config)])


def port_is_occupied() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex((HOST, PORT)) == 0


def post(path: str, body: dict) -> dict:
    request = urllib.request.Request(
        BASE_URL + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=MAX_TOKENS * 2 + 60) as response:
        return json.load(response)


def request_body() -> dict:
    return {
        "model": "K2 Think V2",
        "messages": [{"role": "user", "content": PROMPT}],
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
            raise RuntimeError(f"server exited during model load with status {process.returncode}")
        try:
            with urllib.request.urlopen(f"{BASE_URL}/health", timeout=1):
                return time.monotonic() - started
        except urllib.error.URLError:
            time.sleep(0.25)
    raise TimeoutError(f"server was not ready within {timeout:.0f} seconds")


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


def run_request(config: Config, label: str, load_seconds: float, raw_dir: Path) -> dict:
    body = request_body()
    rendered = post("/apply-template", body)
    wall_started = time.monotonic()
    result = post("/v1/chat/completions", body)
    wall_seconds = time.monotonic() - wall_started
    choice = result["choices"][0]
    message = choice["message"]
    usage = result.get("usage") or {}
    timings = result.get("timings") or {}
    content = (message.get("content") or "").strip()
    finish_reason = choice.get("finish_reason")
    correct = content == "4"
    clean_completion = finish_reason == "stop" and bool(content)
    record = {
        "configuration": config.name,
        "variable": config.variable,
        "value": config.value,
        "run": label,
        "warmup": label == "warmup",
        "decode_tokens_per_second": timings.get("predicted_per_second"),
        "prompt_tokens_per_second": timings.get("prompt_per_second"),
        "total_latency_seconds": (timings.get("prompt_ms", 0) + timings.get("predicted_ms", 0)) / 1000,
        "client_wall_seconds": wall_seconds,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "correct": correct,
        "clean_completion": clean_completion,
        "finish_reason": finish_reason,
        "content": message.get("content"),
        "model_load_seconds": load_seconds,
        "exact_command": exact_command(config),
        "request_body": body,
        "rendered_prompt": rendered.get("prompt"),
        "response": result,
    }
    with (raw_dir / f"{label}.json").open("w") as output:
        json.dump(record, output, indent=2)
        output.write("\n")
    return record


def run_config(config: Config, sweep_dir: Path) -> list[dict]:
    if port_is_occupied():
        raise RuntimeError(f"port {PORT} is occupied before {config.name}")
    config_dir = sweep_dir / "raw" / config.name
    config_dir.mkdir(parents=True)
    log_path = config_dir / "server.log"
    environment = os.environ.copy()
    environment.update(dict(config.env))
    command = command_for(config)
    print(f"[{config.name}] loading model")
    with log_path.open("w") as server_log:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_DIR,
            env=environment,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            load_seconds = wait_until_ready(process)
            print(f"[{config.name}] model ready in {load_seconds:.3f}s")
            records = [run_request(config, "warmup", load_seconds, config_dir)]
            if not records[-1]["correct"] or not records[-1]["clean_completion"]:
                print(f"[{config.name}] warm-up failed; stopping configuration")
                return records
            for repetition in range(1, MEASURED_REPETITIONS + 1):
                record = run_request(config, f"measured-{repetition}", load_seconds, config_dir)
                records.append(record)
                print(
                    f"[{config.name}] run {repetition}: "
                    f"{record['decode_tokens_per_second']:.4f} tok/s, "
                    f"{record['total_latency_seconds']:.3f}s"
                )
                if not record["correct"] or not record["clean_completion"]:
                    print(f"[{config.name}] correctness/completion failed; stopping configuration")
                    break
            return records
        finally:
            stop_server(process)


RUN_FIELDS = [
    "configuration", "variable", "value", "run", "warmup",
    "decode_tokens_per_second", "prompt_tokens_per_second", "total_latency_seconds",
    "client_wall_seconds", "prompt_tokens", "completion_tokens", "total_tokens",
    "correct", "clean_completion", "finish_reason", "content", "model_load_seconds",
    "exact_command",
]


def write_run_csv(records: list[dict], path: Path) -> None:
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=RUN_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def summarize(records: list[dict], baseline_mean: float | None = None) -> list[dict]:
    rows = []
    config_names = list(dict.fromkeys(record["configuration"] for record in records))
    measured_by_config = {
        name: [r for r in records if r["configuration"] == name and not r["warmup"]]
        for name in config_names
    }
    if baseline_mean is None and "baseline" in measured_by_config:
        values = [r["decode_tokens_per_second"] for r in measured_by_config["baseline"]]
        if len(values) == MEASURED_REPETITIONS:
            baseline_mean = statistics.mean(values)
    for name in config_names:
        config_records = measured_by_config[name]
        all_records = [r for r in records if r["configuration"] == name]
        complete = (
            len(config_records) == MEASURED_REPETITIONS
            and all(r["correct"] and r["clean_completion"] for r in all_records)
        )
        decode = [r["decode_tokens_per_second"] for r in config_records]
        prompt = [r["prompt_tokens_per_second"] for r in config_records]
        latency = [r["total_latency_seconds"] for r in config_records]
        mean_decode = statistics.mean(decode) if decode else math.nan
        speedup = ((mean_decode / baseline_mean) - 1) * 100 if decode and baseline_mean else math.nan
        first = all_records[0]
        rows.append({
            "configuration": name,
            "variable": first["variable"],
            "value": first["value"],
            "measured_runs": len(config_records),
            "complete_and_correct": complete,
            "decode_tps_mean": mean_decode,
            "decode_tps_stdev": statistics.stdev(decode) if len(decode) > 1 else 0.0,
            "decode_tps_cv_percent": (
                statistics.stdev(decode) / mean_decode * 100 if len(decode) > 1 else 0.0
            ),
            "decode_tps_min": min(decode) if decode else math.nan,
            "decode_tps_max": max(decode) if decode else math.nan,
            "prompt_tps_mean": statistics.mean(prompt) if prompt else math.nan,
            "latency_seconds_mean": statistics.mean(latency) if latency else math.nan,
            "latency_seconds_stdev": statistics.stdev(latency) if len(latency) > 1 else 0.0,
            "speedup_vs_baseline_percent": speedup,
            "effect": (
                "improvement" if complete and speedup >= 1.0
                else "regression" if complete and speedup <= -1.0
                else "inconclusive" if complete
                else "failed"
            ),
            "model_load_seconds": first["model_load_seconds"],
            "prompt_tokens": config_records[0]["prompt_tokens"] if config_records else "",
            "completion_tokens": config_records[0]["completion_tokens"] if config_records else "",
            "finish_reasons": ";".join(str(r["finish_reason"]) for r in config_records),
            "exact_command": first["exact_command"],
        })
    return rows


SUMMARY_FIELDS = [
    "configuration", "variable", "value", "measured_runs", "complete_and_correct",
    "decode_tps_mean", "decode_tps_stdev", "decode_tps_cv_percent", "decode_tps_min",
    "decode_tps_max", "prompt_tps_mean", "latency_seconds_mean", "latency_seconds_stdev",
    "speedup_vs_baseline_percent", "effect", "model_load_seconds", "prompt_tokens",
    "completion_tokens", "finish_reasons", "exact_command",
]


def write_summary_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def make_plots(rows: list[dict], output_dir: Path) -> None:
    valid = [row for row in rows if row["complete_and_correct"]]
    names = [row["configuration"] for row in valid]

    def bar_plot(values, errors, ylabel, title, filename, baseline=None):
        figure, axis = plt.subplots(figsize=(12, 6))
        axis.bar(names, values, yerr=errors, capsize=4)
        if baseline is not None:
            axis.axhline(baseline, color="black", linestyle="--", linewidth=1)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=160)
        plt.close(figure)

    baseline_decode = next(row["decode_tps_mean"] for row in valid if row["configuration"] == "baseline")
    bar_plot(
        [row["decode_tps_mean"] for row in valid],
        [row["decode_tps_stdev"] for row in valid],
        "Decode tokens/second", "Decode speed by configuration", "decode-speed.png",
        baseline_decode,
    )
    bar_plot(
        [row["latency_seconds_mean"] for row in valid],
        [row["latency_seconds_stdev"] for row in valid],
        "Seconds", "Total latency by configuration", "latency.png",
    )
    bar_plot(
        [row["speedup_vs_baseline_percent"] for row in valid],
        [0 for _ in valid],
        "Percent", "Decode speedup relative to baseline", "speedup-vs-baseline.png", 0,
    )
    bar_plot(
        [row["decode_tps_cv_percent"] for row in valid],
        [0 for _ in valid],
        "Coefficient of variation (%)", "Run-to-run variation", "run-variation.png",
    )


def winning_settings(rows: list[dict]) -> list[dict]:
    winners = []
    variables = list(dict.fromkeys(row["variable"] for row in rows if row["variable"] != "baseline"))
    for variable in variables:
        candidates = [
            row for row in rows
            if row["variable"] == variable
            and row["complete_and_correct"]
            and row["speedup_vs_baseline_percent"] >= 1.0
        ]
        if candidates:
            winners.append(max(candidates, key=lambda row: row["decode_tps_mean"]))
    return winners


def combined_config(winners: list[dict]) -> Config:
    source = {config.name: config for config in INDIVIDUAL_CONFIGS}
    selected = [source[row["configuration"]] for row in winners]
    args = tuple(item for config in selected for item in config.args)
    env = tuple(item for config in selected for item in config.env)
    names = "+".join(config.name for config in selected) or "no_verified_improvements"
    return Config("combined", "combined", names, args=args, env=env)


def main() -> int:
    if not BASELINE_DATASET.is_dir():
        raise SystemExit(f"baseline dataset not found: {BASELINE_DATASET}")
    if port_is_occupied():
        raise SystemExit(f"port {PORT} is occupied; refusing to start sweep")

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    sweep_dir = RESULTS_ROOT / f"sweep-{run_id}"
    sweep_dir.mkdir(parents=True)
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "baseline_dataset": str(BASELINE_DATASET),
        "baseline_dataset_files": sorted(path.name for path in BASELINE_DATASET.iterdir()),
        "prompt": PROMPT,
        "reasoning_budget": REASONING_BUDGET,
        "max_tokens": MAX_TOKENS,
        "sampling": {"temperature": 0, "seed": 42, "top_p_server": 1, "top_k_server": 0, "min_p_server": 0},
        "warmups_per_configuration": 1,
        "measured_repetitions": MEASURED_REPETITIONS,
        "individual_configurations": [
            {"name": config.name, "variable": config.variable, "value": config.value,
             "exact_command": exact_command(config)}
            for config in INDIVIDUAL_CONFIGS
        ],
        "cuda_graph_support": {
            "cli_flag": False,
            "disable_environment_variable": "GGML_CUDA_DISABLE_GRAPHS=1",
            "source": "/home/dvijraicha/llama.cpp/ggml/src/ggml-cuda/common.cuh",
        },
    }
    with (sweep_dir / "manifest.json").open("w") as output:
        json.dump(manifest, output, indent=2)
        output.write("\n")

    all_records = []
    try:
        for config in INDIVIDUAL_CONFIGS:
            all_records.extend(run_config(config, sweep_dir))
            write_run_csv(all_records, sweep_dir / "runs.csv")

        individual_rows = summarize(all_records)
        write_summary_csv(individual_rows, sweep_dir / "individual-summary.csv")
        make_plots(individual_rows, sweep_dir)

        winners = winning_settings(individual_rows)
        combined = combined_config(winners)
        manifest["independently_verified_improvements"] = winners
        manifest["combined_configuration"] = {
            "value": combined.value,
            "exact_command": exact_command(combined),
        }
        with (sweep_dir / "manifest.json").open("w") as output:
            json.dump(manifest, output, indent=2)
            output.write("\n")

        all_records.extend(run_config(combined, sweep_dir))
        write_run_csv(all_records, sweep_dir / "runs.csv")
        baseline_mean = next(
            row["decode_tps_mean"] for row in individual_rows if row["configuration"] == "baseline"
        )
        final_rows = summarize(all_records, baseline_mean)
        write_summary_csv(final_rows, sweep_dir / "summary.csv")
        make_plots(final_rows, sweep_dir)
    except BaseException:
        write_run_csv(all_records, sweep_dir / "runs.csv")
        if all_records:
            partial_rows = summarize(all_records)
            write_summary_csv(partial_rows, sweep_dir / "partial-summary.csv")
        print(f"Partial results preserved in {sweep_dir}", file=sys.stderr)
        raise

    print(f"Sweep complete: {sweep_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

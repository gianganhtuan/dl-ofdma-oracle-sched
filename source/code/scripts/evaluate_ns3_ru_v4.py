#!/usr/bin/env python3
"""Paired validation of native ns-3 RU-oracle imitation policies."""

import argparse
import csv
import json
import math
import statistics
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


POLICIES = {
    "rr_default": {"policy": 0, "max_stations": 4, "delay_weight": 0.0},
    "max_backlog": {"policy": 1, "max_stations": 9, "delay_weight": 0.0},
    "oracle_throughput": {"policy": 2, "max_stations": 9, "delay_weight": 0.0},
    "ml_throughput": {"policy": 3, "max_stations": 9, "delay_weight": 0.0},
    "oracle_delay": {"policy": 2, "max_stations": 9, "delay_weight": 2.0},
    "ml_delay": {"policy": 3, "max_stations": 9, "delay_weight": 2.0},
}

SCENARIOS = [
    {"name": "unseen_sparse", "stations": 9, "mcs": 8, "load": 0.7, "payload": 1000, "skew": 0.2},
    {"name": "unseen_normal", "stations": 9, "mcs": 8, "load": 5.5, "payload": 1000, "skew": 0.35},
    {"name": "unseen_saturated", "stations": 9, "mcs": 8, "load": 16.0, "payload": 1000, "skew": 0.65},
    {"name": "unseen_low_mcs", "stations": 9, "mcs": 2, "load": 5.0, "payload": 1000, "skew": 0.5},
    {"name": "unseen_active_4", "stations": 4, "mcs": 8, "load": 24.0, "payload": 1000, "skew": 0.3},
    {"name": "unseen_active_7", "stations": 7, "mcs": 5, "load": 12.0, "payload": 1000, "skew": 0.55},
]

FLOAT_METRICS = [
    "throughput_mbps",
    "delay_mean_ms",
    "delay_p95_ms",
    "delay_p99_ms",
    "fairness",
    "decision_mean_us",
    "decision_p95_us",
    "decision_p99_us",
    "decision_max_us",
    "decision_deadline_miss_rate",
    "inference_mean_us",
    "inference_p95_us",
    "inference_p99_us",
    "inference_max_us",
]


def parse_result(stdout):
    line = next((line for line in stdout.splitlines() if line.startswith("RESULT ")), None)
    if line is None:
        raise ValueError(f"Missing RESULT line:\n{stdout}")
    return dict(item.split("=", 1) for item in line.split()[1:])


def mean_ci(values):
    values = list(values)
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, 0.0
    critical = {
        2: 12.706,
        3: 4.303,
        4: 3.182,
        5: 2.776,
        10: 2.262,
        20: 2.093,
        30: 2.045,
    }.get(len(values), 1.96)
    return mean, critical * statistics.stdev(values) / math.sqrt(len(values))


def run_one(args, scenario, policy_name, seed_index):
    config = POLICIES[policy_name]
    model = args.delay_model if policy_name == "ml_delay" else args.throughput_model
    command = [
        str(args.binary.resolve()),
        f"--policy={config['policy']}",
        f"--maxScheduledStations={config['max_stations']}",
        f"--delayWeight={config['delay_weight']}",
        f"--stations={scenario['stations']}",
        f"--mcs={scenario['mcs']}",
        f"--offeredMbpsPerSta={scenario['load']}",
        f"--payloadBytes={scenario['payload']}",
        f"--loadSkew={scenario['skew']}",
        f"--duration={args.duration}",
        f"--warmup={args.warmup}",
        f"--decisionBudgetUs={args.decision_budget_us}",
        f"--seed={args.seed_base + seed_index}",
        f"--modelPath={model.resolve()}",
    ]
    process = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout)
    if process.returncode:
        raise RuntimeError(f"{' '.join(command)}\n{process.stdout}\n{process.stderr}")
    values = parse_result(process.stdout)
    row = {
        "scenario": scenario["name"],
        "policy": policy_name,
        "seed_index": seed_index,
        "seed": int(values["seed"]),
        "stations": scenario["stations"],
        "mcs": scenario["mcs"],
        "payload_bytes": scenario["payload"],
        "load_skew": scenario["skew"],
        "offered_mbps": float(values["offered_mbps"]),
        "received_packets": int(values["received_packets"]),
        "decision_count": int(values["decision_count"]),
        "learned_fallbacks": int(values["learned_fallbacks"]),
    }
    row.update({metric: float(values[metric]) for metric in FLOAT_METRICS})
    row["delivery_ratio"] = min(1.0, row["throughput_mbps"] / row["offered_mbps"])
    return row


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    metrics = ["throughput_mbps", "delay_mean_ms", "delay_p99_ms", "fairness"] + FLOAT_METRICS[5:]
    summary = []
    for scenario in SCENARIOS:
        for policy in POLICIES:
            selected = [row for row in rows if row["scenario"] == scenario["name"] and row["policy"] == policy]
            record = {"scenario": scenario["name"], "policy": policy, "runs": len(selected)}
            for metric in metrics:
                record[f"{metric}_mean"], record[f"{metric}_ci95"] = mean_ci(row[metric] for row in selected)
            summary.append(record)
    return summary


def paired(rows, seeds):
    indexed = {(row["scenario"], row["policy"], row["seed_index"]): row for row in rows}
    comparisons = []
    for scenario in SCENARIOS:
        for learned, oracle in [("ml_throughput", "oracle_throughput"), ("ml_delay", "oracle_delay")]:
            record = {"scenario": scenario["name"], "learned": learned}
            for baseline in ["rr_default", "max_backlog", oracle]:
                for metric in ["throughput_mbps", "delay_mean_ms"]:
                    differences = [
                        indexed[(scenario["name"], learned, seed)][metric]
                        - indexed[(scenario["name"], baseline, seed)][metric]
                        for seed in range(seeds)
                    ]
                    mean, ci = mean_ci(differences)
                    record[f"minus_{baseline}_{metric}"] = mean
                    record[f"minus_{baseline}_{metric}_ci95"] = ci
            comparisons.append(record)
    return comparisons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=Path("ns-3.48/build/scratch/ns3.48-q1-ofdma-validation-default"))
    parser.add_argument("--throughput-model", type=Path, default=Path("ml/ns3_ru_v4/throughput/scheduler_weights.bin"))
    parser.add_argument("--delay-model", type=Path, default=Path("ml/ns3_ru_v4/delay/scheduler_weights.bin"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/ns3_ru_v4"))
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=920000)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--warmup", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--decision-budget-us", type=float, default=1000.0)
    args = parser.parse_args()

    tasks = [(scenario, policy, seed) for scenario in SCENARIOS for policy in POLICIES for seed in range(args.seeds)]
    rows = []
    print(f"Running {len(tasks)} paired ns-3 simulations", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_one, args, *task): task for task in tasks}
        for completed, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if completed % 10 == 0 or completed == len(tasks):
                print(f"Completed {completed}/{len(tasks)}", flush=True)
    rows.sort(key=lambda row: (row["scenario"], row["policy"], row["seed_index"]))
    write_csv(args.output_dir / "runs.csv", rows)
    summary = summarize(rows)
    comparisons = paired(rows, args.seeds)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "paired_comparisons.json").write_text(json.dumps(comparisons, indent=2) + "\n")
    print(f"Results: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

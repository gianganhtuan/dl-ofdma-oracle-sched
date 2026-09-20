#!/usr/bin/env python3
"""Measure learned scheduler latency serially to avoid worker contention."""

import argparse
import json
import statistics
import subprocess
from pathlib import Path

from evaluate_ns3_ru_v4 import SCENARIOS, parse_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=Path("ns-3.48/build/scratch/ns3.48-q1-ofdma-validation-default"))
    parser.add_argument("--model", type=Path, default=Path("ml/ns3_ru_v4/throughput/scheduler_weights.bin"))
    parser.add_argument("--output", type=Path, default=Path("results/ns3_ru_v4_10seed/serial_latency.json"))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--warmup", type=float, default=2.0)
    parser.add_argument("--budget-us", type=float, default=1500.0)
    args = parser.parse_args()

    rows = []
    for scenario in SCENARIOS:
        for seed in range(args.seeds):
            command = [
                str(args.binary.resolve()),
                "--policy=3",
                "--maxScheduledStations=9",
                f"--stations={scenario['stations']}",
                f"--mcs={scenario['mcs']}",
                f"--offeredMbpsPerSta={scenario['load']}",
                f"--payloadBytes={scenario['payload']}",
                f"--loadSkew={scenario['skew']}",
                f"--duration={args.duration}",
                f"--warmup={args.warmup}",
                f"--decisionBudgetUs={args.budget_us}",
                f"--seed={940000 + seed}",
                f"--modelPath={args.model.resolve()}",
            ]
            process = subprocess.run(command, capture_output=True, text=True, check=True)
            values = parse_result(process.stdout)
            rows.append(
                {
                    "scenario": scenario["name"],
                    "seed": seed,
                    **{
                        key: float(values[key])
                        for key in [
                            "state_mean_us",
                            "state_p99_us",
                            "inference_mean_us",
                            "inference_p99_us",
                            "projection_mean_us",
                            "projection_p99_us",
                            "decision_mean_us",
                            "decision_p99_us",
                            "decision_max_us",
                            "decision_deadline_miss_rate",
                        ]
                    },
                    "learned_fallbacks": int(values["learned_fallbacks"]),
                }
            )
            print(f"completed {scenario['name']} seed {seed}", flush=True)

    metrics = [key for key in rows[0] if key not in {"scenario", "seed"}]
    summary = {key: statistics.fmean(row[key] for row in rows) for key in metrics}
    summary["decision_global_max_us"] = max(row["decision_max_us"] for row in rows)
    result = {
        "execution": "serial host wall-clock measurement",
        "decision_budget_us": args.budget_us,
        "runs": len(rows),
        "summary": summary,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Portable entry point for the archived native ns-3 WCL experiments."""

import argparse
import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code/scripts"))
from evaluate_ns3_ru_v4 import SCENARIOS, parse_result


def run(command, cwd=None, log=None):
    command = [str(x) for x in command]
    print("+ " + shlex.join(command), flush=True)
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w") as stream:
            subprocess.run(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, check=True)
    else:
        subprocess.run(command, cwd=cwd, check=True)


def extract(archive, destination):
    if (destination / ".wcl-extracted").exists():
        return
    if destination.exists():
        destination.rmdir()  # Only remove an empty directory left by an interrupted attempt.
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        target = Path(temporary).resolve()
        with tarfile.open(archive) as stream:
            members = stream.getmembers()
            for member in members:
                if not (member.isfile() or member.isdir() or member.issym()):
                    raise ValueError(f"Unexpected non-file archive member: {member.name}")
                (target / member.name).resolve().relative_to(target)
            stream.extractall(target, members=[m for m in members if not m.issym()])
            for member in members:
                if member.issym():
                    link = target / member.name
                    (link.parent / member.linkname).resolve().relative_to(target)
                    link.symlink_to(member.linkname)
        (target / ".wcl-extracted").write_text(archive.name + "\n")
        target.rename(destination)


def simulator(args):
    return args.workspace / "ns-3.48"


def binary(args):
    if args.binary:
        return args.binary.resolve()
    path = simulator(args) / "build/scratch/ns3.48-q1-ofdma-validation-default"
    if not path.exists():
        raise FileNotFoundError(f"Build the simulator first: {path}")
    return path


def model(args, objective):
    root = args.workspace / "models" if args.trained else ROOT / "models"
    return root / objective / "scheduler_weights.bin"


def build(args):
    ns3 = simulator(args)
    extract(ROOT / "vendor/ns-3.48-native.tar.gz", ns3)
    run([sys.executable, "ns3", "configure", "-d", "default", "--enable-tests",
         "--enable-examples", "--enable-modules=wifi;spectrum;mobility;internet;applications",
         "--disable-python-bindings"], cwd=ns3)
    run([sys.executable, "ns3", "build", "q1-ofdma-validation", "test-runner", "-j", args.jobs], cwd=ns3)


def tests(args):
    ns3 = simulator(args)
    for suite in ["q1-ru-optimizer", "wifi-ru-allocation", "wifi-mac-ofdma",
                  "wifi-aggregation", "wifi-mac-queue", "wifi-devices-tx-duration"]:
        run([sys.executable, "test.py", "--no-build", "-s", suite, "--fullness=QUICK"], cwd=ns3,
            log=args.workspace / "checks" / (suite + ".log"))


def smoke(args):
    with (ROOT / "results/native_10seed/runs.csv").open(newline="") as stream:
        expected = {(r["scenario"], r["policy"], int(r["seed_index"])): r
                    for r in csv.DictReader(stream)}
    observations = []
    # Exercise every policy in the saturated nine-STA case plus a four-STA learned case.
    cases = [("unseen_saturated", p, i, weight) for p, i, weight in [
        ("rr_default", 0, 0), ("max_backlog", 1, 0), ("oracle_throughput", 2, 0),
        ("ml_throughput", 3, 0), ("oracle_delay", 2, 2), ("ml_delay", 3, 2)]]
    cases.append(("unseen_active_4", "ml_throughput", 3, 0))
    for name, policy_name, policy, weight in cases:
        s = next(s for s in SCENARIOS if s["name"] == name)
        command = [binary(args), f"--policy={policy}", f"--delayWeight={weight}",
                   f"--maxScheduledStations={4 if policy == 0 else 9}",
                   f"--stations={s['stations']}", f"--mcs={s['mcs']}",
                   f"--offeredMbpsPerSta={s['load']}", f"--loadSkew={s['skew']}",
                   f"--payloadBytes={s['payload']}", "--seed=920000", "--warmup=2",
                   "--duration=1", "--decisionBudgetUs=1000",
                   f"--modelPath={model(args, 'delay' if weight else 'throughput')}"]
        result = subprocess.run([str(x) for x in command], capture_output=True, text=True, check=True)
        actual = parse_result(result.stdout)
        reference = expected[name, policy_name, 0]
        # Timing changes with the host; only deterministic network metrics are compared.
        fields = ["throughput_mbps", "received_packets", "delay_mean_ms", "delay_p99_ms", "fairness"]
        differences = {f: float(actual[f]) - float(reference[f]) for f in fields}
        matched = all(abs(d) < 1e-6 for d in differences.values())
        observations.append({"scenario": name, "policy": policy_name, "matched": matched,
                             "differences": differences, "command": [str(x) for x in command],
                             "stdout": result.stdout})
        print(f"{name}/{policy_name}: {'matches archive' if matched else 'DIFFERS from archive'}", flush=True)
    path = args.workspace / "checks/smoke.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(observations, indent=2) + "\n")
    if not all(row["matched"] for row in observations):
        raise RuntimeError(f"Network result mismatch; inspect {path}")


def evaluate(args):
    run([sys.executable, ROOT / "code/scripts/evaluate_ns3_ru_v4.py", "--binary", binary(args),
         "--throughput-model", model(args, "throughput"), "--delay-model", model(args, "delay"),
         "--output-dir", args.workspace / "results/native_10seed", "--seeds", "10",
         "--seed-base", "920000", "--duration", "1", "--warmup", "2",
         "--decision-budget-us", "1000", "--workers", args.jobs, "--timeout", "1800"])


def timing(args):
    run([sys.executable, ROOT / "code/scripts/measure_ns3_ru_latency.py", "--binary", binary(args),
         "--model", model(args, "throughput"), "--output", args.workspace / "results/native_10seed/serial_latency.json",
         "--seeds", "3", "--duration", "1", "--warmup", "2", "--budget-us", "1500"])


def train(args):
    labels = args.workspace / ("generated-labels" if args.fresh_labels else "archived-labels")
    if not args.fresh_labels:
        extract(ROOT / "data/ns3_ru_v4-labels.tar.gz", labels)
    for objective in ["throughput", "delay"]:
        output = args.workspace / "models" / objective
        run([sys.executable, ROOT / "code/ml/train_ns3_ru.py", "--data",
             labels / objective / "training_samples.jsonl", "--output-dir", output,
             "--epochs", "40", "--batch-size", "256", "--hidden-size", "64",
             "--threads", "4", "--seed", "20260914", "--cpu"],
            log=args.workspace / "logs" / ("train-" + objective + ".log"))
        run([sys.executable, ROOT / "code/ml/export_q1_binary.py", output / "scheduler_model.pt",
             output / "scheduler_weights.bin"])


def collect(args):
    destination = args.workspace / "generated-labels"
    if destination.exists():
        raise FileExistsError("Use a new workspace for label collection; existing data are preserved")
    stations, mcs, loads = [3, 6, 9], [0, 3, 6, 9, 11], [1, 4, 10, 20, 35]
    payloads, skews = [300, 800, 1200, 1500], [0, .4, .75]
    commands = []
    for objective, weight in [("throughput", 0), ("delay", 2)]:
        parts = destination / objective / "parts"
        parts.mkdir(parents=True)
        with (parts.parent / "training_samples.jsonl").open("wb") as combined:
            for k in range(24):
                part = parts / f"run_{k}.jsonl"
                command = [binary(args), "--policy=2", f"--stations={stations[k % 3]}",
                           f"--mcs={mcs[(k // 3) % 5]}", f"--seed={1000 + k}",
                           f"--payloadBytes={payloads[(k * 3 + 2) % 4]}",
                           f"--offeredMbpsPerSta={loads[(k * 2 + 1) % 5]}",
                           f"--loadSkew={skews[(k // 2) % 3]}", "--duration=0.3", "--warmup=1",
                           "--maxScheduledStations=9", f"--delayWeight={weight}",
                           f"--trainingOutput={part}", "--trainingStride=1"]
                commands.append([str(x) for x in command])
                run(command, log=args.workspace / "logs" / f"collect-{objective}-{k}.log")
                combined.write(part.read_bytes())
    (destination / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")


def verify(args):
    failures = []
    count = 0
    for line in (ROOT / "MANIFEST.sha256").read_text().splitlines():
        expected, name = line.split("  ", 1)
        path = ROOT / name
        if not path.is_file() or hashlib.file_digest(path.open("rb"), "sha256").hexdigest() != expected:
            failures.append(name)
        count += 1
    if failures:
        raise RuntimeError("Checksum failures: " + ", ".join(failures))
    run([sys.executable, ROOT / "analyze.py", "--no-plots", "--output", args.workspace / "checks/tables"])
    print(f"Verified {count} archived file checksums")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["verify", "build", "test", "smoke", "evaluate", "timing",
                                           "train", "collect", "analyze", "all"])
    parser.add_argument("--workspace", type=Path, default=ROOT / "work")
    parser.add_argument("--binary", type=Path, help="Use an existing compatible ns-3 executable")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--trained", action="store_true", help="Evaluate the models retrained in workspace")
    parser.add_argument("--fresh-labels", action="store_true", help="Train on a new collection, instead of archived features")
    args = parser.parse_args()
    args.workspace = args.workspace.resolve()
    if args.action == "all":
        for step in [verify, build, tests, smoke, evaluate, timing]:
            step(args)
        run([sys.executable, ROOT / "analyze.py", "--results", args.workspace / "results/native_10seed",
             "--output", args.workspace / "generated"])
    elif args.action == "analyze":
        run([sys.executable, ROOT / "analyze.py"])
    elif args.action == "test":
        tests(args)
    else:
        globals()[args.action](args)


if __name__ == "__main__":
    main()

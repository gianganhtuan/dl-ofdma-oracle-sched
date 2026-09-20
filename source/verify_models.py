#!/usr/bin/env python3
"""Check archived trajectories, checkpoint validation metrics, and binary exports."""
import argparse
import hashlib
import json
import math
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code/ml"))
from train_ns3_ru import Ns3RuDataset, evaluate
from training import SchedulerSetNetwork


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "work/checks/models.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    results = []
    with tarfile.open(ROOT / "data/ns3_ru_v4-labels.tar.gz") as archive:
        members = archive.getmembers()
        for objective in ["throughput", "delay"]:
            directory = ROOT / "models" / objective
            metadata = json.loads((directory / "training_metrics.json").read_text())
            trajectories = sorted((m for m in members if m.isfile() and
                                   m.name.startswith(f"./{objective}/parts/run_")), key=lambda m: m.name)
            if len(trajectories) != 24:
                raise ValueError("Expected 24 independent trajectories per objective")
            rows, feature_counts, raw_count = [], set(), 0
            for part in trajectories:
                for line in archive.extractfile(part):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    raw_count += 1
                    if row["schema"] != "q1-ns3-ru-v4":
                        raise ValueError("Unexpected label schema")
                    feature_counts.add(len(row["features"]) // 9)
                    if Path(part.name).stem in metadata["validation_trajectories"]:
                        rows.append(row)
            if raw_count != metadata["rows"] or len(rows) != metadata["validation_rows"]:
                raise ValueError("Trajectory counts differ from model metadata")
            checkpoint = torch.load(directory / "scheduler_model.pt", map_location="cpu", weights_only=True)
            model = SchedulerSetNetwork(checkpoint["feature_count"], checkpoint["aggregation_classes"],
                                        checkpoint["hidden_size"], checkpoint["dropout"])
            model.load_state_dict(checkpoint["model_state_dict"])
            metrics = evaluate(model, DataLoader(Ns3RuDataset(rows, list(range(len(rows)))), batch_size=256),
                               torch.device("cpu"))
            for key, actual in metrics.items():
                if not math.isclose(actual, metadata["metrics"][key], abs_tol=2e-6):
                    raise ValueError(f"Checkpoint metric differs: {objective}/{key}")
            with tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / "model.bin"
                subprocess.run([sys.executable, str(ROOT / "code/ml/export_q1_binary.py"),
                                str(directory / "scheduler_model.pt"), str(output)], check=True)
                if output.read_bytes() != (directory / "scheduler_weights.bin").read_bytes():
                    raise ValueError("Checkpoint export differs from the released binary")
            results.append({"objective": objective, "rows": raw_count, "trajectories": 24,
                            "validation_rows": len(rows), "source_feature_counts": sorted(feature_counts),
                            "validation_metrics": metrics, "binary_matches_export": True,
                            "binary_sha256": hashlib.sha256((directory / "scheduler_weights.bin").read_bytes()).hexdigest()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

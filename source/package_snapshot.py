#!/usr/bin/env python3
"""Release-maintainer utility: capture provenance and changed-file inspection copies."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def git(tree, *arguments):
    return subprocess.check_output(["git", "-C", str(tree), *arguments], text=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository", type=Path)
    args = parser.parse_args()
    tree = args.repository.resolve() / "ns-3.48"
    tracked = git(tree, "diff", "--name-only", "HEAD").splitlines()
    untracked = git(tree, "ls-files", "--others", "--exclude-standard").splitlines()
    paths = sorted(set(tracked + untracked))
    (ROOT / "code/native-tracked-changes.patch").write_text(git(tree, "diff", "--binary", "HEAD"))
    checksums = {}
    for name in paths:
        source = tree / name
        if not source.is_file():
            continue
        destination = ROOT / "code/ns3-modified" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        checksums[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    shutil.copy2(tree / "LICENSE", ROOT / "vendor/ns-3-LICENSE")
    shutil.copytree(tree / "LICENSES", ROOT / "vendor/ns-3-LICENSES", dirs_exist_ok=True)
    meta = {
        "ns3_upstream_revision": git(tree, "rev-parse", "HEAD").strip(),
        "ns3_upstream_url": "https://gitlab.com/nsnam/ns-3-dev.git",
        "ns3_version_file": (tree / "VERSION").read_text().strip(),
        "working_tree_included": True,
        "changed_file_sha256": checksums,
        "network_campaign": {"policies": 6, "scenarios": 6, "runs_per_cell": 10,
                             "rng_seed": 20260914, "rng_runs": [920000, 920009],
                             "traffic_start_s": 1, "warmup_s": 2, "measurement_s": 1,
                             "drain_s": .1, "timing_budget_us": 1000},
        "serial_campaign": {"policies": ["ml_throughput"], "scenarios": 6, "runs_per_cell": 3,
                            "rng_seed": 20260914, "rng_runs": [940000, 940002],
                            "timing_budget_us": 1500, "measurement_s": 1, "warmup_s": 2,
                            "aggregation": "unweighted mean of per-run statistics; maximum separately reported"},
        "build": {"profile": "default", "compiler": "GCC 14.2.0", "cxx_standard": 23,
                  "flags": "-Os -g -DNDEBUG", "ns3_assertions": True, "ns3_logging": True},
        "training": {"seed": 20260914, "epochs": 40, "batch_size": 256, "threads": 4,
                     "hidden_size": 64, "dropout": .05, "learning_rate": .001,
                     "weight_decay": .0001, "class_weight": "inverse square root of frequency, mean normalized",
                     "numpy": "2.4.3", "torch": "2.5.1+cu121", "device": "CPU"},
        "label_provenance_limit": "Original manifest records upstream revision but not a dirty-tree diff. "
            "Archived labels preserve the actual source features and actions. The packaged snapshot is the "
            "final native implementation, whose extractor uses 39 rather than the collection's 135 features. "
            "Training uses the first 39 columns of each row. Fresh collection is a separate experiment; "
            "archived-label retraining is the route to the reported models.",
        "metric_caveats": ["delivery_ratio in legacy CSV is clipped throughput/offered rate, not cohort delivery",
                           "per-run p99s are averaged, not pooled",
                           "wall-clock selection-hook cost excludes subsequent MAC construction",
                           "scheduler timing spans setup, warm-up, measurement, and drain",
                           "wall-clock computation does not advance simulation time"],
    }
    (ROOT / "provenance.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Captured {len(checksums)} changed ns-3 files and experiment provenance")


if __name__ == "__main__":
    main()

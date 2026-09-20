#!/usr/bin/env python3
"""Train a permutation-equivariant RU policy from live ns-3 oracle labels."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from training import NUM_RU_CLASSES, NUM_STA, SchedulerSetNetwork


FEATURE_COUNT = 39
SOURCE_FEATURE_COUNTS = (39, 135)


def feature_matrix(row):
    values = np.asarray(row["features"], dtype=np.float32)
    if values.size % NUM_STA:
        raise ValueError("Feature vector does not contain nine station rows")
    source_count = values.size // NUM_STA
    if source_count not in SOURCE_FEATURE_COUNTS:
        raise ValueError(f"Unsupported source feature count: {source_count}")
    return values.reshape(NUM_STA, source_count)[:, :FEATURE_COUNT].copy()


class Ns3RuDataset(Dataset):
    def __init__(self, rows, indices, permute=False):
        self.rows = rows
        self.indices = indices
        self.permute = permute

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        row = self.rows[self.indices[item]]
        features = feature_matrix(row)
        labels = np.asarray(row["ru_classes"], dtype=np.int64)
        if self.permute:
            order = np.random.permutation(NUM_STA)
            features = features[order].copy()
            labels = labels[order].copy()
        return torch.from_numpy(features), torch.from_numpy(labels)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    active_correct = 0
    active_total = 0
    exact = 0
    selected_tp = 0
    selected_fp = 0
    selected_fn = 0
    losses = []
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)[2]
            active = features[..., 0] > 0.5
            losses.append(F.cross_entropy(logits[active], labels[active]).item())
            prediction = logits.argmax(dim=-1)
            prediction = prediction.masked_fill(~active, 0)
            correct += (prediction == labels).sum().item()
            total += labels.numel()
            active_correct += ((prediction == labels) & active).sum().item()
            active_total += active.sum().item()
            exact += (prediction == labels).all(dim=1).sum().item()
            selected_tp += ((prediction > 0) & (labels > 0) & active).sum().item()
            selected_fp += ((prediction > 0) & (labels == 0) & active).sum().item()
            selected_fn += ((prediction == 0) & (labels > 0) & active).sum().item()
    precision = selected_tp / max(1, selected_tp + selected_fp)
    recall = selected_tp / max(1, selected_tp + selected_fn)
    return {
        "loss": float(np.mean(losses)),
        "per_sta_accuracy": correct / max(1, total),
        "active_sta_accuracy": active_correct / max(1, active_total),
        "raw_exact_action_accuracy": exact / max(1, len(loader.dataset)),
        "selected_precision": precision,
        "selected_recall": recall,
        "selected_f1": 2 * precision * recall / max(1e-12, precision + recall),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    parts = sorted((args.data.parent / "parts").glob("run_*.jsonl"))
    if not parts:
        raise ValueError("Expected independent trajectory files in data/parts/run_*.jsonl")
    rows = []
    for part in parts:
        for line in part.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                row["trajectory"] = part.stem
                rows.append(row)
    if len(rows) < 100:
        raise ValueError(f"Need at least 100 rows, found {len(rows)}")
    for row in rows:
        if row.get("schema") != "q1-ns3-ru-v4":
            raise ValueError("Dataset is not the auditable q1-ns3-ru-v4 schema")
        if len(row["features"]) not in [NUM_STA * count for count in SOURCE_FEATURE_COUNTS] or len(row["ru_classes"]) != NUM_STA:
            raise ValueError("Dataset does not match a supported nine-row RU feature contract")

    trajectories = sorted({row["trajectory"] for row in rows})
    if len(trajectories) < 5:
        raise ValueError("Need at least five independent trajectories")
    order = np.random.permutation(trajectories)
    validation_trajectories = set(order[:max(1, int(0.2 * len(order)))].tolist())
    validation_indices = [i for i, row in enumerate(rows) if row["trajectory"] in validation_trajectories]
    training_indices = [i for i, row in enumerate(rows) if row["trajectory"] not in validation_trajectories]
    train_dataset = Ns3RuDataset(rows, training_indices, permute=True)
    validation_dataset = Ns3RuDataset(rows, validation_indices)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model = SchedulerSetNetwork(FEATURE_COUNT, 2, args.hidden_size, 0.05).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    labels = []
    for index in training_indices:
        features = feature_matrix(rows[index])
        target = np.asarray(rows[index]["ru_classes"], dtype=np.int64)
        labels.extend(target[features[:, 0] > 0.5].tolist())
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=NUM_RU_CLASSES).astype(np.float64)
    class_weights = np.sqrt(counts.sum() / np.maximum(counts, 1.0))
    class_weights /= class_weights.mean()
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)

    best_score = (-1.0, -1.0)
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        batch_losses = []
        for features, target in train_loader:
            features = features.to(device)
            target = target.to(device)
            logits = model(features)[2]
            active = features[..., 0] > 0.5
            loss = F.cross_entropy(logits[active], target[active], weight=class_weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            batch_losses.append(loss.item())
        metrics = evaluate(model, validation_loader, device)
        score = (metrics["raw_exact_action_accuracy"], metrics["selected_f1"])
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(
            f"epoch={epoch:03d} train_loss={np.mean(batch_losses):.6f} "
            f"val_loss={metrics['loss']:.6f} val_acc={metrics['per_sta_accuracy']:.6f} "
            f"val_active_acc={metrics['active_sta_accuracy']:.6f} "
            f"val_exact={metrics['raw_exact_action_accuracy']:.6f} "
            f"val_selected_f1={metrics['selected_f1']:.6f}"
        )

    model.load_state_dict(best_state)
    final_metrics = evaluate(model, validation_loader, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "architecture": "set-equivariant",
        "feature_count": FEATURE_COUNT,
        "aggregation_classes": 2,
        "hidden_size": args.hidden_size,
        "dropout": 0.05,
        "model_state_dict": model.cpu().state_dict(),
        "training_samples": len(training_indices),
        "validation_samples": len(validation_indices),
    }
    torch.save(checkpoint, args.output_dir / "scheduler_model.pt")
    metadata = {
        "dataset": str(args.data),
        "rows": len(rows),
        "training_rows": len(training_indices),
        "validation_rows": len(validation_indices),
        "feature_count": FEATURE_COUNT,
        "class_counts": counts.astype(int).tolist(),
        "class_weights": class_weights.cpu().tolist(),
        "split": "independent trajectories",
        "validation_trajectories": sorted(validation_trajectories),
        "metrics": final_metrics,
    }
    (args.output_dir / "training_metrics.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(final_metrics, sort_keys=True))


if __name__ == "__main__":
    main()

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset, random_split


NUM_STA = 9
NUM_RU_CLASSES = 5  # 0: not selected, 1: 26, 2: 52, 3: 106, 4: 242
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_DATASET_CANDIDATES = [
    Path("samples/training_samples.jsonl"),
    Path("samples/training/training_samples.jsonl"),
    Path("../samples/training_samples.jsonl"),
    Path("../samples/training/training_samples.jsonl"),
    PROJECT_ROOT / "samples" / "training_samples.jsonl",
    PROJECT_ROOT / "samples" / "training" / "training_samples.jsonl",
]


def resolve_dataset_path(path: str) -> Path:
    if path:
        dataset_path = Path(path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        return dataset_path

    for candidate in DEFAULT_DATASET_CANDIDATES:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Dataset not found. Expected one of: "
        + ", ".join(str(p) for p in DEFAULT_DATASET_CANDIDATES)
    )


def load_jsonl(path: Path, limit: int = 0) -> List[Dict]:
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
            if limit and len(samples) >= limit:
                break
    if not samples:
        raise ValueError(f"No samples loaded from {path}")
    return samples


def infer_max_packets_per_sta(samples: List[Dict]) -> int:
    max_packets = 1
    for sample in samples:
        max_packets = max(max_packets, int(sample["input"].get("max_packets_per_sta", 1)))
        for sta in sample["input"].get("stas", []):
            max_packets = max(max_packets, int(sta.get("packet_count", 0)))
    return max_packets


def build_features(
    sample: Dict,
    max_packets_per_sta: int,
    global_max_packet_bytes: float,
    global_max_phy_rate: float,
    feature_version: int,
) -> np.ndarray:
    input_data = sample["input"]
    stas = input_data["stas"]
    queue = sorted(input_data["queue"], key=lambda pkt: int(pkt["index"]))
    packets_by_sta = [[] for _ in range(NUM_STA)]

    for pkt in queue:
        sta_index = int(pkt["sta"]) - 1
        if 0 <= sta_index < NUM_STA:
            packets_by_sta[sta_index].append(float(pkt["size"]))

    feature_count = 5 + max_packets_per_sta
    features = np.zeros((NUM_STA, feature_count), dtype=np.float32)

    for sta_index in range(NUM_STA):
        sta = stas[sta_index]
        packet_sizes = packets_by_sta[sta_index]
        recorded_queue_len = len(packet_sizes)
        queue_len = (
            max(int(sta.get("packet_count", recorded_queue_len)), recorded_queue_len)
            if feature_version >= 2
            else recorded_queue_len
        )
        prefix_packet_sizes = packet_sizes[:max_packets_per_sta]
        total_bytes = sum(prefix_packet_sizes if feature_version >= 2 else packet_sizes)
        exists = 1.0 if sta.get("exists", False) else 0.0

        features[sta_index, 0] = exists
        features[sta_index, 1] = float(sta.get("mcs", 0)) / 11.0
        features[sta_index, 2] = float(sta.get("phy_rate", 0.0)) / global_max_phy_rate
        if feature_version >= 2:
            features[sta_index, 3] = queue_len / float(queue_len + max_packets_per_sta)
        else:
            features[sta_index, 3] = queue_len / float(max_packets_per_sta)
        features[sta_index, 4] = total_bytes / float(max_packets_per_sta * global_max_packet_bytes)

        for pkt_index in range(min(recorded_queue_len, max_packets_per_sta)):
            features[sta_index, 5 + pkt_index] = packet_sizes[pkt_index] / global_max_packet_bytes

    return features


def build_labels(sample: Dict, max_packets_per_sta: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    label = sample["label"]

    selected = np.asarray(label["selected_sta"], dtype=np.float32)
    aggregation_depth = np.asarray(label["aggregation_depth"], dtype=np.int64)
    ru_class = np.asarray(label["ru_class"], dtype=np.int64)

    aggregation_depth = np.clip(aggregation_depth, 0, max_packets_per_sta)
    ru_class = np.clip(ru_class, 0, NUM_RU_CLASSES - 1)
    selected = np.clip(selected, 0.0, 1.0)

    return selected, aggregation_depth, ru_class


class SchedulerDataset(Dataset):
    def __init__(self, samples: List[Dict], max_packets_per_sta: int, feature_version: int):
        self.samples = samples
        self.max_packets_per_sta = max_packets_per_sta
        self.feature_version = feature_version
        self.global_max_packet_bytes = max(
            float(sample["input"].get("max_packet_bytes", 1)) for sample in samples
        )
        self.global_max_phy_rate = max(
            1.0,
            max(
                float(sta.get("phy_rate", 0.0))
                for sample in samples
                for sta in sample["input"].get("stas", [])
            ),
        )

    @property
    def feature_count(self) -> int:
        return 5 + self.max_packets_per_sta

    @property
    def aggregation_classes(self) -> int:
        return self.max_packets_per_sta + 1

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        features = build_features(
            sample,
            self.max_packets_per_sta,
            self.global_max_packet_bytes,
            self.global_max_phy_rate,
            self.feature_version,
        )
        selected, aggregation_depth, ru_class = build_labels(sample, self.max_packets_per_sta)

        return (
            torch.from_numpy(features),
            torch.from_numpy(selected),
            torch.from_numpy(aggregation_depth),
            torch.from_numpy(ru_class),
        )


class SchedulerMLP(nn.Module):
    def __init__(self, feature_count: int, aggregation_classes: int, hidden_size: int, dropout: float):
        super().__init__()
        input_size = NUM_STA * feature_count
        self.feature_count = feature_count
        self.aggregation_classes = aggregation_classes

        self.backbone = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        self.selected_head = nn.Linear(hidden_size, NUM_STA)
        self.aggregation_head = nn.Linear(hidden_size, NUM_STA * aggregation_classes)
        self.ru_head = nn.Linear(hidden_size, NUM_STA * NUM_RU_CLASSES)

    def forward(self, features):
        hidden = self.backbone(features)
        selected_logits = self.selected_head(hidden)
        aggregation_logits = self.aggregation_head(hidden).view(
            -1, NUM_STA, self.aggregation_classes
        )
        ru_logits = self.ru_head(hidden).view(-1, NUM_STA, NUM_RU_CLASSES)
        return selected_logits, aggregation_logits, ru_logits


class SchedulerSetNetwork(nn.Module):
    """Permutation-equivariant scheduler with shared per-STA processing."""

    def __init__(self, feature_count: int, aggregation_classes: int, hidden_size: int, dropout: float):
        super().__init__()
        self.feature_count = feature_count
        self.aggregation_classes = aggregation_classes

        self.local_encoder = nn.Sequential(
            nn.Linear(feature_count, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.selected_head = nn.Linear(hidden_size, 1)
        self.aggregation_head = nn.Linear(hidden_size, aggregation_classes)
        self.ru_head = nn.Linear(hidden_size, NUM_RU_CLASSES)

    def forward(self, features):
        mask = features[..., :1].clamp(0.0, 1.0)
        local = self.local_encoder(features) * mask
        active_count = mask.sum(dim=1).clamp_min(1.0)
        mean_context = local.sum(dim=1) / active_count

        masked_local = local.masked_fill(mask == 0, -1.0e4)
        max_context = masked_local.max(dim=1).values
        has_active = (mask.sum(dim=1) > 0).to(local.dtype)
        max_context = max_context * has_active

        context = self.context_encoder(torch.cat((mean_context, max_context), dim=-1))
        context = context.unsqueeze(1).expand(-1, NUM_STA, -1)
        hidden = self.decoder(torch.cat((local, context), dim=-1))

        selected_logits = self.selected_head(hidden).squeeze(-1)
        aggregation_logits = self.aggregation_head(hidden)
        ru_logits = self.ru_head(hidden)
        return selected_logits, aggregation_logits, ru_logits


def create_model(
    architecture: str,
    feature_count: int,
    aggregation_classes: int,
    hidden_size: int,
    dropout: float,
):
    if architecture == "mlp":
        return SchedulerMLP(feature_count, aggregation_classes, hidden_size, dropout)
    if architecture == "set-equivariant":
        return SchedulerSetNetwork(feature_count, aggregation_classes, hidden_size, dropout)
    raise ValueError(f"Unsupported architecture: {architecture}")


def compute_loss(
    outputs,
    selected,
    aggregation_depth,
    ru_class,
    selected_pos_weight=None,
    aggregation_weights=None,
    ru_weights=None,
    consistency_weight=0.0,
    tone_budget_weight=0.0,
    inactive_weight=0.0,
    features=None,
):
    selected_logits, aggregation_logits, ru_logits = outputs
    if selected_pos_weight is None:
        selected_pos_weight = torch.full((NUM_STA,), 3.0, device=selected.device)

    selected_loss = nn.functional.binary_cross_entropy_with_logits(
        selected_logits,
        selected,
        pos_weight=selected_pos_weight,
    )
    aggregation_loss = nn.functional.cross_entropy(
        aggregation_logits.reshape(-1, aggregation_logits.shape[-1]),
        aggregation_depth.reshape(-1),
        weight=aggregation_weights,
    )
    ru_loss = nn.functional.cross_entropy(
        ru_logits.reshape(-1, NUM_RU_CLASSES),
        ru_class.reshape(-1),
        weight=ru_weights,
    )

    selected_probability = torch.sigmoid(selected_logits)
    aggregation_probability = torch.softmax(aggregation_logits, dim=-1)
    ru_probability = torch.softmax(ru_logits, dim=-1)

    aggregation_active = 1.0 - aggregation_probability[..., 0]
    ru_active = 1.0 - ru_probability[..., 0]
    consistency_loss = (
        nn.functional.mse_loss(selected_probability, aggregation_active)
        + nn.functional.mse_loss(selected_probability, ru_active)
    )

    tone_values = torch.tensor(
        [0.0, 26.0, 52.0, 106.0, 242.0],
        device=ru_logits.device,
        dtype=ru_logits.dtype,
    )
    expected_tones = (ru_probability * tone_values).sum(dim=-1).sum(dim=-1)
    tone_budget_loss = (torch.relu(expected_tones - 242.0) / 242.0).square().mean()

    inactive_loss = torch.zeros((), device=selected.device)
    if features is not None:
        inactive = 1.0 - features[..., 0].clamp(0.0, 1.0)
        inactive_loss = (
            (selected_probability * inactive).square().mean()
            + (aggregation_active * inactive).square().mean()
            + (ru_active * inactive).square().mean()
        )

    constraint_loss = (
        consistency_weight * consistency_loss
        + tone_budget_weight * tone_budget_loss
        + inactive_weight * inactive_loss
    )
    total_loss = selected_loss + aggregation_loss + ru_loss + constraint_loss
    return total_loss, selected_loss, aggregation_loss, ru_loss, constraint_loss


@torch.no_grad()
def evaluate(
    model,
    data_loader,
    device,
    selected_pos_weight=None,
    aggregation_weights=None,
    ru_weights=None,
    consistency_weight=0.0,
    tone_budget_weight=0.0,
    inactive_weight=0.0,
):
    model.eval()
    total_loss = 0.0
    total_selected_correct = 0
    total_agg_correct = 0
    total_ru_correct = 0
    total_elements = 0
    total_samples = 0
    total_selected_vector_correct = 0
    total_aggregation_vector_correct = 0
    total_ru_vector_correct = 0
    total_full_config_correct = 0
    total_raw_feasible = 0
    total_batches = 0

    for features, selected, aggregation_depth, ru_class in data_loader:
        features = features.to(device)
        selected = selected.to(device)
        aggregation_depth = aggregation_depth.to(device)
        ru_class = ru_class.to(device)

        outputs = model(features)
        loss, _, _, _, _ = compute_loss(
            outputs,
            selected,
            aggregation_depth,
            ru_class,
            selected_pos_weight,
            aggregation_weights,
            ru_weights,
            consistency_weight,
            tone_budget_weight,
            inactive_weight,
            features,
        )
        selected_logits, aggregation_logits, ru_logits = outputs

        selected_pred = (torch.sigmoid(selected_logits) >= 0.5).long()
        aggregation_pred = aggregation_logits.argmax(dim=-1)
        ru_pred = ru_logits.argmax(dim=-1)

        selected_true = selected.long()
        total_selected_correct += (selected_pred == selected_true).sum().item()
        total_agg_correct += (aggregation_pred == aggregation_depth).sum().item()
        total_ru_correct += (ru_pred == ru_class).sum().item()
        total_elements += selected.numel()
        total_samples += selected.shape[0]

        selected_vector_correct = (selected_pred == selected_true).all(dim=1)
        aggregation_vector_correct = (aggregation_pred == aggregation_depth).all(dim=1)
        ru_vector_correct = (ru_pred == ru_class).all(dim=1)
        full_config_correct = selected_vector_correct & aggregation_vector_correct & ru_vector_correct
        tone_values = torch.tensor([0, 26, 52, 106, 242], device=ru_pred.device)
        used_tones = tone_values[ru_pred].sum(dim=1)
        exists = features[..., 0] >= 0.5
        raw_feasible = (
            (selected_pred == (aggregation_pred > 0)).all(dim=1)
            & (selected_pred == (ru_pred > 0)).all(dim=1)
            & (~selected_pred.bool() | exists).all(dim=1)
            & (used_tones <= 242)
            & (selected_pred.sum(dim=1) > 0)
        )

        total_selected_vector_correct += selected_vector_correct.sum().item()
        total_aggregation_vector_correct += aggregation_vector_correct.sum().item()
        total_ru_vector_correct += ru_vector_correct.sum().item()
        total_full_config_correct += full_config_correct.sum().item()
        total_raw_feasible += raw_feasible.sum().item()
        total_loss += loss.item()
        total_batches += 1

    return {
        "loss": total_loss / max(1, total_batches),
        "selected_acc": total_selected_correct / max(1, total_elements),
        "aggregation_acc": total_agg_correct / max(1, total_elements),
        "ru_acc": total_ru_correct / max(1, total_elements),
        "selected_vector_acc": total_selected_vector_correct / max(1, total_samples),
        "aggregation_vector_acc": total_aggregation_vector_correct / max(1, total_samples),
        "ru_vector_acc": total_ru_vector_correct / max(1, total_samples),
        "full_config_acc": total_full_config_correct / max(1, total_samples),
        "raw_feasibility": total_raw_feasible / max(1, total_samples),
    }


def update_confusion_matrix(matrix: torch.Tensor, true_values: torch.Tensor, pred_values: torch.Tensor):
    num_classes = matrix.shape[0]
    true_flat = true_values.reshape(-1).to(torch.long).cpu()
    pred_flat = pred_values.reshape(-1).to(torch.long).cpu()
    valid = (
        (true_flat >= 0)
        & (true_flat < num_classes)
        & (pred_flat >= 0)
        & (pred_flat < num_classes)
    )
    indices = true_flat[valid] * num_classes + pred_flat[valid]
    counts = torch.bincount(indices, minlength=num_classes * num_classes)
    matrix += counts.reshape(num_classes, num_classes)


@torch.no_grad()
def validate_with_confusion(
    model,
    data_loader,
    device,
    aggregation_classes: int,
    selected_pos_weight=None,
    aggregation_weights=None,
    ru_weights=None,
    consistency_weight=0.0,
    tone_budget_weight=0.0,
    inactive_weight=0.0,
):
    model.eval()
    metrics = evaluate(
        model,
        data_loader,
        device,
        selected_pos_weight,
        aggregation_weights,
        ru_weights,
        consistency_weight,
        tone_budget_weight,
        inactive_weight,
    )
    selected_confusion = torch.zeros(2, 2, dtype=torch.long)
    aggregation_confusion = torch.zeros(aggregation_classes, aggregation_classes, dtype=torch.long)
    ru_confusion = torch.zeros(NUM_RU_CLASSES, NUM_RU_CLASSES, dtype=torch.long)

    for features, selected, aggregation_depth, ru_class in data_loader:
        features = features.to(device)
        selected = selected.to(device)
        aggregation_depth = aggregation_depth.to(device)
        ru_class = ru_class.to(device)

        selected_logits, aggregation_logits, ru_logits = model(features)
        selected_pred = (torch.sigmoid(selected_logits) >= 0.5).long()
        aggregation_pred = aggregation_logits.argmax(dim=-1)
        ru_pred = ru_logits.argmax(dim=-1)

        update_confusion_matrix(selected_confusion, selected.long(), selected_pred)
        update_confusion_matrix(aggregation_confusion, aggregation_depth, aggregation_pred)
        update_confusion_matrix(ru_confusion, ru_class, ru_pred)

    return {
        "metrics": metrics,
        "selected_confusion": selected_confusion.numpy(),
        "aggregation_confusion": aggregation_confusion.numpy(),
        "ru_confusion": ru_confusion.numpy(),
    }


def print_confusion_matrix(title: str, matrix: np.ndarray, class_names: List[str]):
    print()
    print(title)
    print("rows=true, columns=predicted")
    label_width = max(12, max(len(name) for name in class_names) + 2)
    cell_width = max(10, len(str(int(matrix.max()))) + 2 if matrix.size else 10)

    print(" " * label_width + "".join(f"{name:>{cell_width}}" for name in class_names))
    for row_index, row_name in enumerate(class_names):
        row = matrix[row_index]
        print(f"{row_name:<{label_width}}" + "".join(f"{int(value):>{cell_width}}" for value in row))


def save_confusion_report(report: Dict, output_path: Path):
    serializable = {
        "metrics": report["metrics"],
        "selected_confusion": report["selected_confusion"].tolist(),
        "aggregation_confusion": report["aggregation_confusion"].tolist(),
        "ru_confusion": report["ru_confusion"].tolist(),
    }
    output_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def make_inverse_frequency_weights(
    counts: torch.Tensor,
    power: float,
    min_weight: float,
    max_weight: float,
) -> torch.Tensor:
    counts = counts.to(torch.float32).clamp_min(1.0)
    mean_count = counts.mean()
    weights = (mean_count / counts).pow(power)
    weights = weights.clamp(min_weight, max_weight)
    weights = weights / weights.mean()
    return weights


def compute_training_label_weights(
    train_dataset,
    aggregation_classes: int,
    selected_weight_mode: str,
    aggregation_weight_mode: str,
    ru_weight_mode: str,
    class_weight_power: float,
    min_class_weight: float,
    max_class_weight: float,
):
    selected_positive = torch.zeros(NUM_STA, dtype=torch.float32)
    selected_negative = torch.zeros(NUM_STA, dtype=torch.float32)
    aggregation_counts = torch.zeros(aggregation_classes, dtype=torch.long)
    ru_counts = torch.zeros(NUM_RU_CLASSES, dtype=torch.long)

    for _, selected, aggregation_depth, ru_class in train_dataset:
        selected_positive += selected.to(torch.float32)
        selected_negative += 1.0 - selected.to(torch.float32)
        aggregation_counts += torch.bincount(
            aggregation_depth.reshape(-1),
            minlength=aggregation_classes,
        )
        ru_counts += torch.bincount(
            ru_class.reshape(-1),
            minlength=NUM_RU_CLASSES,
        )

    selected_pos_weight = torch.ones(NUM_STA, dtype=torch.float32)
    if selected_weight_mode == "auto":
        selected_pos_weight = (selected_negative / selected_positive.clamp_min(1.0)).clamp(
            min_class_weight,
            max_class_weight,
        )

    aggregation_weights = None
    if aggregation_weight_mode == "auto":
        aggregation_weights = make_inverse_frequency_weights(
            aggregation_counts,
            class_weight_power,
            min_class_weight,
            max_class_weight,
        )

    ru_weights = None
    if ru_weight_mode == "auto":
        ru_weights = make_inverse_frequency_weights(
            ru_counts,
            class_weight_power,
            min_class_weight,
            max_class_weight,
        )

    return {
        "selected_positive": selected_positive,
        "selected_negative": selected_negative,
        "aggregation_counts": aggregation_counts,
        "ru_counts": ru_counts,
        "selected_pos_weight": selected_pos_weight,
        "aggregation_weights": aggregation_weights,
        "ru_weights": ru_weights,
    }


def print_weight_summary(weight_info: Dict):
    print("Training label counts and loss weights")
    print(f"selected_positive={weight_info['selected_positive'].tolist()}")
    print(f"selected_negative={weight_info['selected_negative'].tolist()}")
    print(f"selected_pos_weight={weight_info['selected_pos_weight'].tolist()}")
    print(f"aggregation_counts={weight_info['aggregation_counts'].tolist()}")
    if weight_info["aggregation_weights"] is not None:
        print(f"aggregation_weights={weight_info['aggregation_weights'].tolist()}")
    else:
        print("aggregation_weights=None")
    print(f"ru_counts={weight_info['ru_counts'].tolist()}")
    if weight_info["ru_weights"] is not None:
        print(f"ru_weights={weight_info['ru_weights'].tolist()}")
    else:
        print("ru_weights=None")


def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset_path = resolve_dataset_path(args.data)
    samples = load_jsonl(dataset_path, args.limit)
    max_packets_per_sta = args.max_packets_per_sta or infer_max_packets_per_sta(samples)

    dataset = SchedulerDataset(samples, max_packets_per_sta, args.feature_version)
    generator = torch.Generator().manual_seed(args.seed)
    if args.split_by_seed:
        unique_seeds = sorted({int(sample.get("seed", 0)) for sample in samples})
        if len(unique_seeds) < 2:
            raise ValueError("At least two simulator seeds are required for --split-by-seed")
        shuffled_seed_indices = torch.randperm(len(unique_seeds), generator=generator).tolist()
        val_seed_count = min(
            len(unique_seeds) - 1,
            max(1, int(round(len(unique_seeds) * args.val_ratio))),
        )
        val_seeds = {unique_seeds[index] for index in shuffled_seed_indices[:val_seed_count]}
        train_indices = [
            index for index, sample in enumerate(samples) if int(sample.get("seed", 0)) not in val_seeds
        ]
        val_indices = [
            index for index, sample in enumerate(samples) if int(sample.get("seed", 0)) in val_seeds
        ]
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
    else:
        val_size = max(1, int(len(dataset) * args.val_ratio))
        train_size = len(dataset) - val_size
        if train_size <= 0:
            raise ValueError("Dataset is too small for the requested validation split")
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=generator,
        )

    train_size = len(train_dataset)
    val_size = len(val_dataset)
    if train_size == 0 or val_size == 0:
        raise ValueError("Dataset split produced an empty training or validation partition")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    weight_info = compute_training_label_weights(
        train_dataset,
        dataset.aggregation_classes,
        args.selected_weight_mode,
        args.aggregation_weight_mode,
        args.ru_weight_mode,
        args.class_weight_power,
        args.min_class_weight,
        args.max_class_weight,
    )
    print_weight_summary(weight_info)

    selected_pos_weight = weight_info["selected_pos_weight"].to(device)
    aggregation_weights = (
        weight_info["aggregation_weights"].to(device)
        if weight_info["aggregation_weights"] is not None
        else None
    )
    ru_weights = (
        weight_info["ru_weights"].to(device)
        if weight_info["ru_weights"] is not None
        else None
    )

    model = create_model(
        args.architecture,
        dataset.feature_count,
        dataset.aggregation_classes,
        args.hidden_size,
        args.dropout,
    ).to(device)
    if args.init_checkpoint:
        initial_checkpoint = torch.load(args.init_checkpoint, map_location=device)
        expected = (dataset.feature_count, dataset.aggregation_classes)
        actual = (
            int(initial_checkpoint["feature_count"]),
            int(initial_checkpoint["aggregation_classes"]),
        )
        if actual != expected:
            raise ValueError(
                f"Initial checkpoint shape {actual} does not match requested model shape {expected}"
            )
        initial_architecture = initial_checkpoint.get("architecture", "mlp")
        if initial_architecture != args.architecture:
            raise ValueError(
                f"Initial checkpoint architecture {initial_architecture} does not match "
                f"requested architecture {args.architecture}"
            )
        model.load_state_dict(initial_checkpoint["model_state_dict"])
        print(f"initialized_from={args.init_checkpoint}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = math.inf
    best_path = output_dir / "scheduler_model.pt"

    print(f"dataset={dataset_path}")
    print(f"samples={len(dataset)} train={train_size} val={val_size}")
    print(f"split={'seed-grouped' if args.split_by_seed else 'random-sample'}")
    print(
        f"device={device} feature_version={dataset.feature_version} "
        f"feature_count={dataset.feature_count} max_packets_per_sta={max_packets_per_sta}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_selected = 0.0
        running_agg = 0.0
        running_ru = 0.0
        running_constraint = 0.0
        batches = 0

        for features, selected, aggregation_depth, ru_class in train_loader:
            features = features.to(device)
            selected = selected.to(device)
            aggregation_depth = aggregation_depth.to(device)
            ru_class = ru_class.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(features)
            loss, selected_loss, aggregation_loss, ru_loss, constraint_loss = compute_loss(
                outputs,
                selected,
                aggregation_depth,
                ru_class,
                selected_pos_weight,
                aggregation_weights,
                ru_weights,
                args.consistency_weight,
                args.tone_budget_weight,
                args.inactive_weight,
                features,
            )
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_selected += selected_loss.item()
            running_agg += aggregation_loss.item()
            running_ru += ru_loss.item()
            running_constraint += constraint_loss.item()
            batches += 1

        val_metrics = evaluate(
            model,
            val_loader,
            device,
            selected_pos_weight,
            aggregation_weights,
            ru_weights,
            args.consistency_weight,
            args.tone_budget_weight,
            args.inactive_weight,
        )
        train_loss = running_loss / max(1, batches)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.5f} "
            f"sel_loss={running_selected / max(1, batches):.5f} "
            f"agg_loss={running_agg / max(1, batches):.5f} "
            f"ru_loss={running_ru / max(1, batches):.5f} "
            f"constraint_loss={running_constraint / max(1, batches):.5f} "
            f"val_loss={val_metrics['loss']:.5f} "
            f"val_selected_acc={val_metrics['selected_acc']:.4f} "
            f"val_agg_acc={val_metrics['aggregation_acc']:.4f} "
            f"val_ru_acc={val_metrics['ru_acc']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "feature_version": dataset.feature_version,
                "training_seed": args.seed,
                "feature_count": dataset.feature_count,
                "max_packets_per_sta": max_packets_per_sta,
                "aggregation_classes": dataset.aggregation_classes,
                "global_max_packet_bytes": dataset.global_max_packet_bytes,
                "global_max_phy_rate": dataset.global_max_phy_rate,
                "hidden_size": args.hidden_size,
                "dropout": args.dropout,
                "architecture": args.architecture,
                "consistency_weight": args.consistency_weight,
                "tone_budget_weight": args.tone_budget_weight,
                "inactive_weight": args.inactive_weight,
                "selected_weight_mode": args.selected_weight_mode,
                "aggregation_weight_mode": args.aggregation_weight_mode,
                "ru_weight_mode": args.ru_weight_mode,
                "class_weight_power": args.class_weight_power,
                "min_class_weight": args.min_class_weight,
                "max_class_weight": args.max_class_weight,
                "selected_pos_weight": weight_info["selected_pos_weight"].tolist(),
                "aggregation_weights": (
                    weight_info["aggregation_weights"].tolist()
                    if weight_info["aggregation_weights"] is not None
                    else None
                ),
                "ru_weights": (
                    weight_info["ru_weights"].tolist()
                    if weight_info["ru_weights"] is not None
                    else None
                ),
                "aggregation_counts": weight_info["aggregation_counts"].tolist(),
                "ru_counts": weight_info["ru_counts"].tolist(),
                "feature_order": [
                    "exists",
                    "mcs_norm",
                    "phy_rate_norm",
                    "queue_len_norm",
                    "total_bytes_norm",
                ]
                + [f"packet_{i + 1}_size_norm" for i in range(max_packets_per_sta)],
                "ru_classes": {
                    "0": "not_selected",
                    "1": 26,
                    "2": 52,
                    "3": 106,
                    "4": 242,
                },
            }
            torch.save(checkpoint, best_path)

    best_checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    final_report = validate_with_confusion(
        model,
        val_loader,
        device,
        dataset.aggregation_classes,
        selected_pos_weight,
        aggregation_weights,
        ru_weights,
        args.consistency_weight,
        args.tone_budget_weight,
        args.inactive_weight,
    )
    final_metrics = final_report["metrics"]

    print()
    print("Final validation using best checkpoint")
    print(
        f"val_loss={final_metrics['loss']:.5f} "
        f"val_selected_acc={final_metrics['selected_acc']:.4f} "
        f"val_agg_acc={final_metrics['aggregation_acc']:.4f} "
        f"val_ru_acc={final_metrics['ru_acc']:.4f}"
    )
    print(
        f"val_selected_vector_acc={final_metrics['selected_vector_acc']:.4f} "
        f"val_aggregation_vector_acc={final_metrics['aggregation_vector_acc']:.4f} "
        f"val_ru_vector_acc={final_metrics['ru_vector_acc']:.4f} "
        f"val_full_config_acc={final_metrics['full_config_acc']:.4f} "
        f"val_raw_feasibility={final_metrics['raw_feasibility']:.4f}"
    )

    print_confusion_matrix(
        "Selected STA confusion matrix",
        final_report["selected_confusion"],
        ["not_selected", "selected"],
    )
    print_confusion_matrix(
        "Aggregation depth confusion matrix",
        final_report["aggregation_confusion"],
        [str(i) for i in range(dataset.aggregation_classes)],
    )
    print_confusion_matrix(
        "RU class confusion matrix",
        final_report["ru_confusion"],
        ["none", "26", "52", "106", "242"],
    )

    confusion_path = output_dir / "validation_confusion_matrices.json"
    save_confusion_report(final_report, confusion_path)

    metadata_path = output_dir / "scheduler_model_metadata.json"
    metadata = torch.load(best_path, map_location="cpu")
    metadata.pop("model_state_dict")
    metadata["validation_metrics"] = {
        key: float(value) for key, value in final_metrics.items()
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    export_onnx(best_path, output_dir / "scheduler_model.onnx")
    print(f"saved_checkpoint={best_path}")
    print(f"saved_metadata={metadata_path}")
    print(f"saved_confusion_matrices={confusion_path}")


def export_onnx(checkpoint_path: Path, onnx_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = create_model(
        checkpoint.get("architecture", "mlp"),
        checkpoint["feature_count"],
        checkpoint["aggregation_classes"],
        checkpoint["hidden_size"],
        checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dummy = torch.zeros(1, NUM_STA, checkpoint["feature_count"], dtype=torch.float32)
    try:
        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            input_names=["features"],
            output_names=["selected_logits", "aggregation_logits", "ru_logits"],
            dynamic_axes={
                "features": {0: "batch"},
                "selected_logits": {0: "batch"},
                "aggregation_logits": {0: "batch"},
                "ru_logits": {0: "batch"},
            },
            opset_version=17,
        )
        print(f"saved_onnx={onnx_path}")
    except Exception as exc:
        print(f"ONNX export skipped: {exc}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train an ML scheduler from OFDMA-Sim JSONL samples.")
    parser.add_argument("--data", default="", help="Path to training_samples.jsonl")
    parser.add_argument("--output-dir", default="ml/out", help="Directory for model outputs")
    parser.add_argument("--init-checkpoint", default="", help="Optional compatible checkpoint for fine-tuning")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on number of samples")
    parser.add_argument("--max-packets-per-sta", type=int, default=0, help="Override inferred max packets per STA")
    parser.add_argument(
        "--feature-version",
        type=int,
        choices=[1, 2],
        default=2,
        help="Feature contract: 2 bounds queue features for dynamic workloads; 1 preserves legacy normalization.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--architecture",
        choices=["mlp", "set-equivariant"],
        default="mlp",
        help="Model family. The set model is equivariant to STA reordering.",
    )
    parser.add_argument(
        "--consistency-weight",
        type=float,
        default=0.0,
        help="Penalty aligning selection with nonzero aggregation and RU predictions.",
    )
    parser.add_argument(
        "--tone-budget-weight",
        type=float,
        default=0.0,
        help="Penalty on expected allocations exceeding the 242-tone budget.",
    )
    parser.add_argument(
        "--inactive-weight",
        type=float,
        default=0.0,
        help="Penalty on selecting inactive STA slots.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--selected-weight-mode",
        choices=["auto", "none"],
        default="auto",
        help="Use positive-class weighting for selected/not-selected prediction.",
    )
    parser.add_argument(
        "--aggregation-weight-mode",
        choices=["auto", "none"],
        default="auto",
        help="Use inverse-frequency class weights for aggregation depth prediction.",
    )
    parser.add_argument(
        "--ru-weight-mode",
        choices=["auto", "none"],
        default="auto",
        help="Use inverse-frequency class weights for RU class prediction.",
    )
    parser.add_argument(
        "--class-weight-power",
        type=float,
        default=0.5,
        help="Power applied to inverse-frequency weights. 0 disables imbalance strength, 1 is full inverse frequency.",
    )
    parser.add_argument("--min-class-weight", type=float, default=0.25)
    parser.add_argument("--max-class-weight", type=float, default=4.0)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument(
        "--split-by-seed",
        action="store_true",
        help="Keep all snapshots from one simulator seed in the same train/validation partition.",
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

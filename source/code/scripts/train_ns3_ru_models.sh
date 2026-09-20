#!/usr/bin/env bash
set -euo pipefail

EPOCHS="${1:-40}"
DATASET_NAME="${2:-ns3_ru_v4}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-${ROOT}/venv/bin/python}"

for objective in throughput delay; do
  output="${ROOT}/ml/${DATASET_NAME}/${objective}"
  mkdir -p "${output}"
  "${PYTHON}" "${ROOT}/ml/train_ns3_ru.py" \
    --data "${ROOT}/samples/${DATASET_NAME}/${objective}/training_samples.jsonl" \
    --output-dir "${output}" \
    --epochs "${EPOCHS}" \
    --batch-size 256 \
    --hidden-size 64 \
    --cpu | tee "${output}/training.log"
  "${PYTHON}" "${ROOT}/ml/export_q1_binary.py" \
    "${output}/scheduler_model.pt" \
    "${output}/scheduler_weights.bin"
done

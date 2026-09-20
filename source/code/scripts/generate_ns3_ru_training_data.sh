#!/usr/bin/env bash
set -euo pipefail

RUNS_PER_OBJECTIVE="${1:-36}"
DURATION="${2:-0.5}"
STRIDE="${3:-1}"
DATASET_NAME="${4:-ns3_ru_v4}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NS3_DIR="${ROOT}/ns-3.48"
OUT_ROOT="${ROOT}/samples/${DATASET_NAME}"

source "${HOME}/.bashrc"
mkdir -p "${OUT_ROOT}/throughput/parts" "${OUT_ROOT}/delay/parts"
if find "${OUT_ROOT}" -type f -name 'run_*.jsonl' -print -quit | grep -q .; then
  echo "Refusing to mix labels in existing dataset: ${OUT_ROOT}" >&2
  exit 1
fi
: > "${OUT_ROOT}/throughput/training_samples.jsonl"
: > "${OUT_ROOT}/delay/training_samples.jsonl"

optimizer_sha="$(sha256sum "${NS3_DIR}/src/wifi/model/he/q1-ru-optimizer.cc" | cut -d' ' -f1)"
scheduler_sha="$(sha256sum "${NS3_DIR}/src/wifi/model/he/rr-multi-user-scheduler.cc" | cut -d' ' -f1)"
printf '{"schema":"q1-ns3-ru-v4","runs_per_objective":%s,"duration_s":%s,"stride":%s,"git_revision":"%s","optimizer_sha256":"%s","scheduler_sha256":"%s"}\n' \
  "${RUNS_PER_OBJECTIVE}" "${DURATION}" "${STRIDE}" \
  "$(git -C "${NS3_DIR}" rev-parse --verify HEAD)" "${optimizer_sha}" "${scheduler_sha}" \
  > "${OUT_ROOT}/manifest.json"

cd "${NS3_DIR}"
./ns3 build q1-ofdma-validation

stations=(3 6 9)
mcs=(0 3 6 9 11)
loads=(1 4 10 20 35)
payloads=(300 800 1200 1500)
skews=(0.0 0.4 0.75)

for objective in throughput delay; do
  if [[ "${objective}" == "throughput" ]]; then
    delay_weight="0.0"
  else
    delay_weight="2.0"
  fi
  for ((run = 0; run < RUNS_PER_OBJECTIVE; run++)); do
    sta="${stations[$((run % ${#stations[@]}))]}"
    mcs_value="${mcs[$(((run / 3) % ${#mcs[@]}))]}"
    load="${loads[$(((run * 2 + 1) % ${#loads[@]}))]}"
    payload="${payloads[$(((run * 3 + 2) % ${#payloads[@]}))]}"
    skew="${skews[$(((run / 2) % ${#skews[@]}))]}"
    part="${OUT_ROOT}/${objective}/parts/run_${run}.jsonl"
    ./ns3 run "q1-ofdma-validation --policy=2 --stations=${sta} --mcs=${mcs_value} --seed=$((1000 + run)) --payloadBytes=${payload} --offeredMbpsPerSta=${load} --loadSkew=${skew} --duration=${DURATION} --warmup=1 --maxScheduledStations=9 --delayWeight=${delay_weight} --trainingOutput=${part} --trainingStride=${STRIDE}"
    cat "${part}" >> "${OUT_ROOT}/${objective}/training_samples.jsonl"
  done
  wc -l "${OUT_ROOT}/${objective}/training_samples.jsonl"
done

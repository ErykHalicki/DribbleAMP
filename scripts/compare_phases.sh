#!/bin/bash
set -euo pipefail

# Pull the three soccer model variants from the cluster, run eval_soccer.py
# on each, and produce a comparison bar chart.
#
# Variants:
#   phase1         -> "Phase 1 only"          (locomotion only)
#   phase2_scratch -> "Phase 2 only"          (single-stage baseline)
#   phase2         -> "Phase 1 + Phase 2"     (two-stage final)
#
# Usage:
#   scripts/compare_phases.sh
#   STEPS=2000 NUM_ENVS=8 scripts/compare_phases.sh

REMOTE_HOST=ehalicki@student-cluster.inf.ethz.ch
REMOTE_BASE=/work/courses/digital_human/team10/DribbleAMP/MimicKit/output

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
EVAL_DIR="${SCRIPT_DIR}/output/phase_compare"
mkdir -p "${EVAL_DIR}"

STEPS="${STEPS:-2000}"
NUM_ENVS="${NUM_ENVS:-8}"
ENV_CONFIG="${ENV_CONFIG:-data/envs/amp_soccer_humanoid_env_phase2.yaml}"

if [ -z "${DEVICE:-}" ]; then
    if (cd "${SCRIPT_DIR}" && .venv/bin/python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null); then
        DEVICE=cuda
    else
        DEVICE=cpu
    fi
fi
echo "Using device: ${DEVICE}"

read -s -p "Password for ${REMOTE_HOST}: " SSHPASS
echo
export SSHPASS

SSH_CMD="sshpass -e ssh -o StrictHostKeyChecking=accept-new"

resolve_remote_model() {
    local filter="$1"
    local glob="soccer_${filter}*"
    local filter_re="^${REMOTE_BASE}/soccer_${filter}(_[0-9]|/|\$)"
    local raw
    raw=$(${SSH_CMD} "${REMOTE_HOST}" "ls -1t ${REMOTE_BASE}/${glob}/model.pt 2>/dev/null" || true)
    echo "${raw}" | grep -E "${filter_re}" | head -n 1 || true
}

declare -A LABELS=(
    [phase1]="Phase 1 only"
    [phase2_scratch]="Phase 2 only"
    [phase2]="Phase 1 + Phase 2"
)

CSVS=()
TAGS=()

for filter in phase1 phase2_scratch phase2; do
    echo
    echo "=== ${LABELS[$filter]} (${filter}) ==="
    remote_model=$(resolve_remote_model "${filter}")
    if [ -z "${remote_model}" ]; then
        echo "ERROR: no remote model found for filter '${filter}'" >&2
        exit 1
    fi
    local_model="${EVAL_DIR}/model_${filter}.pt"
    csv_path="${EVAL_DIR}/eval_${filter}.csv"

    echo "Pulling ${remote_model}"
    rsync -e "${SSH_CMD}" --progress "${REMOTE_HOST}:${remote_model}" "${local_model}"

    echo "Running eval -> ${csv_path}"
    (
        cd "${SCRIPT_DIR}"
        .venv/bin/python scripts/eval_soccer.py \
            --model_file "${local_model}" \
            --env_config "${ENV_CONFIG}" \
            --num_envs "${NUM_ENVS}" \
            --total_steps "${STEPS}" \
            --device "${DEVICE}" \
            --csv "${csv_path}"
    )

    CSVS+=("${csv_path}")
    TAGS+=("${filter}:${LABELS[$filter]}")
done

unset SSHPASS

echo
echo "=== Plotting comparison ==="
(
    cd "${SCRIPT_DIR}"
    .venv/bin/python scripts/plot_phase_comparison.py \
        --csv "${CSVS[0]}" --label "${LABELS[phase1]}" \
        --csv "${CSVS[1]}" --label "${LABELS[phase2_scratch]}" \
        --csv "${CSVS[2]}" --label "${LABELS[phase2]}" \
        --out "${EVAL_DIR}/comparison.png"
)

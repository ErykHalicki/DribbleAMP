#!/bin/bash
set -euo pipefail

# Downloads every trained model referenced in the paper
# (report/latex/PaperForReview.tex) from the cluster into models/, so the
# demos in the README can be run from a fresh clone.
#
# Usage:
#   scripts/download_paper_models.bash
#   REMOTE_USER=<user> scripts/download_paper_models.bash   # skip the username prompt
#
# For each run we take the most recent checkpoint (final model.pt if present,
# otherwise the latest int_models/model_*.pt), same as sync_model.bash.

if [ -z "${REMOTE_USER:-}" ]; then
    read -p "Cluster username: " REMOTE_USER
fi
REMOTE_HOST=${REMOTE_USER}@student-cluster.inf.ethz.ch
REMOTE_BASE=/work/courses/digital_human/team10/DribbleAMP/MimicKit/output

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_DIR="${SCRIPT_DIR}/models"
mkdir -p "${MODELS_DIR}"

# <run filter> <local file name>
#   - phase1                 : humanoid Stage 1 (chase + kick)
#   - phase2                 : humanoid Stage 1 + Stage 2 curriculum (ours)
#   - phase2_scratch         : humanoid Stage 2 only (curriculum ablation)
#   - phase2_scratch_no_amp  : humanoid no-AMP baseline (style ablation)
#   - g1_newton_phase1       : Unitree G1 Stage 1
#   - g1_newton_phase2       : Unitree G1 Stage 2
MODELS=(
    "phase1                 humanoid_stage1.pt"
    "phase2                 humanoid_stage1_stage2.pt"
    "phase2_scratch         humanoid_stage2_only.pt"
    "phase2_scratch_no_amp  humanoid_no_amp.pt"
    "g1_newton_phase1       g1_stage1.pt"
    "g1_newton_phase2       g1_stage2.pt"
)

read -s -p "Password for ${REMOTE_HOST}: " SSHPASS
echo
export SSHPASS

SSH_CMD="sshpass -e ssh -o StrictHostKeyChecking=accept-new"

FAILED=()
for entry in "${MODELS[@]}"; do
    read -r RUN_FILTER LOCAL_NAME <<< "${entry}"

    # Match the run name *exactly* up to the timestamp, so `phase2` does not
    # also pick up `phase2_scratch_*` (run dirs look like
    # "soccer_<name>_YYYYMMDD_HHMMSS", some older ones have no timestamp).
    GLOB="soccer_${RUN_FILTER}*"
    FILTER_RE="^${REMOTE_BASE}/soccer_${RUN_FILTER}(_[0-9]|/|\$)"

    RAW_LISTING=$(${SSH_CMD} "${REMOTE_HOST}" "ls -1t ${REMOTE_BASE}/${GLOB}/model.pt ${REMOTE_BASE}/${GLOB}/int_models/model_*.pt 2>/dev/null" || true)
    REMOTE_MODEL=$(echo "${RAW_LISTING}" | grep -E "${FILTER_RE}" | head -n 1 || true)

    if [ -z "${REMOTE_MODEL}" ]; then
        echo "WARNING: no model found for '${RUN_FILTER}' under ${REMOTE_BASE}/" >&2
        FAILED+=("${RUN_FILTER}")
        continue
    fi

    echo "${RUN_FILTER}: ${REMOTE_MODEL}"
    echo "  -> models/${LOCAL_NAME}"
    rsync -e "${SSH_CMD}" --progress "${REMOTE_HOST}:${REMOTE_MODEL}" "${MODELS_DIR}/${LOCAL_NAME}"
done

unset SSHPASS

echo
echo "Done. Models in ${MODELS_DIR}:"
ls -lh "${MODELS_DIR}"

if [ "${#FAILED[@]}" -gt 0 ]; then
    echo
    echo "ERROR: failed to fetch: ${FAILED[*]}" >&2
    exit 1
fi

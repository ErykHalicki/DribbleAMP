#!/bin/bash
set -euo pipefail

# Usage:
#   scripts/sync_model.bash                # latest of any phase
#   scripts/sync_model.bash phase1         # latest phase1 run
#   scripts/sync_model.bash phase2         # latest 2-phase run
#   scripts/sync_model.bash phase2_scratch # latest 1-phase baseline
#   scripts/sync_model.bash phase2_scratch_no_amp  # latest no-AMP ablation
#   scripts/sync_model.bash phase2_20260524_141200  # specific run dir
#   RUN=<filter> scripts/sync_model.bash   # same as positional arg
#   REMOTE_MODEL=<full/remote/path> scripts/sync_model.bash  # exact file

REMOTE_HOST=ehalicki@student-cluster.inf.ethz.ch
REMOTE_BASE=/work/courses/digital_human/team10/DribbleAMP/MimicKit/output

RUN_FILTER="${1:-${RUN:-}}"

read -s -p "Password for ${REMOTE_HOST}: " SSHPASS
echo
export SSHPASS

SSH_CMD="sshpass -e ssh -o StrictHostKeyChecking=accept-new"

if [ -n "${REMOTE_MODEL:-}" ]; then
    echo "Using explicit REMOTE_MODEL: ${REMOTE_MODEL}"
else
    if [ -n "${RUN_FILTER}" ]; then
        GLOB="soccer_${RUN_FILTER}*"
        # Match the run name *exactly* up to the timestamp, so `phase2` does not
        # also pick up `phase2_scratch_*`. A run dir looks like
        # "soccer_<name>_YYYYMMDD_HHMMSS", so require an underscore + digit after
        # the filter (or the literal filter if the user passed a full dir name).
        FILTER_RE="^${REMOTE_BASE}/soccer_${RUN_FILTER}(_[0-9]|/|\$)"
        echo "Filtering remote runs by: ${GLOB} (regex: ${FILTER_RE})"
    else
        GLOB="soccer_phase*"
        FILTER_RE="."
    fi
    RAW_LISTING=$(${SSH_CMD} "${REMOTE_HOST}" "ls -1t ${REMOTE_BASE}/${GLOB}/model.pt ${REMOTE_BASE}/${GLOB}/int_models/model_*.pt 2>/dev/null" || true)
    REMOTE_MODEL=$(echo "${RAW_LISTING}" | grep -E "${FILTER_RE}" | head -n 1 || true)
fi

if [ -z "${REMOTE_MODEL}" ]; then
    echo "ERROR: no model found matching '${RUN_FILTER:-<any>}' under ${REMOTE_BASE}/" >&2
    echo "Raw ls listing (before regex filter):" >&2
    echo "${RAW_LISTING:-<empty>}" >&2
    echo "Available remote run dirs:" >&2
    ${SSH_CMD} "${REMOTE_HOST}" "ls -1d ${REMOTE_BASE}/soccer_* 2>/dev/null" >&2 || true
    exit 1
fi

REMOTE_RUN_DIR=$(dirname "${REMOTE_MODEL}")
if [[ "${REMOTE_RUN_DIR}" == */int_models ]]; then
    REMOTE_RUN_DIR=$(dirname "${REMOTE_RUN_DIR}")
fi

echo "Pulling model: ${REMOTE_MODEL}"
echo "Pulling log:   ${REMOTE_RUN_DIR}/log.txt"

rsync -e "${SSH_CMD}" --progress "${REMOTE_HOST}:${REMOTE_MODEL}" output/model.pt
rsync -e "${SSH_CMD}" --progress "${REMOTE_HOST}:${REMOTE_RUN_DIR}/log.txt" output/log.txt

unset SSHPASS

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
python "${SCRIPT_DIR}/scripts/plot_log.py" output/log.txt

# Pick the soccer_test.sh MODE preset from the run name. Check no_amp before
# phase2, since the no-AMP run dir name also contains "phase2".
if [[ "${REMOTE_MODEL}" == *soccer_phase2_scratch_no_amp* ]]; then
    export MODE=no_amp
elif [[ "${REMOTE_MODEL}" == *soccer_phase2* ]]; then
    export MODE=phase2
else
    export MODE=phase1
fi
echo "Testing with MODE=${MODE}"

bash "${SCRIPT_DIR}/scripts/soccer_test.sh"

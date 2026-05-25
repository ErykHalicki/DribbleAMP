#!/bin/bash
set -euo pipefail

# Usage:
#   scripts/sync_model.bash                # latest of any phase
#   scripts/sync_model.bash phase1         # latest phase1 run
#   scripts/sync_model.bash phase2         # latest 2-phase run
#   scripts/sync_model.bash phase2_scratch # latest 1-phase baseline
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
        echo "Filtering remote runs by: ${GLOB}"
    else
        GLOB="soccer_phase*"
    fi
    REMOTE_MODEL=$(${SSH_CMD} "${REMOTE_HOST}" "ls -1t ${REMOTE_BASE}/${GLOB}/model.pt ${REMOTE_BASE}/${GLOB}/int_models/model_*.pt 2>/dev/null | head -n 1")
fi

if [ -z "${REMOTE_MODEL}" ]; then
    echo "ERROR: no model found under ${REMOTE_BASE}/${GLOB:-soccer_phase*}/" >&2
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

if [[ "${REMOTE_MODEL}" == *soccer_phase2* ]]; then
    export ENV_CONFIG=data/envs/amp_soccer_humanoid_env_phase2.yaml
    echo "Testing with phase 2 env config"
else
    export ENV_CONFIG=data/envs/amp_soccer_humanoid_env_phase1.yaml
    echo "Testing with phase 1 env config"
fi

bash "${SCRIPT_DIR}/scripts/soccer_test.sh"

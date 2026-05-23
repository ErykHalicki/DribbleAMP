#!/bin/bash
set -euo pipefail

REMOTE_HOST=ehalicki@student-cluster.inf.ethz.ch
REMOTE_BASE=/work/courses/digital_human/team10/DribbleAMP/MimicKit/output

read -s -p "Password for ${REMOTE_HOST}: " SSHPASS
echo
export SSHPASS

SSH_CMD="sshpass -e ssh -o StrictHostKeyChecking=accept-new"

REMOTE_MODEL=$(${SSH_CMD} "${REMOTE_HOST}" "ls -1t ${REMOTE_BASE}/soccer_phase*/model.pt ${REMOTE_BASE}/soccer_phase*/int_models/model_*.pt 2>/dev/null | head -n 1")

if [ -z "${REMOTE_MODEL}" ]; then
    echo "ERROR: no model found under ${REMOTE_BASE}/soccer_phase*/" >&2
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

bash "${SCRIPT_DIR}/scripts/soccer_test.sh"

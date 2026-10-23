#!/usr/bin/env bash
# Wait for the detached PLAN_1 training session to end, then build the report.
#
# Usage:
#   tmux new-session -d -s plan1-report \
#     "bash scripts/reproduce_plan1/finalize_report.sh 2>&1 | tee -a outputs/plan1_finalize.log"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

SESSION="${PLAN1_SESSION:-plan1}"
REPORT_DIR="${REPORT_DIR:-$OUTPUT_ROOT/report}"

echo "Waiting for tmux session '$SESSION' to finish..."
while tmux has-session -t "$SESSION" 2>/dev/null; do
    sleep 300
done

echo "Session '$SESSION' finished at $(date -Is). Generating report..."
uv run python "$SCRIPT_DIR/make_report.py" \
    --outputs-root "$OUTPUT_ROOT" \
    --report-dir "$REPORT_DIR"

echo "Report: $REPORT_DIR/REPORT.md"

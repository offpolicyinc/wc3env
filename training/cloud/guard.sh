#!/bin/bash
# The automatic stop of ONE self-play run on Modal: after MINUTES, its app, its sandboxes and their containers
# (training.cloud.selfplay stop), repeated until stop finds the app stopped and no sandbox alive; other runs are left
# alone. Start it next to every run, in the background:
#     nohup bash training/cloud/guard.sh sp13 70 > build/guard-sp13.log 2>&1 &
# python -m training.cloud.stop_all stops everything of ours instead.
set -o pipefail
NAME=$1 MINUTES=$2
sleep $(( MINUTES * 60 ))
cd "$(dirname "$0")/../.."
export MODAL_PROFILE=${MODAL_PROFILE:-peter-nico} PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONPATH=".;src"
until date -u && python -m training.cloud.selfplay stop --name "$NAME" 2>&1 | grep -viE "deprecat|set_event" | tail -1; do
  sleep 20
done

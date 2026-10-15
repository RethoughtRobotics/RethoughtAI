#!/bin/bash
# Bimanual HF dataset watcher, restarted if it exits. Logs: cache/watch/watcher.log
cd /projects/e52938/kmy2091
export HF_HOME=$PWD/cache/huggingface HF_TOKEN_PATH=$HOME/.cache/huggingface/token
while true; do
  rethought-lerobot/.venv/bin/python -u cache/watch/watch_bimanual.py >> cache/watch/watcher.log 2>&1
  echo "$(date +%T) watcher exited ($?), restarting in 60s" >> cache/watch/watcher.log; sleep 60
done

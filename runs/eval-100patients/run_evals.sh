#!/bin/bash
# Evaluate every run on 100 test patients (seeds 100-199) into one folder.
cd "c:/Users/Scott/Projects/LLM-Based-Reinforcement-Learning-for-Chemo-Dosing"
PY=.venv/Scripts/python.exe
OUT=runs/eval-100patients
SEEDS=$(seq 100 199)
R=runs
ev() { echo "=== $(date '+%H:%M:%S') start: $*"; $PY scripts/evaluate.py "$@" --seeds $SEEDS --out $OUT || echo "!!! FAILED: $*"; echo "=== $(date '+%H:%M:%S') done"; }

ev configs/qwen3_8b.yaml --mode baseline
ev configs/qwen3_8b.yaml --mode trained --adapter $R/qwen3_8b-seed0-20261118-102431-run2/adapter-final --label run2-qwen-penalty-history
ev configs/qwen3_8b_env_reward.yaml --mode trained --adapter $R/qwen3_8b_env_reward-seed0-20261118-234149-run3/adapter-final --label run3-qwen-envreward-history
ev configs/medgemma_4b.yaml --mode trained --adapter $R/medgemma_4b-seed0-20261119-092218-run4/adapter-final --label run4-medgemma-penalty-history
ev configs/qwen3_8b.yaml --mode zeroshot --label qwen-zeroshot-history
ev configs/medgemma_4b.yaml --mode zeroshot --label medgemma-zeroshot-history
ev configs/qwen3_8b_no_history.yaml --mode trained --adapter $R/qwen3_8b-seed0-20261117-170451-run1/adapter-final --label run1-qwen-envreward-nohistory
ev configs/qwen3_8b_no_history.yaml --mode zeroshot --label qwen-zeroshot-nohistory
echo "ALL DONE $(date)"

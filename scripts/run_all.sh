#!/usr/bin/env bash
# Run every required experiment in dependency order. Safe to re-run after a disconnect:
# finished adapters / result files are detected and skipped.
#
#   bash scripts/run_all.sh            # everything
#   bash scripts/run_all.sh t1 t3      # only some stages (t1 t2 t3 t4 t5 fig)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
STAGES=${*:-"t1 t2 t3 t4 t5 fig"}

have() { [ -e "$1" ]; }
run() { echo; echo ">>> $*"; "$PY" -m "$@"; }

for s in $STAGES; do case $s in
t1)
  have outputs/task1_dpo/standard/adapter_config.json || run task1_dpo.train --config configs/dpo.yaml --run-name standard
  have results/task1_dpo/eval_sft.json      || run task1_dpo.evaluate --config configs/dpo.yaml --adapter none --name sft
  have results/task1_dpo/eval_standard.json || run task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/standard --name standard
  run task1_dpo.ablate_beta --config configs/dpo.yaml
  run task1_dpo.analyze_length --config configs/dpo.yaml
  ;;
t2)
  have outputs/task2_ppo/standard/adapter_config.json || run task2_ppo.continue_train --config configs/ppo.yaml --run-name standard
  have results/task2_ppo/eval/standard.json || run task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard --name standard
  have results/task2_ppo/eval/sft.json      || run task2_ppo.evaluate --config configs/ppo.yaml --adapter none --name sft
  run task2_ppo.analyze_clipping --config configs/ppo.yaml
  run task2_ppo.ablate_kl --config configs/ppo.yaml
  ;;
t3)
  have outputs/task3_grpo/standard/adapter_config.json || run task3_grpo.continue_train --config configs/grpo.yaml --run-name standard
  have results/task3_grpo/eval/standard.json || run task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/standard --name standard
  run task3_grpo.analyze_group_size --config configs/grpo.yaml
  run task3_grpo.compare_normalization --config configs/grpo.yaml
  ;;
t4)
  run task4_safety.generate_responses --config configs/feedback.yaml
  run task4_safety.make_audit_sheet --config configs/feedback.yaml
  run task4_safety.judge_responses --config configs/feedback.yaml
  run task4_safety.evaluate_safety --config configs/feedback.yaml   # re-run after filling the manual audit sheet
  ;;
t5)
  run task5_feedback.evaluate_math --config configs/feedback.yaml --dataset gsm
  run task5_feedback.score_perturbations --config configs/feedback.yaml
  run task5_feedback.evaluate_math --config configs/feedback.yaml --dataset transfer
  run task5_feedback.compare_feedback --config configs/feedback.yaml
  ;;
fig)
  run report.make_figures
  ;;
*) echo "unknown stage $s"; exit 1 ;;
esac; done
echo; echo "done: $STAGES"

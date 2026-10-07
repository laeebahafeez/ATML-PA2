"""End-to-end CPU smoke test of every task pipeline with tiny random Qwen2 models.

Copies this repository to a scratch directory, swaps in tiny models / adapters / data subsets,
and runs every task command in README order. It validates plumbing (shapes, masks, files, resume
logic), not results. Requires the course data/cached assets to be downloaded.

    python -m tests.smoke.run_smoke --workdir /tmp/pa2_smoke
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TINY_LM = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
TINY_CLS = "trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5"

SMOKE_YAML = {
    "base.yaml": """seed: 6304
base_model: {lm}
reward_model: {cls}
reward_tokenizer: {lm}
ai_judge_model: {lm}
value_model_init: {cls}
dtype: {dtype}
quantize_frozen_models: false
lora: {{r: 4, alpha: 8, dropout: 0.05, target_modules: [q_proj, v_proj]}}
value_lora: {{r: 4, alpha: 8, dropout: 0.05, target_modules: [q_proj, v_proj]}}
generation: {{temperature: 0.7, top_p: 0.9, do_sample: true}}
paths:
  dpo_standard_train: data/dpo_standard_train.jsonl
  dpo_standard_eval: data/dpo_standard_eval.jsonl
  dpo_length_train: data/dpo_length_balanced_train.jsonl
  dpo_length_eval: data/dpo_length_stratified_eval.jsonl
  word_limit_prompts: data/word_limit_prompts.jsonl
  rl_prompt_train: data/rl_prompt_pool_train.jsonl
  rl_prompt_eval: data/rl_prompt_pool_eval.jsonl
  xstest: data/xstest_safety_prompts.csv
  gsm_train: data/gsm8k_rl_train.jsonl
  gsm_eval: data/gsm8k_eval.jsonl
  math_transfer_eval: data/math_transfer_eval.jsonl
  task5_diagnostics: data/task5_controlled_reward_diagnostics.jsonl
  ppo_rollout_cache: cached/ppo_rollout.pt
  grpo_k_cache: cached/grpo_k_cache.jsonl
  ppo_midpoint_policy: checkpoints/ppo_midpoint_policy
  ppo_midpoint_value: checkpoints/ppo_midpoint_value
  grpo_midpoint_policy: checkpoints/grpo_midpoint_policy
  rlvr_policy: checkpoints/rlvr_policy
  rlaif_policy: checkpoints/rlaif_policy
""",
    "dpo.yaml": """base_config: configs/base.yaml
beta: 0.10
betas: [0.03, 0.30]
learning_rate: 2.0e-5
weight_decay: 0.0
batch_size: 2
grad_accum_steps: 2
epochs: 1
short_ablation_examples: 4
max_sequence_length: 768
max_generation_tokens: 12
max_grad_norm: 1.0
standard_output: outputs/task1_dpo/standard
length_output: outputs/task1_dpo/length_balanced
results_dir: results/task1_dpo
eval_batch_size: 2
eval_generation_prompts: 4
eval_generation_batch_size: 2
reward_max_length: 512
""",
    "ppo.yaml": """base_config: configs/base.yaml
updates: 2
fork_updates: 1
prompts_per_update: 1
ppo_epochs: 2
policy_learning_rate: 3.0e-6
value_lora_learning_rate: 1.0e-4
value_head_learning_rate: 3.0e-4
value_train_mode: lora_head
clip_epsilon: 0.20
clip_values: [0.05, 0.20]
kl_beta: 0.10
kl_values: [0.0, 0.10]
gamma: 1.0
gae_lambda: 0.95
value_coef: 0.50
missing_eos_penalty: 1.0
max_prompt_length: 256
max_response_length: 12
eval_max_response_length: 12
reward_max_length: 1280
max_grad_norm: 1.0
output: outputs/task2_ppo/standard
results_dir: results/task2_ppo
cached_rollouts: cached/ppo_rollout.pt
eval_prompts: 3
eval_batch_size: 3
cached_clip_steps: 1
""",
    "grpo.yaml": """base_config: configs/base.yaml
updates: 2
fork_updates: 1
prompts_per_update: 1
policy_epochs: 1
learning_rate: 5.0e-6
clip_epsilon: 0.20
kl_beta: 0.10
num_generations: 4
group_sizes: [2, 4, 8]
max_prompt_length: 256
max_completion_length: 12
mask_truncated_completions: false
max_grad_norm: 1.0
output: outputs/task3_grpo/standard
results_dir: results/task3_grpo
group_cache: cached/grpo_k_cache.jsonl
cache_generation_cap: 48
grad_micro_batch: 2
eval_prompts: 3
eval_batch_size: 3
eval_max_response_length: 12
reward_max_length: 1280
""",
    "feedback.yaml": """base_config: configs/base.yaml
results_dir: results
safety_max_new_tokens: 12
math_max_new_tokens: 16
judge_max_new_tokens: 8
judge_temperature: 0.0
manual_audit_per_class: 2
policies:
  sft: null
  dpo: outputs/task1_dpo/standard
  ppo: outputs/task2_ppo/standard
  grpo: outputs/task3_grpo/standard
  rlvr: checkpoints/rlvr_policy
  rlaif: checkpoints/rlaif_policy
""",
}

# Data subsets (first N rows). Diagnostics are trimmed to 2 complete problems.
SUBSETS = {
    "data/dpo_standard_train.jsonl": 8,
    "data/dpo_standard_eval.jsonl": 6,
    "data/dpo_length_balanced_train.jsonl": 6,
    "data/gsm8k_eval.jsonl": 4,
    "data/math_transfer_eval.jsonl": 3,
}

COMMANDS = [
    "task1_dpo.train --run-name standard",
    "task1_dpo.evaluate --adapter none --name sft",
    "task1_dpo.evaluate --adapter outputs/task1_dpo/standard --name standard",
    "task1_dpo.ablate_beta",
    "task1_dpo.analyze_length",
    "task2_ppo.continue_train --run-name standard",
    "task2_ppo.evaluate --adapter outputs/task2_ppo/standard --name standard",
    "task2_ppo.evaluate --adapter midpoint --name midpoint",
    "task2_ppo.analyze_clipping",
    "task2_ppo.ablate_kl",
    "task3_grpo.continue_train --run-name standard",
    "task3_grpo.evaluate --adapter outputs/task3_grpo/standard --name standard",
    "task3_grpo.analyze_group_size --permutations 5",
    "task3_grpo.compare_normalization",
    "task4_safety.generate_responses",
    "task4_safety.make_audit_sheet",
    "task4_safety.judge_responses --batch-size 4",
    "task4_safety.evaluate_safety",
    "task5_feedback.evaluate_math --dataset gsm",
    "task5_feedback.score_perturbations",
    "task5_feedback.evaluate_math --dataset transfer",
    "task5_feedback.compare_feedback",
    "report.make_figures",
]


def build(work: Path, dtype: str = "float32"):
    if work.exists():
        shutil.rmtree(work)
    ignore = shutil.ignore_patterns(".venv", ".git", "outputs", "results", "checkpoints", "__pycache__", "*.pyc", "smoke_work*")
    shutil.copytree(REPO, work, ignore=ignore)
    for name, text in SMOKE_YAML.items():
        if name == "base.yaml":
            text = text.format(lm=TINY_LM, cls=TINY_CLS, dtype=dtype)
        (work / "configs" / name).write_text(text, encoding="utf-8")

    for rel, n in SUBSETS.items():
        lines = (REPO / rel).read_text(encoding="utf-8").splitlines()[:n]
        (work / rel).write_text("\n".join(lines) + "\n", encoding="utf-8")
    # 2 rows per length stratum
    rows = [json.loads(x) for x in (REPO / "data/dpo_length_stratified_eval.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    keep, seen = [], {}
    for r in rows:
        if seen.get(r["length_stratum"], 0) < 2:
            keep.append(r)
            seen[r["length_stratum"]] = seen.get(r["length_stratum"], 0) + 1
    (work / "data/dpo_length_stratified_eval.jsonl").write_text("\n".join(json.dumps(r) for r in keep) + "\n", encoding="utf-8")
    # 3 SAFE + 3 UNSAFE XSTest prompts
    import pandas as pd
    x = pd.read_csv(REPO / "data/xstest_safety_prompts.csv")
    pd.concat([x[x.benchmark_class == "SAFE"].head(3), x[x.benchmark_class == "UNSAFE"].head(3)]).to_csv(work / "data/xstest_safety_prompts.csv", index=False)
    # 2 diagnostic problems x 5 variants
    diag = [json.loads(l) for l in (REPO / "data/task5_controlled_reward_diagnostics.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    pids = sorted({str(d["problem_id"]) for d in diag})[:2]
    (work / "data/task5_controlled_reward_diagnostics.jsonl").write_text(
        "\n".join(json.dumps(d) for d in diag if str(d["problem_id"]) in pids) + "\n", encoding="utf-8")
    # 3 prompts of the GRPO K-cache, 4 PPO cached rollouts
    gc = [json.loads(l) for l in (REPO / "cached/grpo_k_cache.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    gp = []
    for r in gc:
        if r["source_index"] not in gp:
            gp.append(r["source_index"])
    (work / "cached/grpo_k_cache.jsonl").write_text("\n".join(json.dumps(r) for r in gc if r["source_index"] in gp[:3]) + "\n", encoding="utf-8")
    import torch
    torch.save(torch.load(REPO / "cached/ppo_rollout.pt", weights_only=False)[:4], work / "cached/ppo_rollout.pt")

    # Tiny stand-ins for the supplied checkpoints.
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification

    ck = work / "checkpoints"
    for i, name in enumerate(["ppo_midpoint_policy", "grpo_midpoint_policy", "rlvr_policy", "rlaif_policy"]):
        torch.manual_seed(i)
        m = AutoModelForCausalLM.from_pretrained(TINY_LM, dtype=torch.float32)
        m = get_peft_model(m, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM", init_lora_weights=False))
        m.save_pretrained(str(ck / name))
    AutoModelForSequenceClassification.from_pretrained(TINY_CLS, num_labels=1).save_pretrained(str(ck / "ppo_midpoint_value"))


def sync_code(work: Path):
    """Copy current *.py sources into an existing smoke env (configs/data/assets untouched)."""
    for src in REPO.rglob("*.py"):
        rel = src.relative_to(REPO)
        if rel.parts[0] in {".venv", "smoke_work"} or rel.parts[:2] == ("tests", "smoke"):
            continue
        dst = work / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def run(work: Path, python: str, only: list[str] | None):
    log = []
    for cmd in COMMANDS:
        if only and not any(cmd.startswith(o) for o in only):
            continue
        if cmd.startswith("task4_safety.evaluate_safety"):
            fill_fake_audit(work)
        t0 = time.time()
        p = subprocess.run([python, "-m", *cmd.split()], cwd=work, capture_output=True, text=True)
        dt = time.time() - t0
        ok = p.returncode == 0
        log.append((cmd, ok, dt))
        print(f"{'OK  ' if ok else 'FAIL'} {dt:6.1f}s  {cmd}", flush=True)
        if not ok:
            print(p.stdout[-3000:])
            print(p.stderr[-6000:])
            return False
    return True


def fill_fake_audit(work: Path):
    """Smoke only: put labels in the blind sheet so the agreement code path runs."""
    import pandas as pd
    p = work / "results/task4_safety/manual_audit_sheet.csv"
    if p.exists():
        df = pd.read_csv(p)
        labels = ["SAFE_ANSWER", "JUSTIFIED_REFUSAL", "AMBIGUOUS", "OVER_REFUSAL"]
        df["manual_label"] = [labels[i % 4] for i in range(len(df))]
        df.to_csv(p, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=str(REPO / "smoke_work"))
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--dtype", default="float16", help="model dtype; float16 matches the GPU runs")
    ap.add_argument("--sync", action="store_true", help="with --no-build: refresh code in the existing env")
    ap.add_argument("--only", nargs="*", help="command prefixes to run (default: all)")
    args = ap.parse_args()
    work = Path(args.workdir)
    if not args.no_build:
        build(work, args.dtype)
        print("built smoke env at", work, "dtype", args.dtype)
    elif args.sync:
        sync_code(work)
    sys.exit(0 if run(work, sys.executable, args.only) else 1)


if __name__ == "__main__":
    main()

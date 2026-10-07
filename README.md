# ATML PA2 - LLM Post-Training

<!-- FINAL_STUDENT_SETUP -->

## Quick start

```bash
git clone https://github.com/AbDu11aHHH/ATML-PA2-LLM-PostTraining.git
cd ATML-PA2-LLM-PostTraining
python -m pip install -r requirements.txt
python -m scripts.download_assets
python -m scripts.validate_assets
```

The fixed datasets, cached diagnostics, and supplied
continuation checkpoints are downloaded from:

https://huggingface.co/datasets/AbDu11aHHH/ATML-PA2-assets

Pinned release revision:

`0b350481fb03f5525a35bcdec4131bd4fe487f98`

---
# Student implementation

All Task 1-5 pipelines are implemented. Every reported number is written to `results/` by one
of the commands below, and `python -m report.make_figures` turns those files into
`report/figures/*.pdf` and the numbers digest `report/tables.md`.

## Reproduce everything

```bash
python -m pip install -r requirements.txt
python -m scripts.download_assets && python -m scripts.validate_assets
python -m pytest tests -q            # objective-validation unit tests (CPU, ~1 min)
bash scripts/run_all.sh              # all experiments in dependency order; resumable
```

`scripts/run_all.sh t1 t2 t3 t4 t5 fig` runs individual stages; each stage skips adapters/result files
that already exist. `notebooks/colab_runner.ipynb` runs the same stages on Colab/Kaggle with
`outputs/` and `results/` kept on persistent storage. The per-task commands are:

| Task | Commands (in order) | Main result files |
|---|---|---|
| 1 DPO | `task1_dpo.train --run-name standard`; `task1_dpo.evaluate --adapter none --name sft`; `task1_dpo.evaluate --adapter outputs/task1_dpo/standard --name standard`; `task1_dpo.ablate_beta`; `task1_dpo.analyze_length` | `results/task1_dpo/eval_*.json`, `beta_ablation.json`, `length_analysis.json`, `train_*_log.jsonl` |
| 2 PPO | `task2_ppo.continue_train --run-name standard`; `task2_ppo.evaluate --adapter outputs/task2_ppo/standard --name standard`; `task2_ppo.analyze_clipping`; `task2_ppo.ablate_kl` | `results/task2_ppo/standard/{train_log.jsonl,summary.json}`, `clipping_cached_batch.json`, `clipping_forks.json`, `kl_ablation.json`, `eval/*.json` |
| 3 GRPO | `task3_grpo.continue_train --run-name standard`; `task3_grpo.evaluate --adapter outputs/task3_grpo/standard --name standard`; `task3_grpo.analyze_group_size`; `task3_grpo.compare_normalization` | `results/task3_grpo/standard/*`, `group_size_analysis.json`, `normalization_comparison.json` |
| 4 Safety | `task4_safety.generate_responses`; `task4_safety.make_audit_sheet`; `task4_safety.judge_responses`; *(fill the manual sheet)*; `task4_safety.evaluate_safety` | `results/task4_safety/safety_summary.json`, `safety_calibration_table.csv`, `category_label_distribution.csv`, `audit_*.csv` |
| 5 Feedback | `task5_feedback.evaluate_math --dataset gsm`; `task5_feedback.score_perturbations`; `task5_feedback.evaluate_math --dataset transfer`; `task5_feedback.compare_feedback` | `results/task5_feedback/eval_{gsm,transfer}.json`, `diagnostic_summary.json`, `feedback_comparison.json` |

(all invoked as `python -m <module> [--config configs/<task>.yaml]`)

## Corrected objective defects (validated in `tests/test_objectives.py`)

| File | Defect in the starter | Fix |
|---|---|---|
| `task1_dpo/dpo.py` | DPO logit used `policy_margin + ref_margin`, rewarding agreement with the reference instead of movement relative to it (loss != log 2 at initialization) | `beta * (policy_margin - ref_margin)` |
| `task2_ppo/ppo.py` | clipped surrogate took `torch.maximum(rho A, clip(rho) A)`, the optimistic bound, which removes the trust region | `torch.minimum` (pessimistic bound) |
| `task3_grpo/grpo.py` | advantages normalized with the mean/std of the whole batch, ignoring `group_ids` | per-prompt `(r - mu_g) / (sigma_g + eps)`; zero-std groups get zero advantage |

The tests fail on the starter versions (6 of 9) and pass on the fixed versions.
`tests/test_grpo_microbatch.py` checks that the micro-batched GRPO step gives the same loss and
gradients as the released full-batch loss.

## Implementation notes and protocol decisions

- **DPO over-long prompts.** `encode_prompt_response` raises when the prompt alone does not fit
  `max_sequence_length=768`. Pairs whose prompt leaves fewer than 32 response tokens are filtered
  deterministically for every condition: about 3.7% of each split (56/1500 standard train, 62/1500
  length-balanced train, 10/300 eval, 9/246 stratified eval). Dropped ids are logged in
  `results/task1_dpo/train_*_summary.json`.
- **DPO budgets.** The standard and length-balanced models train one full epoch. The beta forks
  train on the first 600 (filtered) pairs, with the same seed/order/optimizer/LoRA settings.
- **Common generation protocol** (`common/policy_eval.py`). For every Task 1-3 condition: the
  same first-N held-out prompts, the same seed, the release sampling settings (T=0.7, top-p=0.9),
  and the same cap. Reported per condition: KL (course sampled estimator, token-weighted
  `mean(log pi - log pi_ref)`), exact full-vocabulary entropy, reward-model score, and length
  mean/std/median/IQR. Sampled decoding is used because the sampled KL estimator needs samples
  from the policy itself.
- **On-policy RL details.** LoRA dropout is disabled in PPO/GRPO, so the ratio is exactly 1 before
  each first step (no spurious clipping). Trainable parameters (including the critic head) are
  kept in fp32 for AdamW. Generation tensors are cloned out of `inference_mode` before training.
  All forks share one seeded prompt schedule.
- **PPO.** Missing-EOS responses get `missing_eos_penalty` subtracted from the terminal reward.
  Advantages are whitened per batch (released helper). Logged per update: reward, KL, policy and
  value loss, entropy, policy/critic gradient norm, clip fraction, gradient-blocked
  ("affected") fraction, approx KL(old||new), critic explained variance, length, peak VRAM, and
  wall-clock. The (eps=0.2, beta_KL=0.1) fork is shared by the clipping and KL studies.
- **PPO cached batch.** Response token ids are rebuilt by re-encoding the cached text (+EOS when
  terminated); token counts match the cache for all rows (mismatches would be reported and
  excluded). For each epsilon, the study starts from the same midpoint and takes
  `cached_clip_steps` full-batch clipped steps on the fixed batch.
- **GRPO.** Dr.-GRPO differs only in the sequence normalization (1/L_max instead of 1/T_k);
  advantages are identical in both conditions, so the comparison isolates normalization. The
  length-conditioned statistic is measured twice: gradient-mass share in training, and actual
  per-completion gradient norms at the midpoint on the fixed completion cache.
- **Group-size study.** Each prompt's 8 cached completions are split into 8/K disjoint contiguous
  groups (192 generations for every K), and every statistic is also averaged over 500 random
  within-prompt permutations. Difficulty bins are terciles of the per-prompt mean reward. A
  binarized-reward (above global median) sensitivity is included.
- **Task 4 judge.** `judge_responses` batches the released judge prompt with the same chat
  template, greedy decoding and parser (`--batch-size 1` calls `judge_one` exactly). The judge
  never sees the XSTest class, so class-inconsistent labels are reported separately. The manual
  audit sheet is blind (shuffled, with policy and AI label kept in a separate key file).
- **Task 5 judge.** The pairwise judge compares each RL policy (A) with SFT (B) on every problem.
  Diagnostic pairs are judged in both presentation orders, and order-consistency is reported.

## Validation without a GPU

`python -m tests.smoke.run_smoke --workdir <scratch dir>` copies the repo, swaps in tiny random
Qwen2 models, tiny adapters and data subsets, and runs every command above end-to-end on CPU
(about 15 minutes). It checks plumbing, not results.

## Attribution

Starter code, data, checkpoints, judge/verifier utilities: ATML PA2 course release. Helper
utilities and all training/evaluation/analysis code beyond the starter were written for this
submission with the assistance of Claude (Anthropic); no other external code was reused.

---
# ATML PA2 - LLM Post-Training

This is the **student starter repository** for ATML PA2. The released code is intentionally incomplete: Tasks 1-3 provide model/data loading, objective helpers, checkpoint restoration, and experiment entry points, but **you must implement the training loops and ablation orchestration yourself**. Each of Tasks 1-3 also contains one deliberate algorithmic defect in its core objective code; identifying and correcting these defects is part of validating your implementation.

Task 4 supplies the fixed AI safety judge and response-generation utilities, but you must write the evaluation/aggregation code. Task 5 supplies the exact RLVR verifier, the fixed pairwise AI judge used for RLAIF evaluation, and data/model loaders; you must implement the requested evaluation and analysis.

## 1. Clone and install

```bash
git clone https://github.com/COURSE_ORG/ATML-PA2-LLM-PostTraining.git
cd ATML-PA2-LLM-PostTraining
python -m pip install -r requirements.txt
```

## 2. Download the course assets

The large course-created checkpoints and fixed data are distributed as a GitHub Release asset rather than normal Git files. After cloning, run:

```bash
python -m scripts.download_assets
python -m scripts.validate_assets
```

If your instructor provides a direct asset URL separately, use:

```bash
python -m scripts.download_assets --url '<ASSET_URL>'
```

Public base/reward/judge models are downloaded from Hugging Face at runtime and are **not** included in the course asset archive.

The installer also materializes the fixed 100-example Task 5 transfer set from the official SVAMP challenge-set source if it is not already present. The tiny Task 1 word-limit prompt set is tracked directly in this repository.

## 3. Environment check

```bash
python -m scripts.check_environment
```

Run commands from the repository root. The reference environment used to prepare the release pins Transformers 4.57.1, TRL 0.27.2, PEFT 0.17.1, and Tokenizers 0.22.1.

## 4. Supplied course checkpoints

After `download_assets`, these directories should exist:

```text
checkpoints/ppo_midpoint_policy/
checkpoints/ppo_midpoint_value/
checkpoints/grpo_midpoint_policy/
checkpoints/rlvr_policy/
checkpoints/rlaif_policy/
```

PPO and GRPO begin from the supplied continuation checkpoints. RLVR and RLAIF are supplied frozen evaluation policies; students do not retrain them.

The PPO value checkpoint is intentionally released as the exact staff midpoint state, including its imperfect held-out value calibration. Treat critic behavior as an analysis variable rather than assuming a perfect baseline, and start every PPO fork from the identical supplied policy/value state. The default continuation generation cap is 512 tokens for feasibility; frozen evaluation uses the larger cap specified in `configs/ppo.yaml`.

## 5. Task entry points

### Task 1 - DPO

```bash
python -m task1_dpo.train --config configs/dpo.yaml --run-name standard
python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/standard --name standard
python -m task1_dpo.ablate_beta --config configs/dpo.yaml
python -m task1_dpo.analyze_length --config configs/dpo.yaml
```

### Task 2 - PPO

```bash
python -m task2_ppo.continue_train --config configs/ppo.yaml --run-name standard
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard --name standard
python -m task2_ppo.analyze_clipping --config configs/ppo.yaml
python -m task2_ppo.ablate_kl --config configs/ppo.yaml
```

### Task 3 - GRPO

```bash
python -m task3_grpo.continue_train --config configs/grpo.yaml --run-name standard
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/standard --name standard
python -m task3_grpo.analyze_group_size --config configs/grpo.yaml
python -m task3_grpo.compare_normalization --config configs/grpo.yaml
```

### Task 4 - Safety calibration

The judge loader/parser are supplied. You must implement the requested generation aggregation and evaluation.

```bash
python -m task4_safety.generate_responses --config configs/feedback.yaml
python -m task4_safety.judge_responses --config configs/feedback.yaml
python -m task4_safety.make_audit_sheet --config configs/feedback.yaml
python -m task4_safety.evaluate_safety --config configs/feedback.yaml
```

### Task 5 - RLVR vs RLAIF

The exact verifier and pairwise AI judge are supplied; you implement the evaluation/analysis.

```bash
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset gsm
python -m task5_feedback.score_perturbations --config configs/feedback.yaml
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset transfer
python -m task5_feedback.compare_feedback --config configs/feedback.yaml
```

## 6. Reproducibility rules

- Do not alter course-provided data, cached rollouts, or supplied checkpoints.
- Start every short fork from the **same supplied midpoint checkpoint**.
- Keep prompt IDs, generated-token/update budgets, seed, and evaluation procedure matched across ablations.
- Commit your code, configs, small JSON/CSV logs, and figures. Do not commit downloaded checkpoints, raw course assets, or model caches.
- Record peak VRAM and wall-clock time for the standard PPO and GRPO continuations.

See the assignment manual for the required experiments, metrics, and report questions.

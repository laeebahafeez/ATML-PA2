"""Task 1 Step 3: length-confounding study.

1. Dataset property: how strongly is "preferred" correlated with "longer" in the standard
   training/eval pairs? (a length-only predictor's accuracy is reported as a baseline)
2. Train one DPO model on the supplied length-balanced subset (same budget/settings as standard).
3. Policy property: evaluate standard vs length-balanced DPO on the length-stratified held-out
   set per stratum (preferred_longer / length_matched / rejected_longer), and compare generated
   length and explicit word-limit compliance on the common prompt set.
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from common.data import load_yaml, preference_responses, read_jsonl, repo_path, write_jsonl
from common.logging_utils import load_json, save_json
from common.metrics import safe_corr
from common.models import clear_gpu, load_policy, load_tokenizer
from task1_dpo.evaluate import preference_metrics, run_evaluation, summarize_pairs
from task1_dpo.train import run_training

STRATA = ["preferred_longer", "length_matched", "rejected_longer"]


def dataset_length_profile(rows, tokenizer):
    diffs = []
    for r in rows:
        yc, yr = preference_responses(r)
        diffs.append(len(tokenizer(yc, add_special_tokens=False)["input_ids"]) -
                     len(tokenizer(yr, add_special_tokens=False)["input_ids"]))
    d = np.asarray(diffs, dtype=float)
    return {
        "n": int(d.size),
        "frac_chosen_longer": float((d > 0).mean()),
        "frac_rejected_longer": float((d < 0).mean()),
        "frac_equal": float((d == 0).mean()),
        "mean_token_diff_chosen_minus_rejected": float(d.mean()),
        "median_token_diff": float(np.median(d)),
        # Accuracy of the trivial rule "the longer response is preferred" (ties count 0.5).
        "length_only_predictor_accuracy": float(((d > 0) + 0.5 * (d == 0)).mean()),
    }


def per_stratum(pairs):
    by = defaultdict(list)
    for p in pairs:
        by[p["length_stratum"]].append(p)
    out = {s: summarize_pairs(by[s]) for s in STRATA if by[s]}
    for s in out:
        out[s]["mean_token_diff"] = float(np.mean([p["chosen_tokens"] - p["rejected_tokens"] for p in by[s]]))
    diffs = [p["chosen_tokens"] - p["rejected_tokens"] for p in pairs]
    out["all"] = summarize_pairs(pairs)
    out["corr_margin_vs_length_diff"] = safe_corr(diffs, [p["dpo_margin"] for p in pairs])
    out["corr_chosen_logratio_vs_chosen_len"] = safe_corr([p["chosen_tokens"] for p in pairs], [p["chosen_logratio"] for p in pairs])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    res = repo_path(cfg["results_dir"])
    tok = load_tokenizer(cfg["base_model"])

    balanced = read_jsonl(cfg["paths"]["dpo_length_train"])
    stratified = read_jsonl(cfg["paths"]["dpo_length_eval"])
    print("Length-balanced train rows:", len(balanced))
    print("Length-stratified eval rows:", len(stratified))

    out = {
        "dataset": {
            "standard_train": dataset_length_profile(read_jsonl(cfg["paths"]["dpo_standard_train"]), tok),
            "standard_eval": dataset_length_profile(read_jsonl(cfg["paths"]["dpo_standard_eval"]), tok),
            "length_balanced_train": dataset_length_profile(balanced, tok),
        }
    }

    # Train the length-balanced condition (same budget: one epoch over 1500 pairs).
    lb_out = cfg["length_output"]
    if args.force or not (repo_path(lb_out) / "adapter_config.json").exists():
        run_training(args.config, run_name="length_balanced", dataset_path=cfg["paths"]["dpo_length_train"], output_path=lb_out)
    if args.force or not (res / "eval_length_balanced.json").exists():
        run_evaluation(args.config, lb_out, "length_balanced")

    # Per-stratum held-out preference accuracy for both DPO models.
    out["stratified_eval"] = {}
    for name, adapter in [("standard", cfg["standard_output"]), ("length_balanced", lb_out)]:
        if not (repo_path(adapter) / "adapter_config.json").exists():
            raise FileNotFoundError(f"Missing adapter for {name}: {adapter}. Run task1_dpo.train first.")
        model = load_policy(cfg, adapter_path=adapter, trainable=False)
        pairs, dropped = preference_metrics(model, tok, stratified, float(cfg["beta"]), int(cfg["max_sequence_length"]),
                                            int(cfg.get("eval_batch_size", 4)))
        write_jsonl(res / f"stratified_{name}_pairs.jsonl", pairs)
        out["stratified_eval"][name] = per_stratum(pairs)
        out["stratified_eval"][name]["n_dropped_overlong_prompts"] = len(dropped)
        clear_gpu(model)
        del model
        clear_gpu()

    # Generated length + word-limit compliance (from the common evaluation of each condition).
    out["generation"] = {}
    for name in ["sft", "standard", "length_balanced"]:
        p = res / f"eval_{name}.json"
        if p.exists():
            s = load_json(p)
            out["generation"][name] = {
                "response_tokens": s["generation"]["response_tokens"],
                "reward_mean": s["generation"]["reward"]["mean"],
                "kl": s["generation"]["kl"],
                "truncation_rate": s["generation"]["truncation_rate"],
                "word_limit": s["word_limit"],
            }
        else:
            print(f"(missing {p}; run task1_dpo.evaluate --name {name})")

    save_json(res / "length_analysis.json", out)
    print("\nstratum           | standard acc | balanced acc")
    for s in STRATA:
        a = out["stratified_eval"]["standard"].get(s, {}).get("preference_accuracy", float("nan"))
        b = out["stratified_eval"]["length_balanced"].get(s, {}).get("preference_accuracy", float("nan"))
        print(f"{s:18s}| {a:.3f}        | {b:.3f}")
    for name, g in out["generation"].items():
        print(f"{name:16s} len {g['response_tokens']['mean']:.1f}±{g['response_tokens']['std']:.1f} "
              f"word-limit {g['word_limit']['compliance_rate']:.2f}")


if __name__ == "__main__":
    main()

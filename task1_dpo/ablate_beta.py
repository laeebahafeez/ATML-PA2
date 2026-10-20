"""Task 1 Step 2: matched short-run DPO forks for beta in cfg['betas'].

Every fork starts from the original Qwen2.5-1.5B-Instruct initialization with a fresh LoRA,
sees the same first `short_ablation_examples` (filtered) training pairs in the same seeded
order, and uses the same optimizer/LoRA settings. Only beta changes. Each fork is then
evaluated with the identical Task 1 protocol (task1_dpo.evaluate.run_evaluation).

NOTE: these forks use a 600-pair budget, whereas the standard run is a full 1-epoch run over
1500 pairs; the two budgets are reported separately.
"""
from __future__ import annotations

import argparse

from common.data import load_yaml, repo_path
from common.logging_utils import load_json, save_json
from task1_dpo.evaluate import run_evaluation
from task1_dpo.train import run_training


def beta_tag(beta: float) -> str:
    return f"beta_{beta:g}".replace(".", "p")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--force", action="store_true", help="retrain forks even if the adapter already exists")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    print("Required beta values:", cfg["betas"])
    print("Short-run examples per condition:", cfg["short_ablation_examples"])

    rows = []
    for beta in cfg["betas"]:
        tag = beta_tag(float(beta))
        out = f"outputs/task1_dpo/{tag}"
        if args.force or not (repo_path(out) / "adapter_config.json").exists():
            run_training(args.config, run_name=tag, output_path=out, beta=float(beta),
                         max_examples=int(cfg["short_ablation_examples"]))
        ev = repo_path(cfg["results_dir"]) / f"eval_{tag}.json"
        summary = load_json(ev) if (ev.exists() and not args.force) else run_evaluation(args.config, out, tag, beta=float(beta))
        rows.append({"beta": float(beta), "tag": tag, **summary})

    save_json(repo_path(cfg["results_dir"]) / "beta_ablation.json", rows)
    print("\nbeta | pref-acc | DPO loss | KL | reward | length")
    for r in rows:
        p, g = r["heldout_preference"], r["generation"]
        print(f"{r['beta']:<5g}| {p['preference_accuracy']:.3f}    | {p['dpo_loss']:.4f}  | {g['kl']:.4f} | "
              f"{g['reward']['mean']:.3f} | {g['response_tokens']['mean']:.1f}")


if __name__ == "__main__":
    main()

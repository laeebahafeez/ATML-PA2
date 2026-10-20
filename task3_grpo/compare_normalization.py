"""Task 3 Step 3: canonical GRPO (1/T_k) vs Dr.-GRPO-style (1/L_max) sequence normalization.

1. Matched short forks from the identical midpoint: same prompts, seed, K, reward, beta, eps,
   generation cap and update budget; only `loss_type` differs.
2. Common held-out evaluation of both forks (+ the midpoint as the shared start).
3. Length-conditioned gradient statistics:
   a. From the forks' own training completions: share of policy-term gradient mass assigned to
      short vs long completions (by length tercile), and corr(length, gradient mass).
   b. Controlled measurement on identical data: on the supplied completion cache (first K
      completions per prompt), at the midpoint, measure each completion's actual gradient norm
      ||grad_theta of its policy-term contribution|| under both normalizations. At ratio=1 the two
      differ only by the per-sequence factor T_k / L_max, so a single backward per completion
      gives both; we report per-length-bin mean norms and the long/short ratio.
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.logging_utils import load_json, save_json, set_seed
from common.metrics import safe_corr
from common.models import clear_gpu, load_policy, load_tokenizer, trainable_parameters
from common.rl_utils import disable_dropout, ensure_fp32_trainable, grad_norm, response_logprobs
from task3_grpo.analyze_group_size import load_k8_cache
from task3_grpo.continue_train import run_grpo
from task3_grpo.evaluate import evaluate
from task3_grpo.grpo import group_relative_advantages

LOSS_TYPES = ["grpo", "dr_grpo"]


def fork_name(loss_type):
    return f"fork_{loss_type}"


def length_bins(lengths):
    q1, q2 = np.percentile(lengths, [100 / 3, 200 / 3])
    return q1, q2, lambda n: "short" if n <= q1 else ("medium" if n <= q2 else "long")


def mass_by_length(completions, binner):
    by = defaultdict(list)
    for c in completions:
        if c["loss_tokens"] > 0:
            by[binner(c["response_tokens"])].append(c)
    total = sum(c["gradient_mass"] for cs in by.values() for c in cs) or 1.0
    out = {}
    for b in ["short", "medium", "long"]:
        cs = by.get(b, [])
        out[b] = {
            "n": len(cs),
            "mean_tokens": float(np.mean([c["response_tokens"] for c in cs])) if cs else float("nan"),
            "mean_abs_adv": float(np.mean([abs(c["advantage"]) for c in cs])) if cs else float("nan"),
            "gradient_mass_share": float(sum(c["gradient_mass"] for c in cs) / total),
        }
    valid = [c for c in completions if c["loss_tokens"] > 0]
    out["corr_length_vs_mass"] = safe_corr([c["response_tokens"] for c in valid], [c["gradient_mass"] for c in valid])
    out["corr_length_vs_reward"] = safe_corr([c["response_tokens"] for c in completions], [c["reward"] for c in completions])
    return out


def measured_gradient_norms(cfg, k):
    """Per-completion gradient norms at the midpoint on the fixed cache (identical data for both losses)."""
    set_seed(int(cfg["seed"]))
    tok = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=cfg["paths"]["grpo_midpoint_policy"], trainable=True)
    ensure_fp32_trainable(policy)
    disable_dropout(policy)
    params = trainable_parameters(policy)
    device = next(policy.parameters()).device
    L = float(cfg["max_completion_length"])
    pool = {str(r["prompt_id"]): r for r in read_jsonl(cfg["paths"]["rl_prompt_eval"]) + read_jsonl(cfg["paths"]["rl_prompt_train"])}

    out = []
    for pid, comps in load_k8_cache(cfg["group_cache"]).items():
        comps = comps[:k]
        src = pool.get(str(comps[0]["prompt_id"]))
        if src is None:
            continue
        rewards = torch.tensor([float(c["reward"]) for c in comps])
        adv = group_relative_advantages(rewards, torch.zeros(len(comps), dtype=torch.long))
        p_ids = tok.apply_chat_template(prompt_messages(src), tokenize=True, add_generation_prompt=True)
        for c, a in zip(comps, adv.tolist()):
            r_ids = tok(c["completion"], add_special_tokens=False)["input_ids"]
            if c.get("terminated_with_eos"):
                r_ids = r_ids + [tok.eos_token_id]
            r_ids = r_ids[: int(cfg.get("cache_generation_cap", 768))]
            ids = torch.tensor([p_ids + r_ids], device=device)
            rid = torch.tensor([r_ids], device=device)
            policy.zero_grad(set_to_none=True)
            lp, _ = response_logprobs(policy, ids, torch.ones_like(ids), len(p_ids), rid)
            T = lp.shape[1]
            # canonical GRPO per-sequence policy term at ratio=1: -(A / T) * sum_t log pi
            (-(a / T) * lp.sum()).backward()
            g = grad_norm(params)
            out.append({"prompt": pid, "tokens": T, "advantage": a, "truncated": bool(c.get("clipped_at_max")),
                        "grad_norm_grpo": g, "grad_norm_dr_grpo": g * T / L})
    clear_gpu(policy)
    del policy
    clear_gpu()
    return out


def summarize_measured(rows):
    _, _, binner = length_bins([r["tokens"] for r in rows])
    out = {}
    for lt in LOSS_TYPES:
        key = f"grad_norm_{lt}"
        by = defaultdict(list)
        for r in rows:
            by[binner(r["tokens"])].append(r[key])
        total = sum(r[key] for r in rows) or 1.0
        out[lt] = {b: {"n": len(by[b]), "mean_grad_norm": float(np.mean(by[b])) if by[b] else float("nan"),
                       "share_of_total": float(sum(by[b]) / total)} for b in ["short", "medium", "long"]}
        out[lt]["long_over_short_mean_norm"] = out[lt]["long"]["mean_grad_norm"] / max(out[lt]["short"]["mean_grad_norm"], 1e-12)
        out[lt]["corr_tokens_vs_grad_norm"] = safe_corr([r["tokens"] for r in rows], [r[key] for r in rows])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    res = repo_path(cfg["results_dir"])
    print("Fork updates:", cfg["fork_updates"])
    print("Compare loss_type='grpo' vs loss_type='dr_grpo' from the identical supplied midpoint.")

    def ev(adapter, name):
        p = res / "eval" / f"{name}.json"
        return load_json(p) if (p.exists() and not args.force) else evaluate(args.config, adapter, name)

    out = {"midpoint": ev(cfg["paths"]["grpo_midpoint_policy"], "midpoint"), "forks": {}}
    all_comps = {}
    for lt in LOSS_TYPES:
        name = fork_name(lt)
        adapter = f"outputs/task3_grpo/{name}"
        if args.force or not (repo_path(adapter) / "adapter_config.json").exists():
            run_grpo(args.config, output=adapter, updates=int(cfg["fork_updates"]), loss_type=lt, run_name=name)
        log = read_jsonl(res / name / "train_log.jsonl")
        all_comps[lt] = read_jsonl(res / name / "completions.jsonl")
        out["forks"][lt] = {
            "run": name, "heldout": ev(adapter, name),
            "train_trajectory": {k: [r[k] for r in log] for k in ["reward", "kl", "entropy", "response_length", "grad_norm", "truncated_frac"]},
        }

    # Common length bins over both forks' training completions so the bins are identical.
    q1, q2, binner = length_bins([c["response_tokens"] for cs in all_comps.values() for c in cs])
    out["train_length_bins"] = {"short_max": float(q1), "medium_max": float(q2)}
    for lt in LOSS_TYPES:
        out["forks"][lt]["train_gradient_mass_by_length"] = mass_by_length(all_comps[lt], binner)

    mpath = res / "normalization_measured_grad_norms.json"
    if args.force or not mpath.exists():
        rows = measured_gradient_norms(cfg, int(cfg["num_generations"]))
        save_json(mpath, rows)
    out["measured_gradient_norms_on_cache"] = summarize_measured(load_json(mpath))

    save_json(res / "normalization_comparison.json", out)
    print("\nloss    | held-out R | KL     | len  | train grad-mass share short/long | measured long/short norm")
    for lt in LOSS_TYPES:
        h = out["forks"][lt]["heldout"]
        g = out["forks"][lt]["train_gradient_mass_by_length"]
        m = out["measured_gradient_norms_on_cache"][lt]
        print(f"{lt:8s}| {h['reward']['mean']:.3f}      | {h['kl']:.4f} | {h['response_tokens']['mean']:.0f} | "
              f"{g['short']['gradient_mass_share']:.2f}/{g['long']['gradient_mass_share']:.2f}                       | "
              f"{m['long_over_short_mean_norm']:.2f}")


if __name__ == "__main__":
    main()

"""Task 2 Step 2: clipping study.

Part A - cached rollout batch (immediate geometric effect of epsilon).
  The supplied batch fixes responses, old-policy log-probs, reference log-probs, critic values
  and terminal rewards. We rebuild token ids, compute KL-shaped rewards -> GAE -> normalized
  advantages once, and then, for each epsilon, start from the identical midpoint policy and take
  `cached_clip_steps` full-batch clipped-surrogate gradient steps on that one batch. After every
  step we record, over the same response mask:
    clip_fraction          fraction of valid tokens with rho outside [1-eps, 1+eps] (course def.)
    affected_fraction      fraction of valid tokens whose gradient is zeroed by the clip
                           (rho>1+eps with A>0, or rho<1-eps with A<0)
    clipped_surrogate      E[min(rho A, clip(rho) A)]   and   unclipped_surrogate E[rho A]
    approx_kl_old_new, ratio range.
  Step 0 is the midpoint itself evaluated against the cached old-policy log-probs.

Part B - matched short continuation forks (cfg['fork_updates'] updates, same prompts/seed/beta_KL)
  for each epsilon, followed by the common held-out evaluation and stability statistics.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.logging_utils import load_json, save_json, set_seed
from common.metrics import masked_mean
from common.models import clear_gpu, load_policy, load_tokenizer, trainable_parameters
from common.rl_utils import disable_dropout, ensure_fp32_trainable, response_logprobs
from task2_ppo.continue_train import run_or_load_fork
from task2_ppo.evaluate import evaluate
from task2_ppo.ppo import compute_gae, normalize_advantages, shaped_rewards


def load_cached_rollouts(path):
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")

    # Instructor iterations used two equivalent names for these fields. Normalize once here so
    # the student analysis code sees one stable interface.
    normalized = []
    for row in rows:
        row = dict(row)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        normalized.append(row)

    required = {"source_index", "response", "old_logprobs", "ref_logprobs"}
    if not required.issubset(normalized[0]):
        raise ValueError(f"Unexpected PPO cache schema; need at least {sorted(required)}")
    return normalized


def rebuild_batch(cfg, rows, tokenizer):
    """Re-tokenize each cached (prompt, response) into the exact token layout of the rollout.

    The cache stores response text plus per-token tensors; the response token ids are recovered
    by re-encoding the text (+EOS when the rollout terminated). Rows whose re-encoding does not
    reproduce the cached token count are reported and excluded.
    """
    pool = {}
    for key in ("rl_prompt_train", "rl_prompt_eval"):
        for r in read_jsonl(cfg["paths"][key]):
            pool[str(r["prompt_id"])] = r
    items, skipped = [], []
    for r in rows:
        src = pool.get(str(r.get("prompt_id")))
        if src is None:
            skipped.append({"prompt_id": r.get("prompt_id"), "reason": "prompt not found"})
            continue
        n = int(r["old_logprobs"].shape[0])
        if "full_ids" in r and "prompt_len" in r:
            p_ids = r["full_ids"][: int(r["prompt_len"])].tolist()
            r_ids = r["response_ids"].tolist()
        else:
            p_ids = tokenizer.apply_chat_template(prompt_messages(src), tokenize=True, add_generation_prompt=True)
            r_ids = tokenizer(r["response"], add_special_tokens=False)["input_ids"]
            if bool(r.get("terminated_with_eos", False)):
                r_ids = r_ids + [tokenizer.eos_token_id]
        if len(r_ids) != n:
            skipped.append({"prompt_id": r.get("prompt_id"), "reason": f"retokenized {len(r_ids)} != cached {n}"})
            continue
        reward = float(r.get("effective_terminal_reward", r.get("raw_terminal_reward", 0.0)))
        items.append({"prompt_ids": p_ids, "response_ids": r_ids, "old": r["old_logprobs"].float(),
                      "ref": r["ref_logprobs"].float(), "values": r.get("values"), "reward": reward,
                      "prompt_id": str(r.get("prompt_id"))})
    return items, skipped


def padded(items, key, T, fill=0.0):
    out = torch.full((len(items), T), fill, dtype=torch.float32)
    for i, it in enumerate(items):
        v = it[key]
        out[i, : v.shape[0]] = v.float()
    return out


def cached_advantages(cfg, items):
    T = max(len(it["response_ids"]) for it in items)
    mask = torch.zeros(len(items), T)
    for i, it in enumerate(items):
        mask[i, : len(it["response_ids"])] = 1.0
    old, ref = padded(items, "old", T), padded(items, "ref", T)
    has_values = all(it["values"] is not None for it in items)
    values = padded(items, "values", T) if has_values else torch.zeros(len(items), T)
    rewards = shaped_rewards(torch.tensor([it["reward"] for it in items]), old, ref, mask, float(cfg["kl_beta"]))
    adv, returns = compute_gae(rewards, values * mask, mask, float(cfg["gamma"]), float(cfg["gae_lambda"]))
    return normalize_advantages(adv, mask), mask, old, has_values


def _row_tensors(it, device):
    ids = torch.tensor([it["prompt_ids"] + it["response_ids"]], device=device)
    rid = torch.tensor([it["response_ids"]], device=device)
    return ids, torch.ones_like(ids), len(it["prompt_ids"]), rid


def cached_clip_study(cfg, items, adv, mask, old, eps_values, n_steps):
    set_seed(int(cfg["seed"]))
    policy = load_policy(cfg, adapter_path=cfg["paths"]["ppo_midpoint_policy"], trainable=True)
    ensure_fp32_trainable(policy)
    disable_dropout(policy)
    params = trainable_parameters(policy)
    init_state = [p.detach().clone() for p in params]
    device = next(policy.parameters()).device
    n_tok = float(mask.sum())
    results = {}

    for eps in eps_values:
        with torch.no_grad():
            for p, s in zip(params, init_state):
                p.copy_(s)
        opt = AdamW(params, lr=float(cfg["policy_learning_rate"]))
        steps = []
        for step in range(n_steps + 1):
            do_update = step < n_steps
            agg = {"clip": 0.0, "affected": 0.0, "clipped": 0.0, "unclipped": 0.0, "akl": 0.0}
            rmax, rmin = 0.0, float("inf")
            opt.zero_grad(set_to_none=True)
            for i, it in enumerate(items):
                ids, attn, pw, rid = _row_tensors(it, device)
                n = len(it["response_ids"])
                with torch.set_grad_enabled(do_update):
                    new_lp, _ = response_logprobs(policy, ids, attn, pw, rid)
                    new_lp = new_lp[0]
                    o = old[i, :n].to(device)
                    a = adv[i, :n].to(device)
                    ratio = torch.exp(new_lp - o)
                    s1, s2 = ratio * a, ratio.clamp(1 - eps, 1 + eps) * a
                    obj = torch.minimum(s1, s2)
                    if do_update:
                        # full-batch token-mean objective, accumulated row by row (exact)
                        (-(obj.sum()) / n_tok).backward()
                with torch.no_grad():
                    r = ratio.detach()
                    agg["clip"] += float(((r < 1 - eps) | (r > 1 + eps)).float().sum())
                    agg["affected"] += float((((r > 1 + eps) & (a > 0)) | ((r < 1 - eps) & (a < 0))).float().sum())
                    agg["clipped"] += float(obj.detach().sum())
                    agg["unclipped"] += float(s1.detach().sum())
                    agg["akl"] += float((0.5 * (new_lp.detach() - o) ** 2).sum())
                    rmax, rmin = max(rmax, float(r.max())), min(rmin, float(r.min()))
            rec = {
                "step": step,
                "clip_fraction": agg["clip"] / n_tok,
                "affected_fraction": agg["affected"] / n_tok,
                "clipped_surrogate": agg["clipped"] / n_tok,
                "unclipped_surrogate": agg["unclipped"] / n_tok,
                "approx_kl_old_new": agg["akl"] / n_tok,
                "ratio_max": rmax,
                "ratio_min": rmin,
            }
            if do_update:
                rec["grad_norm"] = float(torch.nn.utils.clip_grad_norm_(params, float(cfg["max_grad_norm"])))
                opt.step()
            steps.append(rec)
            print(f"[clip-cache eps={eps}] step {step} clip {rec['clip_fraction']:.4f} affected {rec['affected_fraction']:.4f} "
                  f"Lclip {rec['clipped_surrogate']:.4f} Lunclip {rec['unclipped_surrogate']:.4f} akl {rec['approx_kl_old_new']:.2e}")
        results[str(eps)] = steps
    clear_gpu(policy)
    del policy
    clear_gpu()
    return results


def stability_stats(log):
    """Stability statistics for one fork, from its per-update training log.

    Primary: max per-update approx-KL(pi_old || pi_new) = size of the largest single policy step.
    Also: std of policy loss, max gradient norm, std of reward, final KL to reference.
    """
    g = lambda k: np.asarray([r[k] for r in log], dtype=float)
    return {
        "max_update_kl": float(g("approx_kl_old_new").max()),
        "mean_update_kl": float(g("approx_kl_old_new").mean()),
        "mean_clip_fraction": float(g("clip_fraction").mean()),
        "mean_active_clip_fraction": float(g("active_clip_fraction").mean()),
        "policy_loss_std": float(g("policy_loss").std()),
        "max_grad_norm": float(g("grad_norm").max()),
        "reward_std": float(g("reward").std()),
        "max_ratio": float(g("ratio_max").max()),
        "min_ratio": float(g("ratio_min").min()),
        "final_kl_to_ref": float(g("kl")[-1]),
        "any_nonfinite": bool(not np.isfinite(g("policy_loss")).all()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--skip-forks", action="store_true", help="only run the cached-batch part")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    rows = load_cached_rollouts(cfg["cached_rollouts"])
    print("Cached PPO rollouts:", len(rows))
    print("Required epsilon values:", cfg["clip_values"])
    print("Cache keys:", sorted(rows[0].keys()))
    res_dir = repo_path(cfg["results_dir"])

    # ---- Part A --------------------------------------------------------------------------
    cache_path = res_dir / "clipping_cached_batch.json"
    if args.force or not cache_path.exists():
        tok = load_tokenizer(cfg["base_model"])
        items, skipped = rebuild_batch(cfg, rows, tok)
        print(f"Rebuilt {len(items)} cached rollouts; skipped {len(skipped)}")
        adv, mask, old, has_values = cached_advantages(cfg, items)
        study = cached_clip_study(cfg, items, adv, mask, old, [float(e) for e in cfg["clip_values"]],
                                  int(cfg.get("cached_clip_steps", 4)))
        save_json(cache_path, {
            "n_rollouts": len(items), "n_tokens": int(mask.sum()), "skipped": skipped,
            "used_cached_values": has_values, "kl_beta_for_shaping": float(cfg["kl_beta"]),
            "advantage_abs_mean": float(masked_mean(adv.abs(), mask)),
            "per_epsilon": study,
        })

    # ---- Part B --------------------------------------------------------------------------
    if not args.skip_forks:
        kl = float(cfg["kl_beta"])
        forks = []
        for eps in cfg["clip_values"]:
            name = run_or_load_fork(args.config, float(eps), kl, force=args.force)
            p = res_dir / "eval" / f"{name}.json"
            ev = load_json(p) if (p.exists() and not args.force) else evaluate(args.config, f"outputs/task2_ppo/{name}", name)
            log = read_jsonl(res_dir / name / "train_log.jsonl")
            forks.append({"clip_epsilon": float(eps), "kl_beta": kl, "run": name, "heldout": ev,
                          "stability": stability_stats(log)})
        save_json(res_dir / "clipping_forks.json", forks)
        print("\neps  | held-out R | KL     | len | max update-KL | mean clip")
        for f in forks:
            h, s = f["heldout"], f["stability"]
            print(f"{f['clip_epsilon']:<5g}| {h['reward']['mean']:.3f}      | {h['kl']:.4f} | {h['response_tokens']['mean']:.0f} | "
                  f"{s['max_update_kl']:.2e}      | {s['mean_clip_fraction']:.3f}")


if __name__ == "__main__":
    main()

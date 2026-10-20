"""Task 3 Step 2: equal-generation group-size study on the supplied K=8 completion/reward cache.

Equal-budget regrouping rule (used for every K): each prompt's 8 cached completions, ordered by
`generation_index`, are split into 8/K disjoint contiguous groups of size K. Every K therefore
uses exactly the same 192 generations (24 prompts x 8); K=2 gives 96 groups, K=4 gives 48 and
K=8 gives 24. Because the contiguous split is one arbitrary partition, every statistic is also
reported as an expectation over R random within-prompt permutations.

Prompt-difficulty bins (defined once): prompts are sorted by their mean reward over all 8 cached
completions and split into terciles: "hard" (lowest mean reward), "medium", "easy".

Reported per K (overall and per difficulty bin):
  informative_rate        fraction of groups with reward std > ZERO_STD_TOL (non-zero advantages)
  weak_rate@tau           fraction of groups with std < tau on the reward-model scale
  mean_group_std          mean within-group reward std
  centered_signal_var     variance of the group-relative signal r - mu_group
  adv_mse_vs_k8 / adv_corr_vs_k8 / sign_agree_vs_k8
                          how well a size-K group's normalized advantage reproduces the advantage
                          computed with the prompt's full 8-sample baseline (signal quality)
A binarized-reward sensitivity (reward > global median) shows the regime the manual warns about,
where low-resolution rewards make whole groups uninformative.
"""
from __future__ import annotations

import argparse
import warnings
from collections import defaultdict

import numpy as np

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json
from task3_grpo.grpo import ZERO_STD_TOL

TAUS = [0.05, 0.10, 0.25]
EPS = 1e-6


def load_k8_cache(path):
    rows = read_jsonl(path)
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[str(row["source_index"])].append(row)
    # Instructor cache has 8 rows per prompt, one row per completion.
    bad = {pid: len(group) for pid, group in by_prompt.items() if len(group) < 8}
    if bad:
        raise ValueError(f"Expected at least K=8 cached completions per prompt; short groups: {bad}")
    for group in by_prompt.values():
        group.sort(key=lambda x: int(x.get("generation_index", 0)))
    return by_prompt


def regroup_equal_generation_budget(by_prompt, k: int, order: dict | None = None):
    """Return K-sized groups while keeping total cached completions fixed.

    Each prompt's first 8 completions (in `generation_index` order, or the permutation given in
    `order[pid]`) are split into 8/k disjoint contiguous groups. Total generations = 8 x #prompts
    for every k.
    """
    if 8 % k:
        raise ValueError("k must divide 8")
    groups = []
    for pid, comps in by_prompt.items():
        comps = comps[:8]
        idx = order[pid] if order else list(range(8))
        rewards = np.asarray([float(comps[i]["reward"]) for i in idx])
        for s in range(0, 8, k):
            groups.append({"prompt": pid, "rewards": rewards[s : s + k], "members": idx[s : s + k]})
    return groups


def group_metrics(groups, full_mean, full_std, binarize_threshold=None):
    stds, centered, adv_k, adv_ref = [], [], [], []
    for g in groups:
        r = g["rewards"]
        if binarize_threshold is not None:
            r = (r > binarize_threshold).astype(float)
        mu, sd = r.mean(), r.std()
        stds.append(sd)
        centered.extend(r - mu)
        adv_k.extend((r - mu) / (sd + EPS))
        fm, fs = full_mean[g["prompt"]], full_std[g["prompt"]]
        adv_ref.extend((r - fm) / (fs + EPS))
    stds, centered = np.asarray(stds), np.asarray(centered)
    adv_k, adv_ref = np.asarray(adv_k), np.asarray(adv_ref)
    nz = np.abs(adv_ref) > 1e-9
    out = {
        "n_groups": len(groups),
        "n_generations": int(sum(len(g["rewards"]) for g in groups)),
        "informative_rate": float((stds > ZERO_STD_TOL).mean()),
        "mean_group_std": float(stds.mean()),
        "centered_signal_var": float(centered.var()),
        "adv_mse_vs_k8": float(((adv_k - adv_ref) ** 2).mean()),
        "adv_corr_vs_k8": float(np.corrcoef(adv_k, adv_ref)[0, 1]) if adv_k.std() > 0 and adv_ref.std() > 0 else float("nan"),
        "sign_agree_vs_k8": float((np.sign(adv_k[nz]) == np.sign(adv_ref[nz])).mean()) if nz.any() else float("nan"),
    }
    if binarize_threshold is None:
        for t in TAUS:
            out[f"weak_rate@{t:g}"] = float((stds < t).mean())
    return out


def averaged(by_prompt, k, full_mean, full_std, n_perm, seed, prompts=None, binarize=None):
    sub = {p: by_prompt[p] for p in (prompts or by_prompt)}
    fixed = group_metrics(regroup_equal_generation_budget(sub, k), full_mean, full_std, binarize)
    rng = np.random.default_rng(seed)
    perms = []
    for _ in range(n_perm):
        order = {p: rng.permutation(8).tolist() for p in sub}
        perms.append(group_metrics(regroup_equal_generation_budget(sub, k, order), full_mean, full_std, binarize))
    keys = [k_ for k_ in fixed if k_ not in ("n_groups", "n_generations")]
    with warnings.catch_warnings():  # sign agreement is undefined (NaN) for all-tied binarized bins
        warnings.simplefilter("ignore", RuntimeWarning)
        exp = {k_: float(np.nanmean([m[k_] for m in perms])) for k_ in keys}
        sd = {k_: float(np.nanstd([m[k_] for m in perms])) for k_ in keys}
    return {"fixed_partition": fixed, "permutation_mean": exp, "permutation_std": sd}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--permutations", type=int, default=500)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    by_prompt = load_k8_cache(cfg["group_cache"])
    print("Cached prompts:", len(by_prompt))
    print("Group sizes to analyze:", cfg["group_sizes"])
    first = next(iter(by_prompt.values()))
    print("Cache row keys:", sorted(first[0].keys()))

    full = {p: np.asarray([float(c["reward"]) for c in comps[:8]]) for p, comps in by_prompt.items()}
    full_mean = {p: v.mean() for p, v in full.items()}
    full_std = {p: v.std() for p, v in full.items()}

    # Difficulty terciles by mean cached reward (lower reward = harder prompt).
    ordered = sorted(by_prompt, key=lambda p: full_mean[p])
    n = len(ordered)
    bins = {"hard": ordered[: n // 3], "medium": ordered[n // 3 : 2 * n // 3], "easy": ordered[2 * n // 3 :]}
    all_rewards = np.concatenate(list(full.values()))
    median = float(np.median(all_rewards))

    out = {
        "cache": {
            "n_prompts": n, "n_generations": int(all_rewards.size),
            "reward_mean": float(all_rewards.mean()), "reward_std": float(all_rewards.std()),
            "clipped_at_max_frac": float(np.mean([bool(c.get("clipped_at_max")) for cs in by_prompt.values() for c in cs[:8]])),
            "per_prompt_std_mean": float(np.mean(list(full_std.values()))),
        },
        "binning_rule": "terciles of per-prompt mean reward over the 8 cached completions (hard = lowest)",
        "bins": {b: {"prompts": ps, "mean_reward": float(np.mean([full_mean[p] for p in ps])),
                     "mean_k8_std": float(np.mean([full_std[p] for p in ps]))} for b, ps in bins.items()},
        "binarize_threshold_median": median,
        "by_k": {},
    }
    for k in cfg["group_sizes"]:
        k = int(k)
        entry = {"all": averaged(by_prompt, k, full_mean, full_std, args.permutations, int(cfg["seed"]))}
        for b, ps in bins.items():
            entry[b] = averaged(by_prompt, k, full_mean, full_std, args.permutations, int(cfg["seed"]), prompts=ps)
        entry["binarized"] = {"all": averaged(by_prompt, k, full_mean, full_std, args.permutations, int(cfg["seed"]), binarize=median)}
        for b, ps in bins.items():
            entry["binarized"][b] = averaged(by_prompt, k, full_mean, full_std, args.permutations, int(cfg["seed"]),
                                             prompts=ps, binarize=median)
        out["by_k"][str(k)] = entry

    save_json(repo_path(cfg["results_dir"]) / "group_size_analysis.json", out)
    print("\nprompt-difficulty bins: " + ", ".join(f"{b}: R={v['mean_reward']:.2f}, std={v['mean_k8_std']:.2f}" for b, v in out["bins"].items()))
    print("K | groups | informative | weak@0.1 | mean std | centered var | adv corr vs K8 | sign agree | binarized informative (hard/med/easy)")
    for k, e in out["by_k"].items():
        a = e["all"]["permutation_mean"]
        bz = e["binarized"]
        print(f"{k} | {e['all']['fixed_partition']['n_groups']:6d} | {a['informative_rate']:.3f}       | {a['weak_rate@0.1']:.3f}    | "
              f"{a['mean_group_std']:.3f}    | {a['centered_signal_var']:.3f}        | {a['adv_corr_vs_k8']:.3f}          | "
              f"{a['sign_agree_vs_k8']:.3f}      | {bz['hard']['permutation_mean']['informative_rate']:.2f}/"
              f"{bz['medium']['permutation_mean']['informative_rate']:.2f}/{bz['easy']['permutation_mean']['informative_rate']:.2f}")


if __name__ == "__main__":
    main()

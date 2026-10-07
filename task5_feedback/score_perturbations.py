"""Task 5 Step 2: controlled reward-sensitivity / robustness study.

Each of the 20 problems has five manually validated variants. Every non-clean variant is paired
with `clean_correct`, which is the diagnostically better response in every pair:

  pair (clean vs X)                   axis       what a good mechanism should do
  corrupt_reasoning_correct_final     reasoning  prefer clean   (S_reason)
  good_reasoning_wrong_final          outcome    prefer clean   (S_outcome)
  gold_distractor_wrong_final         outcome    prefer clean   (S_outcome; gold number only as distractor)
  persuasive_filler_correct           style      prefer clean or tie (filler adds nothing; a strict
                                                 preference FOR the filler variant is a style error)

For each mechanism and pair type we report better-response rate, tie rate and wrong-preference
rate. RLVR compares exact rewards (binary); RLAIF uses the fixed pairwise AI judge. The judge is
queried in both presentation orders (A=clean,B=X and A=X,B=clean); disagreement between the two
orders is reported as judge order-inconsistency, and the per-pair verdict is the order-averaged
score (clean win=1, tie/inconsistent=0.5, loss=0).

We also compute the RLAIF-style group reward used in training, (wins + 0.5 ties)/(K-1), over all
five variants of each problem, next to the exact reward, for every variant category.
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.logging_utils import save_json
from task5_feedback.evaluate_math import JUDGE_CACHE
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward

EXPECTED_VARIANTS = {
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
}
VARIANT_ORDER = ["clean_correct", "corrupt_reasoning_correct_final", "good_reasoning_wrong_final",
                 "persuasive_filler_correct", "gold_distractor_wrong_final"]
PAIRS = {
    "corrupt_reasoning_correct_final": "reasoning",
    "good_reasoning_wrong_final": "outcome",
    "gold_distractor_wrong_final": "outcome",
    "persuasive_filler_correct": "style",
}


def load_diagnostic_groups(path):
    rows = read_jsonl(path)
    by_problem = defaultdict(dict)
    for row in rows:
        by_problem[str(row["problem_id"])][row["variant_type"]] = row
    for pid, variants in by_problem.items():
        missing = EXPECTED_VARIANTS - set(variants)
        if missing:
            raise ValueError(f"Problem {pid} missing variants: {sorted(missing)}")
    return by_problem


def outcome_of(score):
    return "better" if score > 0.5 else ("wrong" if score < 0.5 else "tie")


def rates(outcomes):
    n = len(outcomes)
    return {"n": n, **{f"{k}_rate": (sum(o == k for o in outcomes) / n if n else float("nan")) for k in ["better", "tie", "wrong"]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--single-order", action="store_true", help="query the judge once per pair (no order swap)")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    groups = load_diagnostic_groups(cfg["paths"]["task5_diagnostics"])
    print("Diagnostic problems:", len(groups))
    print("Variants/problem:", sorted(EXPECTED_VARIANTS))
    print("Use exact_reward(...) for RLVR and PairwiseAIJudge(...) for RLAIF.")

    judge = PairwiseAIJudge(cfg, JUDGE_CACHE)
    pair_recs, group_recs = [], []
    for pid, v in groups.items():
        q, gold = v["clean_correct"]["question"], str(v["clean_correct"]["gold_final"])
        ex = {name: exact_reward(v[name]["response"], gold) for name in VARIANT_ORDER}
        for name, row in v.items():
            exp = row.get("expected_exact_reward")
            if exp is not None and float(exp) != ex[name]:
                print(f"WARNING problem {pid} {name}: verifier {ex[name]} != expected {exp}")

        clean = v["clean_correct"]["response"]
        for x, axis in PAIRS.items():
            other = v[x]["response"]
            # RLVR: compare binary exact rewards.
            rv = 1.0 if ex["clean_correct"] > ex[x] else (0.0 if ex["clean_correct"] < ex[x] else 0.5)
            # RLAIF: fixed judge, both presentation orders.
            p1 = judge.compare(q, clean, other)               # A = clean
            s1 = {"A": 1.0, "B": 0.0, "TIE": 0.5}[p1]
            if args.single_order:
                p2, s2 = None, s1
            else:
                p2 = judge.compare(q, other, clean)           # A = perturbed
                s2 = {"A": 0.0, "B": 1.0, "TIE": 0.5}[p2]
            aif = (s1 + s2) / 2
            pair_recs.append({
                "problem_id": pid, "perturbation": x, "axis": axis,
                "exact_clean": ex["clean_correct"], "exact_other": ex[x],
                "rlvr_score": rv, "rlvr_outcome": outcome_of(rv),
                "judge_order1": p1, "judge_order2": p2, "rlaif_score": aif, "rlaif_outcome": outcome_of(aif),
                "judge_order_consistent": (p2 is None) or (s1 == s2),
                "clean_chars": len(clean), "other_chars": len(other),
            })

        # RLAIF group reward over all five variants (the training-time reward form), plus exact reward.
        responses = [v[name]["response"] for name in VARIANT_ORDER]
        aif_group = judge.group_rewards(q, responses)
        for name, r in zip(VARIANT_ORDER, aif_group):
            group_recs.append({"problem_id": pid, "variant": name, "exact_reward": ex[name], "rlaif_group_reward": float(r)})

    out = {"by_perturbation": {}, "by_variant": {}, "judge": {}}
    for x in PAIRS:
        rs = [r for r in pair_recs if r["perturbation"] == x]
        out["by_perturbation"][x] = {
            "axis": PAIRS[x],
            "rlvr": rates([r["rlvr_outcome"] for r in rs]),
            "rlaif": rates([r["rlaif_outcome"] for r in rs]),
            "rlaif_order_consistency": float(np.mean([r["judge_order_consistent"] for r in rs])),
        }
    for mech in ["rlvr", "rlaif"]:
        reason = [r[f"{mech}_score"] for r in pair_recs if r["axis"] == "reasoning"]
        outcome = [r[f"{mech}_score"] for r in pair_recs if r["axis"] == "outcome"]
        out[f"S_reason_{mech}"] = float(np.mean([s > 0.5 for s in reason]))
        out[f"S_outcome_{mech}"] = float(np.mean([s > 0.5 for s in outcome]))
        out[f"S_reason_{mech}_tie_rate"] = float(np.mean([s == 0.5 for s in reason]))
        out[f"S_outcome_{mech}_tie_rate"] = float(np.mean([s == 0.5 for s in outcome]))
        out[f"S_reason_{mech}_wrong_rate"] = float(np.mean([s < 0.5 for s in reason]))
        out[f"S_outcome_{mech}_wrong_rate"] = float(np.mean([s < 0.5 for s in outcome]))
    for name in VARIANT_ORDER:
        rs = [r for r in group_recs if r["variant"] == name]
        out["by_variant"][name] = {"mean_exact_reward": float(np.mean([r["exact_reward"] for r in rs])),
                                   "mean_rlaif_group_reward": float(np.mean([r["rlaif_group_reward"] for r in rs]))}
    out["judge"]["order_consistency_all"] = float(np.mean([r["judge_order_consistent"] for r in pair_recs]))
    out["judge"]["n_queries"] = len(pair_recs) * (1 if args.single_order else 2) + 10 * len(groups)

    res = repo_path(cfg["results_dir"]) / "task5_feedback"
    write_jsonl(res / "diagnostic_pairs.jsonl", pair_recs)
    write_jsonl(res / "diagnostic_group_rewards.jsonl", group_recs)
    save_json(res / "diagnostic_summary.json", out)

    print("\nperturbation (vs clean)         | RLVR better/tie/wrong | RLAIF better/tie/wrong | order-consistent")
    for x, d in out["by_perturbation"].items():
        a, b = d["rlvr"], d["rlaif"]
        print(f"{x:32s}| {a['better_rate']:.2f}/{a['tie_rate']:.2f}/{a['wrong_rate']:.2f}        | "
              f"{b['better_rate']:.2f}/{b['tie_rate']:.2f}/{b['wrong_rate']:.2f}         | {d['rlaif_order_consistency']:.2f}")
    print(f"S_reason: RLVR {out['S_reason_rlvr']:.2f}  RLAIF {out['S_reason_rlaif']:.2f} | "
          f"S_outcome: RLVR {out['S_outcome_rlvr']:.2f}  RLAIF {out['S_outcome_rlaif']:.2f}")
    print("variant group rewards (exact | RLAIF):", {k: (round(v['mean_exact_reward'], 2), round(v['mean_rlaif_group_reward'], 2))
                                                   for k, v in out["by_variant"].items()})


if __name__ == "__main__":
    main()

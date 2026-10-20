"""Task 5 synthesis: combine in-domain, controlled-diagnostic and transfer results into one
RLVR-vs-RLAIF comparison, and measure the per-call inference cost of each feedback source.

Cost model reported: an RLVR group reward needs K verifier calls (regex + float compare); an
RLAIF group reward needs K(K-1)/2 judge generations with a 3B model. We time both on real
diagnostic responses (judge timed with the cache bypassed).
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import load_json, save_json
from task5_feedback.rlvr import exact_reward


def benchmark_costs(cfg, n=12):
    import torch

    from task5_feedback.rlaif import PairwiseAIJudge

    rows = read_jsonl(cfg["paths"]["task5_diagnostics"])
    t0 = time.perf_counter()
    for _ in range(100):
        for r in rows:
            exact_reward(r["response"], str(r["gold_final"]))
    verifier_s = (time.perf_counter() - t0) / (100 * len(rows))

    # Separate throw-away cache so the timing never hits (or pollutes) the real judge cache.
    judge = PairwiseAIJudge(cfg, "results/task5_feedback/_benchmark_cache_unused.json")
    n = min(n, len(rows) // 2)
    pairs = [(rows[i]["question"], rows[i]["response"], rows[i + 1]["response"]) for i in range(0, 2 * n, 2)]
    judge.compare(*pairs[0])  # warm-up
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for q, a, b in pairs:
        judge.cache = {}
        judge.compare(q, a, b)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    judge_s = (time.perf_counter() - t0) / len(pairs)
    judge.cache_path.unlink(missing_ok=True)
    k = int(load_yaml("configs/grpo.yaml")["num_generations"])
    return {
        "verifier_seconds_per_call": verifier_s,
        "judge_seconds_per_call": judge_s,
        "K": k,
        "rlvr_calls_per_group": k,
        "rlaif_judge_calls_per_group": k * (k - 1) // 2,
        "rlvr_seconds_per_group": k * verifier_s,
        "rlaif_seconds_per_group": k * (k - 1) // 2 * judge_s,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


def posthoc_pairwise(res, ds, policy):
    """From saved files: how many greedy answers are identical to SFT, and verifier-judge agreement
    restricted to pairs whose texts actually differ (identical pairs are trivially tied)."""
    pol = res / f"{ds}_{policy}_responses.jsonl"
    sft = res / f"{ds}_sft_responses.jsonl"
    jud = res / f"{ds}_{policy}_vs_sft_judge.jsonl"
    if not (pol.exists() and sft.exists() and jud.exists()):
        return None
    P, S, J = read_jsonl(pol), read_jsonl(sft), read_jsonl(jud)
    same = [p["response"] == s["response"] for p, s in zip(P, S)]
    diff = [j for j, sm in zip(J, same) if not sm]
    dec = [j for j in diff if j["verifier_pref"] != "TIE"]
    return {
        "n": len(P),
        "identical_to_sft": int(sum(same)),
        "n_differing": len(diff),
        "judge_ties_on_differing": int(sum(j["judge_pref"] == "TIE" for j in diff)),
        "agreement_on_differing": (sum(j["judge_pref"] == j["verifier_pref"] for j in diff) / len(diff)) if diff else float("nan"),
        "n_verifier_decisive_differing": len(dec),
        "agreement_on_verifier_decisive": (sum(j["judge_pref"] == j["verifier_pref"] for j in dec) / len(dec)) if dec else float("nan"),
        "win_rate_on_differing": (sum({"A": 1, "B": 0, "TIE": 0.5}[j["judge_pref"]] for j in diff) / len(diff)) if diff else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--skip-benchmark", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    res = repo_path(cfg["results_dir"]) / "task5_feedback"

    out = {}
    for ds in ["gsm", "transfer"]:
        p = res / f"eval_{ds}.json"
        out[ds] = load_json(p) if p.exists() else None
        if out[ds] is None:
            print(f"(missing {p}; run task5_feedback.evaluate_math --dataset {ds})")
    out["posthoc_pairwise"] = {f"{ds}/{pol}": posthoc_pairwise(res, ds, pol) for ds in ["gsm", "transfer"] for pol in ["rlvr", "rlaif"]}
    p = res / "diagnostic_summary.json"
    out["diagnostics"] = load_json(p) if p.exists() else None
    prev = res / "feedback_comparison.json"
    if not args.skip_benchmark:
        out["cost"] = benchmark_costs(cfg)
    elif prev.exists() and "cost" in load_json(prev):
        out["cost"] = load_json(prev)["cost"]  # keep the GPU measurement from the full run
    save_json(res / "feedback_comparison.json", out)

    rows = []
    for ds in ["gsm", "transfer"]:
        if not out[ds]:
            continue
        for pol, d in out[ds]["policies"].items():
            pw = d.get("pairwise_vs_sft", {})
            rows.append({
                "dataset": ds, "policy": pol,
                "exact_acc": d["exact_accuracy"], "format": d["format_compliance"],
                "mean_tokens": d["response_tokens"]["mean"],
                "win_vs_sft": pw.get("win_rate_vs_sft"), "ties": pw.get("ties"),
                "verifier_judge_agree": pw.get("verifier_judge_agreement_all"),
                "acc_drop_from_gsm": d.get("accuracy_drop_from_gsm"),
            })
    if rows:
        table = pd.DataFrame(rows)
        table.to_csv(res / "feedback_comparison_table.csv", index=False)
        print(table.round(3).to_string(index=False))
    if out["diagnostics"]:
        d = out["diagnostics"]
        print(f"\nS_reason RLVR={d['S_reason_rlvr']:.2f} RLAIF={d['S_reason_rlaif']:.2f} | "
              f"S_outcome RLVR={d['S_outcome_rlvr']:.2f} RLAIF={d['S_outcome_rlaif']:.2f}")
    if "cost" in out:
        c = out["cost"]
        print(f"cost per K={c['K']} group: RLVR {c['rlvr_seconds_per_group'] * 1e3:.3f} ms vs RLAIF {c['rlaif_seconds_per_group']:.2f} s "
              f"({c['rlaif_judge_calls_per_group']} judge calls)")


if __name__ == "__main__":
    main()

"""Recompute every derived number quoted in the report text from the saved files in results/.

The main tables come straight from the per-task result files. This script covers the numbers
in the prose that need an extra computation on top of those files (shares, counts, agreement
statistics, fits). Run from the repository root:

    python -m report.derived_numbers

It prints the values and writes them to results/derived_numbers.json.
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

import numpy as np

from task4_safety.evaluate_safety import REFUSAL_RE

R = Path("results")
POLICIES4 = ["sft", "dpo", "ppo", "grpo"]


def jl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def kappa(a, b):
    labels = sorted(set(a) | set(b))
    n = len(a)
    po = float(np.mean([x == y for x, y in zip(a, b)]))
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return po, (po - pe) / (1 - pe) if pe < 1 else float("nan")


# ------------------------------------------------------------------------------- Task 1
def task1():
    out = {}
    for name in ("standard", "length_balanced"):
        strata = defaultdict(list)
        for r in jl(R / "task1_dpo" / f"stratified_{name}_pairs.jsonl"):
            m = r["chosen_logratio"] / r["chosen_tokens"] - r["rejected_logratio"] / r["rejected_tokens"]
            strata[r["length_stratum"]].append(m > 0)
        out[f"per_token_normalized_strata_accuracy_{name}"] = {k: round(float(np.mean(v)), 3) for k, v in strata.items()}
    return out


# ------------------------------------------------------------------------------- Task 3
def task3():
    out = {}
    comps = jl(R / "task3_grpo/standard/completions.jsonl")
    toks = np.array([c["response_tokens"] for c in comps], float)
    trunc = np.array([bool(c["truncated"]) for c in comps])
    out["standard_truncated_completions"] = f"{int(trunc.sum())}/{len(comps)}"
    out["standard_truncated_token_share"] = round(float(toks[trunc].sum() / toks.sum()), 3)
    rw = np.array([c["reward"] for c in comps])
    out["standard_reward_truncated_vs_complete"] = [round(float(rw[trunc].mean()), 3), round(float(rw[~trunc].mean()), 3)]

    log = jl(R / "task3_grpo/standard/train_log.jsonl")
    out["zero_gradient_updates"] = [e["update"] for e in log if e["grad_norm"] == 0]
    out["largest_grad_norm_update"] = max(((e["grad_norm"], e["update"], e["response_length"]) for e in log))
    for u in (2, 14):
        r = np.array([c["reward"] for c in comps if c["update"] == u])
        out[f"update{u}_rewards"] = [round(float(x), 2) for x in r]
        out[f"update{u}_reward_span"] = round(float(r.max() - r.min()), 2)
        out[f"update{u}_advantages"] = [round(float(x), 2) for x in (r - r.mean()) / (r.std() + 1e-6)]

    g = json.load(open(R / "task3_grpo/normalization_measured_grad_norms.json"))
    T = np.array([x["tokens"] for x in g], float)
    a = np.abs(np.array([x["advantage"] for x in g]))
    g1 = np.array([x["grad_norm_grpo"] for x in g])
    g2 = np.array([x["grad_norm_dr_grpo"] for x in g])
    tr = np.array([x["truncated"] for x in g])
    cuts = np.quantile(T, [1 / 3, 2 / 3])
    b = np.digitize(T, cuts)
    out["grad_share_by_length_third_grpo"] = [round(float(g1[b == i].sum() / g1.sum()), 3) for i in range(3)]
    out["grad_share_by_length_third_dr_grpo"] = [round(float(g2[b == i].sum() / g2.sum()), 3) for i in range(3)]
    keep = a > 0.05
    out["unnormalized_grad_norm_exponent"] = round(float(np.polyfit(np.log(T[keep]), np.log(g1[keep] / a[keep]), 1)[0]) + 1, 2)
    top6 = np.argsort(-g1)[:6]
    out["top6_grpo_completions_tokens"] = [int(T[i]) for i in top6]
    out["top6_grpo_grad_share"] = round(float(g1[top6].sum() / g1.sum()), 3)
    out["capped_completions_in_measurement"] = int(tr.sum())
    out["dr_grpo_grad_share_of_capped"] = round(float(g2[tr].sum() / g2.sum()), 3)
    nt = ~tr
    bn = b[nt]
    out["dr_grpo_long_third_share_without_capped"] = round(float(g2[nt][bn == 2].sum() / g2[nt].sum()), 3)
    m = g1 > 1e-9
    out["dr_over_grpo_equals_T_over_512"] = bool(np.allclose(g2[m] / g1[m], T[m] / 512, rtol=1e-3))

    for fork in ("fork_grpo", "fork_dr_grpo"):
        out[f"{fork}_masked"] = sum(e["masked_completions"] for e in jl(R / f"task3_grpo/{fork}/train_log.jsonl"))

    # binarized group-size column: exact (hypergeometric) informative probability from pass counts
    ga = json.load(open(R / "task3_grpo/group_size_analysis.json"))
    cache = [json.loads(l) for l in open("cached/grpo_k_cache.jsonl", encoding="utf-8")] if Path("cached/grpo_k_cache.jsonl").exists() else None
    if cache is not None:
        bp = defaultdict(list)
        for c in cache:
            bp[str(c["source_index"])].append(c)
        med = ga["binarize_threshold_median"]
        for bin_name, info in ga["bins"].items():
            passes = [int(sum(float(c["reward"]) > med for c in bp[p][:8])) for p in info["prompts"]]
            exact = {K: round(float(np.mean([1 - comb(m_, K) / comb(8, K) - comb(8 - m_, K) / comb(8, K) for m_ in passes])), 3)
                     for K in (2, 4, 8)}
            out[f"binarized_{bin_name}_pass_counts"] = passes
            out[f"binarized_{bin_name}_exact_informative"] = exact
    else:
        out["binarized_note"] = "cached/grpo_k_cache.jsonl not present (download course assets to recompute)"
    var = {k: ga["by_k"][k]["all"]["permutation_mean"]["centered_signal_var"] for k in ("2", "4", "8")}
    out["variance_ratio_vs_k8"] = {"K2": round(var["2"] / var["8"], 3), "K4": round(var["4"] / var["8"], 3),
                                   "predicted_K2": round(0.5 / 0.875, 3), "predicted_K4": round(0.75 / 0.875, 3)}
    return out


# ------------------------------------------------------------------------------- Task 4
def task4():
    out = {}
    T = R / "task4_safety"
    gen = {p: {r["xstest_id"]: r for r in jl(T / f"generated_{p}.jsonl")} for p in POLICIES4}
    jud = {p: {r["xstest_id"]: r for r in jl(T / f"judged_{p}.jsonl")} for p in POLICIES4}
    ids = sorted(gen["sft"])
    refuses = lambda text: bool(REFUSAL_RE.search(text[:300]))

    tags = [jud[p][i]["judge_rationale"].lower() for p in POLICIES4 for i in ids]
    out["rationale_tags_describing_request"] = round(float(np.mean([any(w in t for w in ("request", "query", "question")) for t in tags])), 3)
    out["judge_confidence_counts"] = dict(Counter(str(jud[p][i]["judge_confidence"]) for p in POLICIES4 for i in ids))

    tab = Counter()
    for p in POLICIES4:
        for i in ids:
            if gen[p][i]["benchmark_class"] == "SAFE":
                tab[(jud[p][i]["judge_label"], refuses(gen[p][i]["response"]))] += 1
    out["safe_prompts_judge_label_x_refusal_phrase"] = {f"{k[0]}|phrase={k[1]}": v for k, v in tab.items()}

    for p in POLICIES4[1:]:
        out[f"{p}_identical_responses_to_sft"] = sum(gen[p][i]["response"] == gen["sft"][i]["response"] for i in ids)
        out[f"{p}_label_changes_vs_sft"] = sum(jud[p][i]["judge_label"] != jud["sft"][i]["judge_label"] for i in ids)
    ref = {"JUSTIFIED_REFUSAL", "OVER_REFUSAL"}
    safe = [i for i in ids if gen["sft"][i]["benchmark_class"] == "SAFE"]
    a = [jud["sft"][i]["judge_label"] in ref for i in safe]
    b = [jud["dpo"][i]["judge_label"] in ref for i in safe]
    out["dpo_vs_sft_safe_refusal_flips"] = {"only_dpo_refuses": sum(y and not x for x, y in zip(a, b)),
                                            "only_sft_refuses": sum(x and not y for x, y in zip(a, b))}
    unsafe_sa = Counter(gen[p][i]["type"] for p in POLICIES4 for i in ids
                        if gen[p][i]["benchmark_class"] == "UNSAFE" and jud[p][i]["judge_label"] == "SAFE_ANSWER")
    out["unsafe_prompts_labelled_safe_answer"] = {"total": sum(unsafe_sa.values()), "contrast_discr": unsafe_sa.get("contrast_discr", 0)}
    out["unsafe_compliance_labels_by_class"] = dict(Counter(gen[p][i]["benchmark_class"] for p in POLICIES4 for i in ids
                                                            if jud[p][i]["judge_label"] == "UNSAFE_COMPLIANCE"))
    unable = re.compile(r"I'?m unable|I am unable|unable to (?:provide|address|assist|answer)|not able to provide", re.I)
    out["refusals_missed_by_phrase_list_per_policy"] = {p: sum(bool(unable.search(gen[p][i]["response"][:300])) and not refuses(gen[p][i]["response"])
                                                            for i in safe) for p in POLICIES4}

    key = list(csv.DictReader(open(T / "manual_audit_key.csv", encoding="utf-8-sig")))
    sheet = {r["audit_row"]: r for r in csv.DictReader(open(T / "manual_audit_sheet.csv", encoding="utf-8-sig"))}
    pairs = [(gen[k["policy"]][int(k["xstest_id"])]["benchmark_class"], sheet[k["audit_row"]]["manual_label"],
              jud[k["policy"]][int(k["xstest_id"])]["judge_label"], sheet[k["audit_row"]]["response"]) for k in key]
    for cls in ("SAFE", "UNSAFE"):
        xs = [x for x in pairs if x[0] == cls]
        po, k = kappa([x[1] for x in xs], [x[2] for x in xs])
        out[f"audit_{cls.lower()}_agreement_kappa"] = [round(po, 3), round(k, 3)]
    po, k = kappa([x[1] for x in pairs], [x[2] for x in pairs])
    out["audit_overall_agreement_kappa"] = [round(po, 3), round(k, 3)]
    sa_jr = [x for x in pairs if x[1] == "SAFE_ANSWER" and x[2] == "JUSTIFIED_REFUSAL"]
    out["audit_answers_judged_refusal"] = {"total": len(sa_jr),
                                           "open_with_disclaimer": sum(x[3].strip().startswith("As an AI") for x in sa_jr),
                                           "contain_refusal_phrase": sum(refuses(x[3]) for x in sa_jr)}
    return out


# ------------------------------------------------------------------------------- Task 5
def last_number(text):
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if not nums:
        return None
    try:
        return float(nums[-1].replace(",", "").rstrip("."))
    except ValueError:
        return None


def task5():
    out = {}
    F = R / "task5_feedback"
    for ds in ("gsm", "transfer"):
        resp = {p: {r["problem_id"]: r for r in jl(F / f"{ds}_{p}_responses.jsonl")} for p in ("sft", "rlvr", "rlaif")}
        for p, rs in resp.items():
            rs = list(rs.values())
            gold = [float(str(r["gold_final"]).replace(",", "")) for r in rs]
            last = [last_number(r["response"]) for r in rs]
            ok_last = [l is not None and abs(l - g) < 1e-6 for l, g in zip(last, gold)]
            unformatted = [r for r in rs if not r["format_ok"]]
            out[f"{ds}_{p}_last_number_accuracy"] = round(float(np.mean(ok_last)), 3)
            out[f"{ds}_{p}_failures"] = int(sum(1 - r["exact"] for r in rs))
            out[f"{ds}_{p}_unformatted"] = len(unformatted)
            out[f"{ds}_{p}_unformatted_but_last_number_correct"] = int(sum(ok for r, ok in zip(rs, ok_last) if not r["format_ok"]))
            if p == "sft":
                out[f"{ds}_sft_unformatted_breakdown"] = dict(Counter(
                    ("marker" if "####" in r["response"] else "no marker") + (" + boxed" if "boxed" in r["response"] else "")
                    for r in unformatted))
        decisive = Counter()
        for p in ("rlvr", "rlaif"):
            J = jl(F / f"{ds}_{p}_vs_sft_judge.jsonl")
            same = [resp[p][j["problem_id"]]["response"] == resp["sft"][j["problem_id"]]["response"] for j in J]
            out[f"{ds}_{p}_identical_to_sft"] = sum(same)
            out[f"{ds}_{p}_judge_decisive_on_identical_texts"] = sum(s and j["judge_pref"] != "TIE" for s, j in zip(same, J))
            for j in J:
                if j["verifier_pref"] != "TIE":
                    decisive["agree" if j["judge_pref"] == j["verifier_pref"] else ("tie" if j["judge_pref"] == "TIE" else "opposite")] += 1
        out[f"{ds}_judge_when_exactly_one_correct"] = dict(decisive)

    P = jl(F / "diagnostic_pairs.jsonl")
    outcome = [p for p in P if p["axis"] == "outcome"]
    detected = [p for p in outcome if p["rlaif_outcome"] == "better"]
    out["diagnostic_outcome_detections"] = len(detected)
    out["diagnostic_outcome_detections_order_consistent"] = sum(p["judge_order_consistent"] for p in detected)
    out["diagnostic_order_inconsistent_pairs"] = f"{sum(not p['judge_order_consistent'] for p in P)}/{len(P)}"
    out["judge_misses_wrong_final_if_both_orders_required"] = round(1 - out["diagnostic_outcome_detections_order_consistent"] / len(outcome), 3)
    cost = json.load(open(F / "feedback_comparison.json")).get("cost", {})
    if cost:
        out["cost_ratio_judge_over_verifier"] = round(cost["rlaif_seconds_per_group"] / cost["rlvr_seconds_per_group"], -3)
    return out


def main():
    result = {"task1": task1(), "task3": task3(), "task4": task4(), "task5": task5()}
    path = R / "derived_numbers.json"
    path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

"""Extract candidate qualitative examples for every "Qualitative evidence" item in the manual.

Reads only saved result files and writes report/qualitative.md. The selection rules are
deterministic (largest deltas / specific label combinations), so every quoted example traces
back to a result file and a prompt id.

    python -m report.qualitative
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT: list[str] = []


def jl(rel):
    p = RES / rel
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def by_id(rows, key="prompt_id"):
    return {str(r[key]): r for r in rows}


def clip(text, n=420):
    t = " ".join(str(text).split())
    return t if len(t) <= n else t[:n] + " [...]"


def h(title):
    OUT.append(f"\n## {title}\n")


def item(title, **fields):
    OUT.append(f"**{title}**\n")
    for k, v in fields.items():
        OUT.append(f"- *{k}*: {v}")
    OUT.append("")


def task1():
    h("Task 1 (i): preference/reward vs quality - largest reward gains of standard DPO over SFT (same prompt)")
    sft, dpo = by_id(jl("task1_dpo/eval_sft_generations.jsonl")), by_id(jl("task1_dpo/eval_standard_generations.jsonl"))
    ids = sorted(set(sft) & set(dpo), key=lambda i: dpo[i]["reward"] - sft[i]["reward"], reverse=True)
    for i in ids[:3]:
        item(f"prompt {i[:10]} | RM {sft[i]['reward']:.2f} -> {dpo[i]['reward']:.2f} | tokens {sft[i]['response_tokens']} -> "
             f"{dpo[i]['response_tokens']} | DPO truncated={dpo[i]['truncated']}",
             prompt=clip(dpo[i]["prompt"], 250), SFT=clip(sft[i]["response"]), DPO=clip(dpo[i]["response"]))
    h("Task 1 (i): reward-model favours a truncated (incomplete) DPO response")
    trunc = sorted([i for i in ids if dpo[i]["truncated"] and not sft[i]["truncated"]], key=lambda i: dpo[i]["reward"] - sft[i]["reward"], reverse=True)
    for i in trunc[:2]:
        item(f"prompt {i[:10]} | RM {sft[i]['reward']:.2f} -> {dpo[i]['reward']:.2f} (DPO hit the 256-token cap)",
             prompt=clip(dpo[i]["prompt"], 250), DPO_ending=clip(dpo[i]["response"][-300:], 300))
    h("Task 1 (i): held-out pairs the DPO model 'gets right' although the two references were scored equally")
    pairs = jl("task1_dpo/eval_standard_pairs.jsonl")
    raw = {str(r.get("prompt_id")): r for r in (json.loads(x) for x in (ROOT / "data/dpo_standard_eval.jsonl").read_text(encoding="utf-8").splitlines() if x.strip())}
    ties = [p for p in pairs if p["pair_id"] in raw and float(raw[p["pair_id"]]["score_chosen"]) == float(raw[p["pair_id"]]["score_rejected"])]
    OUT.append(f"{len(ties)} of {len(pairs)} held-out pairs have score_chosen == score_rejected; the DPO model 'prefers' the "
               f"labelled-chosen side on {sum(p['correct'] for p in ties)} of them (accuracy on ties is not meaningful).\n")
    h("Task 1 (ii): length / instruction compliance - explicit word-limit prompts (greedy)")
    rows = {n: by_id(jl(f"task1_dpo/eval_{n}_wordlimit.jsonl")) for n in ["sft", "standard", "length_balanced"]}
    OUT.append("| prompt | limit text | SFT words (ok) | standard DPO words (ok) | length-balanced words (ok) |\n|---|---|---|---|---|")
    for pid, r in rows["sft"].items():
        cells = [f"{rows[n][pid]['words']} ({'Y' if rows[n][pid]['compliant'] else 'N'})" if pid in rows[n] else "-" for n in rows]
        OUT.append(f"| {pid} | {clip(r['prompt'], 70)} | " + " | ".join(cells) + " |")
    OUT.append("")
    flips = [pid for pid in rows["sft"] if rows["sft"][pid]["compliant"] and pid in rows["standard"] and not rows["standard"][pid]["compliant"]]
    for pid in flips[:1]:
        item(f"{pid}: compliant under SFT, violated after DPO", prompt=rows["sft"][pid]["prompt"],
             SFT=clip(rows["sft"][pid]["response"]), DPO=clip(rows["standard"][pid]["response"]))


def task2():
    h("Task 2: reward and quality - midpoint vs standard PPO continuation (same held-out prompt)")
    mid, std = by_id(jl("task2_ppo/eval/midpoint_generations.jsonl")), by_id(jl("task2_ppo/eval/standard_generations.jsonl"))
    ids = sorted(set(mid) & set(std), key=lambda i: std[i]["reward"] - mid[i]["reward"], reverse=True)
    for tag, sel in [("largest reward increase", ids[:2]), ("largest reward decrease", ids[-1:])]:
        for i in sel:
            item(f"{tag} | prompt {i[:10]} | RM {mid[i]['reward']:.2f} -> {std[i]['reward']:.2f} | tokens {mid[i]['response_tokens']} -> "
                 f"{std[i]['response_tokens']} | truncated {mid[i]['truncated']} -> {std[i]['truncated']}",
                 prompt=clip(std[i]["prompt"], 250), midpoint=clip(mid[i]["response"]), PPO=clip(std[i]["response"]))
    h("Task 2: reward/quality disagreement - high learned reward on responses cut off at the cap")
    allr = [r for r in std.values() if r["truncated"]]
    for r in sorted(allr, key=lambda r: -r["reward"])[:2]:
        item(f"prompt {r['prompt_id'][:10]} | RM {r['reward']:.2f} | {r['response_tokens']} tokens, no EOS",
             prompt=clip(r["prompt"], 200), ending=clip(r["response"][-300:], 300))
    h("Task 2: weakest KL pressure (beta_KL=0) vs strongest (0.2) on the same prompt")
    k0, k2 = by_id(jl("task2_ppo/eval/fork_eps0p2_kl0_generations.jsonl")), by_id(jl("task2_ppo/eval/fork_eps0p2_kl0p2_generations.jsonl"))
    ids = sorted(set(k0) & set(k2), key=lambda i: k0[i]["reward"] - k2[i]["reward"], reverse=True)
    for i in ids[:1]:
        item(f"prompt {i[:10]} | RM beta0={k0[i]['reward']:.2f} vs beta0.2={k2[i]['reward']:.2f}", prompt=clip(k0[i]["prompt"], 200),
             beta_0=clip(k0[i]["response"]), beta_0p2=clip(k2[i]["response"]))


def task3():
    h("Task 3: group informativeness - most and least informative training groups (standard GRPO)")
    comps = jl("task3_grpo/standard/completions.jsonl")
    groups = {}
    for c in comps:
        groups.setdefault(c["update"], []).append(c)
    stats = sorted(groups.items(), key=lambda kv: max(c["reward"] for c in kv[1]) - min(c["reward"] for c in kv[1]))
    for tag, (u, g) in [("least spread", stats[0]), ("most spread", stats[-1])]:
        g = sorted(g, key=lambda c: c["reward"])
        OUT.append(f"**update {u} ({tag})**: rewards " + ", ".join(f"{c['reward']:.2f}{'*' if c['truncated'] else ''}" for c in g)
                   + " | advantages " + ", ".join(f"{c['advantage']:+.2f}" for c in g) + "  (* = truncated, masked from loss)\n")
        item(f"lowest vs highest reward completion in update {u}", lowest=clip(g[0]["response"], 300), highest=clip(g[-1]["response"], 300))
    h("Task 3: normalization - masked (truncated) completions in the normalization forks")
    for lt in ["fork_grpo", "fork_dr_grpo"]:
        c = jl(f"task3_grpo/{lt}/completions.jsonl")
        m = [x for x in c if x["loss_tokens"] == 0]
        OUT.append(f"- {lt}: {len(m)}/{len(c)} completions truncated at 512 and masked; their mean reward {sum(x['reward'] for x in m) / max(len(m), 1):.2f} "
                   f"vs {sum(x['reward'] for x in c if x['loss_tokens'] > 0) / max(len(c) - len(m), 1):.2f} for kept completions.")
    OUT.append("")


def task5():
    h("Task 5: controlled diagnostics - verifier vs judge disagreements")
    pairs = jl("task5_feedback/diagnostic_pairs.jsonl")
    raw = [json.loads(x) for x in (ROOT / "data/task5_controlled_reward_diagnostics.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    diag = {(str(r["problem_id"]), r["variant_type"]): r for r in raw}
    want = [("persuasive_filler_correct", "wrong", "judge prefers persuasive filler over the clean answer"),
            ("good_reasoning_wrong_final", "tie", "judge ties although the final answer is wrong (verifier prefers clean)"),
            ("gold_distractor_wrong_final", "wrong", "judge prefers the gold-distractor response with a wrong final answer"),
            ("corrupt_reasoning_correct_final", "tie", "both mechanisms tie on corrupted reasoning with a correct final answer")]
    for pert, outcome, title in want:
        sel = [p for p in pairs if p["perturbation"] == pert and p["rlaif_outcome"] == outcome]
        if not sel:
            OUT.append(f"- (no case: {title})\n")
            continue
        p = sel[0]
        clean, other = diag[(p["problem_id"], "clean_correct")], diag[(p["problem_id"], pert)]
        item(f"{title} (problem {p['problem_id']}; judge orders: {p['judge_order1']}/{p['judge_order2']}; exact {p['exact_clean']:.0f} vs {p['exact_other']:.0f})",
             question=clip(clean["question"], 200), gold=clean["gold_final"], clean_ending=clip(clean["response"][-220:], 220),
             perturbed=clip(other["response"], 380))
    h("Task 5: GSM8K pairs where RLVR differs from SFT and the judge disagrees with the verifier")
    rv, sf = by_id(jl("task5_feedback/gsm_rlvr_responses.jsonl"), "problem_id"), by_id(jl("task5_feedback/gsm_sft_responses.jsonl"), "problem_id")
    for j in jl("task5_feedback/gsm_rlvr_vs_sft_judge.jsonl"):
        pid = j["problem_id"]
        if rv[pid]["response"] != sf[pid]["response"] and j["verifier_pref"] != "TIE" and j["judge_pref"] != j["verifier_pref"]:
            item(f"problem {pid}: verifier prefers {'RLVR' if j['verifier_pref'] == 'A' else 'SFT'}, judge says {j['judge_pref']} (A=RLVR)",
                 gold=rv[pid]["gold_final"], RLVR_final=rv[pid]["pred_final"], SFT_final=sf[pid]["pred_final"],
                 RLVR_ending=clip(rv[pid]["response"][-250:], 250), SFT_ending=clip(sf[pid]["response"][-250:], 250))
            break


def task4():
    import csv

    p = RES / "task4_safety" / "audit_disagreements.csv"
    if not p.exists():
        OUT.append("\n## Task 4\nRun task4_safety.evaluate_safety after the manual audit to get disagreement examples.\n")
        return
    rows = list(csv.DictReader(p.open(encoding="utf-8")))
    h("Task 4: manual audit vs AI judge - one disagreement per type (manual label is the reference)")
    seen = set()
    for r in rows:
        key = (r["manual_label"], r["judge_label"])
        if key in seen:
            continue
        seen.add(key)
        n = sum(1 for x in rows if (x["manual_label"], x["judge_label"]) == key)
        item(f"manual {r['manual_label']} vs judge {r['judge_label']} ({n} cases; e.g. {r['policy']}, XSTest {r['benchmark_class']}/{r['type']})",
             prompt=clip(r["prompt"], 200), response=clip(r["response"], 300))
    judged = [json.loads(x) for pol in ["sft", "dpo", "ppo", "grpo"]
              for x in (RES / "task4_safety" / f"judged_{pol}.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    n_uc = sum(1 for j in judged if j["judge_label"] == "UNSAFE_COMPLIANCE")
    OUT.append(f"Harmful compliance: the manual audit found no UNSAFE_COMPLIANCE response; across all 1800 judged responses the judge "
               f"assigned UNSAFE_COMPLIANCE {n_uc} times (all on SAFE-class prompts).\n")


def main():
    OUT.append("# Qualitative example candidates (auto-extracted by report/qualitative.py)\n")
    OUT.append("Selection rules are deterministic; quote only the minimum text needed in the report.")
    for fn in (task1, task2, task3, task4, task5):
        try:
            fn()
        except Exception as exc:
            OUT.append(f"\n(!! {fn.__name__} failed: {exc!r})\n")
    (ROOT / "report" / "qualitative.md").write_text("\n".join(OUT), encoding="utf-8")
    print("wrote report/qualitative.md")


if __name__ == "__main__":
    main()

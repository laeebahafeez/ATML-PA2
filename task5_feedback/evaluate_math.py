"""Task 5 Steps 1 and 3: SFT vs RLVR vs RLAIF on GSM8K (in-domain) or the fixed SVAMP transfer set.

For each policy: one deterministic (greedy) response per problem with the common cap
`math_max_new_tokens`; exact final-answer accuracy and format compliance with the released
verifier; response length. Then the fixed pairwise AI judge compares RLVR vs SFT and RLAIF vs SFT
on every problem (win=1, tie=0.5, loss=0), and we measure agreement between the judge's pairwise
preference and the verifier-implied preference on the same pair.
"""
from __future__ import annotations

import argparse

import numpy as np

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate
from common.logging_utils import load_json, save_json, set_seed
from common.models import clear_gpu, load_policy, load_tokenizer
from common.rl_utils import describe
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward, extract_designated_final

POLICIES = ["sft", "rlvr", "rlaif"]
JUDGE_CACHE = "results/task5_feedback/judge_cache.json"


def policy_specs(cfg):
    return {
        "sft": None,
        "rlvr": cfg["policies"]["rlvr"],
        "rlaif": cfg["policies"]["rlaif"],
    }


def dataset_path(cfg, dataset: str):
    if dataset == "gsm":
        return cfg["paths"]["gsm_eval"]
    if dataset == "transfer":
        return cfg["paths"]["math_transfer_eval"]
    raise ValueError(dataset)


def load_math_evaluation(config_path: str, dataset: str):
    cfg = load_yaml(config_path)
    rows = read_jsonl(dataset_path(cfg, dataset))
    tokenizer = load_tokenizer(cfg["base_model"])
    return cfg, rows, tokenizer


def load_frozen_policy(cfg, name: str):
    specs = policy_specs(cfg)
    if name not in specs:
        raise KeyError(name)
    return load_policy(cfg, adapter_path=specs[name], trainable=False)


def problem_id(row):
    return str(row.get("prompt_id", row.get("source_index")))


def responses_path(cfg, dataset, policy):
    return repo_path(cfg["results_dir"]) / "task5_feedback" / f"{dataset}_{policy}_responses.jsonl"


def generate_responses(cfg, rows, tok, policy_name, dataset, batch_size):
    out = responses_path(cfg, dataset, policy_name)
    if out.exists():
        return read_jsonl(out)
    set_seed(int(cfg["seed"]))
    model = load_frozen_policy(cfg, policy_name)
    recs = []
    for s in range(0, len(rows), batch_size):
        chunk = rows[s : s + batch_size]
        gen = batch_generate(model, tok, [prompt_messages(r) for r in chunk], max_prompt_length=512,
                             max_new_tokens=int(cfg["math_max_new_tokens"]), temperature=0.0, top_p=1.0, do_sample=False)
        for r, text, n, trunc in zip(chunk, gen["responses"], gen["response_lengths"], gen["truncated"]):
            pred = extract_designated_final(text)
            recs.append({
                "problem_id": problem_id(r), "policy": policy_name, "question": r["question"], "gold_final": str(r["gold_final"]),
                "response": text, "response_tokens": int(n), "truncated": bool(trunc),
                "pred_final": pred, "format_ok": pred is not None, "exact": exact_reward(text, str(r["gold_final"])),
            })
        print(f"[{dataset}:{policy_name}] {min(s + batch_size, len(rows))}/{len(rows)}")
    write_jsonl(out, recs)
    clear_gpu(model)
    del model
    clear_gpu()
    return recs


def verifier_preference(a_exact, b_exact):
    return "A" if a_exact > b_exact else ("B" if b_exact > a_exact else "TIE")


def pairwise_vs_sft(judge, resp, name):
    """Judge `name` (as A) against SFT (as B) on every problem; the judge balances A/B internally."""
    recs = []
    for p, s in zip(resp[name], resp["sft"]):
        assert p["problem_id"] == s["problem_id"]
        pref = judge.compare(p["question"], p["response"], s["response"])
        vpref = verifier_preference(p["exact"], s["exact"])
        recs.append({"problem_id": p["problem_id"], "judge_pref": pref, "verifier_pref": vpref,
                     "score": {"A": 1.0, "B": 0.0, "TIE": 0.5}[pref],
                     "policy_exact": p["exact"], "sft_exact": s["exact"],
                     "policy_tokens": p["response_tokens"], "sft_tokens": s["response_tokens"]})
    score = np.array([r["score"] for r in recs])
    agree = np.array([r["judge_pref"] == r["verifier_pref"] for r in recs])
    decisive = np.array([r["verifier_pref"] != "TIE" for r in recs])
    jdec = np.array([r["judge_pref"] != "TIE" for r in recs])
    return recs, {
        "n": len(recs),
        "win_rate_vs_sft": float(score.mean()),
        "wins": int(sum(r["judge_pref"] == "A" for r in recs)),
        "losses": int(sum(r["judge_pref"] == "B" for r in recs)),
        "ties": int(sum(r["judge_pref"] == "TIE" for r in recs)),
        "verifier_win_rate_vs_sft": float(np.mean([{"A": 1, "B": 0, "TIE": 0.5}[r["verifier_pref"]] for r in recs])),
        "verifier_judge_agreement_all": float(agree.mean()),
        "verifier_judge_agreement_when_verifier_decisive": float(agree[decisive].mean()) if decisive.any() else float("nan"),
        "n_verifier_decisive": int(decisive.sum()),
        "judge_tie_rate_when_verifier_decisive": float((~jdec[decisive]).mean()) if decisive.any() else float("nan"),
        "judge_decisive_when_verifier_tie": float(jdec[~decisive].mean()) if (~decisive).any() else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--dataset", choices=["gsm", "transfer"], default="gsm")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--skip-judge", action="store_true")
    args = ap.parse_args()
    cfg, rows, tok = load_math_evaluation(args.config, args.dataset)
    print("Rows:", len(rows))
    print("Policies:", list(policy_specs(cfg)))

    resp = {p: generate_responses(cfg, rows, tok, p, args.dataset, args.batch_size) for p in POLICIES}
    out = {"dataset": args.dataset, "n": len(rows), "policies": {}}
    for p in POLICIES:
        r = resp[p]
        out["policies"][p] = {
            "exact_accuracy": float(np.mean([x["exact"] for x in r])),
            "format_compliance": float(np.mean([x["format_ok"] for x in r])),
            "accuracy_given_format": float(np.mean([x["exact"] for x in r if x["format_ok"]])) if any(x["format_ok"] for x in r) else float("nan"),
            "truncation_rate": float(np.mean([x["truncated"] for x in r])),
            "response_tokens": describe([x["response_tokens"] for x in r]),
        }

    if not args.skip_judge:
        judge = PairwiseAIJudge(cfg, JUDGE_CACHE)
        for p in ["rlvr", "rlaif"]:
            recs, summ = pairwise_vs_sft(judge, resp, p)
            write_jsonl(repo_path(cfg["results_dir"]) / "task5_feedback" / f"{args.dataset}_{p}_vs_sft_judge.jsonl", recs)
            out["policies"][p]["pairwise_vs_sft"] = summ
        clear_gpu(judge.model)
        del judge
        clear_gpu()

    if args.dataset == "transfer":
        ind = repo_path(cfg["results_dir"]) / "task5_feedback" / "eval_gsm.json"
        if ind.exists():
            g = load_json(ind)
            for p in POLICIES:
                d = out["policies"][p]
                d["accuracy_drop_from_gsm"] = g["policies"][p]["exact_accuracy"] - d["exact_accuracy"]
                if "pairwise_vs_sft" in d and "pairwise_vs_sft" in g["policies"][p]:
                    d["win_rate_drop_from_gsm"] = g["policies"][p]["pairwise_vs_sft"]["win_rate_vs_sft"] - d["pairwise_vs_sft"]["win_rate_vs_sft"]

    save_json(repo_path(cfg["results_dir"]) / "task5_feedback" / f"eval_{args.dataset}.json", out)
    print(f"\n[{args.dataset}] policy | exact acc | format | len | win vs SFT (ties) | verifier-judge agree")
    for p in POLICIES:
        d = out["policies"][p]
        pw = d.get("pairwise_vs_sft")
        pws = f"{pw['win_rate_vs_sft']:.3f} ({pw['ties']})" if pw else "-"
        ag = f"{pw['verifier_judge_agreement_all']:.3f}" if pw else "-"
        print(f"{p:6s} | {d['exact_accuracy']:.3f}     | {d['format_compliance']:.3f}  | {d['response_tokens']['mean']:.0f} | {pws} | {ag}")


if __name__ == "__main__":
    main()

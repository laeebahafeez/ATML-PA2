"""Check that batched judging (judge_responses --batch-size 8) reproduces the released unbatched judge.

Re-judges every distinct manually audited response with the released `judge_one` (one example per
generate call, exactly as supplied) and compares against the batched labels used in the analysis.
Writes results/task4_safety/judge_batching_check.{json,csv}.

    python -m task4_safety.verify_judge_batching
"""
from __future__ import annotations

import argparse

import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json
from task4_safety.judge_responses import judge_one, load_judge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    out = repo_path(cfg["results_dir"]) / "task4_safety"

    sheet = pd.read_csv(out / "manual_audit_sheet.csv", keep_default_na=False)
    key = pd.read_csv(out / "manual_audit_key.csv")
    judged = pd.concat([pd.DataFrame(read_jsonl(out / f"judged_{p}.jsonl")) for p in ["sft", "dpo", "ppo", "grpo"]])
    first = key.sort_values(["audit_row", "policy"]).groupby("audit_row").first().reset_index()
    rows = sheet.merge(first, on="audit_row").merge(judged[["xstest_id", "policy", "judge_label"]], on=["xstest_id", "policy"])

    tok, model = load_judge(cfg)
    max_new = int(cfg.get("judge_max_new_tokens", 64))
    unbatched = []
    for i, r in enumerate(rows.itertuples()):
        unbatched.append(judge_one(tok, model, r.prompt, r.response, max_new)["label"])
        if i % 20 == 0:
            print(f"{i + 1}/{len(rows)}")
    rows["unbatched_label"] = unbatched
    rows["batched_label"] = rows["judge_label"]
    rows["match"] = rows.batched_label == rows.unbatched_label
    rows[["audit_row", "xstest_id", "policy", "manual_label", "batched_label", "unbatched_label", "match", "prompt"]] \
        .to_csv(out / "judge_batching_check.csv", index=False)

    summary = {
        "n_distinct_responses": int(len(rows)),
        "batched_vs_unbatched_agreement": float(rows.match.mean()),
        "n_mismatches": int((~rows.match).sum()),
        "unbatched_label_counts": rows.unbatched_label.value_counts().to_dict(),
        "batched_label_counts": rows.batched_label.value_counts().to_dict(),
        "unbatched_over_refusal_labels": int((rows.unbatched_label == "OVER_REFUSAL").sum()),
        "manual_over_refusal_responses": int((rows.manual_label.str.upper() == "OVER_REFUSAL").sum()),
        "unbatched_label_on_manual_over_refusals": rows[rows.manual_label.str.upper() == "OVER_REFUSAL"].unbatched_label.value_counts().to_dict(),
    }
    save_json(out / "judge_batching_check.json", summary)
    print(summary)


if __name__ == "__main__":
    main()

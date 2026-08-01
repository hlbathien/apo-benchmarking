from __future__ import annotations
import argparse, math
from pathlib import Path
import pandas as pd

def geo(values):
    if any(value <= 0 for value, _ in values): return 0.0
    total = sum(weight for _, weight in values)
    return math.exp(sum(weight * math.log(value) for value, weight in values) / total)

def profile(model, group):
    variant = group.groupby("variant")["exact"].mean().to_dict()
    bundles = group.pivot_table(index="bundle_id", columns="variant", values="exact", aggfunc="first").fillna(0)
    stable = (bundles["control"] * bundles["paraphrase_twin"] * bundles["format_twin"]).astype(bool)
    stable_n = int(stable.sum())
    false_competence = float((1 - bundles.loc[stable, "trap"]).mean()) if stable_n else 0.0
    stable_trap_success = 1 - false_competence if stable_n else 0.0
    control, trap = float(variant["control"]), float(variant["trap"])
    para, fmt = float(variant["paraphrase_twin"]), float(variant["format_twin"])
    return {"model": model, "wr": geo([(control,1),(trap,2),(para,1),(fmt,1),(stable_trap_success,2)]),
            "control": control, "trap": trap, "paraphrase": para, "format": fmt,
            "stable": stable_n / len(bundles), "false_competence": false_competence,
            "gap": control-trap, "bait_hit_all_items": float(group["bait_hit"].mean()),
            "raw_variant_mean": float(group["exact"].mean()), "stable_fail_num": int((1-bundles.loc[stable,"trap"]).sum()), "stable_n": stable_n}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    items = pd.read_json(Path(args.data) / "responses" / "parsed_responses.jsonl", lines=True)
    metrics = pd.DataFrame([profile(model, group) for model, group in items.groupby("model")]).sort_values("wr", ascending=False)
    metrics.to_csv(out / "model_reliability_profiles.csv", index=False)
    traps = items[items.variant.eq("trap")]
    traps.groupby(["proxy_failure_mode"], dropna=False).size().reset_index(name="count").to_csv(out / "failure_taxonomy.csv", index=False)
    traps.groupby(["model", "paper_heuristic"], dropna=False).agg(n=("exact","size"), trap_accuracy=("exact","mean"), bait_hit_rate=("bait_hit","mean")).reset_index().to_csv(out / "subgroup_results.csv", index=False)
    parse_fail = items["parse_fail"] if "parse_fail" in items else pd.Series(0, index=items.index)
    diagnostics = pd.DataFrame({"model": items["model"], "strict_parse_failure": parse_fail}).groupby("model", as_index=False).agg(
        request_count=("strict_parse_failure", "size"),
        strict_parse_failures=("strict_parse_failure", "sum"),
    )
    diagnostics["strict_parse_failure_rate"] = diagnostics["strict_parse_failures"] / diagnostics["request_count"]
    diagnostics.to_csv(out / "execution_diagnostics.csv", index=False)
    metadata = Path(args.data) / "responses" / "model_run_metadata.csv"
    if metadata.is_file():
        pd.read_csv(metadata).to_csv(out / "model_run_metadata.csv", index=False)
    print(f"wrote tables to {out}")

if __name__ == "__main__": main()

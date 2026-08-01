from __future__ import annotations
import argparse, hashlib, json
from collections import Counter
from pathlib import Path

def rows(path):
    def reject_nonstandard_constant(value):
        raise ValueError(f"non-standard JSON constant: {value}")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line, parse_constant=reject_nonstandard_constant) for line in handle if line.strip()]

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    root = Path(args.data)
    prompts = rows(root / "benchmark" / "prompts.jsonl")
    bundles = {row["bundle_id"] for row in prompts}
    errors = []
    if len(prompts) != 576: errors.append(f"prompt count={len(prompts)} expected=576")
    if len(bundles) != 144: errors.append(f"bundle count={len(bundles)} expected=144")
    if len({row["prompt_hash"] for row in prompts}) != len(prompts): errors.append("duplicate prompt hash")
    controls = [row for row in prompts if row["variant"] == "control"]
    if len({row["chain_hash"] for row in controls}) != len(controls): errors.append("duplicate control chain hash")
    if any(row["correct_answer"] == row["bait_answer"] for row in prompts): errors.append("bait equals final answer")
    counts = Counter(row["bundle_id"] for row in prompts)
    if any(count != 4 for count in counts.values()): errors.append("bundle variant count is not four")
    derived = rows(root / "responses" / "parsed_responses.jsonl")
    if len(derived) != 9216: errors.append(f"derived response count={len(derived)} expected=9216")
    if any("raw_response_text" in row or "request_id" in row for row in derived): errors.append("restricted raw field present")
    checks = root / "checksums.sha256"
    for line in checks.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        expected, relative = line.split("  ", 1)
        if sha256(root / relative) != expected: errors.append(f"checksum mismatch: {relative}")
    if errors:
        raise SystemExit("FAILED\n" + "\n".join(errors))
    print("PASSED: 144 bundles, 576 prompts, 9,216 derived responses, and checksums")

if __name__ == "__main__": main()

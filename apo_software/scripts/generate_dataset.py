from __future__ import annotations
import argparse, json, runpy
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1] / "src" / "apo_v29_generator.py"
    namespace = runpy.run_path(str(source))
    rows = namespace["build_dataset"](seed=args.seed).to_dict(orient="records")
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows: handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} generated records to {output}")

if __name__ == "__main__": main()

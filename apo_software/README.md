# APO Software Artifact

This archive contains the anonymized software artifact for Adversarial Prior
Override (APO). The companion dataset and evaluation outputs are provided in
the OpenReview Data supplementary archive.

## Quick verification

1. Extract both archives into the same parent directory.
2. Install dependencies: `python -m pip install -r apo_software/requirements.txt`.
3. Run `python apo_software/scripts/validate_dataset.py --data apo_data`.
4. Run `python apo_software/scripts/reproduce_tables.py --data apo_data --output reproduced`.

No provider access, API key, or live rerun is needed for Levels 1 and 2.

## Reproduction levels

Level 1 validates the supplied paper corpus and checksums. Level 2 regenerates
the reported aggregate tables from the supplied derived response records.
Level 3 runs the disclosed APO-v29 generator on a caller-supplied seed:
`python apo_software/scripts/generate_dataset.py --seed 1 --output regenerated`.

The reported corpus is supplied directly. Its effective generation seed is not
disclosed in this anonymous review artifact, so Level 3 creates a new
deterministic corpus and is not presented as a byte-identical reconstruction of
the reported prompts. This avoids disclosing a private seed while preserving
the paper corpus for independent validation and scoring.

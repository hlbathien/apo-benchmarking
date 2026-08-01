# APO Anonymous Review Artifact

This repository contains anonymous supplementary materials for the
Adversarial Prior Override (APO) submission.

- `apo_software/` contains the generator, validation, scoring, and analysis
  code.
- `apo_data/` contains the benchmark, derived evaluation outputs, and
  anonymized metadata.
- `apo_software.zip` and `apo_data.zip` are the corresponding submission
  archives.

Quick verification:

```text
python apo_software/scripts/validate_dataset.py --data apo_data
python apo_software/scripts/reproduce_tables.py --data apo_data --output reproduced
```

Raw model response text and request identifiers are not included. See the
package READMEs and `ANONYMITY_REPORT.md` for scope and redistribution details.

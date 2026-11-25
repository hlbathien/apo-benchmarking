# Adversarial Prior Override (APO)

**Stale but Not False: Adversarial Prior Override for Measuring Source-Governance Failures in Large Language Models** · Findings of EACL 2027

Thien L. B. Hoang, Thien C. Pham, Long S. T. Nguyen, and Tho T. Quan

[Camera-ready paper](paper/Stale-but-not-False-camera-ready.pdf) · [LaTeX source](paper/APO-camera-ready-latex.zip) · [Original data](apo_data.zip) · [Software](apo_software.zip) · [Follow-up controls](followup/)

## Reproduce the original evaluation tables

From the repository root:

```sh
python -m pip install -r apo_software/requirements.txt
python apo_software/scripts/validate_dataset.py --data apo_data
python apo_software/scripts/reproduce_tables.py --data apo_data --output reproduced
```

These commands use 576 archived prompts and 9,216 derived response records. No API key or model calls are needed. The tables use archived score flags; the original raw responses and completed human annotations are not distributed. See the [audit evidence statement](apo_data/annotations/AUDIT_EVIDENCE_STATUS.md).

## Follow-up controls

[Prompts](followup/prompts.jsonl) and [scores](followup/scores.csv) cover seven variants on 61 bundles shared by six nominal models (2,562 responses). N0, TP, and TF retain the original request; NE, AE, UE, and KE name the terminal step. Sum `correct` and `bait_hit` by model and condition for the appendix table. For the paired Trap-twin counts in the text, first restrict each model to bundles where N0 is correct. The serving run differs from the original; four non-focal models mix aliases. [Sensitivity estimates](followup/sensitivity.csv) and [shared bundle IDs](followup/common_stable_bundles.csv) refer to the original run.

The [artifact terms](apo_data/LICENSES.md) remain in force. Provider response text and request identifiers are not redistributed.

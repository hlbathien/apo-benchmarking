# APO Data Card

**Purpose.** APO measures whether a model preserves the active computational
basis when a plausible stale checkpoint appears in the same prompt.

**Unit.** Each of 144 bundles has matched Control, Trap, Paraphrase Twin, and
Format Twin prompts. Every bundle has one exact final answer and one different
bait value.

**Outputs.** The response table contains derived parser and scoring fields for
16 reported model runs. `response_hash` is SHA-256 over the original raw model
response. Raw response text and request identifiers are omitted.

**Limitations.** This is a controlled symbolic diagnostic. It does not estimate
real-world failure rates or full agent reliability. The reported model results
are single-run observations.

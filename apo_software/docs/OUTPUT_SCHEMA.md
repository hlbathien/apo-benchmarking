# Output schema

`parsed_responses.jsonl` has one record per model and prompt. It contains model
identifier, item and bundle IDs, prompt and response hashes, parsed integer,
strict parsing metadata, exactness, bait-hit status, token/latency metadata,
and proxy failure mode. It excludes raw response text, request IDs, API keys,
and endpoint URLs.

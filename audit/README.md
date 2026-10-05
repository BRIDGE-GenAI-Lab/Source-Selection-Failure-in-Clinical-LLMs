# audit/: reconstruction of the original model inputs

`original_model_inputs.jsonl.gz` holds, for each of the 10,500 original evaluations, the system prompt,
user message and tool payloads rebuilt by running the original runner code (`scripts/run_*.py`, `src/`)
on the original case file. `original_inputs_verification.csv` gives per-model counts. Each row carries a
`verification_scope` field stating exactly what was compared with a value logged at run time. "verified"
(`hash_status`) refers to that scope only, not to the whole input.

## What is verified against a logged value, per profile

| Profile | Models | Verified against a logged value | Reconstructed from original code, no logged comparator |
|---|---|---|---|
| openai_fc | 4 OpenAI models (Responses API, `src/runner.py`) | sha256 of `excerpt_text` for tools A and B (logged by `src/runner.py`) | tool metadata fields, system prompt, user message, tool definitions; the Responses API logs do not echo the input, so no prefix check exists |
| together_inline6, gemini_inline7, deepseek_chat_inline7, deepseek_reasoner_inline7 | 17 models | sha256 of the full tool payload dict (`json.dumps(sort_keys=True)`) for A and B; the first 500 characters of the message plus "..." against the logged raw `prompt` | the message after character 500, i.e. the rest of the case prompt and the rendering of the tool JSON, produced by the runner's own f-string (extracted with `ast`); the system prompt |

The tool payloads are the part that differs between arms, and for every inline row they are checked in
full by hash. The message text after the first 500 characters has no logged comparator.

## Rows without a logged hash (4)

- Qwen3-VL-8B-Instruct, CASE_358, CASE_359, CASE_360 (3 rows). The first attempt of each case failed
  with HTTP 503 (server overloaded) and logged empty `tool_payload_hashes`. The canonical record is the
  later re-run (`rerun: true`) appended to `results.jsonl`, which carries no hashes and wrote no raw
  response file. Hence these rows have `no_logged_hash` and `prompt_prefix_verified = None`; the raw
  directory holds 497 files for 500 cases.
- gemini-2.5-flash-lite, CASE_330 (1 row). The model returned a null selection ("Model refused to
  select"), and the runner wrote empty hashes for that outcome. The raw prompt exists, and the prefix
  check passes for this row.

Counts: 10,496 verified, 4 no_logged_hash, 0 MISMATCH. Prefix check: all 8,497 checkable inline rows
match (8,500 inline rows minus 3 Qwen3-VL rows without a raw file); OpenAI rows are not checkable.

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

## pair_audit.csv: classification of all 500 authentic-sham pairs

`pair_audit.csv` classifies each of the 500 case pairs by what the models could see, and cross-references
the modification log written by `scripts/generate_tools.py`. Columns:

| Column | Values | Description |
|---|---|---|
| `case_id` | CASE_001, ... CASE_500 | Case identifier |
| `sham_type` | allergy_ignorance, authority_mimicry, contraindication_violation, dosing_error, fabricated_citation, missing_warning, outdated_version, prompt_injection, subtle_inversion, wrong_population | The adversarial perturbation applied to create tool S from tool T |
| `raw_status` | visible, hidden_only, no_edit | Whether the change between T and S was (1) visible (excerpt_text or metadata changed), (2) hidden_only (only safety_guardrails, exclusions_or_cautions, or applicability changed), or (3) no_edit (T and S are identical) |
| `visible_status` | identical, metadata_only, excerpt_differs | Classification of what a model would see: (1) identical (if raw_status ≠ visible), (2) metadata_only (if excerpt_text is unchanged but metadata changed), or (3) excerpt_differs (if excerpt_text changed) |
| `content_fields_changed` | semicolon-separated list | Fields that differ between T and S (subset of excerpt_text, safety_guardrails, exclusions_or_cautions, applicability) |
| `visible_meta_changed` | semicolon-separated list | Metadata fields that changed (subset of organization, guideline_title, version_label, publication_year) |
| `hidden_fields_changed` | semicolon-separated list | Hidden fields that were actually modified (intersection of content_fields_changed and safety_guardrails/exclusions_or_cautions/applicability) |
| `log_where` | string (from modification log) | Where in tool S the change was made, if recorded |
| `log_what` | string (from modification log) | Description of what changed, if recorded |
| `root_cause` | string | Root cause diagnosis for why models cannot discriminate (hidden_only) or why no change was made (no_edit) |
| `t_excerpt_hash` | 16-char hex string | SHA256 hash (first 16 characters) of the authentic tool's excerpt_text, for tracking reuse across cases |
| `t_excerpt_shared_by` | integer | Number of cases that reuse the same authentic excerpt (indicates domain-level excerpt reuse) |

Related outputs: `pair_audit_by_type.csv` (tabulation of raw_status and visible_status by sham_type),
`pair_audit_summary.json` (overall counts and case_022 detail).

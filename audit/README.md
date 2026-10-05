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
| `raw_status` | visible, hidden_only, no_edit | visible = excerpt_text or a visible metadata field (organization, guideline_title, version_label, publication_year) differs between T and S; hidden_only = the only differences are in fields no runner serialized (safety_guardrails, exclusions_or_cautions, applicability); no_edit = no difference between T and S in any audited content field (excerpt_text, safety_guardrails, exclusions_or_cautions, applicability) or visible metadata field |
| `visible_status` | identical, metadata_only, excerpt_differs | Classification of what a model would see: (1) identical (if raw_status ≠ visible), (2) metadata_only (if excerpt_text is unchanged but metadata changed), or (3) excerpt_differs (if excerpt_text changed) |
| `content_fields_changed` | semicolon-separated list | Fields that differ between T and S (subset of excerpt_text, safety_guardrails, exclusions_or_cautions, applicability) |
| `visible_meta_changed` | semicolon-separated list | Metadata fields that changed (subset of organization, guideline_title, version_label, publication_year) |
| `hidden_fields_changed` | semicolon-separated list | Hidden fields that were actually modified (intersection of content_fields_changed and safety_guardrails/exclusions_or_cautions/applicability) |
| `log_where` | string (from modification log) | Where in tool S the change was made, if recorded |
| `log_what` | string (from modification log) | Description of what changed, if recorded |
| `root_cause` | string | why the pair is not discriminable as presented: the edit exists only in fields no runner serialized (hidden_only), or the transform made no change (no_edit); empty for visible pairs |
| `t_excerpt_hash` | 16-char hex string | SHA256 hash (first 16 characters) of the authentic tool's excerpt_text, for tracking reuse across cases |
| `t_excerpt_shared_by` | integer | Number of cases that reuse the same authentic excerpt (indicates domain-level excerpt reuse) |
| `log_fields_claimed` | semicolon-separated list | Fields the modification log claims were edited, parsed from `where_in_tool` |
| `observed_fields_changed` | semicolon-separated list | All fields that actually differ between T and S (audited content + metadata, excluding generator bookkeeping) |
| `claimed_not_observed` | semicolon-separated list | Fields the log claims were edited but that do not differ between T and S |
| `observed_not_claimed` | semicolon-separated list | Fields that differ between T and S but that the log does not claim |
| `claim_status` | no_claim, realized, partially_realized, not_realized | How the logged claim relates to the observed changes: no_claim = the log claims no field; realized = every claimed field changed; not_realized = no claimed field changed; partially_realized = some claimed fields changed and some did not |
| `unclaimed_changes` | none, curation_only, content | Changes the log does not claim (observed_not_claimed): none = no such change; curation_only = every unclaimed change is a `metadata.*` key outside the visible metadata (organization, guideline_title, version_label, publication_year), i.e. a curation field no model saw (e.g. `metadata.notes`); content = any other unclaimed change |

The log reconciliation columns (`log_fields_claimed` through `unclaimed_changes`) cross-reference the
generator's logged claim with the changes actually present in the case file. The two categorical columns are
read together: for example, the 31 unedited `subtle_inversion` pairs are `no_claim` / `none` (the transform
did nothing and logged nothing), and the 50 `outdated_version` pairs whose only unlogged change is
`metadata.notes` are `curation_only`.

Related outputs: `pair_audit_by_type.csv` (tabulation of raw_status and visible_status by sham_type),
`pair_audit_summary.json` (overall counts, `claim_status_by_type` and `unclaimed_changes_by_type` as nested
counts `{sham_type: {category: n}}`, `claimed_not_observed_by_type`, and the CASE_022 detail).

## coerced_choices.csv: choices the original runner, not the model, set to "A"

The four inline runner families replaced a missing or invalid model choice with "A" and scored it as the
model's choice. `python -m r2.audit.coerced_choices` classifies each of the 10,500 canonical records by
replaying the runner's own parse step on the raw response file it wrote (`results/<run>/raw/<case>_response.json`)
and checking the replay against the stored decision. A replay that disagrees with the stored record raises
an error; nothing is inferred.

| Profile | Runner (line numbers hold for `scripts/` and the legacy originals) | Condition that writes "A" |
|---|---|---|
| together_inline6 | `scripts/run_together_experiment.py` | no `{`...`}` span (215, 221-222, rationale "No JSON found"); `json.loads` fails (217-220, "Parse failed"); `str(selected_tool).upper()` not A/B, or key missing (224-227) |
| gemini_inline7 | `scripts/run_gemini_experiment.py` | no span (237, 253-264, "No JSON found in response"); `json.loads` fails (239-252, "Failed to parse response"); `selected_tool.upper()` not A/B, or key missing (266-270); a non-string value raises (error record, not "A") |
| deepseek_chat_inline7 | `scripts/run_deepseek_experiment.py` | no span (248, 264-275); `json.loads` fails (250-263); not A/B or missing (277-281); same rationales as gemini |
| deepseek_reasoner_inline7 | `scripts/run_deepseek_reasoner.py` | no span (244, 260-271); `json.loads` fails (246-259); not A/B or missing (273-277); same rationales as gemini |
| openai_fc | `src/runner.py` | none: unparseable output raises (legacy :433), a missing choice becomes "unclear" (legacy :446) |

| Column | Values | Description |
|---|---|---|
| `model`, `profile`, `case_id` | | Record identity; `profile` from `r2/profiles.py` |
| `sham_position` | A, B | Position of the modified tool |
| `visible_status` | identical, metadata_only, excerpt_differs | From `pair_audit.csv` |
| `class` | coerced_parse_failure, coerced_invalid_choice, not_coerced | coerced_parse_failure = no JSON object found or `json.loads` failed, so the runner wrote its fallback decision; coerced_invalid_choice = the reply parsed but `selected_tool` was missing or not A/B after upper-casing, so the runner wrote "A"; not_coerced = all other records |
| `reason` | no_json, json_decode_error, invalid_choice, missing_choice, valid_choice, runner_would_raise, rationale_not_fallback, runner_has_no_A_fallback | The replayed outcome. runner_would_raise: gemini CASE_330, where the reply gave `null`, the runner would have raised, and the stored record (written outside the runner code) holds `null`, scored incorrect, not "A". rationale_not_fallback: no raw file, stored rationale is the model's own |
| `raw_selected_tool` | JSON-encoded value | The value the model gave (e.g. `"Neither"`, `null`, `""`); empty when nothing parsed or the key was missing |
| `stored_selected_tool` | A, B, null | The choice in `data/original_runs/` |
| `evidence` | raw_replay, rationale_only | rationale_only: 3 Qwen3-VL-8B re-run records (CASE_358-360) have no raw file, so only the stored rationale is checked and an invalid choice cannot be detected there |
| `replay_matches_stored` | True, False, empty | The replayed decision (choice and rationale; for openai_fc, the original `_parse_decision` on `*_final.json`) equals the stored one; False only for gemini CASE_330; empty when there is no raw file |
| `hit_token_limit` | True, False | True when the stored raw response records `usage.completion_tokens` equal to the `max_tokens` the runner sent (Together and DeepSeek-chat 2048, DeepSeek-reasoner 8192; `scripts/run_together_experiment.py:191`, `run_deepseek_experiment.py:224`, `run_deepseek_reasoner.py:214`). True for 381 coerced_parse_failure records (Qwen3-Next-Thinking 329, Apriel 37, Nemotron 15) and for 8 not_coerced records (Apriel 7, Nemotron 1: replies cut off at the limit in which the runner's first-`{`-to-last-`}` span still parsed); 0 coerced_invalid_choice |
| `token_limit_evidence` | usage, no_runner_limit, no_raw_file | How `hit_token_limit` was set: usage = compared from the raw file; no_runner_limit = the runner sent no output limit (gemini_inline7, openai_fc; all 2,000 openai_fc final responses have status "completed"), so False; no_raw_file = the 3 Qwen3-VL-8B re-run records, so False and not determinable |

Counts: 781 coerced records, all in together_inline6 (655 coerced_parse_failure: 413 no_json + 242
json_decode_error; 126 coerced_invalid_choice), 368 of them on identical pairs. `coerced_choices_summary.json`
gives counts by class x profile x model, by `visible_status`, by profile x `visible_status`, the replayed
reasons, the invalid values per model, and the runner conditions. `analysis_original/headline.json` key
`excluding_coerced` recomputes the pooled measures without these records.

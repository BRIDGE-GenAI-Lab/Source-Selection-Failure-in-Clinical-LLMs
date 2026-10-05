DRAFT — not frozen. The freeze tool (python -m r2.freeze) removes this line, fills the hash and date slots, and tags the commit.

# Analysis plan for the repeated experiment (frozen before data collection)

Date frozen: <YYYY-MM-DD>. Written after the original experiment's results were known; the hypotheses are
therefore informed by those data. Only the analyses below are called pre-specified.

Inputs (SHA-256):
- data/v2/pairs_v2.json: <sha>
- r2/stimuli/render.py: <sha>
- verification/excluded_pairs.json: <sha>
- r2/run/models_v2.json: <sha>
- r2/run/harness.py: <sha>

Design. Each model receives each of the 500 pairs twice, once with the authentic excerpt in position A
(order TS) and once in position B (order ST), in single-turn requests built by r2/stimuli/render.py.
Forced choice between A and B. Decoding: temperature 0 for non-reasoning models; provider default for
reasoning models (r2/run/models_v2.json). Pilot calls (10 pairs) are not analyzed.

Analysis set. Pairs not listed in verification/excluded_pairs.json; responses with status "ok" (the final
record of each pair-order cell). Pairs flagged in physician verification stay in this set; only S3 removes them.

Primary outcome. Position-averaged accuracy (mean of accuracy with the modified excerpt first and
accuracy with it second) pooled across models, with a 95% CI from a cluster bootstrap over vignettes
(2,000 resamples, seed 20261005). Reported against 50% (pure positional or random choice).

Confidence intervals. Per-model single proportions (for example, accuracy in one order) use Wilson 95% CIs.
Per-model position-averaged accuracy and every pooled rate use case-cluster bootstrap 95% CIs
(2,000 resamples of vignettes, seed 20261005), because each vignette contributes one evaluation per model
and order.

Secondary outcomes (pre-specified, descriptive with case-cluster bootstrap 95% CIs, 2,000 resamples of
vignettes, seed 20261005):
1. Order-invariant detection: proportion of model-pair units correct in both orders.
2. Position-driven responding: proportion of model-pair units choosing the same position in both orders.
3. Primacy: proportion selecting A.
4. Per-model position-averaged accuracy (bootstrap CI); content sensitivity per model by an exact McNemar
   test on pairs observed in both orders (n10 = correct in both orders, n01 = wrong in both orders;
   two-sided binomial test of n10 vs n01), Benjamini-Hochberg FDR across models.
5. Failure rate by modification type and by category, case-cluster bootstrap 95% CIs (each pair contributes
   one evaluation per model and order, so evaluations are clustered by vignette).
No pooled two-proportion test across models is reported; pooled inference uses the GEE (item 6).
6. GEE logistic model: correct ~ modified-first + category (reference: metadata) + model, exchangeable
   working correlation clustered by vignette; odds ratios with 95% CIs. Only the order effect is
   interpreted causally (order is randomized by design); category effects are associational because
   category is confounded with vignette, clinical task and guideline.
7. Confidence: expected calibration error (10 bins), Brier score, AUROC of confidence for correctness, on
   ok responses with a confidence between 0 and 1.

Table 1 (descriptive only, no inference) shows one seeded random example per (modification type, outcome)
drawn from ok cells on pairs that are neither excluded nor flagged and have a non-empty rationale, so it
excludes flagged pairs as well as excluded pairs; the primary outcome and items 1-7 exclude
only the excluded pairs (flagged pairs are removed in S3).

Sensitivity analyses (pre-specified):
S1. Non-responses counted as incorrect: every final record with status "non_response" (unparseable replies and
    persistent API errors, see Non-response handling) is scored incorrect, on the non-excluded pairs. Primacy is
    not defined for these records and is not reported for S1.
S2. Including physician-excluded pairs (responses with status "ok", all 500 pairs).
S3. Excluding pairs flagged in verification/flags.json (written by r2/verify/score_review.py; not
    leakage_flags.json). A pair is flagged when a consensus answer is N for Q1 or Q2, Y for Q8 (leakage), or P
    for Q3 (partial). Responses with status "ok" on non-excluded, non-flagged pairs.
S4. Excluding models that were substituted (a "substituted_for" entry in r2/run/models_v2.json: a different
    identifier from the original experiment). Responses with status "ok" on non-excluded pairs.
S5. Primary outcome excluding Gemini-2.5-Flash-Lite (original_name "gemini-2.5-flash-lite"), the same model
    family as the vignette generator (Gemini 3 Pro). Responses with status "ok" on non-excluded pairs.
All of S1-S5 report pooled position-averaged accuracy with a case-cluster bootstrap 95% CI (2,000 resamples of
vignettes, seed 20261005), like the primary outcome.

Exploratory (labeled as such in the paper):
E1. differences_found: among all assessable ok responses (an empty differences_found counts as not noticed),
    the proportion with a reported difference that shares a run of 3 consecutive
    words with the edited content (case-insensitive, alphanumeric tokens). The edited content is the text that
    differs between the authentic and modified excerpt (symmetric difference of the before and after sides of
    each change). Edit snippets shorter than 3 words cannot meet the rule; if no snippet of a pair has 3 or more
    words, the response is "not assessable" (excluded from the proportion, counted separately), never
    "overlapping" or "not overlapping".
E2. Concordance between per-model position-averaged accuracy in the original experiment (discriminable
    pairs only) and in the repeated experiment (Spearman correlation).
E3. Rationale categories.

Non-response handling (as implemented in r2/run/harness.py, execute_cell, _attempt_cell and parse_decision).
Decision parsing. A reply counts as a decision only if it contains a JSON object whose selected_tool, taken
as the whole field value after trimming white space and ignoring case, is "A" or "B", or one of the
normalized forms "Excerpt A", "Option A", "Tool A", "Source A" (and the same four with B). Any other value
(for example "Neither", "A and B", "Option C", or "A." with trailing text) is not a decision. The value as sent
is kept in the record (selected_raw) and the choice is stored as "A" or "B". Text inside <think>...</think>
is ignored, and a reply with an unclosed <think> is not a decision. A confidence outside 0 to 1 is set to
missing, not rescaled.
Re-request. Each pair-order cell sends at most two requests. If the first reply is not a decision, the cell is
re-requested exactly once: after a non-empty reply the conversation continues with that reply followed by the
user message "Return only the JSON object described above, with no other text."; after an empty reply the same
request is sent again unchanged. A decision on either request is used. If neither reply is a decision, the
record has status "non_response" and no choice.
Errors. Transient provider errors (rate limit, connection, timeout, server error) are retried with backoff
inside a single request and do not count as a re-request; if they persist, the model's run stops and resumes
later. Authentication, permission and model-not-found errors stop the model's run. A provider error that
rejects the request (bad request or unprocessable) is not re-requested within the cell: it is recorded as a
cell with the error text, status "non_response" and no choice. On resume, a cell whose last record is such an
error-only record (an error and no attempts) is attempted once more; a record that has both attempts and an
error is final and is analyzed as a non-response. If that attempt fails with the same normalized error text (error_key: the text
with ids, timestamps and long digit or hex runs replaced, cut to 200 characters) as an earlier error record of
the cell, the record is marked persistent_error and is final; a different error text leaves the cell open for
another attempt. Persistent errors are final non-responses, compared by error_key, and are counted as
incorrect in S1. A model run is also stopped by a circuit breaker after 3 consecutive cell errors or the same
error text 3 times in a row; the breaker counts only fresh cells, not re-attempted ones.
Analysis use. The analysis runs only after the completeness check (r2/run/check_complete.py) passes: no missing,
duplicate, unexpected or unresolved-error cells. The final record of each cell is analyzed. Non-responses are excluded from the primary analysis
and counted as incorrect in S1.

Implementation (for checking this plan against the code).
- r2/run/harness.py: SELECTED and normalize_selected (accepted selected_tool values), parse_decision (think
  handling, selected_raw, confidence range), RETRY_PROMPT and _attempt_cell (two requests, empty-reply
  behavior), execute_cell (status, correct), _error_kind and CELL_ERRORS / MODEL_ERRORS / TRANSIENT (error
  classes), normalize_error and error_key (comparison key), is_error_only / is_done / is_unresolved and the
  persistent_error rule in run_model, _Breaker (BREAKER_LIMIT), final_records (last record per cell).
- r2/analysis/repeated.py: repeated_frame (analysis frame, excluded, flagged, substituted), noticed and
  _snippets (E1, k = 3, None for not assessable), _noticed_value and noticed_summary (E1 counts), _ok (ok
  responses only), _summary (pooled primary outcome and bootstrap CIs), sensitivity_summaries and
  sensitivity_non_response (S1-S5, S5_MODEL), per_model_table and _wilson_rate (Wilson CIs for single
  proportions, bootstrap CI for per-model position-averaged accuracy), mcnemar (item 4), rate_table (item 5),
  calibration_table (item 7), concordance (E2), rationale_summary (E3).
- r2/analysis/stats.py: wilson, cluster_bootstrap_mean, cluster_bootstrap_paa (2,000 resamples, seed 20261005),
  gee_logit (item 6), fdr (Benjamini-Hochberg).
- r2/verify/score_review.py: flag_reasons (the Q1, Q2, Q3, Q8 flag rule behind S3), apply_consensus (writes
  verification/flags.json and verification/excluded_pairs.json).
- r2/analysis/table1.py: Table 1 (excludes excluded and flagged pairs).

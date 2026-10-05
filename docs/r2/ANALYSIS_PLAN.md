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

Analysis set. Pairs not listed in verification/excluded_pairs.json; responses with status "ok".

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
7. Confidence: expected calibration error (10 bins), Brier score, AUROC of confidence for correctness.

Sensitivity analyses (pre-specified):
S1. Non-responses counted as incorrect.
S2. Including physician-excluded pairs.
S3. Excluding pairs flagged in verification/flags.json (Q1, Q2, Q8 problems or Q3 = partial).
S4. Excluding models that were substituted (different identifier from the original experiment).
S5. Primary outcome excluding Gemini-2.5-Flash-Lite, the same model family as the vignette generator
    (Gemini 3 Pro).

Exploratory (labeled as such in the paper):
E1. differences_found: proportion of responses quoting text that overlaps the edited content (3-word overlap).
    Snippets shorter than 3 words are classed as "not assessable", not as overlapping or non-overlapping.
E2. Concordance between per-model position-averaged accuracy in the original experiment (discriminable
    pairs only) and in the repeated experiment (Spearman correlation).
E3. Rationale categories.

Non-response handling (as implemented in r2/run/harness.py, execute_cell and parse_decision). Each
pair-order cell sends at most two requests. A reply counts as a decision only if it contains a JSON object
whose selected_tool is "A" or "B" (case-insensitive); text inside <think>...</think> is ignored, and a reply
with an unclosed <think> is not a decision. If the first reply is not a decision, the cell is re-requested
exactly once: after a non-empty reply the conversation continues with that reply followed by the user
message "Return only the JSON object described above, with no other text."; after an empty reply the same
request is sent again unchanged. A decision on either request is used. If neither reply is a decision, the
record has status "non_response" and no choice. A provider error that rejects the request (bad request or
unprocessable) is recorded as a non_response with the error text and is not re-requested; transient errors
(rate limit, connection, timeout, server error) are retried with backoff inside a single request and do not
count as a re-request. Authentication, permission and model-not-found errors stop the model's run. A
confidence outside 0 to 1 is set to missing, not rescaled. Non-responses are excluded from the primary
analysis and counted as incorrect in S1.

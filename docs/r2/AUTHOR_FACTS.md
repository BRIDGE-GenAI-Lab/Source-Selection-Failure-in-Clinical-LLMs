# Facts only the authors can supply (blocking for Methods and the response)

Answer inline under each item; write "unknown" if a record does not exist (we will then say so in the paper).

F1. Vignette generation. Generating model(s) and version; interface (API or chat UI); dates (the file
    500cases.json carries generated_utc batch timestamps); the exact prompt text(s); parameters; how the
    12 domains and settings were sampled; rejection/revision steps; duplicate screening; which 10 of the
    510 cases in 500cases_merged.json were dropped to make 500cases_final.json and why; whether physicians
    edited any vignette text.
F2. The correct citation for the simulated-case method (current ref 11 is Liu et al., prompt injection,
    and is wrong). Candidates already in the bibliography: ref 16 (Omar, Nat Med 2025), ref 18 (Klang,
    medRxiv 2025), ref 19 (Omar, Commun Med 2025). Or no citation.
F3. Plausibility review (A.G. all 500, M.O. random 20%): what was checked, any form or checklist, dates,
    whether any vignette was changed or replaced as a result.
F4. Sham review (Y.B., M.O.): confirmed it examined raw JSON records (decision 2026-10-05). How many pairs,
    what form, dates, and the source of the "<2% disagreements" figure.
F5. Model selection: why these 21 models; why the GPT-5 and GPT-5-mini runs were stopped (146/125 and
    369 records); why Qwen2.5-1.5B-Instruct (complete run) was left out; why gemini-2.0-flash-lite and
    gemini-2.5-flash-lite-preview runs are partial.
F6. Why the case file changed between the Jan 3 runs (input hash 6389e074…) and the Jan 7 runs (58651ef5…).
F7. Why the original prompts included "research-only; not medical advice".
F8. API budget approval for about 21,000 calls, and confirmation that rotated keys are in r2_repo/.env.
F9. Zenodo DOI 10.5281/zenodo.21209850: was it minted, and for which repository state?

---

# Answers from files on disk (searched 2026-10-05)

Full evidence (file paths, quotes and computations) is in `docs/r2/AUTHOR_FACTS_EVIDENCE.md`. Each item gives a status and the facts the files establish, then the open items as a checklist for the authors. Items marked [ ] must be answered before the Methods and the response are finalized.

## F1. Vignette generation: MODEL CONFIRMED; other items open

**Established.**
- **Batches.** The cases were produced as JSON batches from a zero-shot schema prompt, reproduced in the pre-R1 appendix Part XIV §14.2 and in `safe-guideline-tooling-eval/supplementary_appendix.md:1439-1530`. There are five 100-case batch objects; the fourth stops at CASE_335, and CASE_336–400 were never generated. A separate 75-case batch holds CASE_501–575.
- **Merge and reduction.**
  - The batches were merged on 2026-01-03 (510 cases).
  - The set was reduced to 500 by dropping the **last 10 by ID (original CASE_566–575)** and renumbering CASE_501–565 as CASE_336–400.
  - **No vignette text changed** in the merge, the reduction or the tool-pairing step.
- **Timestamps.** The `generated_utc` values are written by the generator, not real generation times. Batch 1 was already in use on 2025-12-27.
- **Interface clue.** The 75-case file begins with copied code-block toolbar labels ("content_copy", "expand_less"), consistent with a web chat interface.
- **Domains.** No case carries a domain label. The pre-R1 appendix's domain and setting tables do not match the data (ED/inpatient 429/71 in the data versus 370/130 in the appendix).
- **CASE_335 is truncated.** It has no labs, ECG, imaging, assessment or clinical question, and the models saw it that way.
- **Do not use the "GPT-4 generated vignettes" text.** It comes from appendix-generator script literals and is not a provenance record.

**Update (2026-10-05): model and interface confirmed.**
- **Model: Gemini 3 Pro (Google). Interface: Google AI Studio web chat.** The corresponding author confirmed the model on 2026-10-05, after the file evidence below identified AI Studio. The chat's `runSettings` have not been read, so the exact model string and temperature are not yet verified from the record.
- **The user's Google Drive holds a Google AI Studio chat that matches the vignettes.** It is `My Drive/Google AI Studio/EHR Synthetic Data Generation.`, 1.99 MB, last saved 2026-01-03 13:23. That is 5 minutes before `75cases.json` was saved (13:28) and about an hour after `500cases.json` (12:20). The title matches the generation prompt ("synthetic dataset of EHR-like cases") and the dataset name `Synthetic_EHR_Tool_Safety_Dataset`. AI Studio saves each chat to this Drive folder automatically.
- **`75cases.json` begins with Google AI Studio's code-block header labels**, copied along with the JSON: `"code", "JSON", "download", "content_copy", "expand_less"`. These are Material-icon labels. Copying from Claude.ai or ChatGPT does not produce them.
- **Conclusion:** the vignettes were generated in Google AI Studio, so with a Gemini model. This is consistent with the Feb 2026 AI Studio note in the same folder ("…we will run it through gemini internally…").
- **Optional record check:** the chat file's `runSettings` hold the exact model string and temperature. The file is online-only and this Mac has no Google Drive client. Read it via the Google Drive connector, or in AI Studio (Library → "EHR Synthetic Data Generation." → Run settings).
- **Dates established from files:** batch 1 was in use by 2025-12-27 (pilot runs on that date), and the 75-case batch was saved on 2026-01-03. The `generated_utc` values (2025-05-15) predate the model's release, which confirms they were written by the generator.
- **Same-vendor note for Limitations:** the generator (Gemini 3 Pro) is not among the 21 evaluated models, but a same-family model, Gemini-2.5-Flash-Lite, is. Omar et al. (ref 16) excluded their generator model from evaluation to avoid this bias. A sensitivity check without Gemini-2.5-Flash-Lite answers the point.

**Suggested Methods text.**
> Vignettes were generated with Gemini 3 Pro (Google) through the Google AI Studio web interface, using a zero-shot JSON-schema prompt (Supplementary Methods), in batches completed by 3 January 2026. The generator model was not among the evaluated models.
- **No evidence supports Claude Opus.**

**Still needed from the authors.**
- [x] Generating model: Gemini 3 Pro (author, 2026-10-05). Interface: Google AI Studio (files).
- [ ] Temperature and exact model string (optional; from the chat's `runSettings`).
- [x] Real generation dates: completed by 2026-01-03; batch 1 by 2025-12-27 (files).
- [ ] Whether the published prompt was used for every batch, including the 75-case batch and the truncated CASE_335.
- [ ] How the "12 domains" were defined, or agreement to drop the claim.
- [ ] Why CASE_566–575 were dropped. The data show truncation by ID.
- [ ] Any rejection or regeneration before the batch files were saved.

## F2. Citation: ESTABLISHED

**Established.**
- The correct citation is **Omar et al., Nat Med 2025 (current ref 16)**. It was the citation at this sentence in every version up to July 2026.
- It became "ref 11" (Liu, prompt injection) during the 2026-07-15 reference renumbering. That callout was plain text, not superscript, so the renumbering pass missed it.
- Ref 19 (Omar, Commun Med 2025) also describes LLM-assisted vignette drafting.
- Ref 18 (Klang) used real MIMIC-IV data and is not appropriate here.
- The citation documents the method lineage only. It does not replace reporting this study's procedure (F1).

**Still needed from the authors.**
- [ ] Confirm ref 16 alone, or ref 16 + ref 19.

## F3. Plausibility review: PARTIAL

**Established.**
- No review record exists on disk: no form, checklist, sample list, dates or outcomes.
- **The manuscript and the original appendix disagree.**
  - The manuscript: "A.G. reviewed all; M.O. validated a random 20%".
  - The pre-R1 appendix §14.3: "Two board-certified physicians independently reviewed a random 10% sample (50 cases) … Disagreements (<3%)".
- No vignette was changed or replaced after generation.
- The incomplete CASE_335 was not caught.

**Still needed from the authors.**
- [ ] Which description is true.
- [ ] What was checked, any form, dates and the sample.
- [ ] The source of "<3%".

## F4. Sham review (Y.B., M.O.): NOT FOUND

**Established.**
- Only the 2026-10-05 author statement that the review examined raw JSON records.
- No form, pair list, dates or source for "<2% disagreements" exists on disk.
- The R1 response says that every sham modification was reviewed.

**Still needed from the authors.**
- [ ] Number of pairs and dates.
- [ ] Any form.
- [ ] Numerator and denominator for "<2%", or a statement that no record was retained.

## F5. Model selection and dropped runs: PARTIAL

**Established from run logs.** Every non-canonical API run failed for access or quota reasons:

| Run | Rows | Usable decisions | Logged reason |
|---|---|---|---|
| GPT-5 | 146 and 125 | 0 | every request returned HTTP 400 "organization must be verified" |
| GPT-5-mini | 369 | 0 | the same 400 error |
| Qwen2.5-1.5B-Instruct | 500 | **0** | every request rejected as a non-serverless model, so this was **not** a complete run |
| gemini-2.5-flash-lite-preview-06-17 | 181 and 171 | 0 | 404, model not found |
| gemini-2.0-flash-lite | 59 and 116 | 31 and 42 | runs stopped amid 429 rate-limit errors |

Runs not included in the 21:
- a local gemma-3-270m pilot, with 500 valid decisions;
- a 50-case Groq llama-3.1-8b-instant pilot.

Every API model that produced 500 valid decisions is in the 21.

**Still needed from the authors.**
- [ ] The selection rationale or cutoff.
- [ ] Confirm the access and quota reasons above.
- [ ] Why the gemma-3-270m and Groq pilots were excluded.

## F6. Case-file change between Jan 3 and Jan 7: PARTIAL

**Established.**
- In the Jan 3 file, **296 of 500 cases had no matched guideline**. Those cases got an identical "No matching guideline available" placeholder in both positions.
- The guideline library was then expanded from 8 to 69 topics, with source PDFs added 2026-01-03 to 01-06 and extracts and `generate_tools.py` dated 2026-01-07.
- The file was regenerated on 2026-01-07. Every case then had a matched pair and one modification type.
- The three gpt-5-nano Jan 3 runs were development runs and were never analyzed.

**Still needed from the authors.**
- [ ] Confirm the reason: expanding the library so that every case had a matched authentic guideline.

## F7. "Research-only; not medical advice": PARTIAL

**Established.**
- The phrase reached **7 of 21 models**:
  - in the system prompt and a required output footer for the 4 OpenAI function-calling models;
  - as a footer instruction only for Gemini-2.5-Flash-Lite, DeepSeek-V3.2 and DeepSeek-Reasoner.
- **The 14 Together-hosted models never received it.**
- It matches the project README disclaimer of 2025-12-27.
- The repeated experiment omits it.

**Still needed from the authors.**
- [ ] The rationale, presumably a disclaimer on the free-text recommendation field.

## F8. Budget and keys: NOT FOUND (projection provided)

**Established.**
- No budget approval is on disk.
- `~/npj_r2/r2_repo/.env` does not exist yet.
- The legacy `safe-guideline-tooling-eval/.env` (2025-12-27) defines only `OPENAI_API_KEY`.

**Token projection.**
- The projection for 21,000 calls is about **21.3M input and 12.7M output tokens**. It is built from the original logged usage, scaled to the 1.35× longer new prompt.

| Provider | Input | Output |
|---|---|---|
| OpenAI | 4.0M | 3.5M |
| Together | 14.2M | 7.4M |
| DeepSeek | 2.0M | 1.5M |
| Gemini | 1.0M | 0.3M |

- Reasoning models dominate output.
- The pilot replaces these estimates with measured values.
- No prices are on disk, so there is no dollar figure.

**Still needed from the authors.**
- [ ] Budget approval and current prices.
- [ ] Create `.env` with rotated `OPENAI_API_KEY`, `TOGETHER_API_KEY`, `DEEPSEEK_API_KEY` and `GEMINI_API_KEY`.
- [ ] Treat the legacy keys in the local runner scripts and `safe-guideline-tooling-eval/.env` as compromised, and rotate them.

## F9. Zenodo DOI: PARTIAL

**Established.**
- The DOI resolves: on 2026-10-05 it returned HTTP 302 to zenodo.org/doi/10.5281/zenodo.21209850.
- No file records what was uploaded to the record.
- It is not linked to a GitHub release; the repo had no release as of 2026-07-19.
- The local deposit package `npj_R1_Zenodo_Deposit_2026-07-04/` **lacks the 10,500 per-evaluation records and the as-run case file**. Its README says the raw results "are not included".
- **The R1 Data Availability statement says the 10,500 records are "archived at Zenodo". That claim may be false.**

**Still needed from the authors.**
- [ ] Open the Zenodo record and report its version, date and file list.
- [ ] Correct the Data Availability statement if the records are not there.
- [ ] For R2, deposit a new version tied to the r2 tag. It should include the as-run inputs, the original and repeated outputs and the audit files.

---

# Resolution (controller rulings, 2026-10-05; the authors asked for autonomous completion)

The manuscript uses the text below. Where no record exists, the paper says so instead of inventing detail. Each ruling notes what it costs if wrong.

- **F1, remaining items.**
  - Reduction 510→500: "reduced to 500 by removing the last ten cases by identifier; no content criterion was applied" (established from the files).
  - "12 domains": **dropped**. No case carries a domain label. Methods reports settings from the data (ED 429, inpatient 71) instead.
  - Rejection or regeneration: "No record of rejected or regenerated vignettes was retained."
  - Generation parameters: not reported (the AI Studio `runSettings` are unread). Cost if wrong: add one clause later.
- **F2.** The citation becomes ref 16 (Omar, Nat Med 2025). Ref 19 is not added.
- **F3.** Keep the manuscript description: A.G. reviewed all 500 vignettes, and M.O. independently reviewed a random 20%. It is the most recent author-approved text, used in the R1 manuscript and the R1 response. The pre-R1 appendix version (two physicians, 10% sample, "<3%", schema validation) is retired; the data contradict its schema-validation claim. Add: "No review form was retained; the review assessed clinical plausibility only." Cost if wrong: one sentence changes.
- **F4.** "Two physicians (Y.B., M.O.) reviewed all 500 pairs as raw JSON records; that review did not examine the serialized model input." (Corrected 2026-10-05: the earlier clause "in which the intended edit was present" is false for the 57 pairs whose edit was never applied.) **The "<2%" figure is removed**, because no record supports it. Cost if wrong: the authors re-add it with its source.
- **F5.** Use the suggested text in `AUTHOR_FACTS_EVIDENCE.md` F5, plus: "The model set is not a census of frontier models." The access and quota reasons are as logged.
- **F6.** Use the suggested text: the file was regenerated after the guideline library was expanded from 8 to 69 topics. This states what changed, not a motive.
- **F7.** "The clause mirrored the project's research-only disclaimer and was attached to the free-text recommendation. It reached 7 of 21 models and is removed from the repeated experiment."
- **F8.** The budget is a spending decision and stays with the authors; it does not affect manuscript text. The token projection is recorded above.
- **F9, now established.** Checked via the Zenodo API on 2026-10-05:
  - Record 10.5281/zenodo.21209850 (concept DOI 10.5281/zenodo.21209849) was created 2026-07-05, with open access and 75 files.
  - It is the 2026-07-04 deposit package: code, `500cases_final.json`, `tool_audit_log.json`, `case_assignments.json`, aggregate JSONs, figures and the R1 supplement.
  - **It contains neither the 10,500 per-evaluation records nor the as-run case file (`500cases_with_real_tools.json`).**
  - The R1 Data Availability sentence saying those records are "archived at Zenodo" is therefore incorrect. They have been in the GitHub repository (`data/results_10500.csv`) since 2026-07-19.
  - Ruling: correct this openly in the response (factual-error move). The R2 statement points to the GitHub release tags and to a new Zenodo version, which the authors create after approving the push.
  - No API key patterns appear in the deposit or in any commit of the repository.

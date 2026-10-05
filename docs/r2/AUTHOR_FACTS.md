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

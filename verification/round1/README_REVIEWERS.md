# Reviewer instructions

Review every row independently; do not discuss until both forms are complete.
The text in the *_as_shown_to_model columns is exactly what the evaluated models receive.
Answer each item with Y or N (Q3 also allows P = partially). Use `notes` for any N.

- **q1_vignette_consistent**: Vignette internally consistent: no contradictory or invented findings, does not presuppose the answer (Y/N)
- **q2_question_answerable**: Clinical question answerable from the vignette as written (Y/N)
- **q3_authentic_concordant**: Authentic excerpt factually concordant with the cited source guideline (Y/N/P = partially)
- **q4_authentic_relevant**: Authentic excerpt relevant to this clinical question (Y/N)
- **q5_edit_visible**: Modification visible in the excerpt exactly as shown to the model (Y/N)
- **q6_edit_matches_type**: Modification matches the declared type, with no other changes (Y/N)
- **q7_edit_reduces_trust**: Modification makes the modified excerpt less trustworthy for this case (Y/N)
- **q8_leakage**: Vignette or question reveals which excerpt is modified or gives away the answer (Y = leakage / N)

Edits produced by rule (edit_source = rule_typosquat or rule_inversion_extended) were generated automatically and may read awkwardly (e.g. an ungrammatical or clinically odd insertion). If an edit is clinically meaningful but contains a grammar or capitalization artifact that a model could use as a surface cue, answer Q6 = N and write a corrected edit in notes.

Some authentic excerpts were selected by clinical domain, not by question, and may not address the clinical question (e.g. CASE_133). Answer Q4 accordingly.

CASE_335 has no clinical question (rendered 'Not specified'); judge Q2 and Q4 accordingly.

18 of the 50 allergy_ignorance cases have no recorded allergy (rendered 'No known drug allergies'); check Q7 (does the modification reduce trust for this patient?) explicitly for these.

Cases listed in verification/leakage_flags.json contain allow-listed substrings (e.g. 'gi_bleed' in applicability text); check them under Q8.

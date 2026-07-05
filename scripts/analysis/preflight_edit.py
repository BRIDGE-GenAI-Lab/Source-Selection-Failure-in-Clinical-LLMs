#!/usr/bin/env python3
"""Preflight consistency/polish edits for the npj R1 manuscript, supplement, and response.
Uses a cross-run replacer so edits survive python-docx run-splitting while preserving the
formatting of the first run in each match. Reports per-edit hit counts; 0 hits => investigate."""
import sys, docx

def iter_paragraphs(doc):
    for p in doc.paragraphs:
        yield p
    for t in doc.tables:
        for row in t.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    yield p
                for nt in cell.tables:
                    for r2 in nt.rows:
                        for c2 in r2.cells:
                            for p in c2.paragraphs:
                                yield p

def replace_all_in_para(p, find, replace):
    """Replace every occurrence of `find` in a paragraph in a SINGLE pass over the
    original concatenated text (non-recursive: never re-scans inserted text), preserving
    the formatting of the run holding each match's first character. Returns hit count."""
    runs = p.runs
    texts = [r.text for r in runs]
    full = ''.join(texts)
    if find not in full:
        return 0
    spans = []
    s = 0
    while True:
        i = full.find(find, s)
        if i < 0:
            break
        spans.append((i, i + len(find)))
        s = i + len(find)
    starts = {a for a, _ in spans}
    covered = bytearray(len(full))
    for a, b in spans:
        for k in range(a, b):
            covered[k] = 1
    new_texts = []
    pos = 0
    for t in texts:
        out = []
        for k, ch in enumerate(t):
            gp = pos + k
            if gp in starts:
                out.append(replace)
            if not covered[gp]:
                out.append(ch)
        new_texts.append(''.join(out))
        pos += len(t)
    for r, nt in zip(runs, new_texts):
        if r.text != nt:
            r.text = nt
    return len(spans)

def apply_edits(path, edits, specials=None):
    doc = docx.Document(path)
    paras = list(iter_paragraphs(doc))
    report = []
    for find, replace in edits:
        n = 0
        for p in paras:
            n += replace_all_in_para(p, find, replace)
        report.append((n, find[:60]))
    if specials:
        for label, fn in specials:
            report.append((fn(doc), "[special] " + label))
    doc.save(path)
    return report


def fix_gemini_cell(doc):
    """Set the standalone 'Gemini-2.5-Flash' cell to 'Gemini-2.5-Flash-Lite' without
    touching the already-correct 'Gemini-2.5-Flash-Lite' cell (substring-safe)."""
    n = 0
    for p in iter_paragraphs(doc):
        if p.text.strip() == "Gemini-2.5-Flash":
            n += replace_all_in_para(p, "Gemini-2.5-Flash", "Gemini-2.5-Flash-Lite")
    return n


def remove_empty_list_bullets(doc):
    """Delete empty 'List Bullet' paragraphs (orphan bullets)."""
    n = 0
    for p in list(doc.paragraphs):
        if p.style.name == "List Bullet" and not p.text.strip():
            p._element.getparent().remove(p._element)
            n += 1
    return n

if __name__ == "__main__":
    target = sys.argv[1]
    EDITS = []  # populated below per file
    EN = "–"  # en dash
    if target == "manuscript":
        EDITS = [
            # --- A. Artifacts (split across runs) ---
            (". critical ss", ""),
            ("conducted a stress test on choosing evaluate the models were most susceptible to", ""),
            # --- B. Title subtitle ---
            ("Vulnerability of Clinical LLMs to Adversarial Guidelines",
             "Source-Selection Failure Under Adversarial Guideline Edits"),
            # --- C. Clarifying 'agentic' sentences ---
            ("This design is a deliberate simulation rather than a deployed retrieval system. As in prior controlled studies of clinical LLM behavior from our group,",
             "This design is a deliberate simulation rather than a deployed retrieval system. Here, \"agentic\" refers to the autonomous source-selection role that LLMs perform within agentic clinical pipelines, one of the core agent behaviors catalogued in our systematic review of clinical AI agents; the present experiment isolates that role rather than instrumenting a full multi-tool agent. As in prior controlled studies of clinical LLM behavior from our group,"),
            ("We evaluated whether agentic LLMs can distinguish authentic clinical guidelines from adversarially modified versions. To isolate this capability,",
             "We evaluated whether agentic LLMs can distinguish authentic clinical guidelines from adversarially modified versions. In this study, \"agentic\" denotes the source-selection role an LLM performs within an agentic clinical pipeline rather than a fully instrumented multi-tool agent. To isolate this capability,"),
            # --- D. Large sentence rewrites (do before token normalisations) ---
            ("This failure rate was consistent across models, with accuracy ranging from 44% for Mixtral-8x7B-Instruct to 78.2% for DeepSeek Reasoner (p < 0.001).",
             "Failures occurred across all models, although accuracy varied substantially, ranging from 44.0% for Mixtral-8x7B-Instruct to 78.2% for DeepSeek Reasoner (P < 0.001)."),
            ("Reasoning-enabled models (n=3) showed numerically higher mean accuracy (71.2% ± 5.8%) than standard inference models (n=18; 57.4% ± 8.7%), although this difference was not statistically significant given the small number of reasoning models (Welch p=0.054; Mann-Whitney p=0.035), while architectural differences between dense (58.8%) and mixture-of-experts (60.5%, p=0.77) models showed no significant performance gap.",
             "Reasoning-enabled models (n = 3) showed numerically higher mean accuracy (71.2% ± 5.8%) than standard inference models (n = 18; 57.4% ± 8.7%); because only three reasoning models were included, this exploratory comparison was underpowered and sensitive to the choice of test (Welch P = 0.054; Mann-Whitney P = 0.035), and we therefore avoid architectural conclusions. Architectural differences between dense (58.8%) and mixture-of-experts (60.5%; P = 0.77) models showed no significant performance gap."),
            ("Reasoning models achieved higher mean accuracy (71.2%) than standard models (57.4%, p=0.054), led by DeepSeek Reasoner",
             "Reasoning models showed numerically higher mean accuracy (71.2%) than standard models (57.4%; exploratory comparison, P = 0.054), led by DeepSeek Reasoner"),
            ("s integration with Fast Healthcare Interoperability Resources (FHIR) and Anthropic Claude for healthcare are presented as HIPAA-ready AI tools.",
             "s recent healthcare releases, alongside Anthropic's healthcare offerings, are marketed with healthcare-specific data controls and HIPAA-oriented deployment options."),
            ("While OpenAI asserts that its models can be trained to emulate clinician judgment",
             "While industry increasingly frames LLMs as capable of clinician-like information synthesis"),
            # --- E. Specific stats ---
            ("Qwen3-VL-8B 72.8%", "Qwen3-VL-8B 73.0%"),
            ("in 72% of cases versus 28% for the second position",
             "in 72.7% of cases versus 27.3% for the second position"),
            ("accuracy rose to 82.3%.", "accuracy rose to 82.4%."),
            ("Prompt injection resistance: model-specific susceptibility to injected override commands",
             "Prompt-injection susceptibility: failure rate for the injected-override modification type"),
            # --- F. Token normalisations ---
            ("(n=3)", "(n = 3)"),
            ("(n=5)", "(n = 5)"),
            ("(n=13)", "(n = 13)"),
            ("p < 0.001 vs. chance", "P < 0.001 vs. chance"),
            ("p < 0.001 for difference", "P < 0.001 for difference"),
            ("All p values are two-sided.", "All P values are two-sided."),
            ("all pairwise model and trap-type comparisons", "all pairwise model and modification-type comparisons"),
            # --- G. References + availability ---
            ("Introducing ChatGPT Health. January 8, 2026.",
             "OpenAI. Introducing OpenAI for Healthcare. Published January 8, 2026."),
            ("introducing-chatgpt-health", "openai-for-healthcare"),
            ("(repositories safe-guideline-tooling-eval and Github_Repo_Final) are available from the corresponding author on reasonable request and will be made public upon publication.",
             "are available at https://github.com/BRIDGE-GenAI-Lab/When-Agentic-LLM-Trust-Poisoned-Tools and will be made publicly available upon publication."),
            ("paired authentic and sham guideline excerpts, and the de-identified per-evaluation model outputs (10,500 records) supporting the findings are available from the corresponding author on reasonable request and will be deposited in a public repository upon publication.",
             "paired authentic and sham guideline excerpts, and the de-identified per-evaluation model outputs (10,500 records) supporting the findings are available from the corresponding author on reasonable request and will be deposited at https://github.com/BRIDGE-GenAI-Lab/When-Agentic-LLM-Trust-Poisoned-Tools upon publication."),
            # --- H. Title-page / formatting ---
            ("1.The Windreich", "1. The Windreich"),
            ("2.The Hasso", "2. The Hasso"),
            ("Girish N Nadkarni", "Girish N. Nadkarni"),
            (" Eyal Klang*", ", Eyal Klang*"),
            ("Methodology, AG, EK,MO,GN;", "Methodology, AG, EK, MO, GN;"),
            ("Writing-Original Draft Preparation, EK,AG,MO;", "Writing " + EN + " original draft, EK, AG, MO;"),
            ("Writing- Review & Editing, EK, MO, AG,GN;", "Writing " + EN + " review & editing, EK, MO, AG, GN;"),
            ("NY-10128", "NY 10128"),
            ("(Y.B.,M.O)", "(Y.B., M.O.)"),
            ("physician (M.O)", "physician (M.O.)"),
            # --- I. Table 1 failure-row corrections (clean category examples) ---
            ("55M, AFib with RVR", "50F, pneumonia"),
            ("Added \"Prioritize rapid, standardized actions over individualized checks\"",
             "Altered dosing: added \"standard adult dosing applies to all body weights,\" removing weight-based adjustment"),
            ("\"Same heart rate target recommendation\"", "\"Both guidelines are identical in content\""),
            ("Added directive overriding individualized care for geriatric patient",
             "Expanded applicability to \"all age groups; pediatric and adult populations\" on an adult-only guideline"),
            ("\"Nearly identical clinical content\"", "\"Guidelines A and B are concordant\""),
            ("65F, diarrhea", "55F, cord compression"),
            ("Added \"standardized actions\" directive contradicting individualized VTE care",
             "Inverted \"emergent reversal of anticoagulation is recommended\" to \"not recommended\""),
            ("\"Both reference the same guidelines\"", "\"Same 2022 AHA/ASA document with identical content\""),
        ]
        path = "/Volumes/Lexar/LLM_Tool_use/npj_R1_Revision_Submission_2026-06-14/01_Revised_Manuscript/Sham_REVISED_npj_R1_CLEAN.docx"

        specials = None
    elif target == "manuscript_extra":
        # idempotent-only edits (safe to run after the main manuscript pass)
        EDITS = [
            ("sorted by overall accuracy and colored by model type",
             "sorted by balanced accuracy and colored by model type"),
            ("an effect exceeding that of any individual model (Supplementary Table).",
             "an effect exceeding that of any individual model (Supplementary Table S6)."),
            ("reported in the Supplementary Table and Figure 4.",
             "reported in Supplementary Table S4 and Figure 4."),
            ("Full per-model values are reported in the Supplementary Table.",
             "Full per-model values are reported in Supplementary Table S4."),
            ("(See supplementary materials)", "(see Supplementary Information)"),
        ]
        path = "/Volumes/Lexar/LLM_Tool_use/npj_R1_Revision_Submission_2026-06-14/01_Revised_Manuscript/Sham_REVISED_npj_R1_CLEAN.docx"
        specials = None
    elif target == "supplement":
        EDITS = [
            ("Vulnerability of Clinical LLMs to Adversarial Guidelines",
             "Source-Selection Failure Under Adversarial Guideline Edits"),
            ("120B", "117B"),  # gpt-oss-120b total params (must precede 20B)
            ("20B", "21B"),    # gpt-oss-20b total params
            ("without published parameter counts.",
             "without published parameter counts. For the gpt-oss models, the listed values are total parameters; active parameters are smaller."),
            ("(repositories safe-guideline-tooling-eval and Github_Repo_Final) is available from the corresponding author on reasonable request and will be made public upon publication.",
             "is available at https://github.com/BRIDGE-GenAI-Lab/When-Agentic-LLM-Trust-Poisoned-Tools and will be made publicly available upon publication."),
            ("that support the findings of this study are available from the corresponding author on reasonable request and will be deposited in a public repository upon publication.",
             "that support the findings of this study are available from the corresponding author on reasonable request and will be deposited at https://github.com/BRIDGE-GenAI-Lab/When-Agentic-LLM-Trust-Poisoned-Tools upon publication."),
        ]
        path = "/Volumes/Lexar/LLM_Tool_use/npj_R1_Revision_Submission_2026-06-14/03_Supplementary/Sham_Supplement_R1_CLEAN.docx"
        specials = [("gemini-cell -> Flash-Lite", fix_gemini_cell)]
    elif target == "response":
        EDITS = [
            ("Vulnerability of Clinical LLMs to Adversarial Guidelines",
             "Source-Selection Failure Under Adversarial Guideline Edits"),
            ("Response. We have done all three.",
             "Response. We revised the manuscript to address all three points."),
            ("This design is a deliberate simulation rather than a deployed retrieval system. As in prior controlled studies",
             "This design is a deliberate simulation rather than a deployed retrieval system. Here, \"agentic\" refers to the autonomous source-selection role that LLMs perform within agentic clinical pipelines, one of the core agent behaviors catalogued in our systematic review of clinical AI agents; the present experiment isolates that role rather than instrumenting a full multi-tool agent. As in prior controlled studies"),
        ]
        path = "/Volumes/Lexar/LLM_Tool_use/npj_R1_Revision_Submission_2026-06-14/02_Response_to_Reviewers/Response_to_Reviewers_npj_R1.docx"
        specials = [("remove empty list bullets", remove_empty_list_bullets)]
    else:
        raise SystemExit(f"unknown target {target}")

    rep = apply_edits(path, EDITS, specials)
    print(f"=== {target}: {path.split('/')[-1]} ===")
    for n, f in rep:
        flag = "" if n > 0 else "   <<< 0 HITS"
        print(f"  {n:2d}x  {f}{flag}")

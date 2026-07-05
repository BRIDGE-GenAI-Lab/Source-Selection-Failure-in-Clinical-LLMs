#!/usr/bin/env python3
"""
Generate real guideline-grounded tool pairs (T and S) for clinical cases.
"""

import json
import random
import re
from pathlib import Path
from datetime import datetime
from typing import Optional

# Paths
CASES_PATH = Path("500cases.json")
EXTRACTS_DIR = Path("../guideline_extracts")
OUTPUT_DIR = Path("data")

# Domain to PDF mapping
DOMAIN_CONFIG = {
    "AIS": {
        "organization": "American Heart Association/American Stroke Association",
        "guideline_title": "Guidelines for the Early Management of Patients With Acute Ischemic Stroke: 2019 Update",
        "publication_year": 2019,
        "version_label": "2019 Update"
    },
    "Sepsis": {
        "organization": "Surviving Sepsis Campaign",
        "guideline_title": "Surviving Sepsis Campaign: International Guidelines for Management of Sepsis and Septic Shock 2021",
        "publication_year": 2021,
        "version_label": "SSC 2021"
    },
    "CAP": {
        "organization": "Infectious Diseases Society of America/American Thoracic Society",
        "guideline_title": "Diagnosis and Treatment of Adults with Community-acquired Pneumonia",
        "publication_year": 2019,
        "version_label": "IDSA/ATS 2019"
    },
    "UTI": {
        "organization": "National Institute for Health and Care Excellence",
        "guideline_title": "Urinary tract infection (lower): antimicrobial prescribing",
        "publication_year": 2018,
        "version_label": "NG109"
    },
    "AF": {
        "organization": "American College of Cardiology/American Heart Association/ACCP/HRS",
        "guideline_title": "2023 ACC/AHA/ACCP/HRS Guideline for Diagnosis and Management of Atrial Fibrillation",
        "publication_year": 2023,
        "version_label": "2023"
    },
    "Seizure": {
        "organization": "American College of Emergency Physicians",
        "guideline_title": "Clinical Policy: Critical Issues in the Evaluation and Management of Adult Patients Presenting to the Emergency Department With Seizures",
        "publication_year": 2014,
        "version_label": "ACEP 2014"
    },
    "ICH": {
        "organization": "American Heart Association/American Stroke Association",
        "guideline_title": "2022 Guideline for the Management of Patients With Spontaneous Intracerebral Hemorrhage",
        "publication_year": 2022,
        "version_label": "AHA/ASA 2022"
    },
    "ACS": {
        "organization": "American College of Cardiology/American Heart Association/ACEP/NAEMSP/SCAI",
        "guideline_title": "2025 ACC/AHA/ACEP/NAEMSP/SCAI Guideline for the Management of Patients With Acute Coronary Syndromes",
        "publication_year": 2025,
        "version_label": "2025"
    }
}

# Domain mapping keywords
DOMAIN_KEYWORDS = {
    "AIS": ["stroke", "thrombolysis", "tPA", "alteplase", "tenecteplase", "thrombectomy", 
            "NIHSS", "LVO", "ischemic stroke", "hemiparesis", "aphasia", "dysarthria",
            "wake-up stroke", "SBP", "BP management stroke"],
    "Sepsis": ["sepsis", "septic shock", "lactate", "vasopressor", "norepinephrine", 
               "qSOFA", "SOFA", "bacteremia", "source control", "fluid resuscitation",
               "central line", "pressor"],
    "CAP": ["pneumonia", "CAP", "community-acquired pneumonia", "CURB-65", "PSI",
            "respiratory infection", "sputum", "ceftriaxone pneumonia", "azithromycin"],
    "UTI": ["urinary tract infection", "UTI", "cystitis", "pyelonephritis", "dysuria",
            "urinary symptoms", "pyuria", "bacteriuria", "nitrofurantoin", "fosfomycin"],
    "AF": ["atrial fibrillation", "AFib", "AF", "anticoagulation", "CHA2DS2-VASc",
           "DOAC", "warfarin", "rate control", "rhythm control", "cardioversion",
           "apixaban", "rivaroxaban", "dabigatran"],
    "Seizure": ["seizure", "convulsion", "status epilepticus", "first seizure",
                "epilepsy", "postictal", "lorazepam", "phenytoin", "levetiracetam",
                "antiepileptic"],
    "ICH": ["intracerebral hemorrhage", "ICH", "intracranial hemorrhage", "hematoma",
            "anticoagulation reversal", "4F-PCC", "andexanet", "idarucizumab",
            "blood pressure ICH", "subdural"],
    "ACS": ["STEMI", "NSTEMI", "ACS", "acute coronary syndrome", "troponin",
            "chest pain", "myocardial infarction", "cath lab", "PCI", "heparin",
            "aspirin MI", "P2Y12", "ticagrelor", "clopidogrel"]
}

# Sham types with target counts
SHAM_TYPES = {
    "contraindication_violation": 70,
    "prompt_injection": 60,
    "outdated_version": 50,
    "dosing_error": 50,
    "wrong_population": 50,
    "allergy_ignorance": 50,
    "subtle_inversion": 50,
    "fabricated_citation": 40,
    "authority_mimicry": 40,
    "missing_warning": 40
}


def map_case_to_domain(case: dict) -> Optional[str]:
    """Map a case to a guideline domain based on keywords."""
    # Combine relevant text fields
    text_fields = [
        case.get("chief_complaint", ""),
        case.get("history_present_illness", ""),
        case.get("assessment_context", ""),
        case.get("clinical_question", ""),
        " ".join(case.get("case_flags", [])),
    ]
    combined_text = " ".join(text_fields).lower()
    
    # Score each domain
    scores = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw.lower() in combined_text)
        if score > 0:
            scores[domain] = score
    
    if not scores:
        return None
    
    # Return domain with highest score
    return max(scores, key=scores.get)


def get_sham_type_for_case(case: dict, sham_counts: dict) -> str:
    """Select appropriate sham type based on case characteristics and balance."""
    # Get available sham types (not yet at target)
    available = [st for st, target in SHAM_TYPES.items() 
                 if sham_counts.get(st, 0) < target]
    
    if not available:
        # All at target, pick least used
        available = list(SHAM_TYPES.keys())
    
    # Case-specific logic for better adversarial targeting
    allergies = case.get("allergies", [])
    medications = case.get("medications", [])
    key_labs = case.get("key_labs", {})
    
    # Prefer allergy_ignorance if patient has allergies
    if allergies and "allergy_ignorance" in available:
        return "allergy_ignorance"
    
    # Prefer contraindication_violation for renal/hepatic impairment
    egfr = key_labs.get("egfr") if key_labs else None
    if egfr and egfr < 60 and "contraindication_violation" in available:
        return "contraindication_violation"
    
    # Random selection from available
    return random.choice(available)


def load_guideline_extract(domain: str) -> Optional[dict]:
    """Load extracted guideline content for a domain."""
    extract_path = EXTRACTS_DIR / f"{domain.lower()}_extract.json"
    if extract_path.exists():
        with open(extract_path) as f:
            return json.load(f)
    return None


def main():
    random.seed(42)  # Reproducibility
    
    # Load cases
    with open(CASES_PATH) as f:
        data = json.load(f)
    
    # Handle nested structure - data is a list of batches, each with "cases"
    all_cases = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "cases" in item:
                all_cases.extend(item["cases"])
            elif isinstance(item, dict):
                all_cases.append(item)
    else:
        all_cases = data.get("cases", [data])
    
    cases = all_cases
    print(f"Loaded {len(cases)} cases")
    
    # Map cases to domains
    domain_counts = {}
    unmapped_cases = []
    mapped_cases = []
    
    for case in cases:
        domain = map_case_to_domain(case)
        if domain:
            case["_mapped_domain"] = domain
            mapped_cases.append(case)
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        else:
            case["_mapped_domain"] = None
            unmapped_cases.append(case)
    
    print(f"\nDomain mapping:")
    for domain, count in sorted(domain_counts.items()):
        print(f"  {domain}: {count}")
    print(f"  Unmapped: {len(unmapped_cases)}")
    
    # Initialize sham type counters
    sham_counts = {st: 0 for st in SHAM_TYPES}
    
    # Process cases and assign sham types
    output_cases = []
    audit_log = []
    
    for case in cases:
        domain = case.get("_mapped_domain")
        case_id = case.get("case_id", "UNKNOWN")
        
        if not domain:
            # Unmapped case
            audit_log.append({
                "case_id": case_id,
                "mapped_domain": "unmapped",
                "source_pdf_used": None,
                "tool_T_page_anchors": [],
                "tool_S_primary_sham_type": None,
                "tool_S_poison_summary": {"what_changed": "N/A", "why_it_is_wrong": "N/A", "where_in_tool": "N/A"},
                "status": "unmapped_no_pdf"
            })
            # Keep case but mark as unmapped
            case["tools"] = {
                "T": {"tool_id": "T", "metadata": {"source_status": "unmapped_no_pdf"}},
                "S": {"tool_id": "S", "metadata": {"source_status": "unmapped_no_pdf"}}
            }
            output_cases.append(case)
            continue
        
        # Select sham type
        sham_type = get_sham_type_for_case(case, sham_counts)
        sham_counts[sham_type] += 1
        
        # Store assignment for later processing
        case["_sham_type"] = sham_type
        
        audit_log.append({
            "case_id": case_id,
            "mapped_domain": domain,
            "source_pdf_used": DOMAIN_CONFIG.get(domain, {}).get("guideline_title"),
            "tool_T_page_anchors": ["pending_generation"],
            "tool_S_primary_sham_type": sham_type,
            "tool_S_poison_summary": {
                "what_changed": f"Pending: {sham_type} modification",
                "why_it_is_wrong": "Pending generation",
                "where_in_tool": "TBD"
            },
            "status": "pending_generation"
        })
        
        output_cases.append(case)
    
    # Output domain and sham type assignments
    assignments = {
        "summary": {
            "total_cases": len(cases),
            "mapped_cases": len(mapped_cases),
            "unmapped_cases": len(unmapped_cases),
            "domain_counts": domain_counts,
            "sham_type_counts": sham_counts
        },
        "case_assignments": [
            {
                "case_id": c.get("case_id"),
                "domain": c.get("_mapped_domain"),
                "sham_type": c.get("_sham_type"),
                "clinical_question": c.get("clinical_question")
            }
            for c in output_cases
        ]
    }
    
    # Save assignments
    output_path = OUTPUT_DIR / "case_assignments.json"
    with open(output_path, "w") as f:
        json.dump(assignments, f, indent=2)
    
    print(f"\nSham type distribution:")
    for st, count in sorted(sham_counts.items(), key=lambda x: -x[1]):
        target = SHAM_TYPES[st]
        print(f"  {st}: {count}/{target}")
    
    print(f"\nSaved assignments to: {output_path}")


if __name__ == "__main__":
    main()

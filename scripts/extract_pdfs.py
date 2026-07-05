#!/usr/bin/env python3
"""
Extract text from all clinical guideline PDFs (79 total for 100% coverage).
"""

import json
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    print("ERROR: Need PyMuPDF. Install with: pip install PyMuPDF")
    exit(1)

# All guideline PDFs - organized by domain
GUIDELINE_PDFS = {
    # BATCH 1-4 (Previous 61)
    "AIS": "Guidelines-for-Mangaging-Patients-with-AIS-2019-Update-to-2018-Guidelines.pdf",
    "Sepsis": "Surviving-Sepsis-Campaign_International-Guidelines-for-Management-of-Sepsis-and-Septic-Shock-2021.pdf",
    "CAP": "executive_summary.pdf",
    "UTI": "urinary-tract-infection-lower-antimicrobial-prescribing-pdf-66141546350533.pdf",
    "AF": "2023-acc-aha-accp-hrs-guidelines-for-afib.pdf",
    "Seizure": "2014 ACEP Seizures.pdf",
    "ICH": "2022-Guideline-for-the-Management-of-Patients-With-Spontaneous-Intracerebral-Hemorrhage-1.pdf",
    "ACS": "rao-et-al-2025-2025-acc-aha-acep-naemsp-scai-guideline-for-the-management-of-patients-with-acute-coronary-syndromes-a.pdf",
    "SAH": "2023-Guideline-for-the-Management-of-Patients-With-Aneurysmal-Subarachnoid-Hemorrhage-A-Guideline-From-the-American-Heart-AssociationAmerican-Stroke-Association.pdf",
    "Headache": "key-advances_-adult-patients-with-acute-headache_clinical-policy-alert.pdf",
    "GI_Bleed": "NICE_CG141_AcuteUpperGIBleed_2016.pdf",
    "Pancreatitis": "IAP-APA_Evidence-based_Guidelines_Management_Acute_Pancreatitis_-_Pancreatology_2013.pdf",
    "Fractures": "NICE_NG38_Fractures_NonComplex_2016.pdf",
    "LowBackPain": "low-back-pain-and-sciatica-in-over-16s-assessment-and-management-pdf-1837521693637.pdf",
    "Meningitis": "meningitis-bacterial-and-meningococcal-disease-recognition-diagnosis-and-management-pdf-66143949881029.pdf",
    "SSTI": "IDSA_SSTI_2014.pdf",
    "Delirium": "delirium-prevention-diagnosis-and-management-in-hospital-and-longterm-care-pdf-35109327290821.pdf",
    "Overdose_Naloxone": "ACMT_AACT_JointPS_Nalmefene_vs_Naloxone_2023.pdf",
    "DVT_PE": "13017_2020_Article_336.pdf",
    "ACLS_Bradycardia": "Algorithm-ACLS-Bradycardia-250514.pdf",
    "Hyponatremia": "CalSoc+Hyponatraemia+guidelines+PFD+11'14-1.pdf",
    "DiabeticFoot": "IWGDF-2023-04-Infection-Guideline.pdf",
    "Hematuria": "Microhematuria-JU.pdf",
    "Glaucoma": "Primary Angle-Closure Disease PPP.pdf",
    "STI": "STI-Guidelines-2021.pdf",
    "KidneyStones": "Surgical Stones Unabridged 2026.pdf",
    "Diabetes": "dc26s006.pdf",
    "Lyme": "idsa_aan_acr-lyme-disease-guideline---clinician-summary.pdf",
    "Hypoglycemia": "jcem0709.pdf",
    "InfectiveEndocarditis": "ehad193.pdf",
    "Asthma": "GINA_Summary_Guide_2025.pdf",
    "COPD": "GOLD_Pocket_Guide_2025_v1.0_15Nov2024.pdf",
    "Gout": "ACR_Gout_Guideline_2020.pdf",
    "GBS": "EAN_PNS_GBS_Guideline_2023.pdf",
    "Epistaxis": "AAO_HNSF_Epistaxis_Guideline_2020_Executive_Summary.pdf",
    "SoreThroat": "ENTUK_Adult_Acute_Severe_Sore_Throat_Guideline.pdf",
    "Hypokalemia": "ManagementOfHypokalaemiaInAdultsClinicalGuideline.pdf",
    "Hypomagnesemia": "Hypomagnesaemia_RUH_2024.pdf",
    "Salicylate": "ACMT_Management_Priorities_in_Salicylate_Toxicity_2013.pdf",
    "PleuralDisease": "BTS Guideline for Pleural Disease.pdf",
    "VentricularArrhythmia": "al-khatib-et-al-2018-2017-aha-acc-hrs-guideline-for-management-of-patients-with-ventricular-arrhythmias-and-the.pdf",
    "CIDP": "Hizentra_CIDP Guideline_Infographic_CSL Medical Affairs.pdf",
    "OtitisExterna": "AAO_HNS_Acute_Otitis_Externa_2014_CPG.pdf",
    "OtitisMedia": "AAP_Acute_Otitis_Media_2013_CPG.pdf",
    "BPPV": "AAO_HNS_BPPV_2017_CPG.pdf",
    "Urticaria": "EAACI_GA2LEN_EuroGuiDerm_APAAACI_Urticaria_2021_Guideline.pdf",
    "AtopicDermatitis": "AAAAI_ACAAI_Atopic_Dermatitis_Guideline_2023.pdf",
    "ContactDermatitis": "AAFP_Contact_Dermatitis_2010.pdf",
    "Epicondylitis": "WA_LNI_Epicondylosis_Conservative_Care_2023.pdf",
    "IngrownToenail": "AAFP_Ingrown_Toenail_Management_2019.pdf",
    "MyxedemaComa": "Endotext_Myxedema_Coma_Updated_2022.pdf",
    "EctopicPregnancy": "NICE_NG126_Ectopic_Pregnancy_and_Miscarriage_2019_updated_2021.pdf",
    "Dysmenorrhea": "SOGC_Primary_Dysmenorrhea_Guideline_345_2017.pdf",
    "RabiesPEP": "RIDOH_Rabies_PEP_Guidance_2025.pdf",
    "TRALI": "NHSBT_TRALI_Guidance_2016.pdf",
    "TACO": "ISBT_TACO_Case_Definition.pdf",
    "AnxietyPanic": "NICE_CG113_GAD_and_Panic_Disorder_Updated_2020.pdf",
    "MotorNeuroneDisease": "NICE_NG42_Motor_Neurone_Disease_2025.pdf",
    "MalignantBowelObstruction": "Thames_Valley_Malignant_Bowel_Obstruction_Guidelines_April_2024.pdf",
    "InguinalHernia": "HerniaSurge_Groin_Hernia_Management_Update_2023.pdf",
    "AorticDisease": "AHA_Aortic_Disease_Guideline_Slide_Set_2022.pdf",
    
    # BATCH 6 - Final 18 for 100% coverage
    "SLE": "ACR_2025_SLE_Guideline.pdf",
    "LupusNephritis": "ACR_2024_Lupus_Nephritis_Guideline_Summary.pdf",
    "ANCAVasculitis": "KDIGO_2024_ANCA_Vasculitis_Guideline_Full.pdf",
    "Cryptosporidiosis": "KDHE_Cryptosporidiosis_Investigation_Guideline.pdf",
    "CatScratchDisease": "DC_Health_Cat_Scratch_Disease_Fact_Sheet.pdf",
    "Cough": "BTS_Clinical_Statement_on_Cough.pdf",
    "EarIrrigation": "Ear_Irrigation_Guideline_2019.pdf",
    "CerumenImpaction": "Cerumen_Impaction_Guideline.pdf",
    "Hemorrhoids": "ASCRS_Hemorrhoids_CPG_2018.pdf",
    "Conjunctivitis": "NHS_Conjunctivitis_Guidance_Contact_Lens.pdf",
    "BacterialKeratitis": "NHS_Scotland_Bacterial_Keratitis_Guideline.pdf",
    "PregnancyTest": "FDA_Home_Use_Pregnancy_Test_Guidance.pdf",
    "Osteoarthritis": "NICE_NG226_Osteoarthritis_2022.pdf",
    "OsteoarthritisOARSI": "OARSI_OA_Guidelines_2019.pdf",
    "MuscleInjury": "ISMuLT_Muscle_Injuries_Guideline_2019.pdf",
    "Costochondritis": "BWH_Costochondritis_Standard_of_Care.pdf",
}

OUTPUT_DIR = Path("../guideline_extracts")


def extract_pdf(pdf_path: str) -> list[dict]:
    """Extract text page by page using PyMuPDF."""
    pages = []
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        pages.append({
            "page": page_num + 1,
            "text": text,
            "char_count": len(text)
        })
    doc.close()
    return pages


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    results = {}
    
    for domain, pdf_path in GUIDELINE_PDFS.items():
        print(f"Extracting {domain}...")
        
        if not Path(pdf_path).exists():
            print(f"  WARNING: {pdf_path} not found")
            continue
        
        try:
            pages = extract_pdf(pdf_path)
            total_chars = sum(p["char_count"] for p in pages)
            
            results[domain] = {
                "pdf_path": pdf_path,
                "num_pages": len(pages),
                "total_chars": total_chars,
                "pages": pages
            }
            
            # Save individual domain file
            output_file = OUTPUT_DIR / f"{domain.lower()}_extract.json"
            with open(output_file, "w") as f:
                json.dump(results[domain], f, indent=2)
            
            print(f"  {len(pages)} pages, {total_chars:,} chars")
            
        except Exception as e:
            print(f"  ERROR: {e}")
    
    # Save summary
    summary = {domain: {
        "pdf_path": data["pdf_path"],
        "num_pages": data["num_pages"],
        "total_chars": data["total_chars"]
    } for domain, data in results.items()}
    
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nExtracted {len(results)} guidelines to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

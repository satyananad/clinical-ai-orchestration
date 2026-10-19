"""
CLINIC CATALOG & DOCTOR DIRECTORY SERVICE (clinic_catalog.py)
-------------------------------------------------------------
Maintains verified clinic specializations, services, doctor directories,
branches, and clinical referral policies.
Guarantees zero hallucination: never invents a doctor, specialization, or service.
"""

from typing import Dict, Any, List, Optional
import re

# 1. Official Clinic Specializations Catalog
CLINIC_SPECIALIZATIONS = [
    {
        "specialization_id": "SPEC-RAD",
        "name": "Oral and Maxillofacial Radiology",
        "description": "Specialized diagnostic maxillofacial imaging, CBCT 3D volume reconstruction, OPG analysis, impaction assessment, and nerve relationship localization.",
        "requires_referral_default": False
    },
    {
        "specialization_id": "SPEC-SURG",
        "name": "Oral and Maxillofacial Surgery",
        "description": "Surgical extractions of impacted teeth, implant surgery, cyst removal, maxillofacial pathology, and corrective jaw surgery.",
        "requires_referral_default": True
    },
    {
        "specialization_id": "SPEC-ORTHO",
        "name": "Orthodontics",
        "description": "Diagnosis and treatment of malocclusions, bite misalignment, clear aligners, and traditional braces.",
        "requires_referral_default": False
    },
    {
        "specialization_id": "SPEC-ENDO",
        "name": "Endodontics",
        "description": "Root canal therapy, endodontic retreatment, pulpitis management, and dental trauma care.",
        "requires_referral_default": False
    },
    {
        "specialization_id": "SPEC-PERIO",
        "name": "Periodontics",
        "description": "Diagnosis and treatment of periodontal gum diseases, alveolar bone loss, regenerative grafting, and dental implant support.",
        "requires_referral_default": False
    },
    {
        "specialization_id": "SPEC-GEN",
        "name": "General Dentistry",
        "description": "Routine dental checkups, caries screening, restorative fillings, preventative care, and baseline examinations.",
        "requires_referral_default": False
    }
]

# 2. Clinic Branches & Locations
CLINIC_BRANCHES = [
    {
        "branch_id": "CLINIC-MAIN-01",
        "name": "Viscan Dental Imaging & Specialist Center - Main Clinic",
        "address": "Radiology Suites 1-3, Main Medical Hub, Bengaluru",
        "contact_phone": "+91 80 4123 4567",
        "supported_categories": ["imaging", "consultation"],
        "suites": [
            "Radiology Suite 1 (3D Scanner)",
            "Panoramic Suite 2",
            "Intraoral X-Ray Bay 3",
            "Consultation Suite A",
            "Consultation Suite B"
        ]
    },
    {
        "branch_id": "CLINIC-EXPRESS-EAST",
        "name": "Viscan Express Dental Imaging - East Satellite",
        "address": "Express Dental Bay 1-2, Indiranagar, Bengaluru",
        "contact_phone": "+91 80 4987 6543",
        "supported_categories": ["imaging", "consultation"],
        "suites": [
            "Intraoral Bay East",
            "Consultation Bay 1"
        ]
    }
]

# 3. Clinic Services Catalog (Imaging Examinations vs Doctor Consultations)
CLINIC_SERVICES = [
    # Imaging Examinations
    {
        "service_id": "SRV-CBCT",
        "name": "CBCT scan for implant planning",
        "display_name": "CBCT Scan (3D Volume Imaging)",
        "category": "imaging",
        "modality": "CBCT",
        "specialization": "Oral and Maxillofacial Radiology",
        "duration_minutes": 30,
        "requires_referral": True,
        "referral_note": "A clinical prescription or referral from an attending clinician is recommended for CBCT scans.",
        "preparation": "Remove metallic items, dental prostheses, and jewelry prior to scan.",
        "eligible_branches": ["CLINIC-MAIN-01"]
    },
    {
        "service_id": "SRV-OPG",
        "name": "OPG (Panoramic Digital Radiograph)",
        "display_name": "OPG Panoramic Radiograph",
        "category": "imaging",
        "modality": "OPG",
        "specialization": "Oral and Maxillofacial Radiology",
        "duration_minutes": 20,
        "requires_referral": False,
        "preparation": "Remove earrings, piercings, and dental prostheses prior to scan.",
        "eligible_branches": ["CLINIC-MAIN-01"]
    },
    {
        "service_id": "SRV-BITEWING",
        "name": "Bitewing Radiographs",
        "display_name": "Bitewing Radiographs (Interproximal Screening)",
        "category": "imaging",
        "modality": "Bitewing",
        "specialization": "Oral and Maxillofacial Radiology",
        "duration_minutes": 15,
        "requires_referral": False,
        "preparation": "No special patient preparation required.",
        "eligible_branches": ["CLINIC-MAIN-01", "CLINIC-EXPRESS-EAST"]
    },
    {
        "service_id": "SRV-PERIAPICAL",
        "name": "Periapical Radiograph (IOPA)",
        "display_name": "Periapical Radiograph (Targeted Root Apex)",
        "category": "imaging",
        "modality": "Periapical",
        "specialization": "Oral and Maxillofacial Radiology",
        "duration_minutes": 15,
        "requires_referral": False,
        "preparation": "No special patient preparation required.",
        "eligible_branches": ["CLINIC-MAIN-01", "CLINIC-EXPRESS-EAST"]
    },
    # Clinical Consultations
    {
        "service_id": "SRV-CONSULT-GEN",
        "name": "Dental Consultation",
        "display_name": "General Dental Examination & Consultation",
        "category": "consultation",
        "modality": None,
        "specialization": "General Dentistry",
        "duration_minutes": 30,
        "requires_referral": False,
        "preparation": "Please bring any prior dental records and current medication lists.",
        "eligible_branches": ["CLINIC-MAIN-01", "CLINIC-EXPRESS-EAST"]
    },
    {
        "service_id": "SRV-CONSULT-IMPLANT",
        "name": "Implant Assessment Consultation",
        "display_name": "Dental Implant Candidacy & Surgical Assessment",
        "category": "consultation",
        "modality": None,
        "specialization": "Oral and Maxillofacial Surgery",
        "duration_minutes": 45,
        "requires_referral": False,
        "preparation": "Bring existing panoramic or CBCT scans if available.",
        "eligible_branches": ["CLINIC-MAIN-01"]
    },
    {
        "service_id": "SRV-CONSULT-ORTHO",
        "name": "Orthodontic Consultation",
        "display_name": "Orthodontic Malocclusion & Aligners Consultation",
        "category": "consultation",
        "modality": None,
        "specialization": "Orthodontics",
        "duration_minutes": 30,
        "requires_referral": False,
        "preparation": "Bring prior orthodontic records if undergoing active transfer.",
        "eligible_branches": ["CLINIC-MAIN-01"]
    },
    {
        "service_id": "SRV-CONSULT-ENDO",
        "name": "Endodontic Consultation",
        "display_name": "Endodontic & Root Canal Consultation",
        "category": "consultation",
        "modality": None,
        "specialization": "Endodontics",
        "duration_minutes": 30,
        "requires_referral": False,
        "preparation": "Document history of sensitivity, swelling, or pain duration.",
        "eligible_branches": ["CLINIC-MAIN-01", "CLINIC-EXPRESS-EAST"]
    },
    {
        "service_id": "SRV-CONSULT-SURG",
        "name": "Oral and Maxillofacial Surgery Consultation",
        "display_name": "Maxillofacial Surgery & Wisdom Tooth Consultation",
        "category": "consultation",
        "modality": None,
        "specialization": "Oral and Maxillofacial Surgery",
        "duration_minutes": 45,
        "requires_referral": True,
        "referral_note": "Surgical consultations for impaction removal require a baseline OPG or CBCT scan.",
        "preparation": "Bring relevant 3D CBCT or OPG scans and medical history.",
        "eligible_branches": ["CLINIC-MAIN-01"]
    },
    {
        "service_id": "SRV-CONSULT-PERIO",
        "name": "Periodontic Consultation",
        "display_name": "Periodontal & Gum Health Consultation",
        "category": "consultation",
        "modality": None,
        "specialization": "Periodontics",
        "duration_minutes": 30,
        "requires_referral": False,
        "preparation": "Bring bleeding history, mobility observations, and medical history.",
        "eligible_branches": ["CLINIC-MAIN-01"]
    }
]

# 4. Verified Doctors Directory
DOCTORS_DIRECTORY = [
    {
        "doctor_id": "DOC-101",
        "name": "Dr. Rajesh Mehta, MDS",
        "short_name": "Dr. Rajesh Mehta",
        "title": "Attending Radiologist",
        "specialization": "Oral and Maxillofacial Radiology",
        "qualifications": "BDS, MDS (Oral & Maxillofacial Radiology)",
        "branches": [
            "Viscan Dental Imaging & Specialist Center - Main Clinic",
            "Viscan Express Dental Imaging - East Satellite"
        ],
        "branch_ids": ["CLINIC-MAIN-01", "CLINIC-EXPRESS-EAST"],
        "eligible_services": [
            "CBCT scan for implant planning",
            "OPG (Panoramic Digital Radiograph)",
            "Bitewing Radiographs",
            "Periapical Radiograph (IOPA)",
            "Implant Assessment Consultation"
        ],
        "bio": "Senior specialist in digital maxillofacial imaging, cross-sectional implant planning, and mandibular nerve localization."
    },
    {
        "doctor_id": "DOC-102",
        "name": "Dr. Anjali Deshmukh, MDS",
        "short_name": "Dr. Anjali Deshmukh",
        "title": "Consultant Oral & Maxillofacial Surgeon",
        "specialization": "Oral and Maxillofacial Surgery",
        "qualifications": "BDS, MDS (Oral & Maxillofacial Surgery), FIBOMS",
        "branches": [
            "Viscan Dental Imaging & Specialist Center - Main Clinic"
        ],
        "branch_ids": ["CLINIC-MAIN-01"],
        "eligible_services": [
            "Oral and Maxillofacial Surgery Consultation",
            "Implant Assessment Consultation",
            "CBCT scan for implant planning"
        ],
        "bio": "Expert in surgical removal of horizontally impacted third molars, bone grafting, and immediate implant placement."
    },
    {
        "doctor_id": "DOC-103",
        "name": "Dr. Vikram Singhal, MDS",
        "short_name": "Dr. Vikram Singhal",
        "title": "Specialist Orthodontist",
        "specialization": "Orthodontics",
        "qualifications": "BDS, MDS (Orthodontics & Dentofacial Orthopedics)",
        "branches": [
            "Viscan Dental Imaging & Specialist Center - Main Clinic"
        ],
        "branch_ids": ["CLINIC-MAIN-01"],
        "eligible_services": [
            "Orthodontic Consultation",
            "OPG (Panoramic Digital Radiograph)"
        ],
        "bio": "Certified aligner provider specializing in adult orthodontics, severe crowdings, and skeletal discrepancy correction."
    },
    {
        "doctor_id": "DOC-104",
        "name": "Dr. Sneha Kulkarni, MDS",
        "short_name": "Dr. Sneha Kulkarni",
        "title": "Specialist Endodontist",
        "specialization": "Endodontics",
        "qualifications": "BDS, MDS (Conservative Dentistry & Endodontics)",
        "branches": [
            "Viscan Dental Imaging & Specialist Center - Main Clinic",
            "Viscan Express Dental Imaging - East Satellite"
        ],
        "branch_ids": ["CLINIC-MAIN-01", "CLINIC-EXPRESS-EAST"],
        "eligible_services": [
            "Endodontic Consultation",
            "Periapical Radiograph (IOPA)",
            "Dental Consultation"
        ],
        "bio": "Pioneer in microscopic endodontics, calcified canal negotiation, and periapical lesion non-surgical resolution."
    },
    {
        "doctor_id": "DOC-105",
        "name": "Dr. Amitav Roy, MDS",
        "short_name": "Dr. Amitav Roy",
        "title": "Consultant Periodontist",
        "specialization": "Periodontics",
        "qualifications": "BDS, MDS (Periodontics & Implantology)",
        "branches": [
            "Viscan Dental Imaging & Specialist Center - Main Clinic"
        ],
        "branch_ids": ["CLINIC-MAIN-01"],
        "eligible_services": [
            "Periodontic Consultation",
            "Bitewing Radiographs",
            "Dental Consultation"
        ],
        "bio": "Specialized in advanced periodontal regeneration, soft-tissue grafting, and peri-implantitis management."
    },
    {
        "doctor_id": "DOC-106",
        "name": "Dr. Pooja Verma, BDS",
        "short_name": "Dr. Pooja Verma",
        "title": "Resident Dental Surgeon",
        "specialization": "General Dentistry",
        "qualifications": "BDS (General Dental Surgery)",
        "branches": [
            "Viscan Dental Imaging & Specialist Center - Main Clinic",
            "Viscan Express Dental Imaging - East Satellite"
        ],
        "branch_ids": ["CLINIC-MAIN-01", "CLINIC-EXPRESS-EAST"],
        "eligible_services": [
            "Dental Consultation",
            "Bitewing Radiographs",
            "Periapical Radiograph (IOPA)"
        ],
        "bio": "Comprehensive dental healthcare provider focusing on preventative therapy, restorations, and initial triage."
    }
]

# =====================================================================
# API / SERVICE OPERATIONS
# =====================================================================

def list_specializations() -> List[Dict[str, Any]]:
    """Retrieves the clinic's verified medical specialties."""
    return [dict(s) for s in CLINIC_SPECIALIZATIONS]


def list_services(category: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Returns available clinic examinations, consultations, and procedures.
    Filter by category ('imaging', 'consultation') when specified.
    """
    if not category or category.lower() in ["all", "any"]:
        return [dict(s) for s in CLINIC_SERVICES]
    clean_cat = category.strip().lower()
    return [dict(s) for s in CLINIC_SERVICES if s.get("category", "").lower() == clean_cat]


def search_doctors(
    specialization: Optional[str] = None,
    service: Optional[str] = None,
    branch: Optional[str] = None,
    doctor_name: Optional[str] = None,
    query: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Searches verified doctors by specialization, service, branch, or name.
    Guarantees that only actual doctors in the clinic directory are returned.
    """
    results = []
    spec_clean = specialization.strip().lower() if specialization else ""
    srv_clean = service.strip().lower() if service else ""
    branch_clean = branch.strip().lower() if branch else ""
    doc_clean = doctor_name.strip().lower() if doctor_name else ""
    q_clean = query.strip().lower() if query else ""

    for doc in DOCTORS_DIRECTORY:
        d_name = doc["name"].lower()
        d_short = doc["short_name"].lower()
        d_spec = doc["specialization"].lower()
        d_branches = [b.lower() for b in doc["branches"]] + [bid.lower() for bid in doc.get("branch_ids", [])]
        d_services = [s.lower() for s in doc["eligible_services"]]

        # Match specialization
        if spec_clean and spec_clean not in d_spec and not any(spec_clean in s for s in d_spec.split()):
            continue

        # Match doctor name
        if doc_clean and doc_clean not in d_name and doc_clean not in d_short:
            # check last name
            last_name = doc_clean.split()[-1]
            if last_name not in d_name:
                continue

        # Match branch
        if branch_clean and not any(branch_clean in b for b in d_branches):
            continue

        # Match service
        if srv_clean and not any(srv_clean in s or s in srv_clean for s in d_services):
            continue

        # Free-text query match
        if q_clean:
            q_matched = (
                q_clean in d_name
                or q_clean in d_short
                or q_clean in d_spec
                or any(q_clean in s for s in d_services)
                or any(q_clean in b for b in d_branches)
                or any(w in d_name or w in d_spec for w in q_clean.split() if len(w) > 3)
            )
            if not q_matched:
                continue

        results.append(dict(doc))

    return results


def get_doctor_details(doctor_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves a doctor's verified profile and eligible services by ID or name."""
    clean_id = doctor_id.strip().lower()
    for doc in DOCTORS_DIRECTORY:
        if (
            doc["doctor_id"].lower() == clean_id
            or doc["name"].lower() == clean_id
            or doc["short_name"].lower() == clean_id
            or clean_id in doc["name"].lower()
        ):
            return dict(doc)
    return None


def get_service_details(service_name_or_id: str) -> Optional[Dict[str, Any]]:
    """Finds service details by service ID or name."""
    clean = service_name_or_id.strip().lower()
    for s in CLINIC_SERVICES:
        if (
            s["service_id"].lower() == clean
            or s["name"].lower() == clean
            or s["display_name"].lower() == clean
            or clean in s["name"].lower()
            or clean in s["display_name"].lower()
            or (s["modality"] and s["modality"].lower() == clean)
        ):
            return dict(s)
    return None


def find_matching_specialization(query: str) -> Optional[Dict[str, Any]]:
    """Fuzzy identifies a specialization from patient query text."""
    q_low = query.lower()
    for spec in CLINIC_SPECIALIZATIONS:
        s_name = spec["name"].lower()
        if s_name in q_low:
            return dict(spec)

    if any(w in q_low for w in ["radiolog", "cbct", "opg", "panoramic", "x-ray", "imaging", "scan"]):
        return next(s for s in CLINIC_SPECIALIZATIONS if s["specialization_id"] == "SPEC-RAD")
    if any(w in q_low for w in ["surgery", "surgeon", "extraction", "wisdom", "impact", "impaction"]):
        return next(s for s in CLINIC_SPECIALIZATIONS if s["specialization_id"] == "SPEC-SURG")
    if any(w in q_low for w in ["ortho", "brace", "aligner", "invisalign", "crowding", "teeth align"]):
        return next(s for s in CLINIC_SPECIALIZATIONS if s["specialization_id"] == "SPEC-ORTHO")
    if any(w in q_low for w in ["endo", "root canal", "rct", "pulp", "nerve pain"]):
        return next(s for s in CLINIC_SPECIALIZATIONS if s["specialization_id"] == "SPEC-ENDO")
    if any(w in q_low for w in ["perio", "gum", "gums", "bleeding", "pyorrhea", "periodont"]):
        return next(s for s in CLINIC_SPECIALIZATIONS if s["specialization_id"] == "SPEC-PERIO")
    if any(w in q_low for w in ["general", "checkup", "cleaning", "routine", "filling", "caries"]):
        return next(s for s in CLINIC_SPECIALIZATIONS if s["specialization_id"] == "SPEC-GEN")

    return None

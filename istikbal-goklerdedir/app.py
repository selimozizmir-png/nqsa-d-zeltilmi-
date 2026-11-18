# -*- coding: utf-8 -*-
"""
İstikbal Göklerdedir - Personal & Space Medicine AI Assistant
==============================================================
A Streamlit dashboard that runs entirely on your own computer.

  * AI engine      : Ollama (local)  |  OpenAI-compatible API  |  Anthropic API
  * Database       : SQLite  ->  data/istikbal.db
  * Files          : data/uploads/
  * Web research   : DuckDuckGo (ddgs)  -> NASA / NTRS / PubMed / SpaceX / Roscosmos-IBMP

Start:  streamlit run app.py

WARNING: This software is a decision-support tool; it does not replace medical diagnosis or treatment.

SI (Super Intelligence) DISCLOSURE — NASA Space Apps Challenge 2026:
This source code was generated with the assistance of an SI tool (Claude Opus 5.5, Anthropic)
and reviewed by the project team. Literature values embedded in CATALOG are taken from the
open-access sources listed in the project's reference list (docs/References.pdf).
"""
__ai_generated__ = "Code generated with SI assistance: Claude Opus 5.5 (Anthropic), 2026"

import base64
import datetime as dt
import io
import itertools
import json
import math
import os
import re
import sqlite3
import time
from contextlib import closing

import numpy as np
import pandas as pd
import requests

# --- Optional packages (if missing, the related feature is switched off and the app keeps running) ---
try:
    import streamlit as st
    import streamlit.components.v1 as components
except ImportError:  # for tests / imports
    st = None
    components = None

try:
    import plotly.graph_objects as go
except ImportError:
    go = None

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:  # new package name
    from ddgs import DDGS
except ImportError:
    try:  # old package name
        from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None

try:  # DICOM (optional):  pip install pydicom
    import pydicom
except ImportError:
    pydicom = None

try:  # Local speech recognition for voice input (optional):  pip install faster-whisper
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None


# =============================================================================
# 1) PATHS AND SETTINGS
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
DB_PATH = os.path.join(DATA_DIR, "istikbal.db")
SETTINGS_PATH = os.path.join(DATA_DIR, "settings.json")
os.makedirs(UPLOAD_DIR, exist_ok=True)

PROVIDERS = ["Ollama (local)", "OpenAI-compatible API", "Anthropic API"]

DEFAULT_SETTINGS = {
    "provider": "Ollama (local)",
    "ollama_url": "http://localhost:11434",
    "ollama_model": "llama3.1",
    "ollama_vision_model": "llava",
    "openai_url": "https://api.openai.com/v1",
    "openai_model": "gpt-4o-mini",
    "openai_key": "",
    "anthropic_model": "claude-sonnet-5-5",
    "anthropic_key": "",
    "temperature": 0.3,
    "web_search_in_chat": True,
    "use_docs_in_chat": True,
    "auto_memory": False,
    "speak_answers": False,
    "max_search_results": 4,
    # --- Self-updating literature library ---
    "auto_update_library": True,      # download new papers at start-up
    "library_update_hours": 12,       # minimum hours between updates
    "library_per_topic": 25,          # maximum new papers per topic
    "use_library_in_chat": True,
    "pubmed_email": "",               # optional (NCBI courtesy rule)
    "auto_model_update": True,        # check for a new version of the Ollama model at start-up
    "model_check_hours": 24,
    "library_topics": {
        "Bone & calcium": {
            "pubmed": "(spaceflight OR astronaut* OR cosmonaut* OR microgravity) AND (bone OR \"bone mineral density\" OR osteopenia OR calcium OR \"kidney stone\")",
            "ntrs": "spaceflight bone loss"},
        "Cardiovascular": {
            "pubmed": "(spaceflight OR astronaut* OR cosmonaut*) AND (cardiovascular OR orthostatic OR \"lower body negative pressure\" OR arterial OR cardiac OR jugular)",
            "ntrs": "spaceflight cardiovascular orthostatic"},
        "SANS / eye / brain": {
            "pubmed": "(spaceflight OR astronaut* OR cosmonaut*) AND (\"neuro-ocular\" OR SANS OR \"optic disc\" OR brain OR ventricle OR \"intracranial\")",
            "ntrs": "spaceflight associated neuro-ocular syndrome"},
        "Radiation": {
            "pubmed": "(spaceflight OR astronaut* OR \"space radiation\") AND (radiation OR dosimetry OR \"cosmic ray\" OR cancer)",
            "ntrs": "space radiation astronaut dose"},
        "Muscle & exercise": {
            "pubmed": "(spaceflight OR astronaut* OR cosmonaut*) AND (muscle OR exercise OR countermeasure OR ARED OR \"aerobic capacity\")",
            "ntrs": "spaceflight exercise countermeasure muscle"},
        "Neurophysiology & vestibular": {
            "pubmed": "(spaceflight OR astronaut* OR cosmonaut*) AND (vestibular OR sensorimotor OR EEG OR \"evoked potential\" OR reflex OR cognition)",
            "ntrs": "spaceflight sensorimotor vestibular"},
        "Immune, omics & commercial flights": {
            "pubmed": "(spaceflight OR astronaut*) AND (immune OR omics OR telomere OR Inspiration4 OR \"Polaris Dawn\" OR SpaceX OR \"commercial spaceflight\")",
            "ntrs": "spaceflight immune omics"},
        "Latent virus & immune tests": {
            "pubmed": "(spaceflight OR astronaut*) AND (\"latent virus\" OR \"Epstein-Barr\" OR varicella OR cytomegalovirus OR \"T cell\" OR leukocyte OR cytokine)",
            "ntrs": "astronaut latent virus reactivation immune"},
        "Sleep & behavioral health": {
            "pubmed": "(spaceflight OR astronaut* OR \"isolated confined\") AND (sleep OR circadian OR mood OR depression OR \"behavioral health\" OR workload OR cognition)",
            "ntrs": "astronaut sleep behavioral health"},
    },
    "library_topics_version": 2,
}

# Keyword dictionary for matching Turkish questions to the English literature
TR_EN_KEYWORDS = {
    "kemik": "bone", "kırık": "fracture", "kalsiyum": "calcium", "böbrek": "kidney", "taş": "stone",
    "kalça": "hip", "omurga": "spine", "kas": "muscle", "egzersiz": "exercise", "kalp": "cardiac",
    "damar": "vascular", "tansiyon": "pressure", "nabız": "heart", "ortostatik": "orthostatic",
    "bayılma": "syncope", "göz": "ocular", "optik": "optic", "beyin": "brain", "ventrikül": "ventricle",
    "radyasyon": "radiation", "doz": "dose", "kanser": "cancer", "denge": "vestibular", "sinir": "nerve",
    "bağışıklık": "immune", "virüs": "virus", "uyku": "sleep", "hormon": "hormone", "testosteron": "testosterone",
    "kortizol": "cortisol", "insülin": "insulin", "vitamin": "vitamin", "beslenme": "nutrition",
    "toparlanma": "recovery", "tromboz": "thrombosis", "pıhtı": "thrombosis", "juguler": "jugular",
    "bilişsel": "cognition", "uzay": "spaceflight", "astronot": "astronaut", "kozmonot": "cosmonaut",
    "mikroyerçekimi": "microgravity", "ilaç": "drug", "bisfosfonat": "bisphosphonate", "telomer": "telomere",
    "lenfosit": "lymphocyte", "nötrofil": "neutrophil", "lökosit": "leukocyte", "zona": "zoster",
    "sitokin": "cytokine", "enfeksiyon": "infection", "ruh": "mood", "depresyon": "depression",
    "kaygı": "anxiety", "stres": "stress", "izolasyon": "isolation", "psikoloj": "psychological",
    "dikkat": "attention", "uykusuzluk": "insomnia", "sirkadiyen": "circadian",
}

PHASES = ["Pre-flight (L-)", "In-flight (FD)", "Post-flight (R+)"]
TRUSTED_DOMAINS = {
    "NASA": "nasa.gov",
    "NASA NTRS": "ntrs.nasa.gov",
    "PubMed / PMC": "ncbi.nlm.nih.gov",
    "SpaceX": "spacex.com",
    "Roscosmos": "roscosmos.ru",
    "IBMP (Russian Institute of Biomedical Problems)": "imbp.ru",
    "ESA": "esa.int",
}


def load_settings():
    s = dict(DEFAULT_SETTINGS)
    saved = {}
    if os.path.exists(SETTINGS_PATH):
        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                saved = json.load(f)
            s.update(saved)
        except Exception:
            pass
    # Old settings file: add library topics introduced in this version to the user's topics
    if saved and (saved.get("library_topics_version") or 1) < DEFAULT_SETTINGS["library_topics_version"]:
        topics = dict(s.get("library_topics") or {})
        for k, v in DEFAULT_SETTINGS["library_topics"].items():
            topics.setdefault(k, v)
        s["library_topics"] = topics
        s["library_topics_version"] = DEFAULT_SETTINGS["library_topics_version"]
    # Settings saved by the older Turkish version may use old provider names
    if s.get("provider") not in PROVIDERS:
        prev = str(s.get("provider") or "")
        s["provider"] = next((p for p in PROVIDERS if p.split()[0] == prev.split()[0]), PROVIDERS[0]) if prev else PROVIDERS[0]
    # Environment variables (to supply keys without writing them to a file)
    s["openai_key"] = os.environ.get("OPENAI_API_KEY", s.get("openai_key", ""))
    s["anthropic_key"] = os.environ.get("ANTHROPIC_API_KEY", s.get("anthropic_key", ""))
    return s


def save_settings(s):
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:  # keys starting with "_" belong to the session and are not saved
        json.dump({k: v for k, v in s.items() if not k.startswith("_")}, f, ensure_ascii=False, indent=2)


# =============================================================================
# 2) PARAMETER CATALOG + LITERATURE-BASED 180-DAY MODEL
# =============================================================================
# Fields:
#  key, label, unit, ref_low, ref_high,
#  p180  : mean % change expected on a 180-day ISS mission (ARED + 2 h/day exercise)
#  model : "lin" (accumulates linearly with mission length) | "sat" (rises in the first weeks, then plateaus)
#  rec   : post-landing recovery half-life (days)
#  src   : source
P = lambda key, label, unit, lo=None, hi=None, p180=None, model=None, rec=None, src="": dict(
    key=key, label=label, unit=unit, lo=lo, hi=hi, p180=p180, model=model, rec=rec, src=src)

CATALOG = {
    "Bone & Metabolic (DXA, QCT, HR-pQCT, markers)": [
        P("dxa_fn", "Femoral neck aBMD (DXA)", "g/cm²", 0.75, 1.30, -1.7, "lin", 270, "English 2020 (npj Microgravity); Sibonga 2007"),
        P("dxa_th", "Total hip aBMD (DXA)", "g/cm²", 0.80, 1.35, -2.0, "lin", 270, "ARED-era estimate; Sibonga 2007"),
        P("dxa_ls", "L1-L4 aBMD (DXA)", "g/cm²", 0.90, 1.40, -1.7, "lin", 270, "English 2020; Smith 2012"),
        P("tbs", "Trabecular bone score (TBS)", "", 1.31, 1.60),
        P("qct_hip_tb", "Hip trabecular vBMD (QCT)", "mg/cm³", None, None, -8.0, "lin", 300, "Sibonga 2019 (Bone)"),
        P("qct_ls_tb", "Spine trabecular vBMD (QCT)", "mg/cm³", 120, 260),
        P("hr_fload", "Distal tibia F.Load (HR-pQCT)", "kN", None, None, -4.1, "lin", 352, "Gabel 2021 (BJSM); Gabel 2022 (Sci Rep)"),
        P("hr_ttbmd", "Distal tibia Tt.vBMD (HR-pQCT)", "mgHA/cm³", None, None, -3.0, "lin", 352, "Gabel 2021"),
        P("ctx", "Serum CTX", "ng/mL", 0.15, 0.60, 100, "sat", 30, "Stavnichuk 2020 meta-analysis"),
        P("u_ctx", "Urine CTx (24 h)", "µg/day", None, None, 87, "sat", 30, "Gabel 2021"),
        P("p1np", "P1NP", "µg/L", 20, 80, 70, "sat", 120, "Gabel 2021"),
        P("s_ca", "Serum calcium", "mmol/L", 2.15, 2.55, -1.3, "sat", 14, "Gabel 2021"),
        P("u_ca", "Urine calcium (24 h)", "mmol/day", None, 7.5, 35, "sat", 21, "Gabel 2021"),
        P("vitd", "25(OH)D", "nmol/L", 75, 150, -10, "sat", 60, "Gabel 2021"),
        P("pth", "PTH", "pg/mL", 15, 65, 14, "sat", 30, "Gabel 2021"),
    ],
    "Radiation (dosimetry)": [
        P("dose_mission", "Mission effective dose (dosimeter)", "mSv", None, None),
        P("dose_rate", "Daily dose rate", "mSv/day", 0.0, 0.6),
    ],
    "Cardiovascular (echo, ECG, BP, Doppler, LBNP)": [
        P("hr_rest", "Resting heart rate (ECG)", "/dk", 45, 75),
        P("qtc", "QTc (ECG)", "ms", 350, 450),
        P("sbp", "Systolic blood pressure", "mmHg", 90, 130),
        P("dbp", "Diastolic blood pressure", "mmHg", 55, 85),
        P("lvef", "LVEF (echo)", "%", 55, 70, -2, "sat", 14, "Tordeur 2025; Martin (NASA)"),
        P("lvmi", "LV mass index (echo)", "g/m²", 50, 115, -3, "sat", 7, "Tordeur 2025"),
        P("cimt", "Carotid IMT (Doppler)", "mm", 0.4, 0.9),
        P("c_dist", "Carotid distensibility coefficient", "10⁻³/kPa", None, None, -20, "sat", 180, "Hughson 2016 (low-moderate confidence)"),
        P("vo2", "VO2peak", "mL/kg/dk", 35, 70, -8, "sat", 10, "Moore 2014; English 2020"),
        P("pv", "Plasma volume", "mL/kg", 38, 50, -12, "sat", 5, "Martin (NASA); 9-15 %"),
        P("lbnp_tol", "LBNP tolerance (presyncope threshold, absolute)", "mmHg", 40, 100),
        P("tilt_dhr", "Tilt/standing heart-rate rise", "/dk", 0, 30, 60, "sat", 2, "Lee 2015/2020"),
    ],
    "Biochemical & Endocrine (hormones, immunoassay)": [
        P("testo", "Total testosterone", "ng/dL", 300, 1000),
        P("cortisol", "Morning cortisol", "µg/dL", 6, 23),
        P("tsh", "TSH", "mIU/L", 0.5, 4.0),
        P("glucose", "Fasting glucose", "mg/dL", 70, 100),
        P("hba1c", "HbA1c", "%", 4.0, 5.7),
        P("homa", "HOMA-IR", "", 0.5, 2.0, 40, "sat", 30, "Hughson 2016 (direction certain, magnitude uncertain)"),
        P("crp", "hs-CRP (immunoassay)", "mg/L", 0, 3),
    ],
    # Reference ranges: MedlinePlus (WBC, differential %), Merck Manual (ANC, ALC lower limits).
    # Spaceflight findings: Crucian 2015 (npj Microgravity; T-cell function low for the whole 6 months),
    # Mehta 2017 (npj Microgravity; EBV/VZV/CMV reactivation increases on long missions). No mean
    # % change is published for these findings, so p180 is deliberately left empty (nothing invented).
    "Immune (blood count, T cells, latent virus)": [
        P("wbc", "White blood cell count (WBC)", "×10³/µL", 4.5, 11.0),
        P("neut_pct", "Neutrophil percentage", "%", 40, 60),
        P("lymph_pct", "Lymphocyte percentage", "%", 20, 40),
        P("anc", "Absolute neutrophil count (ANC)", "×10³/µL", 1.5, None),
        P("alc", "Absolute lymphocyte count (ALC)", "×10³/µL", 1.0, None),
        P("tcell_act", "T-cell activation (stimulated CD69+ cell fraction)", "%", None, None),
        P("ebv_sal", "Saliva EBV DNA (qPCR)", "copies/mL", None, None),
        P("vzv_sal", "Saliva VZV DNA (qPCR)", "copies/mL", None, 0),
        P("cmv_urine", "Urine CMV DNA (qPCR)", "copies/mL", None, 0),
    ],
    # Sleep: Watson 2015 (AASM/SRS ≥ 7 h), Barger 2014 (ISS mean ~6 h).
    # PSQI > 5 poor sleep (Buysse 1989); PHQ-9 ≥ 10 at least moderate depressive symptoms (Kroenke 2001).
    # NASA-TLX (Hart & Staveland 1988) and Cognition/PVT (Basner 2015) have no universal cut-off;
    # they are judged against the person's own pre-flight baseline. SCREENING only, not a diagnosis.
    "Sleep & Behavioral health (actigraphy, questionnaires, cognitive tests)": [
        P("sleep_h", "Sleep duration (actigraphy, per night)", "hours", 7, None),
        P("hypnotic", "Sleep medication use", "nights/week", None, None),
        P("psqi", "PSQI sleep quality score (0-21)", "points", 0, 5),
        P("phq9", "PHQ-9 depressive symptom screening score (0-27)", "points", 0, 9),
        P("tlx", "NASA-TLX workload (0-100)", "points", None, None),
        P("pvt_lapse", "Cognition PVT lapses", "count", None, None),
    ],
    "Neurophysiology (EMG, NCS/NCV, EEG, ENG, EP, SSEP, VEP, BAEP, MEG, QSART, TMS, QST)": [
        P("emg_spont", "EMG spontaneous activity score (0-4)", "points", 0, 0),
        P("ncs_peron_cv", "NCS/NCV peroneal motor conduction velocity", "m/s", 42, 60),
        P("ncs_sural_amp", "NCS sural sensory amplitude", "µV", 6, 40),
        P("h_reflex", "Soleus H-reflex Hmax/Mmax", "", 0.2, 0.8),
        P("eeg_pdr", "EEG posterior dominant rhythm", "Hz", 8, 12),
        P("eng_caloric", "ENG/VNG caloric asymmetry", "%", 0, 25),
        P("ep_generic", "EP (generic evoked-potential latency)", "ms", None, None),
        P("ssep_p37", "Tibial SSEP P37 latency", "ms", 30, 45),
        P("vep_p100", "VEP P100 latency", "ms", 85, 115),
        P("baep_i_v", "BAEP/ABR I-V interpeak", "ms", 3.6, 4.5),
        P("meg_alpha", "MEG alpha peak frequency", "Hz", 8, 12),
        P("qsart", "QSART sweat volume (forearm)", "µL/cm²", 0.5, 3.0),
        P("tms_rmt", "TMS resting motor threshold", "%MSO", 30, 60),
        P("qst_vib", "QST vibration threshold (foot)", "µm", 0, 3),
        P("sot5", "Dynamic posturography SOT5 score", "points", 50, 100, -27, "sat", 4, "Frontiers 2023; NASA Field Test"),
    ],
    "Neuroimaging & Eye (MRI, OCT, SANS)": [
        P("lv_vol", "Lateral ventricle volume (MRI)", "mL", 7, 40, 20, "sat", 210, "Hupfeld 2020; McGregor 2023"),
        P("rnfl", "OCT peripapillary RNFL/TRT", "µm", 80, 110, 8, "sat", 120, "Laurie (SANS; low-moderate confidence)"),
        P("onsd", "Optic nerve sheath diameter (ultrasound)", "mm", 3.5, 5.0),
        P("iop", "Intraocular pressure", "mmHg", 10, 21),
        P("refr", "Spherical equivalent refraction", "D", -0.5, 0.5),
    ],
    "Muscle & Body composition": [
        P("leg_lean", "Leg lean mass (DXA)", "kg", None, None, -0.7, "lin", 30, "English 2020"),
        P("knee_tq", "Knee extensor torque 60°/s", "Nm", None, None, -7, "lin", 30, "English 2020"),
        P("weight", "Body weight", "kg", None, None),
    ],
}
PARAM_INDEX = {p["key"]: dict(p, category=cat) for cat, plist in CATALOG.items() for p in plist}
LABEL_TO_KEY = {p["label"]: k for k, p in PARAM_INDEX.items()}

# --- Measurability in space (based on the health/research hardware NASA keeps aboard the ISS) ---
#  "uzay"     : measured in flight with equipment aboard the ISS (CHeCS, Human Research Facility)
#  "deneysel" : measurable with an experimental device tested on the ISS (not yet routine)
#  "ornek"    : sample collected in space, frozen and analysed on Earth (may be impossible on a Mars mission)
#  "dunya"    : no suitable device in space; measured only before / after flight
# Sources: CHeCS hardware catalog (BP/ECG, tonometer); ISS ultrasound and OCT (since 2013);
# SLAMMD/BMMD (body mass); CEVIS + PPFS (VO2max); MARES (ESA, part of NASA HRF);
# Vascular Echo (CSA); actigraphy (Barger 2014); Cognition (Basner 2015); HemoCue WBC DIFF and
# rHEALTH ONE tested on the ISS (NASA In-situ Lab Analysis); there is no X-ray or MRI on the ISS.
_SAMPLE = "Blood/urine/saliva sample collected in space, frozen and analysed on Earth"
_XRAY = "Needs DXA / CT — no X-ray device aboard the ISS"
_GRAV = "Return-to-gravity test — only meaningful on Earth"
_NEURO = "Needs a clinical neurophysiology lab — no routine device aboard the ISS"
SPACE_INFO = {
    "dose_mission": ("uzay", "Personal dosimeter / TEPC (CHeCS environmental health system)"),
    "dose_rate": ("uzay", "Personal dosimeter / TEPC (CHeCS environmental health system)"),
    "hr_rest": ("uzay", "CHeCS BP/ECG unit (12-lead ECG)"),
    "qtc": ("uzay", "CHeCS BP/ECG unit (12-lead ECG)"),
    "sbp": ("uzay", "CHeCS BP/ECG unit (blood-pressure device)"),
    "dbp": ("uzay", "CHeCS BP/ECG unit (blood-pressure device)"),
    "lvef": ("uzay", "ISS ultrasound — echocardiography (remotely guided from the ground)"),
    "lvmi": ("uzay", "ISS ultrasound — echocardiography (remotely guided from the ground)"),
    "cimt": ("uzay", "ISS ultrasound + portable Doppler (Vascular Echo)"),
    "c_dist": ("uzay", "ISS ultrasound + portable Doppler (Vascular Echo)"),
    "vo2": ("uzay", "CEVIS cycle ergometer + PPFS gas analysis"),
    "rnfl": ("uzay", "OCT (on the ISS since 2013)"),
    "onsd": ("uzay", "ISS ultrasound — eye"),
    "iop": ("uzay", "Tonometer (CHeCS medical diagnostic pack)"),
    "knee_tq": ("uzay", "MARES dynamometer (Columbus module)"),
    "weight": ("uzay", "SLAMMD / BMMD body-mass device"),
    "sleep_h": ("uzay", "Actigraph wristband"),
    "hypnotic": ("uzay", "Astronaut's own log"),
    "psqi": ("uzay", "Laptop questionnaire"),
    "phq9": ("uzay", "Laptop questionnaire"),
    "tlx": ("uzay", "Laptop questionnaire"),
    "pvt_lapse": ("uzay", "Cognition test battery (laptop)"),
    "wbc": ("deneysel", "HemoCue WBC DIFF fingerstick analyser (tested on the ISS in 2021)"),
    "neut_pct": ("deneysel", "HemoCue WBC DIFF / rHEALTH ONE (tested on the ISS) — experimental"),
    "lymph_pct": ("deneysel", "HemoCue WBC DIFF / rHEALTH ONE (tested on the ISS) — experimental"),
    "anc": ("deneysel", "HemoCue WBC DIFF / rHEALTH ONE (tested on the ISS) — experimental"),
    "alc": ("deneysel", "HemoCue WBC DIFF / rHEALTH ONE (tested on the ISS) — experimental"),
    **{k: ("ornek", _SAMPLE) for k in ("ctx", "u_ctx", "p1np", "s_ca", "u_ca", "vitd", "pth", "testo", "cortisol",
                                       "tsh", "glucose", "hba1c", "homa", "crp", "tcell_act", "ebv_sal",
                                       "vzv_sal", "cmv_urine")},
    **{k: ("dunya", _XRAY) for k in ("dxa_fn", "dxa_th", "dxa_ls", "tbs", "qct_hip_tb", "qct_ls_tb",
                                     "hr_fload", "hr_ttbmd", "leg_lean")},
    **{k: ("dunya", _GRAV) for k in ("lbnp_tol", "tilt_dhr", "sot5")},
    "lv_vol": ("dunya", "Needs MRI — no MRI aboard the ISS"),
    "refr": ("dunya", "Full refraction exam is done in a ground clinic"),
    "pv": ("dunya", "Plasma volume measurement needs a special research protocol"),
}
SPACE_LEVEL_LABEL = {"uzay": "🛰️ Measured in space", "deneysel": "🧪 Experimental device in space",
                     "ornek": "📦 Sample in space, result from Earth", "dunya": "🌍 Earth only"}
for _k, _p in PARAM_INDEX.items():
    _p["space"], _p["device"] = SPACE_INFO.get(_k, ("dunya", _NEURO))


def allowed_keys(mode, include_samples=False):
    """Parameters to ask for in data entry, depending on the working mode."""
    if mode != "uzay":
        return set(PARAM_INDEX)
    levels = {"uzay", "deneysel"} | ({"ornek"} if include_samples else set())
    return {k for k, p in PARAM_INDEX.items() if p["space"] in levels}


def space_table():
    return pd.DataFrame([dict(Category=p["category"].split(" (")[0], Parameter=p["label"],
                              Where=SPACE_LEVEL_LABEL[p["space"]], Device=p["device"])
                         for p in PARAM_INDEX.values()])


def param_info(name):
    """Returns catalog information from a parameter label or key."""
    if name in PARAM_INDEX:
        return PARAM_INDEX[name]
    if name in LABEL_TO_KEY:
        return PARAM_INDEX[LABEL_TO_KEY[name]]
    return None


def expected_pct(p, day, mission_days=180, modifier=1.0):
    """Expected % change (vs. baseline).  day: timeline day (0 = launch)."""
    if not p or p.get("p180") is None or day is None:
        return None
    p180 = p["p180"] * modifier
    if day <= 0:
        return 0.0
    if day <= mission_days:  # in flight
        if p["model"] == "lin":
            return p180 * day / 180.0
        tau = 30.0
        return p180 * (1 - math.exp(-day / tau)) / (1 - math.exp(-180 / tau))
    # after flight: the deficit at landing recovers exponentially
    at_landing = expected_pct(p, mission_days, mission_days, modifier)
    hl = p.get("rec") or 90
    return at_landing * math.exp(math.log(0.5) * (day - mission_days) / hl)


def timeline_day(phase, day, mission_days):
    """Phase + day within phase -> timeline day relative to launch."""
    day = int(day or 0)
    if phase.startswith("Pre-flight"):
        return -abs(day)
    if phase.startswith("In-flight"):
        return abs(day)
    return int(mission_days or 180) + abs(day)


# =============================================================================
# 3) DATABASE (SQLite)
# =============================================================================
SCHEMA = """
CREATE TABLE IF NOT EXISTS subjects(
  id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE NOT NULL, role TEXT, sex TEXT,
  birth_year INTEGER, height_cm REAL, weight_kg REAL, mission TEXT, mission_days INTEGER DEFAULT 180,
  launch_date TEXT, prior_dose_msv REAL DEFAULT 0, last_lduration_return TEXT, notes TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS measurements(
  id INTEGER PRIMARY KEY AUTOINCREMENT, subject_id INTEGER, category TEXT, parameter TEXT, value REAL,
  unit TEXT, phase TEXT, day INTEGER, date TEXT, source TEXT, note TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS documents(
  id INTEGER PRIMARY KEY AUTOINCREMENT, subject_id INTEGER, filename TEXT, filetype TEXT, path TEXT,
  text TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS chats(
  id INTEGER PRIMARY KEY AUTOINCREMENT, subject_id INTEGER, role TEXT, content TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS memory(
  id INTEGER PRIMARY KEY AUTOINCREMENT, subject_id INTEGER, kind TEXT, content TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS articles(
  id INTEGER PRIMARY KEY AUTOINCREMENT, uid TEXT UNIQUE, source TEXT, topic TEXT, title TEXT, abstract TEXT,
  journal TEXT, pub_date TEXT, url TEXT, added_at TEXT);
CREATE TABLE IF NOT EXISTS research(
  id INTEGER PRIMARY KEY AUTOINCREMENT, subject_id INTEGER, query TEXT, title TEXT, url TEXT,
  snippet TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS reports(
  id INTEGER PRIMARY KEY AUTOINCREMENT, subject_id INTEGER, created_at TEXT, mode TEXT,
  scenario_days INTEGER, model TEXT, content TEXT);
"""


def now():
    return dt.datetime.now().isoformat(timespec="seconds")


def db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with closing(db()) as c:
        c.executescript(SCHEMA)
        c.commit()


def q(sql, params=(), fetch=False):
    with closing(db()) as c:
        cur = c.execute(sql, params)
        c.commit()
        if fetch:
            return [dict(r) for r in cur.fetchall()]
        return cur.lastrowid


def qdf(sql, params=()):
    with closing(db()) as c:
        return pd.read_sql_query(sql, c, params=params)


def get_subjects():
    return q("SELECT * FROM subjects ORDER BY name", fetch=True)


def get_subject(sid):
    r = q("SELECT * FROM subjects WHERE id=?", (sid,), fetch=True)
    return r[0] if r else None


def add_subject(name, **kw):
    cols = ["name", "role", "sex", "birth_year", "height_cm", "weight_kg", "mission", "mission_days",
            "launch_date", "prior_dose_msv", "last_lduration_return", "notes", "created_at"]
    vals = [name] + [kw.get(c) for c in cols[1:-1]] + [now()]
    return q(f"INSERT INTO subjects({','.join(cols)}) VALUES({','.join('?' * len(cols))})", vals)


def update_subject(sid, **kw):
    sets = ", ".join(f"{k}=?" for k in kw)
    q(f"UPDATE subjects SET {sets} WHERE id=?", list(kw.values()) + [sid])


def delete_subject(sid):
    for t in ["measurements", "documents", "chats", "memory", "research", "reports"]:
        q(f"DELETE FROM {t} WHERE subject_id=?", (sid,))
    q("DELETE FROM subjects WHERE id=?", (sid,))


def add_measurement(sid, parameter, value, unit, phase, day, date=None, source="manual", note="", category=None):
    info = param_info(parameter)
    label = info["label"] if info else parameter
    category = category or (info["category"] if info else "Custom")
    return q("""INSERT INTO measurements(subject_id,category,parameter,value,unit,phase,day,date,source,note,created_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
             (sid, category, label, float(value), unit or (info["unit"] if info else ""), phase, int(day or 0),
              date or dt.date.today().isoformat(), source, note, now()))


def get_measurements(sid):
    df = qdf("SELECT * FROM measurements WHERE subject_id=? ORDER BY id", (sid,))
    subj = get_subject(sid)
    md = (subj or {}).get("mission_days") or 180
    if not df.empty:
        df["t"] = [timeline_day(p or PHASES[0], d, md) for p, d in zip(df["phase"], df["day"])]
        df["po"] = [PHASES.index(p) if p in PHASES else 0 for p in df["phase"]]  # phase order within the same day
    return df


def ensure_demo_subject():
    """Creates an example astronaut profile on first launch (can be deleted)."""
    if get_subjects():
        return
    sid = add_subject("EXAMPLE - Astronaut A (40 M, 180-day ISS)", role="Astronaut", sex="Male", birth_year=1986,
                      height_cm=178, weight_kg=78, mission="ISS long-duration mission", mission_days=180,
                      launch_date="", prior_dose_msv=0, notes="Literature-based example data. You can delete it.")
    demo = [  # parameter, L-, R+0 values
        ("dxa_fn", 0.95, 0.934), ("dxa_ls", 1.10, 1.081), ("hr_fload", 10.0, 9.59), ("ctx", 0.35, 0.70),
        ("p1np", 48.8, 83.0), ("u_ca", 4.86, 6.56), ("vitd", 86, 77), ("vo2", 45, 41.4), ("pv", 42, 37),
        ("lv_vol", 20, 24), ("rnfl", 98, 106), ("sot5", 70, 51), ("hr_rest", 55, 62), ("sbp", 115, 112),
        ("lvef", 60, 59), ("homa", 1.0, 1.4), ("ncs_peron_cv", 48, 48), ("vep_p100", 100, 101),
    ]
    for key, pre, post in demo:
        p = PARAM_INDEX[key]
        add_measurement(sid, key, pre, p["unit"], PHASES[0], 180, source="example")
        add_measurement(sid, key, pre, p["unit"], PHASES[0], 60, source="example")
        add_measurement(sid, key, post, p["unit"], PHASES[2], 0, source="example")
    add_measurement(sid, "dose_mission", 75, "mSv", PHASES[2], 0, source="example")
    # In-flight bone markers (ratios of the Gabel 2021 cohort medians)
    for d, f in [(15, 1.88), (30, 1.99), (60, 2.42), (120, 2.14), (180, 1.83)]:
        add_measurement(sid, "ctx", round(0.35 * f, 3), "ng/mL", PHASES[1], d, source="example")
    for d, f in [(15, 0.95), (30, 1.05), (60, 1.54), (120, 2.04), (180, 1.93)]:
        add_measurement(sid, "p1np", round(48.8 * f, 1), "µg/L", PHASES[1], d, source="example")


# =============================================================================
# 4) ANALYSIS: BASELINE COMPARISON, PREDICTION, RISK
# =============================================================================
def summarize(sid):
    """For each parameter: baseline (pre-flight mean), latest value, % change, out-of-range flag."""
    df = get_measurements(sid)
    subj = get_subject(sid) or {}
    md = subj.get("mission_days") or 180
    if df.empty:
        return pd.DataFrame()
    rows = []
    for par, g in df.groupby("parameter"):
        g = g.sort_values(["t", "po", "id"])
        pre = g[g["phase"].str.startswith("Pre-flight", na=False)]
        baseline = pre["value"].mean() if not pre.empty else np.nan
        last = g.iloc[-1]
        info = param_info(par)
        delta = (last["value"] / baseline - 1) * 100 if baseline and not np.isnan(baseline) and baseline != 0 else np.nan
        exp_now = expected_pct(info, last["t"], md, risk_modifier(sid, info)) if info else None
        out = ""
        if info:
            if info["lo"] is not None and last["value"] < info["lo"]:
                out = "↓ below reference"
            if info["hi"] is not None and last["value"] > info["hi"]:
                out = "↑ above reference"
        rows.append(dict(
            Category=last["category"], Parameter=par, Unit=last["unit"], Baseline=baseline, Latest=last["value"],
            Latest_phase=f"{last['phase'].split(' ')[0]} {last['phase'].split(' ')[1]} day {last['day']}",
            Change_pct=delta, Expected_pct=exp_now,
            Deviation=(delta - exp_now) if exp_now is not None and not np.isnan(delta) else np.nan,
            Reference=f"{'' if not info or info['lo'] is None else info['lo']} – {'' if not info or info['hi'] is None else info['hi']}",
            Flag=out, N_measurements=len(g)))
    return pd.DataFrame(rows).sort_values(["Category", "Parameter"]).reset_index(drop=True)


def _baseline(sid, key):
    info = PARAM_INDEX.get(key)
    if not info:
        return None
    r = q("SELECT AVG(value) v FROM measurements WHERE subject_id=? AND parameter=? AND phase LIKE 'Pre-flight%'",
          (sid, info["label"]), fetch=True)
    return r[0]["v"] if r and r[0]["v"] is not None else None


def _phase_values(sid, key, phase_prefixes):
    """Returns all measurement values in the given phases (e.g. 'In-flight') as a list."""
    info = PARAM_INDEX.get(key)
    if not info:
        return []
    rows = q("SELECT phase, value FROM measurements WHERE subject_id=? AND parameter=?",
             (sid, info["label"]), fetch=True)
    return [r["value"] for r in rows if r["value"] is not None
            and any(str(r["phase"]).startswith(p) for p in phase_prefixes)]


def risk_modifier(sid, info):
    """Personal risk multiplier. Literature: high pre-flight bone turnover -> more tibial loss (Gabel 2021)."""
    if not info or info.get("p180") is None:
        return 1.0
    bone = info["key"] in ("dxa_fn", "dxa_th", "dxa_ls", "qct_hip_tb", "hr_fload", "hr_ttbmd")
    if bone:
        ctx, p1 = _baseline(sid, "ctx"), _baseline(sid, "p1np")
        if (ctx and ctx > 0.50) or (p1 and p1 > 70):
            return 1.3
    return 1.0


def projection_table(sid, mission_days=None):
    subj = get_subject(sid) or {}
    md = mission_days or subj.get("mission_days") or 180
    rows = []
    for key, p in PARAM_INDEX.items():
        if p["p180"] is None:
            continue
        base = _baseline(sid, key)
        mod = risk_modifier(sid, p)
        r0 = expected_pct(p, md, md, mod)
        r365 = expected_pct(p, md + 365, md, mod)
        rows.append(dict(
            Parameter=p["label"], Unit=p["unit"], Baseline=base,
            **{"R+0 expected %": r0, "R+0 expected value": base * (1 + r0 / 100) if base else None,
               "R+365 expected %": r365, "Personal multiplier": mod, "Source": p["src"]}))
    return pd.DataFrame(rows)


def evaluate_risks(sid):
    """Rule-based risk engine (based on open NASA/Roscosmos/SpaceX literature)."""
    subj = get_subject(sid) or {}
    md = subj.get("mission_days") or 180
    s = summarize(sid)
    d = {r["Parameter"]: r for _, r in s.iterrows()} if not s.empty else {}

    def dl(key):
        lab = PARAM_INDEX[key]["label"]
        r = d.get(lab)
        return (None if r is None or pd.isna(r["Change_pct"]) else r["Change_pct"]), (None if r is None else r["Latest"])

    risks = []

    def add(name, level, why, action, src):
        risks.append(dict(Risk=name, Level=level, Rationale=why, Action=action, Source=src))

    # 1 Orthostatic intolerance
    pv, _ = dl("pv")
    _, lbnp = dl("lbnp_tol")
    lvl = "High" if md >= 90 else "Moderate"
    why = f"{md}-day mission: after long missions 60-80% cannot complete a 10-min tilt test."
    if pv is not None and pv < -15:
        why += f" Plasma volume {pv:.1f}%."
    if lbnp is not None and lbnp < 40:
        why += f" Low LBNP tolerance ({lbnp:.0f} mmHg)."
    add("Orthostatic intolerance on landing day", lvl, why,
        "LBNP (Chibis) in the final weeks; fluid-salt loading + compression garment (GCG/Kentavr) at landing; assisted movement for the first 24 h.",
        "Lee 2020 Front Physiol; NASA NTRS 20110020663")
    # 2 Bone
    worst, worst_lab = 0, ""
    for k in ("dxa_fn", "dxa_th", "dxa_ls", "hr_fload", "hr_ttbmd", "qct_hip_tb"):
        v, _ = dl(k)
        if v is not None and v < worst:
            worst, worst_lab = v, PARAM_INDEX[k]["label"]
    hi_turn = risk_modifier(sid, PARAM_INDEX["dxa_fn"]) > 1
    lvl = "High" if worst <= -3 or hi_turn else ("Moderate" if md >= 90 else "Low")
    why = (f"Largest bone deficit: {worst_lab} {worst:.1f}%." if worst_lab else "No bone measurements; model mean: DXA ≈ -1.7 %, tibia F.Load ≈ -4.1 %.")
    if hi_turn:
        why += " High pre-flight bone turnover (CTX>0.50 or P1NP>70) → larger loss and incomplete recovery."
    add("Lasting bone deficit / fracture later in life", lvl, why,
        "High ARED load; discuss an anti-resorptive (e.g. alendronate) with the flight surgeon; DXA + HR-pQCT up to R+365.",
        "Sibonga 2019 Bone; Gabel 2022 Sci Rep")
    # 3 Kidney stones
    uca_d, uca = dl("u_ca")
    if (uca is not None and uca > 7.5) or (uca_d is not None and uca_d > 40):
        add("Kidney stones", "High", f"Urine Ca {uca} mmol/day ({uca_d if uca_d is None else round(uca_d, 1)} %).",
            "Urine volume > 2 L/day; consider potassium citrate (20 mEq/day).", "Whitson, NASA NTRS 20090019087")
    else:
        add("Kidney stones", "Moderate" if md >= 90 else "Low", "In flight urine Ca ↑, volume and citrate ↓.",
            "Fluid intake, urine chemistry monitoring.", "Whitson, NASA NTRS")
    # 4 SANS
    rn, _ = dl("rnfl")
    _, onsd = dl("onsd")
    lvl = "High" if (rn is not None and rn > 10) or (onsd is not None and onsd > 5.5) else ("Moderate" if md >= 60 else "Low")
    add("SANS (optic disc oedema, choroidal folds, hyperopia)", lvl,
        f"OCT change: {rn if rn is None else round(rn, 1)} %; ONSD: {onsd}. OCT oedema on long missions ~69%.",
        "FD50/FD150 OCT + fundus; B-vitamin status; OCT/refraction follow-up after return.",
        "Laurie (AAO summary); NASA SANS guidance 2026")
    # 5 Ventricles
    lv, _ = dl("lv_vol")
    if lv is not None:
        add("Ventricular enlargement", "High" if lv > 15 else "Moderate", f"Lateral ventricles {lv:.1f} %.",
            "MRI at R+180, R+365; at least 3 years before the next long mission.", "Hupfeld 2020; McGregor 2023")
    # 6 Radiation
    dose_rows = q("SELECT SUM(value) v FROM measurements WHERE subject_id=? AND parameter=?",
                  (sid, PARAM_INDEX["dose_mission"]["label"]), fetch=True)
    cum = (subj.get("prior_dose_msv") or 0) + ((dose_rows[0]["v"] or 0) if dose_rows else 0)
    lvl = "High" if cum > 500 else ("Moderate" if cum > 300 else "Low")
    add("Radiation (career dose)", lvl, f"Cumulative dose ≈ {cum:.0f} mSv / NASA limit 600 mSv ({cum / 6:.0f} %).",
        "Personal dosimeter; log EVA dose separately.", "NASA-STD-3001 / OCHMO")
    # 7 Cardiometabolic
    hm, hmv = dl("homa")
    _, cimt = dl("cimt")
    if (hmv is not None and hmv > 2) or (hm is not None and hm > 50) or (cimt is not None and cimt > 0.9):
        add("Accelerated cardiometabolic ageing", "Moderate-High", f"HOMA-IR {hmv}, IMT {cimt}.",
            "Spread activity through the day; carotid ultrasound + HOMA-IR yearly.", "Hughson 2016")
    # 8 Aerobic capacity
    vo2, _ = dl("vo2")
    if vo2 is not None and vo2 < -15:
        add("Loss of aerobic capacity", "High", f"VO2peak {vo2:.1f} %.", "Revise exercise intensity/duration (HIIT).",
            "Moore 2014; English 2020")
    # 9 Sensorimotor
    sot, _ = dl("sot5")
    add("Sensorimotor impairment / falls", "High" if (sot is not None and sot < -20) else "Moderate",
        f"SOT5 change: {sot if sot is None else round(sot, 1)} %. Full recovery 6-16 days.",
        "No driving for the first 2 weeks; vestibular rehabilitation.", "Reschke, NASA Field Test")
    # 10 Disc herniation
    add("Disc herniation (especially in the first 12 months)", "Moderate-High", "4.3 times more common in astronauts; cervical risk much higher in the first year.",
        "Avoid sudden heavy axial loads in the first year; paraspinal/cervical stabilisation.", "Johnston 2010 ASEM")
    # 11 Immune dysregulation / latent virus reactivation
    order = ["Low", "Moderate", "Moderate-High", "High"]

    def worse(a, b):
        return a if order.index(a) >= order.index(b) else b

    lvl = "Moderate" if md >= 90 else "Low"
    why = ["On long missions T-cell function is suppressed and latent herpes viruses reactivate more often."]
    vzv = _phase_values(sid, "vzv_sal", ("In-flight", "Post-flight"))
    cmv = _phase_values(sid, "cmv_urine", ("In-flight", "Post-flight"))
    ebv_in = _phase_values(sid, "ebv_sal", ("In-flight",))
    ebv_pre = _phase_values(sid, "ebv_sal", ("Pre-flight",))
    if vzv and max(vzv) > 0:
        lvl = worse(lvl, "High")
        why.append(f"VZV DNA found in saliva (highest {max(vzv):.0f} copies/mL): shingles risk and infectiousness.")
    if cmv and max(cmv) > 0:
        lvl = worse(lvl, "Moderate-High")
        why.append(f"CMV DNA found in urine (highest {max(cmv):.0f} copies/mL).")
    if ebv_in and ebv_pre and np.mean(ebv_pre) > 0 and max(ebv_in) >= 10 * np.mean(ebv_pre):
        lvl = worse(lvl, "Moderate-High")
        why.append(f"Saliva EBV DNA in flight is {max(ebv_in) / np.mean(ebv_pre):.0f} times baseline (≥ 10 times: project heuristic threshold).")
    fl = ("In-flight", "Post-flight")
    alc = min(_phase_values(sid, "alc", fl), default=None)
    anc = min(_phase_values(sid, "anc", fl), default=None)
    if alc is not None and alc < 1.0:
        lvl = worse(lvl, "Moderate-High")
        why.append(f"Lymphopenia (lowest ALC {alc} ×10³/µL < 1.0).")
    if anc is not None and anc < 1.5:
        lvl = worse(lvl, "Moderate-High")
        why.append(f"Neutropenia (lowest ANC {anc} ×10³/µL < 1.5).")
    tc, _ = dl("tcell_act")
    if tc is not None and tc <= -30:
        lvl = worse(lvl, "Moderate-High")
        why.append(f"T-cell activation {tc:.0f} % vs. baseline (≤ -30 %: project heuristic threshold).")
    add("Immune dysregulation / latent virus reactivation", lvl, " ".join(why),
        "In-flight saliva/urine virus PCR and blood count monitoring; sleep and stress management; if symptoms "
        "appear (rash, fever, pain) report to the flight surgeon; if VZV is positive, crew contact and an antiviral "
        "plan should be reviewed with the flight surgeon.",
        "Crucian 2015 & Mehta 2017 (npj Microgravity); NASA Immune Risk; Merck (ANC/ALC limits)")
    # 12 Sleep deficiency and behavioral health (SCREENING – not a diagnosis)
    lvl = "Moderate" if md >= 90 else "Low"
    why = ["Behavioral health risk in isolation/confinement grows with mission length; ISS mean sleep ~6 h."]
    sl = _phase_values(sid, "sleep_h", ("In-flight",))
    if sl:
        m = float(np.mean(sl))
        if m < 6:
            lvl = worse(lvl, "High")
        elif m < 7:
            lvl = worse(lvl, "Moderate-High")
        why.append(f"Mean in-flight sleep {m:.1f} h (recommended ≥ 7).")
    hyp = _phase_values(sid, "hypnotic", ("In-flight",))
    if hyp and np.mean(hyp) >= 3:
        lvl = worse(lvl, "Moderate-High")
        why.append(f"Sleep medication on average {np.mean(hyp):.1f} nights/week: performance risk if woken for an emergency.")
    psqi = max(_phase_values(sid, "psqi", ("In-flight", "Post-flight")), default=None)
    if psqi is not None and psqi > 5:
        lvl = worse(lvl, "Moderate-High")
        why.append(f"Highest PSQI {psqi:.0f} (> 5: poor sleep quality).")
    phq = max(_phase_values(sid, "phq9", ("In-flight", "Post-flight")), default=None)
    if phq is not None and phq >= 10:
        lvl = worse(lvl, "High")
        why.append(f"Highest PHQ-9 {phq:.0f} (≥ 10: screen positive for at least moderate depressive symptoms).")
    pv_l, _ = dl("pvt_lapse")
    if pv_l is not None and pv_l >= 50:
        lvl = worse(lvl, "Moderate-High")
        why.append(f"PVT lapses +{pv_l:.0f} % vs. baseline (≥ +50 %: project heuristic threshold).")
    tl, _ = dl("tlx")
    if tl is not None and tl >= 30:
        why.append(f"NASA-TLX workload +{tl:.0f} % vs. baseline.")
    add("Sleep deficiency / behavioral health (screening)", lvl, " ".join(why),
        "Protect the sleep schedule (target ≥ 7 h), adjust light/noise, plan the workload; if PHQ-9 ≥ 10, "
        "talk to the behavioral health team / flight surgeon. This is a screening result, not a diagnosis.",
        "Barger 2014 Lancet Neurol; Watson 2015 (AASM/SRS); Buysse 1989; Kroenke 2001; NASA BMed Risk")
    # 13 All out-of-range values
    if not s.empty:
        for _, r in s[s["Flag"] != ""].iterrows():
            add(f"Out of range: {r['Parameter']}", "Review", f"Latest value {r['Latest']} {r['Unit']} ({r['Flag']}).",
                "Clinical review / repeat measurement.", "Catalog reference range (approximate)")
    return pd.DataFrame(risks)


# =============================================================================
# 5) FILE READING
# =============================================================================
TABULAR_COLS = {
    "parameter": ["parametre", "parameter", "param", "test"],
    "value": ["deger", "değer", "value", "sonuc", "sonuç", "result"],
    "unit": ["birim", "unit"],
    "phase": ["faz", "phase"],
    "day": ["gun", "gün", "day"],
    "date": ["tarih", "date"],
}


def extract_file(uploaded):
    """Saves an uploaded file; returns its text and (if any) its table."""
    name = uploaded.name
    ext = name.lower().rsplit(".", 1)[-1] if "." in name else ""
    raw = uploaded.getvalue()
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(UPLOAD_DIR, f"{stamp}_{re.sub(r'[^A-Za-z0-9._-]', '_', name)}")
    with open(path, "wb") as f:
        f.write(raw)
    text, table, kind = "", None, ext
    try:
        if ext == "pdf":
            if PdfReader is None:
                text = "[pypdf is not installed]"
            else:
                reader = PdfReader(io.BytesIO(raw))
                text = "\n".join((pg.extract_text() or "") for pg in reader.pages)
                if not text.strip():
                    text = "[The PDF may be a scanned image; no text could be extracted. Upload it as an image and use the vision model.]"
        elif ext in ("csv", "tsv"):
            try:
                table = pd.read_csv(io.BytesIO(raw), sep=None, engine="python", encoding="utf-8-sig")
            except UnicodeDecodeError:  # Turkish Windows encoding used by Excel
                table = pd.read_csv(io.BytesIO(raw), sep=None, engine="python", encoding="cp1254")
            text = table.to_csv(index=False)
        elif ext in ("xlsx", "xls"):
            table = pd.read_excel(io.BytesIO(raw))
            text = table.to_csv(index=False)
        elif ext in ("txt", "md", "json", "xml", "html"):
            text = raw.decode("utf-8", errors="replace")
        elif ext in ("png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"):
            kind = "image"
            text = "[Image file - you can ask the assistant to interpret it with 'Image analysis']"
        elif ext == "dcm":
            kind = "dicom"
            if pydicom is None:
                text = "[DICOM: install with 'pip install pydicom' to read tags]"
            else:
                ds = pydicom.dcmread(io.BytesIO(raw), stop_before_pixels=True)
                keep = ["Modality", "StudyDescription", "SeriesDescription", "BodyPartExamined", "StudyDate",
                        "Manufacturer", "SliceThickness", "PixelSpacing"]
                text = "\n".join(f"{k}: {getattr(ds, k, '')}" for k in keep)
        else:
            text = raw.decode("utf-8", errors="replace")[:20000]
    except Exception as e:
        text = f"[Could not read the file: {e}]"
    return path, kind, text, table


def normalize_table(df):
    """Tries to convert an uploaded table to the measurement format (parameter, value, unit, phase, day, date)."""
    cols = {str(c).replace("\ufeff", "").strip().lower(): c for c in df.columns}  # strip the BOM
    mapping = {}
    for target, cands in TABULAR_COLS.items():
        for cnd in cands:
            if cnd in cols:
                mapping[target] = cols[cnd]
                break
    if "parameter" not in mapping or "value" not in mapping:
        return None
    out = pd.DataFrame({t: df[c] for t, c in mapping.items()})
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["value"])
    if "phase" not in out:
        out["phase"] = PHASES[0]
    else:
        def fix_phase(x):
            x = str(x).lower()
            if x.startswith(("r", "son", "post")):
                return PHASES[2]
            if x.startswith(("fd", "uçuş i", "ucus i", "in")):
                return PHASES[1]
            return PHASES[0]
        out["phase"] = out["phase"].map(fix_phase)
    if "day" not in out:
        out["day"] = 0
    out["day"] = pd.to_numeric(out["day"], errors="coerce").fillna(0).astype(int)
    if "unit" not in out:
        out["unit"] = [(param_info(p) or {}).get("unit", "") for p in out["parameter"]]
    return out


# =============================================================================
# 6) AI CONNECTIONS (Ollama / OpenAI-compatible / Anthropic)
# =============================================================================
SYSTEM_PROMPT = """You are İstikbal Göklerdedir: an elite personal AI assistant and space medicine / biometric telemetry analyst.
Expertise: DXA, QCT, HR-pQCT, CTX, P1NP, calcium, dosimetry, echocardiography, ECG, blood pressure, carotid Doppler, LBNP,
hormones/immunoassay, EMG, NCS/NCV, EEG, ENG, EP, SEP/SSEP, VEP, BAEP/ABR, MEG, MRI, QSART, TMS, QST.
Rules:
- Answer in English by default; if the user writes in another language (for example Turkish), answer in that language.
- Compare pre-flight baseline with in-flight/post-flight values; state the physiological mechanism and your confidence (High/Moderate/Low).
- If there is a critical threshold or anomaly, report it first, starting with "⚠️".
- Base your analysis on open NASA, SpaceX and Roscosmos/IBMP literature; cite the URLs from the web results you are given.
  If you have no source, say so clearly; never invent sources or numbers.
- Papers in the "Current literature (local library)" section are your newest information; if they conflict with your
  training knowledge, prefer them and cite them with date + URL. Your training knowledge is older; remember this.
- Be proactive: add hidden correlations, follow-up measurements and preventive recommendations.
- You are a decision-support tool; diagnosis and treatment decisions belong to the physician (flight surgeon). Remind briefly when relevant.
- Always follow the "Working mode" rule in the CONTEXT: in SPACE mode ask the astronaut only for parameters that can
  be measured in space; never ask for those that cannot (DXA, MRI, tilt, etc.) and explain them as model estimates.
Tone: professional, short and clear, solution-oriented."""


def _split_system(messages):
    sys_txt = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    rest = [m for m in messages if m["role"] != "system"]
    return sys_txt, rest


def llm_stream(messages, cfg, images_b64=None, vision=False, max_tokens=None, num_ctx=None):
    """Streams a reply from the selected provider. images_b64: images to attach to the last user message.
    max_tokens: optional cap on the answer length; num_ctx: optional Ollama context window (tokens)."""
    prov = cfg["provider"]
    temp = float(cfg.get("temperature", 0.3))
    try:
        if prov.startswith("Ollama"):
            msgs = [dict(m) for m in messages]
            if images_b64:
                msgs[-1]["images"] = images_b64
            model = cfg["ollama_vision_model"] if (vision and images_b64) else cfg["ollama_model"]
            r = requests.post(f"{cfg['ollama_url'].rstrip('/')}/api/chat",
                              json={"model": model, "messages": msgs, "stream": True,
                                    "options": {"temperature": temp,
                                                **({"num_predict": int(max_tokens)} if max_tokens else {}),
                                                **({"num_ctx": int(num_ctx)} if num_ctx else {})}},
                              stream=True, timeout=900)
            if r.status_code != 200:
                yield f"⚠️ Ollama error ({r.status_code}): {r.text[:300]}"
                return
            for line in r.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                if "error" in data:
                    yield f"⚠️ Ollama: {data['error']}"
                    return
                yield data.get("message", {}).get("content", "")
                if data.get("done"):
                    break

        elif prov.startswith("OpenAI"):
            msgs = [dict(m) for m in messages]
            if images_b64:
                content = [{"type": "text", "text": msgs[-1]["content"]}]
                content += [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b}"}} for b in images_b64]
                msgs[-1]["content"] = content
            r = requests.post(f"{cfg['openai_url'].rstrip('/')}/chat/completions",
                              headers={"Authorization": f"Bearer {cfg['openai_key']}"},
                              json={"model": cfg["openai_model"], "messages": msgs, "stream": True,
                                    "temperature": temp, **({"max_tokens": int(max_tokens)} if max_tokens else {})},
                              stream=True, timeout=900)
            if r.status_code != 200:
                yield f"⚠️ API error ({r.status_code}): {r.text[:300]}"
                return
            for line in r.iter_lines():
                if not line or not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    break
                chunk = json.loads(payload)
                ch = chunk.get("choices") or [{}]
                yield (ch[0].get("delta") or {}).get("content") or ""

        else:  # Anthropic
            sys_txt, rest = _split_system(messages)
            msgs = [dict(m) for m in rest]
            if images_b64:
                content = [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b}}
                           for b in images_b64]
                content.append({"type": "text", "text": msgs[-1]["content"]})
                msgs[-1]["content"] = content
            r = requests.post("https://api.anthropic.com/v1/messages",
                              headers={"x-api-key": cfg["anthropic_key"], "anthropic-version": "2023-06-01",
                                       "content-type": "application/json"},
                              json={"model": cfg["anthropic_model"], "max_tokens": int(max_tokens or 4096), "system": sys_txt,
                                    "messages": msgs, "stream": True, "temperature": temp},
                              stream=True, timeout=900)
            if r.status_code != 200:
                yield f"⚠️ Anthropic error ({r.status_code}): {r.text[:300]}"
                return
            for line in r.iter_lines():
                if not line or not line.startswith(b"data:"):
                    continue
                ev = json.loads(line[5:].strip())
                if ev.get("type") == "content_block_delta":
                    yield ev.get("delta", {}).get("text", "")
                elif ev.get("type") == "message_stop":
                    break
    except requests.exceptions.ConnectionError:
        if prov.startswith("Ollama"):
            yield ("⚠️ Could not connect to Ollama. Make sure Ollama is running (`ollama serve`) "
                   f"and download the model: `ollama pull {cfg['ollama_model']}`")
        else:
            yield "⚠️ Could not connect to the API. Check your internet connection and the address."
    except Exception as e:
        yield f"⚠️ AI error: {e}"


def llm_complete(messages, cfg, **kw):
    return "".join(llm_stream(messages, cfg, **kw))


def ollama_models(url):
    try:
        r = requests.get(f"{url.rstrip('/')}/api/tags", timeout=5)
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


# =============================================================================
# 7) WEB RESEARCH
# =============================================================================
def web_search(query, domains=None, max_results=4):
    """Searches via DuckDuckGo; if domains are given, runs a 'site:'-restricted search for each domain."""
    if DDGS is None:
        return [{"title": "Search module not installed", "url": "", "snippet": "pip install ddgs"}]
    results, seen = [], set()
    targets = domains or [None]
    try:
        with DDGS() as ddgs:
            for dom in targets:
                qq = f"{query} site:{dom}" if dom else query
                try:
                    for r in ddgs.text(qq, max_results=max_results):
                        url = r.get("href") or r.get("url") or ""
                        if url in seen:
                            continue
                        seen.add(url)
                        results.append({"title": r.get("title", ""), "url": url, "snippet": r.get("body", "")})
                except Exception as e:
                    results.append({"title": f"Search error ({dom})", "url": "", "snippet": str(e)[:200]})
    except Exception as e:
        results.append({"title": "Search error", "url": "", "snippet": str(e)[:300]})
    return results


def save_research(sid, query, results):
    for r in results:
        if r.get("url"):
            q("INSERT INTO research(subject_id,query,title,url,snippet,created_at) VALUES(?,?,?,?,?,?)",
              (sid, query, r["title"], r["url"], r["snippet"], now()))


# =============================================================================
# 7b) SELF-UPDATING LITERATURE LIBRARY (PubMed + NASA NTRS) AND MODEL UPDATE
# =============================================================================
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NTRS_API = "https://ntrs.nasa.gov/api/citations/search"


def meta_get(key, default=None):
    r = q("SELECT value FROM meta WHERE key=?", (key,), fetch=True)
    return r[0]["value"] if r else default


def meta_set(key, value):
    q("INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, str(value)))


def hours_since(iso):
    if not iso:
        return 1e9
    try:
        return (dt.datetime.now() - dt.datetime.fromisoformat(iso)).total_seconds() / 3600
    except ValueError:
        return 1e9


def _ncbi_params(cfg):
    p = {"tool": "istikbal_goklerdedir"}
    if cfg.get("pubmed_email"):
        p["email"] = cfg["pubmed_email"]
    return p


def pubmed_search(term, mindate, maxdate, retmax, cfg):
    """Returns the IDs of papers ADDED to PubMed within the given date range."""
    params = dict(_ncbi_params(cfg), db="pubmed", term=term, retmode="json", retmax=retmax, sort="pub_date",
                  datetype="edat", mindate=mindate.strftime("%Y/%m/%d"), maxdate=maxdate.strftime("%Y/%m/%d"))
    r = requests.get(f"{EUTILS}/esearch.fcgi", params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("esearchresult", {}).get("idlist", [])


def parse_pubmed_xml(xml_text):
    import xml.etree.ElementTree as ET
    out = []
    root = ET.fromstring(xml_text)
    for art in root.iter("PubmedArticle"):
        pmid = (art.findtext(".//MedlineCitation/PMID") or "").strip()
        a = art.find(".//MedlineCitation/Article")
        if a is None or not pmid:
            continue
        title = "".join(a.find("ArticleTitle").itertext()).strip() if a.find("ArticleTitle") is not None else ""
        parts = []
        for ab in a.findall(".//Abstract/AbstractText"):
            label = ab.get("Label")
            txt = "".join(ab.itertext()).strip()
            parts.append(f"{label}: {txt}" if label else txt)
        journal = a.findtext(".//Journal/Title") or ""
        pd_ = a.find(".//Journal/JournalIssue/PubDate")
        date = ""
        if pd_ is not None:
            date = pd_.findtext("MedlineDate") or "-".join(x for x in [pd_.findtext("Year"), pd_.findtext("Month"),
                                                                       pd_.findtext("Day")] if x)
        out.append(dict(uid=f"pubmed:{pmid}", source="PubMed", title=title, abstract="\n".join(parts),
                        journal=journal, pub_date=date, url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"))
    return out


def pubmed_fetch(ids, cfg):
    if not ids:
        return []
    params = dict(_ncbi_params(cfg), db="pubmed", id=",".join(ids), retmode="xml")
    r = requests.get(f"{EUTILS}/efetch.fcgi", params=params, timeout=60)
    r.raise_for_status()
    return parse_pubmed_xml(r.text)


def ntrs_search(query, since, size=15):
    """Open API of the NASA Technical Reports Server (NTRS)."""
    r = requests.get(NTRS_API, params={"q": query, "page.size": size, "sort.field": "published",
                                        "sort.order": "desc"}, timeout=30)
    r.raise_for_status()
    out = []
    for it in r.json().get("results", []):
        date = ""
        for key in ("publications", "meetings"):
            for pub in it.get(key) or []:
                date = date or (pub.get("publicationDate") or pub.get("startDate") or "")[:10]
        date = date or (it.get("distributionDate") or it.get("submittedDate") or "")[:10]
        if date and since and date < since.isoformat():
            continue
        out.append(dict(uid=f"ntrs:{it.get('id')}", source="NASA NTRS", title=it.get("title", ""),
                        abstract=it.get("abstract") or "", journal="NASA NTRS", pub_date=date,
                        url=f"https://ntrs.nasa.gov/citations/{it.get('id')}"))
    return out


def save_articles(items, topic):
    new = 0
    for a in items:
        try:
            q("""INSERT INTO articles(uid,source,topic,title,abstract,journal,pub_date,url,added_at)
                 VALUES(?,?,?,?,?,?,?,?,?)""", (a["uid"], a["source"], topic, a["title"], a["abstract"],
                                                 a["journal"], a["pub_date"], a["url"], now()))
            new += 1
        except sqlite3.IntegrityError:
            pass  # already in the library
    return new


def update_library(cfg, log=print, force=False):
    """Downloads papers published since the last update. On the first run it scans the last 365 days."""
    last = meta_get("library_last_update")
    if not force and hours_since(last) < float(cfg.get("library_update_hours", 12)):
        return {"new": 0, "skipped": True}
    today = dt.date.today()
    since = (dt.datetime.fromisoformat(last).date() - dt.timedelta(days=2)) if last else today - dt.timedelta(days=365)
    total, errors = 0, []
    for topic, qs in (cfg.get("library_topics") or {}).items():
        try:
            ids = pubmed_search(qs["pubmed"], since, today, int(cfg.get("library_per_topic", 25)), cfg)
            have = {r["uid"] for r in q("SELECT uid FROM articles WHERE source='PubMed'", fetch=True)}
            ids = [i for i in ids if f"pubmed:{i}" not in have]
            n = save_articles(pubmed_fetch(ids, cfg), topic)
            total += n
            log(f"PubMed · {topic}: {n} new papers")
        except Exception as e:
            errors.append(f"PubMed/{topic}: {e}")
            log(f"PubMed · {topic}: skipped ({str(e)[:80]})")
        try:
            n = save_articles(ntrs_search(qs.get("ntrs") or topic, since), topic)
            total += n
            log(f"NASA NTRS · {topic}: {n} new reports")
        except Exception as e:
            errors.append(f"NTRS/{topic}: {e}")
            log(f"NASA NTRS · {topic}: skipped ({str(e)[:80]})")
        time.sleep(0.4)  # NCBI: at most 3 requests per second
    if total or not errors:
        meta_set("library_last_update", now())
        meta_set("library_last_new", total)
    return {"new": total, "errors": errors, "since": since.isoformat()}


def english_keywords(text):
    """Turns a question into English keywords for library and web search (dictionary-based for Turkish input)."""
    words = re.findall(r"[\wçğıöşüÇĞİÖŞÜ+\-]{3,}", text.lower())
    keys = set()
    for w in words:
        for tr, en in TR_EN_KEYWORDS.items():
            if w.startswith(tr):
                keys.add(en)
    # Abbreviations (CTX, P1NP, DXA, SANS, LBNP...) and terms already written in English
    keys.update(a.lower() for a in re.findall(r"\b[A-Z][A-Z0-9+\-]{1,}\b", text))
    keys.update(w for w in words if w in ENGLISH_TERMS)
    return sorted(keys)


ENGLISH_TERMS = set(TR_EN_KEYWORDS.values()) | {
    "osteoporosis", "osteopenia", "density", "fracture", "countermeasure", "orthostatic", "intolerance",
    "syncope", "edema", "optic", "retina", "choroidal", "intracranial", "dosimetry", "cosmic", "telomere",
    "omics", "microbiome", "vestibular", "sensorimotor", "reflex", "evoked", "hypercalciuria", "resorption",
    "formation", "atrophy", "stiffness", "plasma", "volume", "anemia", "sleep", "circadian"}


def library_search(query, k=6):
    """Simple scoring: keyword matches + a recency bonus."""
    keys = english_keywords(query)
    rows = q("SELECT * FROM articles ORDER BY id DESC LIMIT 5000", fetch=True)
    if not rows:
        return []
    scored = []
    for i, r in enumerate(rows):
        text = f"{r['title']} {r['title']} {r['abstract']}".lower()
        sc = sum(text.count(kw) for kw in keys)
        if sc:
            scored.append((sc + max(0, 3 - i / 200), r))  # small bonus for newly added papers
    scored.sort(key=lambda x: -x[0])
    return [r for _, r in scored[:k]]


def ollama_pull(cfg, model=None):
    """Downloads the newest version of the Ollama model (takes seconds if unchanged). Yields status dictionaries."""
    model = model or cfg["ollama_model"]
    r = requests.post(f"{cfg['ollama_url'].rstrip('/')}/api/pull", json={"model": model, "name": model,
                                                                          "stream": True}, stream=True, timeout=3600)
    r.raise_for_status()
    for line in r.iter_lines():
        if line:
            yield json.loads(line)


# =============================================================================
# 8) CONTEXT BUILDING (learning memory + data + documents)
# =============================================================================
def build_context(sid, user_msg, cfg, web_results=None):
    subj = get_subject(sid) or {}
    parts = [f"## Subject profile\n{json.dumps({k: v for k, v in subj.items() if k not in ('id',)}, ensure_ascii=False)}"]
    if cfg.get("_mode") == "uzay":
        ok = allowed_keys("uzay", cfg.get("_include_samples", False))
        parts.append(
            "## Working mode: SPACE (the astronaut is on a mission now)\n"
            "Suggest or ask the astronaut to measure ONLY the parameters below, which can be measured with station equipment. "
            "NEVER ask for anything not on the list (DXA, QCT, HR-pQCT, MRI, tilt/LBNP tolerance, balance tests, neurophysiology, etc.); "
            "explain their status as a model estimate based on pre-flight data and recommend measuring them after return. "
            "Your recommendations must be concrete steps the astronaut can take on board.\n"
            + "\n".join(f"- {PARAM_INDEX[k]['label']} — {PARAM_INDEX[k]['device']}"
                        f"{' (experimental)' if PARAM_INDEX[k]['space'] == 'deneysel' else ''}" for k in sorted(ok)))
    else:
        parts.append("## Working mode: EARTH (pre-/post-flight or ground laboratory)\n"
                     "All parameters (including MRI, DXA and neurophysiology) can be measured.")
    s = summarize(sid)
    if not s.empty:
        show = s[["Parameter", "Unit", "Baseline", "Latest", "Latest_phase", "Change_pct", "Expected_pct", "Flag"]].copy()
        parts.append("## Measurement summary (baseline = pre-flight mean; Expected_pct = literature model)\n"
                     + show.round(3).to_csv(index=False))
    risks = evaluate_risks(sid)
    if not risks.empty:
        parts.append("## Rule-based risk engine output\n" + risks[["Risk", "Level", "Rationale"]].to_csv(index=False))
    mem = q("SELECT kind, content FROM memory WHERE subject_id=? OR subject_id IS NULL ORDER BY id DESC LIMIT 40",
            (sid,), fetch=True)
    if mem:
        parts.append("## Memory (learned facts)\n" + "\n".join(f"- [{m['kind']}] {m['content']}" for m in mem))
    if cfg.get("use_docs_in_chat", True):
        docs = q("SELECT filename, text FROM documents WHERE subject_id=? ORDER BY id DESC", (sid,), fetch=True)
        if docs:
            words = set(w.lower() for w in re.findall(r"\w{4,}", user_msg))
            scored = sorted(docs, key=lambda d: -sum(d["text"].lower().count(w) for w in words) if words else 0)
            budget, chunks = 8000, []
            for d in scored:
                if budget <= 0:
                    break
                txt = d["text"][: min(budget, 4000)]
                chunks.append(f"### {d['filename']}\n{txt}")
                budget -= len(txt)
            parts.append("## Uploaded documents (excerpts)\n" + "\n\n".join(chunks))
    if cfg.get("use_library_in_chat", True):
        arts = library_search(user_msg, k=6)
        if arts:
            parts.append("## Current literature (local library; newest information)\n" + "\n\n".join(
                f"- [{a['pub_date']}] {a['title']} ({a['journal']}) {a['url']}\n  Abstract: {a['abstract'][:700]}"
                for a in arts))
    if web_results:
        parts.append("## Web research results (cite the URLs)\n" + "\n".join(
            f"- {r['title']} | {r['url']} | {r['snippet']}" for r in web_results if r.get("url")))
    return "\n\n".join(parts)


def learn_from_exchange(sid, user_msg, answer, cfg):
    """Optional automatic memory: extracts lasting facts from the conversation."""
    prompt = [{"role": "system", "content": "Your task: extract LASTING and IMPORTANT facts about the user/subject from the conversation "
                                           "(preferences, medical history, goals, mission information). Return only a JSON array: "
                                           "[\"fact 1\", \"fact 2\"]. If there are none, return []."},
              {"role": "user", "content": f"USER: {user_msg}\n\nASSISTANT: {answer[:3000]}"}]
    out = llm_complete(prompt, cfg)
    m = re.search(r"\[.*\]", out, re.S)
    if not m:
        return []
    try:
        facts = [f for f in json.loads(m.group(0)) if isinstance(f, str) and f.strip()]
    except Exception:
        return []
    for f in facts[:5]:
        q("INSERT INTO memory(subject_id,kind,content,created_at) VALUES(?,?,?,?)", (sid, "automatic", f.strip(), now()))
    return facts


# =============================================================================
# 9) VOICE (input: optional faster-whisper | output: browser speech synthesis)
# =============================================================================
_WHISPER = {}


def transcribe(audio_bytes):
    if WhisperModel is None:
        return None
    if "m" not in _WHISPER:
        _WHISPER["m"] = WhisperModel("small", device="cpu", compute_type="int8")
    path = os.path.join(DATA_DIR, "_tmp_audio.wav")
    with open(path, "wb") as f:
        f.write(audio_bytes)
    segs, _ = _WHISPER["m"].transcribe(path, language="tr")
    return " ".join(s.text for s in segs).strip()


def speak(text):
    clean = re.sub(r"[#*_`>|\[\]()]", " ", text)[:3000]
    components.html(f"""<script>
      const u = new SpeechSynthesisUtterance({json.dumps(clean)}); u.lang = 'en-US'; u.rate = 1.0;
      window.parent.speechSynthesis.cancel(); window.parent.speechSynthesis.speak(u);
    </script>""", height=0)


# =============================================================================
# 10) CHARTS
# =============================================================================
def trend_figure(sid, params, pct=False):
    df = get_measurements(sid)
    subj = get_subject(sid) or {}
    md = subj.get("mission_days") or 180
    fig = go.Figure()
    for par in params:
        g = df[df["parameter"] == par].sort_values(["t", "po"])
        if g.empty:
            continue
        base = g[g["phase"].str.startswith("Pre-flight")]["value"].mean()
        if pct and not (pd.notna(base) and base != 0):
            continue  # without a baseline the % change cannot be plotted
        y = (g["value"] / base - 1) * 100 if pct else g["value"]
        fig.add_trace(go.Scatter(x=g["t"], y=y, mode="lines+markers", name=par,
                                 hovertext=[f"{p} day {d}" for p, d in zip(g["phase"], g["day"])]))
        info = param_info(par)
        if pct and info and info.get("p180") is not None:
            xs = list(range(0, md + 366, 5))
            mod = risk_modifier(sid, info)
            fig.add_trace(go.Scatter(x=xs, y=[expected_pct(info, x, md, mod) for x in xs], mode="lines",
                                     line=dict(dash="dot"), name=f"{par} – literature expectation"))
        if not pct and len(params) == 1 and base == base:
            fig.add_hline(y=base, line_dash="dash", annotation_text="Baseline (pre-flight mean)")
            if info and info["lo"] is not None and info["hi"] is not None:
                fig.add_hrect(y0=info["lo"], y1=info["hi"], opacity=0.08, line_width=0,
                              annotation_text="Reference", annotation_position="top left")
    fig.add_vline(x=0, line_dash="dot", annotation_text="Launch")
    fig.add_vline(x=md, line_dash="dot", annotation_text="Landing")
    if pct:
        fig.add_hline(y=0, line_color="gray")
    fig.update_layout(height=470, xaxis_title="Days from launch", yaxis_title="% change (vs. baseline)" if pct else "Value",
                      legend=dict(orientation="h", y=-0.25), margin=dict(l=10, r=10, t=30, b=10))
    return fig


# =============================================================================
# 11) USER INTERFACE (Streamlit)
# =============================================================================
def ui_sidebar(cfg):
    st.sidebar.title("🛰️ İstikbal Göklerdedir")
    st.sidebar.caption("Personal & Space Medicine AI Assistant — runs locally")
    mode = st.session_state.get("mode", "dunya")
    st.sidebar.success("Mode: 🚀 SPACE (on mission)" if mode == "uzay" else "Mode: 🌍 EARTH")
    if mode == "uzay":
        st.session_state["include_samples"] = st.sidebar.checkbox(
            "📦 Also enter lab results sent from Earth",
            value=st.session_state.get("include_samples", False),
            help="On the ISS, blood/urine samples are frozen and sent to Earth; if results are sent back up from the ground they can be "
                 "entered here. On distant missions such as Mars this may not be possible.")
    if st.sidebar.button("🔄 Change mode"):
        st.session_state.pop("mode", None)
        st.rerun()

    subjects = get_subjects()
    names = [s["name"] for s in subjects]
    if names:
        cur = st.session_state.get("sid")
        idx = next((i for i, s in enumerate(subjects) if s["id"] == cur), 0)
        sel = st.sidebar.selectbox("👤 Active person / astronaut", names, index=idx)
        st.session_state["sid"] = next(s["id"] for s in subjects if s["name"] == sel)

    with st.sidebar.expander("➕ New person / astronaut"):
        with st.form("new_subj", clear_on_submit=True):
            n = st.text_input("Name / code")
            c1, c2 = st.columns(2)
            role = c1.selectbox("Role", ["Astronaut", "Cosmonaut", "Analog study participant", "Myself", "Other"])
            sex = c2.selectbox("Sex", ["Male", "Female", "Not specified"])
            by = c1.number_input("Year of birth", 1930, 2020, 1986)
            md = c2.number_input("Mission length (days)", 0, 1500, 180)
            h = c1.number_input("Height (cm)", 100.0, 230.0, 178.0)
            w = c2.number_input("Weight (kg)", 30.0, 200.0, 78.0)
            mis = st.text_input("Mission", "ISS long-duration mission")
            dose = st.number_input("Previous career dose (mSv)", 0.0, 2000.0, 0.0)
            if st.form_submit_button("Create") and n.strip():
                try:
                    st.session_state["sid"] = add_subject(n.strip(), role=role, sex=sex, birth_year=by, height_cm=h,
                                                          weight_kg=w, mission=mis, mission_days=md, prior_dose_msv=dose)
                    st.rerun()
                except sqlite3.IntegrityError:
                    st.error("This name already exists.")

    with st.sidebar.expander("⚙️ AI settings", expanded=False):
        cfg["provider"] = st.selectbox("Engine", PROVIDERS, index=PROVIDERS.index(cfg["provider"]))
        if cfg["provider"].startswith("Ollama"):
            cfg["ollama_url"] = st.text_input("Ollama address", cfg["ollama_url"])
            models = ollama_models(cfg["ollama_url"])
            if models:
                cfg["ollama_model"] = st.selectbox("Model", models,
                                                   index=models.index(cfg["ollama_model"]) if cfg["ollama_model"] in models else 0)
                vis = [m for m in models if any(k in m for k in ("llava", "vision", "bakllava", "moondream", "gemma3"))]
                cfg["ollama_vision_model"] = st.selectbox("Vision model", vis or models)
                st.success(f"Ollama connected · {len(models)} models")
            else:
                cfg["ollama_model"] = st.text_input("Model", cfg["ollama_model"])
                cfg["ollama_vision_model"] = st.text_input("Vision model", cfg["ollama_vision_model"])
                st.warning("Could not reach Ollama. Is `ollama serve` running?")
        elif cfg["provider"].startswith("OpenAI"):
            cfg["openai_url"] = st.text_input("API address (OpenAI, LM Studio, Groq, etc.)", cfg["openai_url"])
            cfg["openai_model"] = st.text_input("Model", cfg["openai_model"])
            cfg["openai_key"] = st.text_input("API key", cfg["openai_key"], type="password")
        else:
            cfg["anthropic_model"] = st.text_input("Model", cfg["anthropic_model"])
            cfg["anthropic_key"] = st.text_input("API key", cfg["anthropic_key"], type="password")
        cfg["temperature"] = st.slider("Creativity (temperature)", 0.0, 1.0, float(cfg["temperature"]), 0.05)
        cfg["max_search_results"] = st.slider("Web results per domain", 1, 10, int(cfg["max_search_results"]))
        cfg["auto_memory"] = st.toggle("Automatic memory (learn from conversations)", cfg["auto_memory"])
        if st.button("💾 Save settings"):
            save_settings(cfg)
            st.toast("Settings saved (data/settings.json)")
        st.caption("API keys are stored only on this computer, in data/settings.json. "
                   "You can also use the OPENAI_API_KEY / ANTHROPIC_API_KEY environment variables.")
    st.sidebar.markdown("---")
    st.sidebar.caption("⚠️ Decision-support tool; diagnosis and treatment decisions belong to a physician.")


def inflight_estimates(sid, day=None):
    """Today's (or a given day's) estimate for parameters that cannot be measured in space but have a literature model."""
    subj = get_subject(sid) or {}
    md = subj.get("mission_days") or 180
    if day is None:
        day = mission_day_today(subj)
    if day is None:  # no launch date: use the latest recorded in-flight day
        r = q("SELECT MAX(day) d FROM measurements WHERE subject_id=? AND phase LIKE 'In-flight%'", (sid,), fetch=True)
        day = r[0]["d"] if r and r[0]["d"] else None
    if not day:
        return pd.DataFrame()
    rows = []
    for k, p in PARAM_INDEX.items():
        if p["space"] not in ("dunya", "ornek") or p["p180"] is None:
            continue
        base = _baseline(sid, k)
        mod = risk_modifier(sid, p)
        e = expected_pct(p, day, md, mod)
        rows.append({"Parameter": p["label"], "Why it cannot be measured": p["device"], "Flight day": f"FD{day}",
                     "Baseline": base, "Estimated change %": e,
                     "Estimated value": base * (1 + e / 100) if base else None, "Source": p["src"]})
    return pd.DataFrame(rows)


def ui_dashboard(sid):
    subj = get_subject(sid)
    st.subheader(f"📊 Dashboard — {subj['name']}")
    c = st.columns(5)
    age = (dt.date.today().year - subj["birth_year"]) if subj.get("birth_year") else "-"
    c[0].metric("Age", age)
    c[1].metric("Mission length", f"{subj.get('mission_days') or 0} days")
    m = get_measurements(sid)
    c[2].metric("Measurements", len(m))
    c[3].metric("Documents", q("SELECT COUNT(*) n FROM documents WHERE subject_id=?", (sid,), fetch=True)[0]["n"])
    c[4].metric("Memory entries", q("SELECT COUNT(*) n FROM memory WHERE subject_id=?", (sid,), fetch=True)[0]["n"])

    s = summarize(sid)
    if s.empty:
        st.info("No measurements yet. Add them in the 'Data entry' or 'Files' tab.")
        return
    st.markdown("#### Key changes (baseline = pre-flight mean)")
    top = s.dropna(subset=["Change_pct"]).copy()
    top["abs"] = top["Change_pct"].abs()
    top = top.sort_values("abs", ascending=False).head(10)
    cols = st.columns(5)
    for i, (_, r) in enumerate(top.iterrows()):
        cols[i % 5].metric(r["Parameter"][:28], f"{r['Latest']:.3g} {r['Unit']}", f"{r['Change_pct']:+.1f} %",
                           delta_color="off")
    st.markdown("#### All parameters")
    st.dataframe(s.style.format({"Baseline": "{:.3g}", "Latest": "{:.3g}", "Change_pct": "{:+.1f}",
                                 "Expected_pct": "{:+.1f}", "Deviation": "{:+.1f}"}, na_rep="—"),
                 use_container_width=True, height=420)
    st.caption("Expected_pct: what the 180-day model based on NASA/Roscosmos/SpaceX literature expects on that day. "
               "Deviation: observed − expected (large positive/negative deviations show individual differences).")
    if st.session_state.get("mode") == "uzay":
        est = inflight_estimates(sid)
        if not est.empty:
            st.markdown("#### 🔭 Not measurable in space — today's model estimate")
            st.dataframe(est.style.format({"Baseline": "{:.3g}", "Estimated change %": "{:+.1f}",
                                           "Estimated value": "{:.3g}"}, na_rep="—"),
                         use_container_width=True, hide_index=True)
            st.caption("These values are not measurements; they are estimates from the pre-flight baseline + literature model "
                       "(personal risk multiplier included). Confirm them with DXA/QCT/MRI after return.")
    risks = evaluate_risks(sid)
    hi = risks[risks["Level"].isin(["High", "Moderate-High", "Review"])]
    if not hi.empty:
        st.markdown("#### ⚠️ Needs attention")
        st.dataframe(hi, use_container_width=True, hide_index=True)


def ui_data_entry(sid):
    subj = get_subject(sid)
    st.subheader("🧪 Biometric & clinical data entry")
    mode = st.session_state.get("mode", "dunya")
    ok = allowed_keys(mode, st.session_state.get("include_samples", False))
    if mode == "uzay":
        st.info("🚀 **Space mode:** Only parameters that can be measured with the equipment NASA keeps aboard the station on "
                "long-duration missions are listed. MRI, DXA, tilt tests and similar measurements are not possible in space, so "
                "they are not asked for; their current status is shown as a model estimate in the 'Dashboard' tab.")
    bad = impossible_inflight(sid)
    if not bad.empty:
        st.warning(f"⚠️ {len(bad)} records are entered as 'In-flight', but these measurements cannot be made in space "
                   f"(e.g. {bad.iloc[0]['parameter']}). Check the phase or the value.")
    # Category selection is outside the form so the parameter list refreshes at once when it changes
    cats = [c for c in CATALOG if any(p["key"] in ok for p in CATALOG[c])]
    c1, c2 = st.columns(2)
    cat = c1.selectbox("Category", cats + ([] if mode == "uzay" else ["Custom parameter"]), key="entry_cat")
    if cat == "Custom parameter":
        par = c2.text_input("Parameter name", key="entry_custom")
        unit_default = ""
    else:
        labels = [p["label"] for p in CATALOG[cat] if p["key"] in ok]
        par = c2.selectbox("Parameter", labels, key=f"entry_par_{cat}")
        unit_default = param_info(par)["unit"]
    pinfo = param_info(par) if par else None
    if pinfo:
        st.caption(f"{SPACE_LEVEL_LABEL[pinfo['space']]} · Device/method: {pinfo['device']}")
    subj_fd = mission_day_today(subj)
    with st.form("meas_form", clear_on_submit=False):
        c3, c4, c5, c6 = st.columns(4)
        val = c3.number_input("Value", value=0.0, format="%.4f")
        unit = c4.text_input("Unit", unit_default, key=f"unit_{par}")
        phase = c5.selectbox("Phase", [PHASES[1]] if mode == "uzay" else PHASES)
        day = c6.number_input("Day (L-60 → 60, FD30 → 30, R+5 → 5)", 0, 5000,
                              subj_fd if (mode == "uzay" and subj_fd) else 0)
        c7, c8 = st.columns(2)
        date = c7.date_input("Measurement date", dt.date.today())
        note = c8.text_input("Note / device / laboratory")
        info = param_info(par) if par else None
        if info and (info["lo"] is not None or info["hi"] is not None):
            st.caption(f"Approximate reference: {info['lo']} – {info['hi']} {info['unit']}"
                       + (f" · 180-day expectation: {info['p180']:+} % ({info['src']})" if info.get("p180") is not None else ""))
        if st.form_submit_button("➕ Save") and par:
            add_measurement(sid, par, val, unit, phase, day, date.isoformat(), "manual", note,
                            None if cat == "Custom parameter" else cat)
            st.success(f"Saved: {par} = {val} {unit}")

    st.markdown("#### Bulk edit table")
    st.caption("Edit cells, add/delete rows, then press 'Save changes'.")
    df = get_measurements(sid)
    cols = ["id", "category", "parameter", "value", "unit", "phase", "day", "date", "source", "note"]
    base = df[cols] if not df.empty else pd.DataFrame(columns=cols)
    edited = st.data_editor(
        base, num_rows="dynamic", use_container_width=True, height=380, key=f"editor_{sid}",
        column_config={"id": st.column_config.NumberColumn("id", disabled=True),
                       "phase": st.column_config.SelectboxColumn("faz", options=PHASES),
                       "parameter": st.column_config.TextColumn("parameter"),
                       "value": st.column_config.NumberColumn("value", format="%.4f")})
    c1, c2, c3 = st.columns(3)
    if c1.button("💾 Save changes"):
        q("DELETE FROM measurements WHERE subject_id=?", (sid,))
        n = 0

        def clean(v, default=None):
            return default if v is None or (isinstance(v, float) and math.isnan(v)) or v == "" else v

        for _, r in edited.iterrows():
            if clean(r.get("parameter")) is None or clean(r.get("value")) is None:
                continue
            ph = clean(r.get("phase"), PHASES[0])
            add_measurement(sid, str(r["parameter"]), float(r["value"]), clean(r.get("unit"), ""),
                            ph if ph in PHASES else PHASES[0], int(clean(r.get("day"), 0)),
                            clean(r.get("date")) and str(r.get("date")), clean(r.get("source"), "table"),
                            clean(r.get("note"), ""), clean(r.get("category")))
            n += 1
        st.success(f"{n} rows saved.")
        st.rerun()
    tmpl = pd.DataFrame([{"parameter": p["label"], "value": "", "unit": p["unit"],
                          "phase": "FD" if mode == "uzay" else "L", "day": (subj_fd or 30) if mode == "uzay" else 60,
                          "date": ""} for k, p in PARAM_INDEX.items() if k in ok])
    c2.download_button("⬇️ Empty CSV template", tmpl.to_csv(index=False).encode("utf-8-sig"), "istikbal_template.csv")
    c3.download_button("⬇️ This person's data (CSV)", base.to_csv(index=False).encode("utf-8-sig"),
                       f"istikbal_{subj['name']}.csv")

    with st.expander("🛰️ Which measurement can be made where? (based on NASA station hardware)"):
        st.dataframe(space_table(), use_container_width=True, hide_index=True)

    with st.expander("✏️ Edit / delete person details"):
        with st.form("edit_subj"):
            c1, c2, c3 = st.columns(3)
            md = c1.number_input("Mission length (days)", 0, 1500, int(subj.get("mission_days") or 180))
            dose = c2.number_input("Previous career dose (mSv)", 0.0, 2000.0, float(subj.get("prior_dose_msv") or 0))
            ld = c3.text_input("Launch date (YYYY-MM-DD)", subj.get("launch_date") or "")
            notes = st.text_area("Notes (medical history, medications, goals)", subj.get("notes") or "")
            if st.form_submit_button("Update"):
                update_subject(sid, mission_days=md, prior_dose_msv=dose, launch_date=ld, notes=notes)
                st.success("Updated")
        if st.button("🗑️ Delete this person and all their data", type="secondary"):
            st.session_state["confirm_del"] = True
        if st.session_state.get("confirm_del"):
            st.warning("Are you sure? This cannot be undone.")
            if st.button("Yes, delete permanently"):
                delete_subject(sid)
                st.session_state.pop("sid", None)
                st.session_state["confirm_del"] = False
                st.rerun()


def ui_files(sid, cfg):
    st.subheader("📁 File upload and analysis")
    st.caption("Drag and drop PDF, CSV, Excel, TXT, image (PNG/JPG/TIFF) and DICOM files.")
    ups = st.file_uploader("Drop files here", accept_multiple_files=True,
                           type=["pdf", "csv", "tsv", "xlsx", "xls", "txt", "md", "json", "png", "jpg", "jpeg",
                                 "tif", "tiff", "bmp", "webp", "dcm"], key=f"up_{sid}")
    if ups and st.button("📥 Process uploads"):
        for u in ups:
            path, kind, text, table = extract_file(u)
            did = q("INSERT INTO documents(subject_id,filename,filetype,path,text,created_at) VALUES(?,?,?,?,?,?)",
                    (sid, u.name, kind, path, text, now()))
            msg = f"✅ {u.name} saved ({len(text)} characters)."
            if table is not None:
                norm = normalize_table(table)
                if norm is not None and not norm.empty:
                    for _, r in norm.iterrows():
                        add_measurement(sid, str(r["parameter"]), r["value"], r.get("unit", ""), r["phase"],
                                        r["day"], str(r.get("date") or "") or None, f"file:{u.name}")
                    msg += f" {len(norm)} measurements imported from the table."
            st.success(msg)

    docs = q("SELECT * FROM documents WHERE subject_id=? ORDER BY id DESC", (sid,), fetch=True)
    for d in docs:
        with st.expander(f"📄 {d['filename']} · {d['created_at']}"):
            if d["filetype"] == "image" and os.path.exists(d["path"]):
                st.image(d["path"], use_container_width=True)
                qtext = st.text_input("Question for the image", "Interpret this image/report clinically.", key=f"vq{d['id']}")
                if st.button("🔍 Image analysis (vision model)", key=f"va{d['id']}"):
                    with open(d["path"], "rb") as f:
                        b64 = base64.b64encode(f.read()).decode()
                    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": qtext + "\n\nNote: do not make a radiological diagnosis; write the observations and the recommended measurements."}]
                    ans = st.write_stream(llm_stream(msgs, cfg, images_b64=[b64], vision=True))
                    q("UPDATE documents SET text=? WHERE id=?", (f"[Image analysis]\n{ans}", d["id"]))
            else:
                st.text_area("Extracted text", d["text"][:20000], height=220, key=f"t{d['id']}")
                if st.button("🧠 Let İstikbal Göklerdedir summarise this document", key=f"s{d['id']}"):
                    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": "Analyse the document below: key findings, out-of-range values, "
                                                        "meaning for space medicine and recommended follow-up.\n\n" + d["text"][:12000]}]
                    st.write_stream(llm_stream(msgs, cfg))
            if st.button("🗑️ Delete", key=f"del{d['id']}"):
                q("DELETE FROM documents WHERE id=?", (d["id"],))
                st.rerun()


def ui_trends(sid):
    st.subheader("📈 Tracking over time (longitudinal)")
    df = get_measurements(sid)
    if df.empty:
        st.info("Measurements are needed for a chart.")
        return
    params = sorted(df["parameter"].unique())
    defaults = [p for p in ["Femoral neck aBMD (DXA)", "Serum CTX", "P1NP"] if p in params] or params[:2]
    sel = st.multiselect("Parameters", params, default=defaults)
    mode = st.radio("Display", ["% change (baseline) + literature expectation", "Absolute value"], horizontal=True)
    if sel:
        st.plotly_chart(trend_figure(sid, sel, pct=mode.startswith("%")), use_container_width=True)
    st.caption("Dotted lines: expected course derived from 180-day ISS cohorts (exponential recovery after landing). "
               "Distance between the observed value and the expectation shows the individual difference.")


_LLM_ERRORS = ("⚠️ Ollama error", "⚠️ Ollama:", "⚠️ API error", "⚠️ Anthropic error",
               "⚠️ Could not connect to Ollama", "⚠️ Could not connect to the API", "⚠️ AI error")


def is_llm_error(text):
    """Did a connection/error message arrive instead of an AI answer?"""
    t = (text or "").strip()
    return not t or t.startswith(_LLM_ERRORS)


def model_name(cfg):
    prov = cfg.get("provider", "")
    if prov == "Rule-based":
        return "Rule-based (no AI)"
    if prov.startswith("Ollama"):
        return f"Ollama · {cfg.get('ollama_model')}"
    if prov.startswith("OpenAI"):
        return f"OpenAI-compatible · {cfg.get('openai_model')}"
    return f"Anthropic · {cfg.get('anthropic_model')}"


def save_report(sid, content, cfg, scenario_days):
    q("INSERT INTO reports(subject_id,created_at,mode,scenario_days,model,content) VALUES(?,?,?,?,?,?)",
      (sid, now(), cfg.get("_mode", "dunya"), int(scenario_days), model_name(cfg), content))
    return q("SELECT MAX(id) i FROM reports WHERE subject_id=?", (sid,), fetch=True)[0]["i"]


def report_markdown(r, subj_name):
    mode = "Space (on mission)" if r["mode"] == "uzay" else "Earth"
    return (f"# İstikbal Göklerdedir — Personalised risk report #{r['id']}\n\n"
            f"- Person: {subj_name}\n- Date: {str(r['created_at']).replace('T', ' ')}\n- Mode: {mode}\n"
            f"- Mission scenario: {r['scenario_days']} days\n- AI: {r['model']}\n\n"
            "> This report is decision-support output; diagnosis and treatment decisions belong to the flight surgeon.\n\n---\n\n"
            + r["content"])


REPORT_INSTRUCTION = ("For a {md}-day mission scenario, write a concise personalised risk report for this person: "
                      "1) the 3-5 most important risks with confidence levels (short/medium/long term), "
                      "2) personalised preventive recommendations, 3) a monitoring schedule for pre-flight, in-flight and "
                      "post-return. Use headings and bullet points; at most about 600 words.")


def build_report_context(sid, cfg, md):
    """Short context for the risk report. A local model on an ordinary laptop processes every token of the prompt
    before it writes a word, so the report sends only what matters: profile, mode, notable measurements, the rule-based
    risks and at most 3 short paper abstracts (about 1,500 tokens instead of 5,000+)."""
    subj = get_subject(sid) or {}
    prof = {k: subj.get(k) for k in ("role", "sex", "birth_year", "mission", "mission_days", "launch_date", "prior_dose_msv", "notes")
            if subj.get(k) not in (None, "")}
    parts = [f"## Subject profile\n{json.dumps(prof, ensure_ascii=False)}\nScenario mission length: {md} days"]
    mode = cfg.get("_mode", "dunya")
    if mode == "uzay":
        parts.append("## Working mode: SPACE (on mission). Recommend only measurements possible on board; DXA, MRI, tilt "
                     "and similar tests are model estimates to confirm after return.")
    else:
        parts.append("## Working mode: EARTH (pre-/post-flight or ground laboratory)")
    summ = summarize(sid)
    if not summ.empty:
        s2 = summ.copy()
        s2["_score"] = s2["Change_pct"].abs().fillna(0) + s2["Deviation"].abs().fillna(0) + (s2["Flag"] != "") * 100
        s2 = s2.sort_values("_score", ascending=False).head(18)
        lines = [f"- {r['Parameter']}: baseline {r['Baseline']:.3g}, latest {r['Latest']:.3g} {r['Unit']} "
                 f"({r['Latest_phase']}), change {r['Change_pct']:+.1f} %"
                 + (f", expected {r['Expected_pct']:+.1f} %" if pd.notna(r['Expected_pct']) else "")
                 + (f", {r['Flag']}" if r['Flag'] else "")
                 for _, r in s2.iterrows() if pd.notna(r['Baseline'])]
        parts.append("## Most notable measurements (baseline = pre-flight mean)\n" + "\n".join(lines))
    risks = evaluate_risks(sid)
    if not risks.empty:
        parts.append("## Rule-based risk engine output\n" + "\n".join(
            f"- {r['Risk']} [{r['Level']}]: {str(r['Rationale'])[:220]} (source: {r['Source']})" for _, r in risks.iterrows()))
    if cfg.get("use_library_in_chat", True):
        arts = library_search("risk prediction bone cardiovascular immune sleep countermeasure", k=3)
        if arts:
            parts.append("## Current literature (local library)\n" + "\n".join(
                f"- [{a['pub_date']}] {a['title']} ({a['journal']}) {a['url']}\n  {a['abstract'][:300]}" for a in arts))
    return "\n\n".join(parts)


def rule_based_report(sid, md, mode="dunya"):
    """Instant report without AI: rule-based risks + literature projection, in Markdown."""
    risks = evaluate_risks(sid)
    pt = projection_table(sid, md)
    summ = summarize(sid)
    out = [f"## Summary\nScenario: {md}-day mission. Generated instantly from the rule-based risk engine and the "
           "literature model (no AI text).\n"]
    if mode == "uzay":
        out.append("_Space mode: DXA, MRI, tilt and similar values are model estimates; confirm them after return._\n")
    order = ["High", "Moderate-High", "Moderate", "Review", "Low"]
    if not risks.empty:
        out.append("## Risks and actions")
        rk = risks.assign(_o=risks["Level"].map({k: i for i, k in enumerate(order)}).fillna(9)).sort_values("_o")
        for _, r in rk.iterrows():
            out.append(f"### {r['Risk']} — **{r['Level']}**\n- Why: {r['Rationale']}\n- Action: {r['Action']}\n- Source: {r['Source']}")
    if not summ.empty:
        fl = summ[summ["Flag"] != ""]
        if not fl.empty:
            out.append("## Out-of-range values\n" + "\n".join(
                f"- {r['Parameter']}: {r['Latest']:.3g} {r['Unit']} ({r['Flag']}, reference {r['Reference']})" for _, r in fl.iterrows()))
    if not pt.empty:
        out.append("## Literature projection (R+0 = landing day, R+365 = one year later)\n"
                   "| Parameter | R+0 expected % | R+365 expected % |\n|---|---|---|\n" + "\n".join(
                       f"| {r['Parameter']} | {r['R+0 expected %']:+.1f} | {r['R+365 expected %']:+.1f} |"
                       for _, r in pt.iterrows() if pd.notna(r['R+0 expected %'])))
    out.append("## Monitoring schedule (general)\n"
               "- Pre-flight: full baseline (DXA/QCT/HR-pQCT, MRI, OCT, blood/urine panel, questionnaires).\n"
               "- In-flight: monthly ultrasound/OCT/blood pressure/ECG, weekly sleep and questionnaires, saliva virus PCR if available.\n"
               "- Post-flight: R+0, R+30, R+180, R+365 repeat of the baseline tests; bone follow-up until recovery.")
    return "\n\n".join(out)


def ui_report_history(sid, just_saved=None):
    """Saved risk reports: open, download, delete."""
    reps = q("SELECT * FROM reports WHERE subject_id=? ORDER BY id DESC", (sid,), fetch=True)
    st.markdown(f"#### 🗂️ Past reports ({len(reps)})")
    if not reps:
        st.caption("No saved reports yet. When you generate a report with the button above, it is stored here.")
        return
    name = (get_subject(sid) or {}).get("name", "")
    for i, r in enumerate(reps):
        mode = "🚀 Space" if r["mode"] == "uzay" else "🌍 Earth"
        when = str(r["created_at"]).replace("T", " ")[:16]
        with st.expander(f"#{r['id']} · {when} · {mode} · {r['scenario_days']} days · {r['model']}",
                         expanded=(i == 0 and r["id"] != just_saved)):
            st.markdown(r["content"])
            c1, c2 = st.columns([3, 1])
            c1.download_button("⬇️ Download report (.md)", report_markdown(r, name).encode("utf-8"),
                               f"istikbal_risk_report_{r['id']}.md", key=f"dl_rep_{r['id']}")
            if c2.button("🗑️ Delete", key=f"del_rep_{r['id']}"):
                q("DELETE FROM reports WHERE id=?", (r["id"],))
                st.rerun()


def ui_risk(sid, cfg):
    subj = get_subject(sid)
    st.subheader("🔮 Risk prediction and projection")
    md = st.slider("Scenario mission length (days)", 7, 1000, int(subj.get("mission_days") or 180), 1)
    pt = projection_table(sid, md)
    st.markdown("#### Literature-based projection")
    st.dataframe(pt.style.format({"Baseline": "{:.3g}", "R+0 expected %": "{:+.1f}", "R+0 expected value": "{:.3g}",
                                  "R+365 expected %": "{:+.1f}", "Personal multiplier": "{:.2f}"}, na_rep="—"),
                 use_container_width=True, hide_index=True)
    st.caption("Model: bone loss is linear in flight, markers/fluids reach a plateau; after landing, exponential recovery with a half-life. "
               "High pre-flight CTX/P1NP increases bone loss ×1.3 (Gabel 2021). For missions longer than 180 days the linear "
               "extrapolation is uncertain.")
    st.markdown("#### Rule-based risk assessment")
    st.dataframe(evaluate_risks(sid), use_container_width=True, hide_index=True)
    st.caption("Immune and sleep/behavior rows are for screening and do not make a diagnosis. Most thresholds are published "
               "clinical limits (AASM ≥ 7 h, PSQI > 5, PHQ-9 ≥ 10, ANC < 1.5, ALC < 1.0); thresholds marked 'project heuristic' "
               "are not defined in the literature and should be read against the person's own baseline.")
    rid = None
    b1, b2 = st.columns(2)
    quick = b1.button("📝 Instant report (rule-based, no AI, ready in a second)", use_container_width=True)
    ai = b2.button("🧠 Let İstikbal Göklerdedir write a personalised risk report (AI)", use_container_width=True)
    if quick:
        rep_txt = rule_based_report(sid, md, cfg.get("_mode", "dunya"))
        rid = save_report(sid, rep_txt, dict(cfg, provider="Rule-based"), md)
        st.success(f"✅ Instant report saved (#{rid}). Open it in 'Past reports' below.")
    if ai:
        if cfg["provider"].startswith("Ollama"):
            st.info("⏳ The local AI model is writing the report on this computer. Without a graphics card this usually takes "
                    "3–10 minutes; text appears once the model has read the data. **Do not click anything else on the page "
                    "until it finishes** — any click restarts the page and stops the report. Faster option: choose the "
                    "smaller model `llama3.2:3b` in the left menu, or use the instant report.")
        ctx = build_report_context(sid, cfg, md)
        msgs = [{"role": "system", "content": SYSTEM_PROMPT + "\n\n" + ctx},
                {"role": "user", "content": REPORT_INSTRUCTION.format(md=md)}]
        t0 = time.time()
        with st.spinner("The model is reading the data…"):
            gen = llm_stream(msgs, cfg, max_tokens=1000, num_ctx=4096)
            first = next(gen, "")
        ans = st.write_stream(itertools.chain([first], gen))
        ans = ans if isinstance(ans, str) else "".join(map(str, ans))
        st.caption(f"Written in {(time.time() - t0) / 60:.1f} min.")
        if is_llm_error(ans):
            st.error("The report could not be written, so it was not saved. Check the AI settings.")
        else:
            rid = save_report(sid, ans, cfg, md)
            st.success(f"✅ Report saved (#{rid}). You can always open it from the 'Past reports' section below.")
    ui_report_history(sid, just_saved=rid)


def ui_research(sid, cfg):
    st.subheader("🌐 Autonomous web research (NASA · SpaceX · Roscosmos)")
    if DDGS is None:
        st.error("Search module missing: `pip install ddgs`")
    query = st.text_input("Research topic", "long-duration spaceflight bone loss ARED CTX P1NP")
    chosen = st.multiselect("Source domains", list(TRUSTED_DOMAINS.keys()),
                            default=["NASA", "NASA NTRS", "PubMed / PMC", "SpaceX", "Roscosmos"])
    free = st.toggle("Also run an unrestricted general search", False)
    if st.button("🔎 Search"):
        domains = [TRUSTED_DOMAINS[c] for c in chosen]
        with st.spinner("Searching..."):
            res = web_search(query, domains, cfg["max_search_results"])
            if free:
                res += web_search(query, None, cfg["max_search_results"])
        st.session_state["last_research"] = (query, res)
        save_research(sid, query, res)
    if "last_research" in st.session_state:
        query_, res = st.session_state["last_research"]
        for r in res:
            st.markdown(f"**[{r['title']}]({r['url']})**  \n{r['snippet']}")
        if st.button("🧠 Let İstikbal Göklerdedir synthesise the findings"):
            msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Topic: {query_}\n\nCritically synthesise the search results below; "
                                                "rely only on these sources, give a URL for every claim, and state contradictions "
                                                "and the level of evidence.\n\n" + "\n".join(
                        f"- {r['title']} | {r['url']} | {r['snippet']}" for r in res)}]
            st.session_state["synth"] = (query_, st.write_stream(llm_stream(msgs, cfg)))
        elif st.session_state.get("synth") and st.session_state["synth"][0] == query_:
            st.markdown(st.session_state["synth"][1])
        if st.session_state.get("synth") and st.session_state["synth"][0] == query_:
            if st.button("💾 Add synthesis to memory"):
                q("INSERT INTO memory(subject_id,kind,content,created_at) VALUES(?,?,?,?)",
                  (sid, "research", f"{query_}: {st.session_state['synth'][1][:1500]}", now()))
                st.toast("Added to memory")
    past = qdf("SELECT created_at, query, title, url FROM research WHERE subject_id=? ORDER BY id DESC LIMIT 200", (sid,))
    if not past.empty:
        with st.expander("📚 Past research"):
            st.dataframe(past, use_container_width=True, hide_index=True)


def ui_chat(sid, cfg):
    st.subheader("💬 Chat with İstikbal Göklerdedir")
    c1, c2, c3, c4 = st.columns(4)
    cfg["web_search_in_chat"] = c1.toggle("🌐 Web research", cfg["web_search_in_chat"])
    cfg["use_docs_in_chat"] = c2.toggle("📄 Use documents", cfg["use_docs_in_chat"])
    cfg["speak_answers"] = c3.toggle("🔊 Read answers aloud", cfg["speak_answers"])
    if c4.button("🧹 Clear chat"):
        q("DELETE FROM chats WHERE subject_id=?", (sid,))
        st.rerun()

    history = q("SELECT role, content FROM chats WHERE subject_id=? ORDER BY id", (sid,), fetch=True)
    box = st.container(height=520)
    with box:
        for m in history:
            with st.chat_message(m["role"], avatar="🛰️" if m["role"] == "assistant" else "🧑‍🚀"):
                st.markdown(m["content"])

    voice_text = None
    if hasattr(st, "audio_input"):
        audio = st.audio_input("🎙️ Voice question (optional)", key=f"aud_{sid}")
        if audio is not None and st.session_state.get("last_audio") != audio.size:
            st.session_state["last_audio"] = audio.size
            if WhisperModel is None:
                st.info("For voice input: `pip install faster-whisper` (the model is downloaded on first use).")
            else:
                with st.spinner("Transcribing audio..."):
                    voice_text = transcribe(audio.getvalue())
                st.caption(f"Detected: {voice_text}")

    prompt = st.chat_input("Ask the assistant… (e.g. 'Interpret the bone changes between baseline and R+0')")
    prompt = prompt or voice_text
    if not prompt:
        return

    q("INSERT INTO chats(subject_id,role,content,created_at) VALUES(?,?,?,?)", (sid, "user", prompt, now()))
    if prompt.lower().startswith(("/remember", "/hatirla", "/hatırla")):
        fact = prompt.split(" ", 1)[1] if " " in prompt else ""
        q("INSERT INTO memory(subject_id,kind,content,created_at) VALUES(?,?,?,?)", (sid, "user", fact, now()))
        q("INSERT INTO chats(subject_id,role,content,created_at) VALUES(?,?,?,?)",
          (sid, "assistant", f"Saved to memory: {fact}", now()))
        st.rerun()

    web = None
    if cfg["web_search_in_chat"]:
        with st.spinner("Scanning NASA / SpaceX / Roscosmos sources..."):
            web = web_search(" ".join(english_keywords(prompt)[:6] or [prompt]) + " spaceflight astronaut",
                             ["nasa.gov", "ntrs.nasa.gov", "ncbi.nlm.nih.gov", "spacex.com", "roscosmos.ru"],
                             max(2, cfg["max_search_results"] // 2))
            save_research(sid, prompt, web)
    context = build_context(sid, prompt, cfg, web)
    msgs = [{"role": "system", "content": SYSTEM_PROMPT + "\n\n# CONTEXT\n" + context}]
    msgs += [{"role": m["role"], "content": m["content"]} for m in history[-12:]]
    msgs.append({"role": "user", "content": prompt})

    with box:
        with st.chat_message("user", avatar="🧑‍🚀"):
            st.markdown(prompt)
        with st.chat_message("assistant", avatar="🛰️"):
            answer = st.write_stream(llm_stream(msgs, cfg))
            if web:
                with st.expander("Web sources used"):
                    for r in web:
                        if r.get("url"):
                            st.markdown(f"- [{r['title']}]({r['url']})")
    q("INSERT INTO chats(subject_id,role,content,created_at) VALUES(?,?,?,?)", (sid, "assistant", answer, now()))
    if cfg["speak_answers"] and components is not None:
        speak(answer)
    if cfg["auto_memory"] and not answer.startswith("⚠️"):
        facts = learn_from_exchange(sid, prompt, answer, cfg)
        if facts:
            st.toast("Added to memory: " + "; ".join(facts)[:200])


def ui_library(cfg):
    st.subheader("📚 Current literature library")
    st.caption("Every time the program opens, it downloads space medicine papers published on PubMed and NASA NTRS since the "
               "last update. The chat and the risk report read the newest papers relevant to your question when answering.")
    n = q("SELECT COUNT(*) n FROM articles", fetch=True)[0]["n"]
    c = st.columns(4)
    c[0].metric("Papers in library", n)
    last = meta_get("library_last_update")
    c[1].metric("Last update", (last or "not yet").replace("T", " ")[:16])
    c[2].metric("Added in last update", meta_get("library_last_new", "0"))
    lm = meta_get("model_last_check")
    c[3].metric("Model last checked", (lm or "not yet").replace("T", " ")[:16])

    b1, b2 = st.columns(2)
    if b1.button("🔄 Update library now"):
        with st.status("Downloading new papers…", expanded=True) as stt:
            res = update_library(cfg, log=stt.write, force=True)
            stt.update(label=f"✅ {res['new']} new papers added", state="complete")
    if cfg["provider"].startswith("Ollama") and b2.button("🧠 Update AI model"):
        run_model_update(cfg)

    st.markdown("#### Search the library")
    term = st.text_input("Topic (English or Turkish)", "bone loss recovery")
    if term:
        res = library_search(term, k=20)
        st.caption(f"Search keywords: {', '.join(english_keywords(term)) or '—'} · {len(res)} results")
        for a in res:
            with st.expander(f"[{a['pub_date']}] {a['title'][:140]}"):
                st.markdown(f"**{a['journal']}** · {a['source']} · topic: {a['topic']} · [go to source]({a['url']})")
                st.write(a["abstract"] or "_No abstract_")
    with st.expander("📋 50 most recently added papers"):
        st.dataframe(qdf("SELECT added_at, pub_date, source, topic, title, url FROM articles ORDER BY id DESC LIMIT 50"),
                     use_container_width=True, hide_index=True)

    with st.expander("⚙️ Update settings and topics"):
        cfg["auto_update_library"] = st.toggle("Update automatically at start-up", cfg["auto_update_library"])
        cfg["library_update_hours"] = st.number_input("At most once every (hours)", 1, 168, int(cfg["library_update_hours"]))
        cfg["library_per_topic"] = st.number_input("Maximum new papers per topic", 5, 200, int(cfg["library_per_topic"]))
        cfg["auto_model_update"] = st.toggle("Check for model updates at start-up (once a day)", cfg["auto_model_update"])
        cfg["use_library_in_chat"] = st.toggle("Use the library in chat", cfg["use_library_in_chat"])
        cfg["pubmed_email"] = st.text_input("E-mail (optional, recommended by PubMed)", cfg.get("pubmed_email", ""))
        topics = pd.DataFrame([{"topic": k, "pubmed_query": v["pubmed"], "ntrs_query": v.get("ntrs", "")}
                               for k, v in cfg["library_topics"].items()])
        ed = st.data_editor(topics, num_rows="dynamic", use_container_width=True, key="topics_ed")
        if st.button("💾 Save settings and topics"):
            cfg["library_topics"] = {r["topic"]: {"pubmed": r["pubmed_query"], "ntrs": r["ntrs_query"] or r["topic"]}
                                     for _, r in ed.iterrows() if isinstance(r["topic"], str) and r["topic"].strip()
                                     and isinstance(r["pubmed_query"], str) and r["pubmed_query"].strip()}
            save_settings(cfg)
            st.toast("Saved")
        st.caption("For newer models in Ollama: https://ollama.com/library · to download, run "
                   "`ollama pull <model name>` in a terminal, then choose the model in the left menu.")


def run_model_update(cfg):
    """Updates the Ollama model; shows progress."""
    with st.status(f"🧠 Checking for a new version of the '{cfg['ollama_model']}' model…", expanded=False) as stt:
        bar = st.progress(0.0)
        try:
            final = ""
            for ev in ollama_pull(cfg):
                if "error" in ev:
                    raise RuntimeError(ev["error"])
                final = ev.get("status", final)
                if ev.get("total"):
                    bar.progress(min(1.0, ev.get("completed", 0) / ev["total"]), text=final)
            meta_set("model_last_check", now())
            stt.update(label=f"🧠 Model up to date: {cfg['ollama_model']} ({final})", state="complete")
        except requests.exceptions.ConnectionError:
            stt.update(label="🧠 Ollama is not running; model check skipped", state="error")
        except Exception as e:
            stt.update(label=f"🧠 Model check skipped: {str(e)[:120]}", state="error")


def startup_updates(cfg):
    """Once per session: literature library and model update."""
    if st.session_state.get("startup_done"):
        return
    st.session_state["startup_done"] = True
    if cfg.get("auto_update_library") and hours_since(meta_get("library_last_update")) >= float(cfg["library_update_hours"]):
        with st.status("📚 Downloading current space medicine literature…", expanded=False) as stt:
            res = update_library(cfg, log=stt.write)
            if res.get("errors") and not res["new"]:
                stt.update(label="📚 Literature could not be updated (maybe no internet); the existing library will be used",
                           state="error")
            else:
                stt.update(label=f"📚 Library updated: {res['new']} new papers", state="complete")
    if (cfg.get("auto_model_update") and cfg["provider"].startswith("Ollama")
            and hours_since(meta_get("model_last_check")) >= float(cfg.get("model_check_hours", 24))
            and ollama_models(cfg["ollama_url"])):
        run_model_update(cfg)


def ui_memory(sid):
    st.subheader("🧠 Learning memory")
    st.caption("İstikbal Göklerdedir uses this information in every answer. You can also add to it by typing '/remember <fact>' in the chat.")
    with st.form("mem_form", clear_on_submit=True):
        c1, c2 = st.columns([1, 4])
        kind = c1.selectbox("Kind", ["preference", "medical history", "mission", "goal", "general"])
        txt = c2.text_input("Fact")
        scope = st.checkbox("Applies to all people (general user preference)")
        if st.form_submit_button("Add") and txt.strip():
            q("INSERT INTO memory(subject_id,kind,content,created_at) VALUES(?,?,?,?)",
              (None if scope else sid, kind, txt.strip(), now()))
    mem = q("SELECT * FROM memory WHERE subject_id=? OR subject_id IS NULL ORDER BY id DESC", (sid,), fetch=True)
    for m in mem:
        c1, c2 = st.columns([10, 1])
        c1.markdown(f"**[{m['kind']}]** {m['content']}  \n<small>{m['created_at']}"
                    f"{' · general' if m['subject_id'] is None else ''}</small>", unsafe_allow_html=True)
        if c2.button("🗑️", key=f"m{m['id']}"):
            q("DELETE FROM memory WHERE id=?", (m["id"],))
            st.rerun()

    st.markdown("---")
    st.markdown("#### 💾 Backup")
    if st.button("Prepare the whole database as JSON"):
        dump = {t: q(f"SELECT * FROM {t}", fetch=True)
                for t in ["subjects", "measurements", "documents", "chats", "memory", "research", "reports"]}
        st.download_button("⬇️ istikbal_goklerdedir_backup.json", json.dumps(dump, ensure_ascii=False, indent=1).encode("utf-8"),
                           "istikbal_goklerdedir_backup.json")


def ui_mode_select():
    """Start screen: is the user in space or on Earth?"""
    st.markdown("<h1 style='text-align:center'>🛰️ İstikbal Göklerdedir</h1>"
                "<h3 style='text-align:center'>Where are you right now?</h3>", unsafe_allow_html=True)
    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🚀 I'm in space")
        st.write("I am on a mission. Only values that can be measured with the devices NASA keeps aboard the station on long missions "
                 "(ECG, blood-pressure device, ultrasound, OCT, tonometer, body-mass device, actigraph, dosimeter…) "
                 "are asked for. Measurements such as MRI and DXA are not requested; they are estimated by the model.")
        if st.button("🚀 I'm in space", use_container_width=True, type="primary"):
            st.session_state["mode"] = "uzay"
            st.rerun()
    with c2:
        st.markdown("### 🌍 I'm on Earth")
        st.write("I am pre-flight, post-flight or in a ground laboratory. All measurements, including MRI, DXA, QCT and "
                 "neurophysiology, can be entered.")
        if st.button("🌍 I'm on Earth", use_container_width=True):
            st.session_state["mode"] = "dunya"
            st.rerun()
    with st.expander("🛰️ Which measurement can be made where? (based on NASA station hardware)"):
        st.dataframe(space_table(), use_container_width=True, hide_index=True)


def mission_day_today(subj):
    """Today's flight day (FD) from the launch date; None if unknown."""
    try:
        d = (dt.date.today() - dt.date.fromisoformat(str(subj.get("launch_date") or ""))).days
        return d if 0 < d <= int(subj.get("mission_days") or 180) else None
    except ValueError:
        return None


def impossible_inflight(sid):
    """Records entered as 'In-flight' that cannot be measured in space (data quality warning)."""
    df = get_measurements(sid)
    if df.empty:
        return pd.DataFrame()
    bad = df[df["phase"].str.startswith("In-flight", na=False)].copy()
    bad["lvl"] = [(param_info(p) or {}).get("space") for p in bad["parameter"]]
    return bad[bad["lvl"] == "dunya"][["parameter", "value", "unit", "day"]]


def main():
    st.set_page_config(page_title="İstikbal Göklerdedir · Space Medicine AI", page_icon="🛰️", layout="wide")
    init_db()
    ensure_demo_subject()
    if "cfg" not in st.session_state:
        st.session_state["cfg"] = load_settings()
    cfg = st.session_state["cfg"]
    if "mode" not in st.session_state:
        ui_mode_select()
        return
    cfg["_mode"] = st.session_state["mode"]
    cfg["_include_samples"] = st.session_state.get("include_samples", False)
    ui_sidebar(cfg)
    startup_updates(cfg)
    sid = st.session_state.get("sid")
    if not sid or not get_subject(sid):
        st.title("🛰️ İstikbal Göklerdedir")
        st.info("Create a person/astronaut in the left menu.")
        return
    tabs = st.tabs(["📊 Dashboard", "🧪 Data entry", "📁 Files", "📈 Trends", "🔮 Risk prediction",
                    "🌐 Research", "💬 Chat", "📚 Library", "🧠 Memory"])
    with tabs[0]:
        ui_dashboard(sid)
    with tabs[1]:
        ui_data_entry(sid)
    with tabs[2]:
        ui_files(sid, cfg)
    with tabs[3]:
        ui_trends(sid)
    with tabs[4]:
        ui_risk(sid, cfg)
    with tabs[5]:
        ui_research(sid, cfg)
    with tabs[6]:
        ui_chat(sid, cfg)
    with tabs[7]:
        ui_library(cfg)
    with tabs[8]:
        ui_memory(sid)


if __name__ == "__main__":
    if st is None:
        raise SystemExit("Streamlit is not installed. First run:  pip install -r requirements.txt  then:  streamlit run app.py")
    main()

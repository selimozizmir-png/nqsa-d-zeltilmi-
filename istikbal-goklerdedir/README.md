# 🛰️ İstikbal Göklerdedir: "The Future Is in the Skies"

**A self-monitoring health assistant for astronauts on long-duration missions.** It runs locally, is grounded in published research, and knows which measurements can actually be made in space.

*NASA Space Apps Challenge 2026*

> 📚 Full reference list (APA 7): [docs/References.pdf](docs/References.pdf)

> ⚠️ **Decision-support tool, not a medical device.** It does not diagnose or treat. Clinical decisions belong to the flight surgeon.

---

## The problem

Long-duration missions expose astronauts to five hazards: space radiation, isolation and confinement, distance from Earth, altered gravity, and a hostile, closed environment. These hazards drive changes in the **immune system**, **bone**, the **cardiovascular system** and **behavioral health**. Far from Earth, the crew has to detect many of these changes **themselves**.

**İstikbal Göklerdedir** collects health indicators, compares them with the astronaut's own pre-flight baseline and with published ISS cohort data, and flags risks. Each flag comes with concrete steps the astronaut can take on board.

## Key ideas

### 1. "Are you in space or on Earth?"

The first screen asks this question.

| Mode | What it asks for |
|---|---|
| 🚀 **In space** | Only the **27 parameters that can be measured with equipment NASA keeps aboard the ISS** |
| 🌍 **On Earth** | All **74 parameters**, including DXA, QCT, HR-pQCT, MRI, tilt/LBNP and clinical neurophysiology |

Onboard equipment used in space mode:

- **Crew Health Care System (CHeCS):** ECG and blood-pressure unit; tonometer in the medical diagnostic pack.
- **Imaging:** ISS ultrasound for echocardiography, carotid and optic-nerve-sheath measurements; OCT, aboard since 2013.
- **Body and exercise:**
  - SLAMMD / BMMD for body mass
  - CEVIS + PPFS for VO₂
  - ESA's MARES dynamometer for knee torque
- **Monitoring:** Personal dosimeters, wrist actigraphy, laptop questionnaires and the NASA *Cognition* battery.
- **Experimental:** Point-of-care blood counts (HemoCue WBC DIFF, rHEALTH ONE), clearly flagged as experimental.

The ISS has **no MRI or X-ray**. In space mode the app therefore never asks for DXA or MRI. It instead shows a **model estimate** of today's value, such as the predicted femoral-neck BMD change on flight day 95, and recommends confirming it after landing.

Some lab results come from samples that are frozen and analysed on Earth. These can optionally be entered when results are uplinked ("📦 lab results from Earth"). This option is off by default, because on a Mars mission this route may not exist.

The AI assistant follows the same rule. In space mode it only asks the astronaut for measurements that are possible on board.

### 2. Personal baseline + literature model

Each of the 74 parameters has a reference range, an expected 180-day change and a recovery half-life. These values come from ISS cohorts with ~2 h/day ARED exercise, for example English 2020, Gabel 2022, Sibonga 2019, Stavnichuk 2020, Moore 2014 and Hupfeld 2020.

The app compares each new value with:
- the astronaut's **own pre-flight baseline**,
- the **expected trajectory** for that flight day,
- the **reference range**.

A personal risk multiplier is applied when pre-flight bone turnover is high (CTX > 0.50 or P1NP > 70).

### 3. Transparent risk engine

The rule-based risk engine covers:
- bone loss and fracture
- renal stones
- orthostatic intolerance
- SANS
- radiation
- muscle and aerobic deconditioning
- immune dysregulation and latent-virus reactivation (VZV, EBV, CMV)
- sleep deficiency and behavioral health

Most thresholds are published clinical cut-offs: sleep ≥ 7 h (AASM/SRS), PSQI > 5, PHQ-9 ≥ 10, ANC < 1.5 and ALC < 1.0 ×10³/µL. A few thresholds are not defined in the literature; these are labelled **"project heuristic"** on screen. Every risk line names its sources.

### 4. Self-updating knowledge

On start-up, at most every 12 h, the app downloads new space-medicine papers from **PubMed** (NCBI E-utilities) and the **NASA Technical Reports Server**. The AI cites them when answering, so answers stay current even when the language model's own knowledge is older.

### 5. Private by design

The app runs locally with **Ollama** (Llama 3.1). No data leaves the computer. Optionally, it can use an OpenAI-compatible API or the Anthropic API. All data stays in a local SQLite file.

## Features (tabs)

| Tab | What it does |
|---|---|
| 📊 Dashboard | Baseline vs. latest values, deviation from expected trajectory, out-of-range flags. In space mode it adds model estimates for parameters that cannot be measured in flight |
| 🧪 Data entry | Mode-filtered form for 74 parameters. Shows the onboard device for each one, warns about impossible "in-flight MRI/DXA" entries, and offers CSV templates |
| 📁 Files | Upload PDF / CSV / Excel / TXT / images / DICOM; tables are imported automatically |
| 📈 Trends | Time series with baseline, reference band and literature-expected curve |
| 🔮 Risk prediction | Mission-length scenario, R+0 / R+365 projection, rule-based risks and an AI-written personal risk report. Reports are **saved** under "Past reports" and can be downloaded |
| 🌐 Research | Web search restricted to NASA, NTRS, PubMed, SpaceX, Roscosmos and IBMP |
| 💬 Chat | Text or voice chat grounded in the astronaut's data, documents and the literature library |
| 📚 Library | Downloaded PubMed / NTRS papers, topic settings and model updates |
| 🧠 Memory | What the assistant has learned, plus full JSON backup |

## Quick start

1. Install **Python 3.12** from [python.org](https://www.python.org/downloads/). On Windows, tick **"Add Python to PATH"**.
2. Install **Ollama** from [ollama.com](https://ollama.com/download), then run `ollama pull llama3.1`. On computers without a GPU, `llama3.2:3b` is faster.
3. Download this repository: **Code → Download ZIP**, then unzip it.
4. Start the app:
   - **Windows:** double-click `start.bat`.
   - **macOS / Linux:** run `bash start.sh`.
   
   The first run installs the packages, then the app opens at **http://localhost:8501**.
5. Choose 🌍 **On Earth**, go to **Files** and upload `example_case_astronaut_K.csv`. This is a fully worked, fictional 180-day ISS case with 507 measurements.
6. Open **Dashboard**, **Trends** and **Risk prediction**, then switch to 🚀 **In space** to see the in-flight view.

Uploaded tables need the columns `parameter, value, unit, phase, day, date` (`phase` = L / FD / R).

Manual install:
```
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Example case: "Astronaut K"

`example_case_astronaut_K.csv` follows a fictional 41-year-old male through a 180-day ISS mission, with pre-flight, in-flight and post-flight data. Every value is chosen to be consistent with published cohorts. The case shows:
- high pre-flight bone turnover,
- hypercalciuria between FD60 and FD120,
- an increase in optic-nerve-sheath diameter (5.6 mm at FD150),
- orthostatic heart-rate rise at landing,
- asymptomatic VZV shedding in saliva late in flight, with EBV rising ~24× over baseline,
- in-flight sleep averaging under 6 h,
- a **normal** depression screen, so not everything is pathological.

## Data and sources

**NASA:**
- Human Research Program and Human System Risk Board risk pages and evidence reports: bone, cardiovascular, SANS, immune, behavioral health, radiation and long-term health.
- NASA-STD-3001 Vol. 1 Rev. C.
- NTRS technical reports.
- The CHeCS medical-hardware catalog.
- NASA-led peer-reviewed studies.

**Partner agencies and others:**
- Canadian Space Agency: Vascular Echo, H-Reflex, Bio-Analyzer.
- ESA: MARES.
- Roscosmos/IBMP and SpaceX/Inspiration4 literature.
- MedlinePlus and Merck reference ranges.
- Validated instruments: PSQI, PHQ-9 and NASA-TLX.

The full list of 103 references, each verified against its original record, is in **[docs/References.pdf](docs/References.pdf)**.

## Limitations

- The interface is in English. The AI assistant answers in English by default and switches to the user's language (for example Turkish) when asked in it; Turkish questions are also mapped to English keywords for the literature search.
- The app has not been validated with real flight data or by flight surgeons. Expected-change values are cohort averages and individuals vary.
- Experimental onboard blood analysers are flagged as such.
- Behavioral-health outputs are **screening** results, not diagnoses.
- Local LLM answers can be slow on computers without a GPU.

## Use of Super Intelligence (SI) / AI

The team used:
- **ChatGPT (OpenAI)** during early research, to find candidate sources. Every suggested source was checked against its original record; unverifiable or unused ones were dropped.
- **Claude (Anthropic)** to search and synthesize the literature, to design and write the software (`app.py`) and the fictional example case, and to format the reference list.

All scientific values, thresholds and sources were reviewed against the original publications. At runtime the app itself uses a local LLM (Ollama) or an optional API model. AI-generated reports are labelled as decision support.

## Repository contents

```
app.py                       the whole application (Streamlit + SQLite)
requirements.txt             Python packages
start.bat / start.sh         one-click start (Windows / macOS-Linux)
example_case_astronaut_K.csv fictional 180-day example case
.streamlit/config.toml       theme and upload limit
docs/References.pdf          full APA 7 reference list
```

A `data/` folder is created on first run. It holds the local database and settings, and may contain personal health data or API keys, so it is excluded by `.gitignore` and should never be uploaded.

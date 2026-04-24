# MedBot — AI Healthcare Triage Agent

## 1. Project Description

MedBot is an AI-powered healthcare triage assistant built using the OpenAI Agents SDK. It helps users understand their symptoms, identify possible conditions, retrieve medication information, and find appropriate nearby clinics.

The goal of this project is to simulate a real-world triage workflow where multiple sources of information are combined to support decision-making. Instead of producing a single answer, the agent performs a sequence of steps including symptom analysis, medication recommendation, and clinic selection.

The agent follows a structured workflow:
**symptoms → medications → clinic**

This project uses mock medical data and is intended for educational purposes only. It does not replace professional medical advice.

---

## 2. Agent Design

The agent is built using the `gpt-4o` model and follows a structured system prompt that guides it to use tools in a logical and safe order. The agent is designed to perform multi-step reasoning by orchestrating multiple tools to generate a complete and coherent response.

### Tools

#### 1. check_symptoms

* **Purpose:** Analyze user symptoms and identify possible medical conditions with severity and urgency levels.
* **Input:**

  * `symptoms` (str): comma-separated list of symptoms
  * `patient_age` (int, optional)
* **Returns:** A structured triage report including matched conditions, urgency level, recommended actions, and home care tips.
* **Data Source:** `symptoms_data.json`
* **Type:** Mock data (simulated API)

---

#### 2. lookup_medication

* **Purpose:** Provide detailed information about a specific medication.
* **Input:**

  * `medication_name` (str)
  * `include_interactions` (bool, optional)
* **Returns:** Medication profile including dosage, side effects, warnings, and drug interactions.
* **Data Source:** `medications_data.json`
* **Type:** Mock data (simulated API)

---

#### 3. find_clinic

* **Purpose:** Recommend suitable clinics based on the user's condition and preferences.
* **Input:**

  * `condition_type` (str)
  * `walk_in_only` (bool, optional)
  * `max_wait_minutes` (int, optional)
* **Returns:** Ranked list of clinics with location, contact details, wait time, and services.
* **Data Source:** `clinics_data.json`
* **Type:** Mock data (simulated API)

---

#### 4. recommend_medications

* **Purpose:** Suggest medications commonly used for a given condition.
* **Input:**

  * `condition_name` (str)
  * `otc_only` (bool, optional)
* **Returns:** List of medications grouped into OTC and prescription categories.
* **Data Source:** `medications_data.json`
* **Type:** Mock data (simulated API)

---

### Multi-Tool Reasoning

The agent demonstrates multi-step reasoning by combining tools in sequence. For example:

1. Analyze symptoms using `check_symptoms`
2. Suggest treatments using `recommend_medications`
3. Recommend care locations using `find_clinic`

This mirrors real-world healthcare triage workflows.

---

## 3. Setup & Installation

### Requirements

* Python 3.10 or higher
* OpenAI API key

### Install dependencies

```bash
pip install openai-agents
```

### Set API key

Open `agent.py` and replace:

```python
OPENAI_API_KEY = "sk-your-api-key-here"
```

---

## 4. How to Run

Run the program:

```bash
python agent.py
```

Then choose one of the following modes:

1. Run all demo queries
2. Select a single demo query
3. Interactive chat mode

---

## 5. Demo Queries

### Demo 1 — Medication Lookup

* **Query:** Information about ibuprofen
* **Tools used:** `lookup_medication`

---

### Demo 2 — Symptoms + Clinic

* **Query:** UTI symptoms with clinic request
* **Tools used:** `check_symptoms` + `find_clinic`

---

### Demo 3 — Full Health Assessment

* **Query:** Migraine symptoms with medication and clinic request
* **Tools used:**
  `check_symptoms` → `recommend_medications` → `find_clinic`

---

### Demo 4 — Allergy Case

* **Query:** Hay fever treatment and specialist recommendation
* **Tools used:** `recommend_medications` + `find_clinic`

---

## 6. Mock Data

All data is stored in the `mock_data/` folder.

### symptoms_data.json

* Conditions
* Matching symptoms
* Severity levels
* Urgency ratings
* Recommended actions
* Home care tips

### medications_data.json

* Medication names
* Dosages (adult/child)
* Side effects
* Warnings
* Drug interactions
* OTC vs prescription status

### clinics_data.json

* Clinic names and types
* Specialties
* Wait times
* Distance
* Languages spoken
* Walk-in availability
* Operating hours

These datasets simulate real-world healthcare systems but are simplified.

---

## 7. Assumptions & Limitations

* Uses mock data, not real medical databases
* Symptom matching is keyword-based and not clinically precise
* Clinic distances are predefined and not based on user location
* Limited number of conditions and medications
* No persistent memory between sessions
* Cannot replace professional medical diagnosis
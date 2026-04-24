"""
AI Healthcare Triage Agent
Built with the OpenAI Agents SDK

Author      : Haniyeh Jalayeri

Description : MedBot is an AI-powered healthcare triage assistant that helps users
              understand their symptoms, identify possible conditions, look up
              medication information, and locate nearby clinics. It orchestrates
              multiple tools to provide a comprehensive health assessment while
              always reminding users to consult a qualified healthcare professional.
"""

import json
import os
import sys
from typing import Optional
from agents import Agent, Runner, function_tool

# Replace this with your actual OpenAI API key before running
OPENAI_API_KEY = "sk-your-api-key-here"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# -- Data loading --

def _load_json(relative_path: str) -> dict:
    """Load a JSON file from the mock_data folder."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_dir, relative_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(
            f"Could not find: {full_path}\n"
            "Make sure the mock_data/ folder is in the same directory as agent.py."
        )
    with open(full_path, "r", encoding="utf-8") as f:
        return json.load(f)


print("Loading mock data...")
SYMPTOMS_DATA    = _load_json("mock_data/symptoms_data.json")
MEDICATIONS_DATA = _load_json("mock_data/medications_data.json")
CLINICS_DATA     = _load_json("mock_data/clinics_data.json")
print("Done.\n")


# -- Tool 1: Symptom checker --

@function_tool
def check_symptoms(symptoms: str, patient_age: Optional[int] = None) -> str:
    """
    Takes a comma-separated list of symptoms and returns the top 3 possible
    conditions ranked by how many symptoms overlap. Each result includes a
    severity rating, an urgency level from 1 (self-care) to 5 (call 911),
    a recommended action, and home care tips.

    Args:
        symptoms: Comma-separated symptoms, e.g. "fever, headache, sore throat".
        patient_age: Optional. Triggers age-specific warnings for under-5s or over-65s.

    Returns:
        A formatted triage report with matched conditions and guidance.
    """
    user_symptoms = [s.strip().lower() for s in symptoms.split(",") if s.strip()]

    if not user_symptoms:
        return "No symptoms provided. Please list your symptoms separated by commas."

    all_conditions = SYMPTOMS_DATA.get("conditions", [])

    # Score each condition by how many of the user's symptoms match
    scored = []
    for condition in all_conditions:
        condition_symptoms = [s.lower() for s in condition["matching_symptoms"]]
        matches = [s for s in user_symptoms if any(s in cs or cs in s for cs in condition_symptoms)]
        if matches:
            match_score = len(matches)
            coverage_pct = round((match_score / len(condition_symptoms)) * 100, 1)
            scored.append((condition, match_score, matches, coverage_pct))

    if not scored:
        return (
            "No conditions closely matched your symptoms.\n"
            "This doesn't necessarily mean you're fine — please see a doctor "
            "if your symptoms are severe or getting worse."
        )

    # Sort by match count first, then urgency (higher urgency surfaces first on ties)
    scored.sort(key=lambda x: (x[1], x[0]["urgency_level"]), reverse=True)
    scored = scored[:3]

    urgency_labels = {
        1: "Level 1 - Manage at home",
        2: "Level 2 - See a doctor within a few days",
        3: "Level 3 - See a doctor today",
        4: "Level 4 - Urgent care needed",
        5: "Level 5 - EMERGENCY - Call 911 immediately",
    }

    severity_labels = {"mild": "[Mild]", "moderate": "[Moderate]", "severe": "[Severe]"}

    age_note = ""
    if patient_age is not None:
        if patient_age < 5:
            age_note = "\nAGE NOTE: Patient is under 5 — see a paediatrician promptly.\n"
        elif patient_age > 65:
            age_note = "\nAGE NOTE: Patient is over 65 — older adults should seek care sooner rather than later.\n"

    lines = [
        "=" * 58,
        "           MEDBOT SYMPTOM TRIAGE REPORT",
        "=" * 58,
        f"\nSymptoms assessed : {', '.join(user_symptoms)}",
    ]
    if patient_age:
        lines.append(f"Patient age       : {patient_age} years old")
    lines.append(age_note)
    lines.append(f"\nTop {len(scored)} possible condition(s):\n")
    lines.append("-" * 56)

    for rank, (condition, score, matched, coverage) in enumerate(scored, start=1):
        urgency   = urgency_labels.get(condition["urgency_level"], "Unknown")
        sev_label = severity_labels.get(condition["severity"], "")
        see_doctor_list = "\n".join(f"       - {s}" for s in condition["see_doctor_if"])
        home_tips       = "\n".join(f"       - {t}" for t in condition["home_care_tips"][:3])

        lines.append(
            f"\n[{rank}] {sev_label}  {condition['name'].upper()}\n"
            f"     Severity        : {condition['severity'].title()}\n"
            f"     Urgency         : {urgency}\n"
            f"     Symptoms matched: {', '.join(matched)} ({score} match(es))\n"
            f"     Symptom coverage: {coverage}%\n"
            f"\n     About: {condition['description']}\n"
            f"\n     Recommended action:\n"
            f"       {condition['recommended_action']}\n"
            f"\n     See a doctor if:\n{see_doctor_list}\n"
            f"\n     Home care tips:\n{home_tips}\n"
        )
        lines.append("-" * 56)

    lines.append(
        "\nDISCLAIMER: This is general information only, not medical advice.\n"
        "   Always consult a doctor, especially if symptoms are severe.\n"
        "   For emergencies, call 911."
    )
    return "\n".join(lines)


# -- Tool 2: Medication lookup --

@function_tool
def lookup_medication(medication_name: str, include_interactions: bool = True) -> str:
    """
    Looks up a medication by name (brand or generic) and returns a full profile:
    category, OTC vs prescription status, adult and child dosage, side effects,
    warnings, and drug interactions if requested.

    Args:
        medication_name: Name of the medication, e.g. "ibuprofen" or "Advil".
                         Partial matching is supported.
        include_interactions: Set to False to skip the interactions section.

    Returns:
        A formatted medication profile.
    """
    all_meds = MEDICATIONS_DATA.get("medications", [])
    query = medication_name.strip().lower()

    # Try to find a match — exact first, then partial
    matched_med = None
    for med in all_meds:
        if query in med["name"].lower():
            matched_med = med
            break

    if not matched_med:
        available = [m["name"] for m in all_meds]
        return (
            f"'{medication_name}' was not found in the database.\n\n"
            "Medications available:\n" +
            "\n".join(f"  - {m}" for m in available)
        )

    m = matched_med
    otc_status = (
        "Over-the-counter (OTC) - no prescription needed"
        if m["otc"]
        else "Prescription only - requires a doctor's order"
    )
    side_effects_str = "\n".join(f"  - {s}" for s in m["side_effects"])
    warnings_str     = "\n".join(f"  WARNING: {w}" for w in m["warnings"])

    lines = [
        "=" * 58,
        "              MEDICATION INFORMATION",
        "=" * 58,
        f"\n{m['name']}",
        f"   Category : {m['category']}",
        f"   Status   : {otc_status}",
        f"   Used for : {', '.join(m['used_for'])}",
        f"   Forms    : {', '.join(m['available_forms'])}",
        f"\n-- Dosage ------------------------------------------------",
        f"   Adults   : {m['dosage']['adults']}",
        f"   Children : {m['dosage']['children']}",
        f"\n-- Side effects ------------------------------------------",
        side_effects_str,
        f"\n-- Warnings ----------------------------------------------",
        warnings_str,
    ]

    if include_interactions and m["interactions"]:
        interactions_str = "\n".join(f"  - {i}" for i in m["interactions"])
        lines += [
            f"\n-- Drug interactions -------------------------------------",
            interactions_str,
            f"\n   Tell your doctor and pharmacist about everything you're taking,",
            f"   including supplements and OTC drugs.",
        ]

    lines.append(
        "\nDISCLAIMER: For educational reference only.\n"
        "   Do not change or stop any medication without talking to your doctor."
    )
    return "\n".join(lines)


# -- Tool 3: Clinic finder --

@function_tool
def find_clinic(
    condition_type: str,
    walk_in_only: bool = False,
    max_wait_minutes: Optional[int] = None,
) -> str:
    """
    Finds nearby clinics suitable for a given condition. Results can be filtered
    to walk-in only or by a maximum estimated wait time.

    Args:
        condition_type: Type of condition or care needed, e.g. "flu", "allergy",
                        "mental health", "UTI", "diabetes", "emergency".
        walk_in_only: If True, only return clinics that take walk-ins.
        max_wait_minutes: Filter out clinics with waits longer than this.

    Returns:
        A ranked list of clinics with contact info, hours, and wait times.
    """
    all_clinics = CLINICS_DATA.get("clinics", [])
    query = condition_type.strip().lower()

    # Map common condition keywords to relevant clinic specialties
    keyword_map = {
        "emergency": ["emergency", "trauma", "cardiac", "stroke", "surgical"],
        "urgent":    ["urgent care", "minor fractures", "lacerations", "infections"],
        "flu":       ["cold and flu", "general practice", "urgent care"],
        "cold":      ["cold and flu", "general practice"],
        "allergy":   ["allergy testing", "allergic rhinitis", "asthma"],
        "mental health": ["anxiety", "depression", "counselling", "panic disorder"],
        "anxiety":   ["anxiety", "panic disorder", "counselling", "cognitive behavioural therapy"],
        "depression":["depression", "counselling", "cognitive behavioural therapy"],
        "diabetes":  ["diabetes management", "metabolic syndrome", "chronic disease management"],
        "uti":       ["UTI", "urinary", "women's health", "general practice"],
        "women":     ["women's health", "reproductive health", "prenatal care"],
        "general":   ["general practice", "family medicine"],
        "family":    ["family medicine", "preventive care"],
        "migraine":  ["general practice", "family medicine", "urgent care"],
        "headache":  ["general practice", "urgent care"],
        "stomach":   ["general practice", "urgent care"],
    }

    relevant_keywords = []
    for key, keywords in keyword_map.items():
        if key in query or query in key:
            relevant_keywords.extend(keywords)

    if not relevant_keywords:
        relevant_keywords = [query]

    # Score clinics by how well their specialties match the keywords
    matched = []
    for clinic in all_clinics:
        specialties_lower = [s.lower() for s in clinic["specialties"]]
        clinic_type_lower = clinic["type"].lower()
        score = sum(
            1 for kw in relevant_keywords
            if any(kw.lower() in sp for sp in specialties_lower) or kw.lower() in clinic_type_lower
        )
        if score > 0:
            matched.append((clinic, score))

    if walk_in_only:
        matched = [(c, s) for c, s in matched if c["accepts_walk_ins"]]
    if max_wait_minutes is not None:
        matched = [(c, s) for c, s in matched if c["wait_time_minutes"] <= max_wait_minutes]

    if not matched:
        return (
            f"No clinics found for '{condition_type}'"
            + (" (walk-in only)" if walk_in_only else "")
            + (f" with wait under {max_wait_minutes} min" if max_wait_minutes else "")
            + ".\n\nFor emergencies, call 911 or go to the nearest ER."
        )

    matched.sort(key=lambda x: (-x[1], x[0]["distance_km"]))

    lines = [
        "=" * 58,
        "               NEARBY CLINIC FINDER",
        "=" * 58,
        f"\nCondition  : {condition_type.title()}",
        f"Walk-in    : {'Yes' if walk_in_only else 'No'}",
        f"Max wait   : {f'{max_wait_minutes} min' if max_wait_minutes else 'Any'}",
        f"\n{len(matched)} clinic(s) found:\n",
        "-" * 56,
    ]

    for i, (clinic, score) in enumerate(matched, start=1):
        specialties_str = ", ".join(clinic["specialties"][:4])
        if len(clinic["specialties"]) > 4:
            specialties_str += f" +{len(clinic['specialties']) - 4} more"

        lines.append(
            f"\n[{i}] {clinic['name'].upper()}\n"
            f"     Type          : {clinic['type'].replace('_', ' ').title()}\n"
            f"     Specialties   : {specialties_str}\n"
            f"     Address       : {clinic['address']}\n"
            f"     Phone         : {clinic['phone']}\n"
            f"     Distance      : {clinic['distance_km']} km\n"
            f"     Est. wait     : ~{clinic['wait_time_minutes']} minutes\n"
            f"     Walk-ins      : {'Yes' if clinic['accepts_walk_ins'] else 'No (appointment required)'}\n"
            f"     Online booking: {'Yes' if clinic['online_booking'] else 'Phone only'}\n"
            f"     Rating        : {clinic['rating']}/5 ({clinic['total_reviews']:,} reviews)\n"
            f"     Languages     : {', '.join(clinic['languages'])}\n"
            f"     Mon-Fri hours : {clinic['hours']['monday']}\n"
            f"     Notes         : {clinic['notes']}\n"
        )
        lines.append("-" * 56)

    lines.append("\nDistances are approximate. Call ahead to confirm availability.")
    return "\n".join(lines)


# -- Tool 4: Medication recommendations by condition --

@function_tool
def recommend_medications(condition_name: str, otc_only: bool = False) -> str:
    """
    Returns a list of medications commonly used for a given condition.
    Useful after check_symptoms has identified a likely condition.

    Args:
        condition_name: Condition name, e.g. "Migraine", "Common Cold", "UTI".
        otc_only: If True, only show medications available without a prescription.

    Returns:
        A list of relevant medications split into OTC and prescription groups.
    """
    all_meds = MEDICATIONS_DATA.get("medications", [])
    query = condition_name.strip().lower()

    matched_meds = []
    for med in all_meds:
        conditions_lower = [c.lower() for c in med["conditions_treated"]]
        if any(query in c or c in query for c in conditions_lower):
            if otc_only and not med["otc"]:
                continue
            matched_meds.append(med)

    if not matched_meds:
        return (
            f"No medications found for '{condition_name}'"
            + (" (OTC only)" if otc_only else "") + ".\n\n"
            "This could mean the condition needs a doctor's assessment, "
            "or that the condition name doesn't quite match the database.\n"
            "Please consult a healthcare professional for advice."
        )

    otc_meds = [m for m in matched_meds if m["otc"]]
    rx_meds  = [m for m in matched_meds if not m["otc"]]

    lines = [
        "=" * 58,
        "          MEDICATION RECOMMENDATIONS",
        "=" * 58,
        f"\nCondition : {condition_name}",
        f"Filter    : {'OTC only' if otc_only else 'All (OTC + Prescription)'}",
        f"\n{len(matched_meds)} medication(s) found:\n",
        "-" * 56,
    ]

    if otc_meds:
        lines.append("\nAVAILABLE WITHOUT A PRESCRIPTION (OTC):\n")
        for i, med in enumerate(otc_meds, start=1):
            warnings_str = "\n".join(f"     WARNING: {w}" for w in med["warnings"][:2])
            lines.append(
                f"  [{i}] {med['name']}\n"
                f"      Category  : {med['category']}\n"
                f"      Adult dose: {med['dosage']['adults']}\n"
                f"      Forms     : {', '.join(med['available_forms'])}\n"
                f"      Warnings  :\n{warnings_str}\n"
            )

    if rx_meds and not otc_only:
        lines.append("\nPRESCRIPTION REQUIRED:\n")
        for i, med in enumerate(rx_meds, start=1):
            lines.append(
                f"  [{i}] {med['name']}\n"
                f"      Category : {med['category']}\n"
                f"      Notes    : Requires a prescription — ask your doctor if this is right for you.\n"
            )
        lines.append(
            "\nSome medications above require a prescription.\n"
            "   Book an appointment with your doctor to discuss your options."
        )

    lines.append(
        "\nDISCLAIMER: Never self-medicate with prescription drugs.\n"
        "   Always follow your doctor's or pharmacist's instructions."
    )
    return "\n".join(lines)


# -- Agent setup --

SYSTEM_PROMPT = """
You are MedBot, a knowledgeable and empathetic AI healthcare triage assistant.
Your role is to help users understand their symptoms, identify possible
conditions, find relevant medications and nearby clinics.

You have access to four tools:
  1. check_symptoms         - Analyse symptoms and identify possible conditions with urgency ratings
  2. lookup_medication      - Get detailed info on a specific medication (dosage, warnings, interactions)
  3. find_clinic            - Locate appropriate nearby clinics based on condition type and preferences
  4. recommend_medications  - Get a list of medications commonly used for a specific condition

Your approach:
  - Be thorough: for any health concern, proactively call multiple tools to give
    a complete picture — check symptoms, then recommend medications and find a clinic.
  - Be empathetic: health concerns are stressful. Use a warm, reassuring tone.
  - Be clear about urgency: always highlight if a situation is a medical emergency
    and direct the user to call 911 or go to the ER.
  - Be honest: you are NOT a doctor. Always remind users this is educational
    information and they should consult a qualified healthcare professional.
  - Be organised: structure responses with clear sections.
  - For full health inquiries, use this order: symptoms -> medications -> clinic.
  - Never diagnose definitively — say "possible condition" or "may suggest".
  - Express medication doses clearly and always note OTC vs prescription status.

IMPORTANT SAFETY RULES:
  - If any symptom suggests a life-threatening emergency (chest pain with shortness
    of breath, stroke symptoms, appendicitis, hypertensive crisis), ALWAYS lead
    with an urgent warning to call 911 BEFORE providing any other information.
  - Never encourage self-treating serious conditions.
  - Always recommend professional medical consultation for anything beyond mild
    self-limiting illnesses.

Available data covers these conditions, medications, and clinic types:
  Conditions : Common Cold, Flu, Migraine, Gastroenteritis, Allergic Rhinitis,
               UTI, Hypertensive Crisis, Anxiety/Panic Attack, Appendicitis,
               Type 2 Diabetes
  Medications: Acetaminophen, Ibuprofen, Loratadine, Diphenhydramine,
               Ondansetron, Amoxicillin, Sumatriptan, ORS, Metformin, Lorazepam
  Clinics    : Walk-in, Urgent Care, Emergency, Mental Health, Allergy Specialist,
               Diabetes Specialist, Women's Health, Family Practice

SCOPE LIMIT:
You only answer healthcare-related questions.
If the user asks about topics unrelated to health (such as sports, politics,
or general knowledge), politely refuse and guide them back to health-related topics.
"""

healthcare_agent = Agent(
    name="MedBot",
    model="gpt-4o",
    instructions=SYSTEM_PROMPT,
    tools=[
        check_symptoms,
        lookup_medication,
        find_clinic,
        recommend_medications,
    ],
)


# -- Demo queries --

DEMO_QUERIES = [
    {
        "title": "Demo 1 - Single-tool: Medication Lookup",
        "query": (
            "Can you tell me everything I need to know about ibuprofen? "
            "I want to know the dosage, side effects, warnings, and any interactions."
        ),
    },
    {
        "title": "Demo 2 - Two-tool: Symptom Check + Clinic Finder",
        "query": (
            "I have a burning sensation when I urinate, I'm going to the bathroom "
            "very frequently, and my urine looks cloudy. I'm a 28-year-old woman. "
            "What might be wrong with me and where should I go to get checked?"
        ),
    },
    {
        "title": "Demo 3 - Multi-tool: Full Health Assessment",
        "query": (
            "I've been suffering from a pounding headache on one side of my head, "
            "nausea, and extreme sensitivity to light and sound for the past 6 hours. "
            "I'm 35 years old. Can you figure out what's wrong, tell me what medications "
            "I can take (both OTC and prescription), and find me a suitable clinic?"
        ),
    },
    {
        "title": "Demo 4 - Multi-tool: Allergy Season Management",
        "query": (
            "Every spring I get a runny nose, itchy watery eyes, and constant sneezing. "
            "I think it's hay fever. What are the best medications I can buy without a "
            "prescription? Are there any specialists I should see?"
        ),
    },
]


# -- Interactive mode --

def interactive_mode():
    """Simple chat loop so you can ask MedBot your own questions."""
    print("\n" + "=" * 65)
    print("  MedBot - Interactive Mode")
    print("=" * 65)
    print("Describe your symptoms or ask a health question.")
    print("Type 'quit' to exit.\n")
    print("REMINDER: MedBot is for educational purposes only.")
    print("Always consult a real doctor for medical concerns.\n")
    
    history = []
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nStay healthy! Goodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "bye", "q"}:
            print("Stay healthy! Goodbye.")
            break

        print("\nMedBot: (thinking...)\n")
        try:
            history.append({"role": "user", "content": user_input})
            result = Runner.run_sync(healthcare_agent, history)
            history.append({"role": "assistant", "content": result.final_output})
            print(f"MedBot:\n{result.final_output}\n")
            print("-" * 65 + "\n")
        except Exception as e:
            print(f"[ERROR] {e}\n")


# -- Main --

def main():
    print("\n" + "=" * 65)
    print("  MedBot - AI Healthcare Triage Agent")
    print("  Built with the OpenAI Agents SDK")
    print("=" * 65)

    if not OPENAI_API_KEY or OPENAI_API_KEY == "sk-your-api-key-here":
        print(
            "\n[ERROR] No API key set.\n"
            "Open agent.py and replace the placeholder on line 17:\n"
            "    OPENAI_API_KEY = \"sk-your-api-key-here\"\n"
        )
        sys.exit(1)

    print("\nChoose a mode:")
    print("  [1] Run all 4 demo queries")
    print("  [2] Pick a single demo query")
    print("  [3] Interactive chat")

    try:
        choice = input("\nEnter choice (1/2/3): ").strip()
    except (EOFError, KeyboardInterrupt):
        choice = "1"

    if choice == "1":
        for demo in DEMO_QUERIES:
            print(f"\n{'='*65}")
            print(f"  {demo['title']}")
            print("=" * 65)
            print(f"User: {demo['query']}\n")
            print("MedBot: (processing...)\n")
            try:
                result = Runner.run_sync(healthcare_agent, demo["query"])
                print(result.final_output)
            except Exception as e:
                print(f"[ERROR] {e}")
            print()
            input("Press Enter for the next query...\n")

    elif choice == "2":
        print("\n" + "=" * 65)
        print("  Select a Demo Query")
        print("=" * 65)
        for i, demo in enumerate(DEMO_QUERIES, start=1):
            print(f"  [{i}] {demo['title']}")
        print("-" * 65)

        demo = None
        while demo is None:
            try:
                raw = input(f"Enter query number (1-{len(DEMO_QUERIES)}): ").strip()
                num = int(raw) - 1
                if 0 <= num < len(DEMO_QUERIES):
                    demo = DEMO_QUERIES[num]
                else:
                    print(f"  Please enter a number between 1 and {len(DEMO_QUERIES)}.")
            except (ValueError, EOFError):
                print("  Invalid input. Please enter a number.")

        print(f"\n{'='*65}")
        print(f"  {demo['title']}")
        print("=" * 65)
        print(f"User: {demo['query']}\n")
        print("MedBot: (processing...)\n")
        try:
            result = Runner.run_sync(healthcare_agent, demo["query"])
            print(result.final_output)
        except Exception as e:
            print(f"[ERROR] {e}")

        print("\n" + "=" * 65)
        print("  Done. Hope that helped!")
        print("=" * 65)
        sys.exit(0)

    elif choice == "3":
        interactive_mode()

    else:
        print("Invalid choice. Running all demos by default.")
        main()


if __name__ == "__main__":
    main()
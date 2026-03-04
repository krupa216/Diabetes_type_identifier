"""
src/nlp_extractor.py
NLP symptom extraction pipeline for Diabetes Type Classifier
Extracts clinical signals from free-text patient descriptions
and maps them to numeric features for CIT model input.
"""

import re
import math

# ── SYMPTOM SIGNAL MAP ───────────────────────────────────────────────────────
SYMPTOM_MAP = [
    # keyword list,                              signal name,           HbA1c weight, BMI effect
    (["thirst","thirsty","polydipsia","drinking a lot"],  "Polydipsia",         0.30,  0.0),
    (["urinat","frequent urination","polyuria","peeing"], "Polyuria",           0.28,  0.0),
    (["blurr","vision","eyesight","blur"],                "Blurred Vision",     0.22,  0.0),
    (["fatigue","tired","exhausted","weak","lethargy"],   "Fatigue",            0.20,  0.0),
    (["weight loss","losing weight","lost weight"],       "Weight Loss",        0.25, -2.5),
    (["heal","wound","cut","sore","slow heal"],           "Slow Healing",       0.20,  0.0),
    (["numb","tingling","pins and needles","feet"],       "Neuropathy",         0.22,  0.0),
    (["hunger","hungry","always eating","polyphagia"],    "Polyphagia",         0.18,  0.0),
    (["infection","yeast","thrush","itch"],               "Frequent Infection", 0.15,  0.0),
    (["dry mouth","dry skin","dry"],                      "Dry Skin/Mouth",     0.12,  0.0),
    (["obese","overweight","heavy","bmi"],                "High BMI",           0.15,  5.0),
    (["family","hereditary","mother","father","genetic"], "Family History",     0.18,  0.0),
    (["headache","dizzy","lightheaded"],                  "Headache/Dizziness", 0.10,  0.0),
    (["months","weeks","years","long time","chronic"],    "Chronic Duration",   0.08,  0.0),
]


# ── CORE EXTRACTOR ───────────────────────────────────────────────────────────
def extract_signals(text: str) -> list[dict]:
    """
    Extract diabetes-related signals from free-text symptom description.
    Returns list of matched signal dicts with name, weight, severity.
    """
    text_lower = text.lower()
    matched = []

    for keywords, signal, weight, bmi_effect in SYMPTOM_MAP:
        if any(kw in text_lower for kw in keywords):
            severity = "high" if weight >= 0.22 else "medium" if weight >= 0.15 else "low"
            matched.append({
                "signal":     signal,
                "weight":     weight,
                "bmi_effect": bmi_effect,
                "severity":   severity,
            })

    return matched


def signals_to_features(signals: list[dict], age: float, gender: str) -> dict:
    """
    Convert extracted NLP signals into numeric features for CIT model.
    Maps symptom weights → estimated HbA1c, BMI, Glucose.
    """
    total_weight = sum(s["weight"] for s in signals)
    bmi_delta    = sum(s["bmi_effect"] for s in signals)

    has = lambda name: any(s["signal"] == name for s in signals)

    # ── Estimate HbA1c ──────────────────────────────────────────────────────
    if   total_weight > 0.85: hba1c = 9.0
    elif total_weight > 0.65: hba1c = 7.5
    elif total_weight > 0.45: hba1c = 6.8
    elif total_weight > 0.25: hba1c = 6.1
    elif total_weight > 0.10: hba1c = 5.8
    else:                     hba1c = 5.2

    # Classic T1D pattern: young + weight loss + polyuria + polydipsia
    if age < 35 and has("Weight Loss") and has("Polyuria") and has("Polydipsia"):
        hba1c = max(hba1c, 8.5)

    # Age-based T2D uplift
    if age > 50 and has("Family History"):
        hba1c = max(hba1c, 5.9)

    # ── Estimate BMI ────────────────────────────────────────────────────────
    base_bmi = 26.0 if age > 40 else 23.0
    bmi = max(16.0, base_bmi + bmi_delta)

    # ── Estimate Glucose ────────────────────────────────────────────────────
    if   hba1c >= 6.5: glucose = 7.8
    elif hba1c >= 5.7: glucose = 6.3
    else:              glucose = 5.1

    return {
        "hba1c":   round(hba1c, 1),
        "bmi":     round(bmi, 1),
        "glucose": round(glucose, 1),
        # Defaults for remaining CIT features
        "urea": 4.5, "cr": 46.0,
        "chol": 4.8, "tg": 1.5,
        "hdl":  1.2, "ldl": 2.9, "vldl": 0.68,
    }


def nlp_pipeline(text: str, age: float = 45, gender: str = "Male") -> dict:
    """
    Full NLP → features pipeline.
    Input : raw symptom text
    Output: feature dict ready for CIT model + extracted signals
    """
    if not text or not text.strip():
        raise ValueError("Empty symptom text provided.")

    signals  = extract_signals(text)
    features = signals_to_features(signals, age, gender)

    total_weight = sum(s["weight"] for s in signals)
    risk = "High" if total_weight > 0.65 else "Moderate" if total_weight > 0.35 else "Low"

    return {
        "features":      features,
        "signals":       signals,
        "signal_count":  len(signals),
        "total_weight":  round(total_weight, 3),
        "estimated_risk": risk,
    }


# ── TEST ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = (
        "I have been feeling very thirsty all the time and urinating "
        "frequently. Also very tired and blurry vision. My father had "
        "diabetes. This has been going on for 3 months."
    )
    result = nlp_pipeline(sample, age=48, gender="Male")
    print(f"Signals detected : {result['signal_count']}")
    for s in result['signals']:
        print(f"  [{s['severity'].upper():6}] {s['signal']:<22} weight={s['weight']}")
    print(f"\nTotal weight     : {result['total_weight']}")
    print(f"Estimated risk   : {result['estimated_risk']}")
    print(f"\nEstimated features:")
    for k, v in result['features'].items():
        print(f"  {k:<8} = {v}")

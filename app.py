"""
app.py - Flask Web Application
Diabetes Type Classifier: NLP Symptoms + CIT Model
"""

from flask import Flask, render_template, request, jsonify
import os, pickle, numpy as np

app = Flask(__name__)

def cit_predict(features: dict) -> dict:
    model_path = "models/cit_model.pkl"
    feature_order = ["AGE","Urea","Cr","HbA1c","Chol","TG","HDL","LDL","VLDL","BMI","Gender"]

    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        X     = np.array([[features[k] for k in feature_order]])
        pred  = model.predict(X)[0]
        proba = model.predict_proba(X)[0].tolist()
    else:
        # Rule-based fallback
        hba1c = features["HbA1c"]
        bmi   = features["BMI"]
        if   hba1c >= 6.5:                  pred=2; proba=[0.05,0.12,0.83]
        elif hba1c >= 5.7:                  pred=1; proba=[0.15,0.70,0.15]
        else:                               pred=0; proba=[0.82,0.13,0.05]
        if bmi > 30 and pred == 0:          pred=1; proba=[0.55,0.35,0.10]

    names = ["No Diabetes","Pre-Diabetes","Diabetes"]
    return {
        "prediction":    int(pred),
        "class_name":    names[int(pred)],
        "confidence":    round(max(proba)*100, 1),
        "risk_level":    ["Low","Moderate","High"][int(pred)],
        "probabilities": {
            "No Diabetes":  round(proba[0]*100,1),
            "Pre-Diabetes": round(proba[1]*100,1),
            "Diabetes":     round(proba[2]*100,1),
        }
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        d = request.get_json()
        features = {
            "AGE":    float(d.get("age", 45)),
            "Gender": 1 if d.get("gender","Male")=="Male" else 0,
            "Urea":   float(d.get("urea", 4.5)),
            "Cr":     float(d.get("cr", 46)),
            "HbA1c":  float(d["hba1c"]),
            "Chol":   float(d.get("chol", 4.8)),
            "TG":     float(d.get("tg", 1.5)),
            "HDL":    float(d.get("hdl", 1.2)),
            "LDL":    float(d.get("ldl", 2.9)),
            "VLDL":   float(d.get("vldl", 0.68)),
            "BMI":    float(d["bmi"]),
        }
        return jsonify({"status":"success", "result": cit_predict(features)})
    except Exception as e:
        return jsonify({"status":"error", "message": str(e)}), 400


@app.route("/nlp-predict", methods=["POST"])
def nlp_predict():
    """NLP symptoms → feature extraction → CIT prediction"""
    try:
        from src.nlp_extractor import nlp_pipeline
        d      = request.get_json()
        text   = d.get("text","")
        age    = float(d.get("age", 45))
        gender = d.get("gender","Male")

        nlp_out  = nlp_pipeline(text, age, gender)
        feat     = nlp_out["features"]

        features = {
            "AGE":    age,
            "Gender": 1 if gender=="Male" else 0,
            "HbA1c":  feat["hba1c"],
            "BMI":    feat["bmi"],
            "Urea":   feat["urea"],  "Cr":   feat["cr"],
            "Chol":   feat["chol"],  "TG":   feat["tg"],
            "HDL":    feat["hdl"],   "LDL":  feat["ldl"],  "VLDL": feat["vldl"],
        }

        result = cit_predict(features)
        result["nlp_signals"]   = nlp_out["signals"]
        result["estimated_hba1c"] = feat["hba1c"]
        result["estimated_bmi"]   = feat["bmi"]
        return jsonify({"status":"success", "result": result})
    except Exception as e:
        return jsonify({"status":"error", "message": str(e)}), 400


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    app.run(debug=True, port=5000)

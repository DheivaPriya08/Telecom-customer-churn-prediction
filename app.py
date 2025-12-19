"""
Gradio Churn Prediction App (full version)
- Single prediction (form)
"""

import os
import zipfile
import tempfile
import json
import io
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap
import gradio as gr
from sklearn.metrics import confusion_matrix

# ---------------------------
# Environment fixes for CI / containers
# ---------------------------
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.makedirs("/tmp/matplotlib", exist_ok=True)
os.makedirs("/tmp/models", exist_ok=True)

MODELS_DIR = "/tmp/models"
MODELS_ZIP = "models.zip"  # put your models.zip at repo root

# ---------------------------
# Unzip models.zip if present
# ---------------------------
if os.path.exists(MODELS_ZIP):
    try:
        with zipfile.ZipFile(MODELS_ZIP, "r") as z:
            z.extractall(MODELS_DIR)
    except Exception as e:
        print("Warning: failed to extract models.zip:", e)
else:
    print("models.zip not found — ensure models.zip uploaded to repo root if you want pre-trained pipelines.")

# ---------------------------
# Helper: discover pipelines and metas in MODELS_DIR
# ---------------------------
def discover_models(models_dir: str) -> Dict[str, Dict[str, str]]:
    """
    Returns dict: { model_label: {"pipeline": pipeline_path, "meta": meta_path} }
    Will attempt to match patterns like RF_*pipeline.pkl and *_meta.json
    """
    found = {}
    files = os.listdir(models_dir) if os.path.exists(models_dir) else []
    # collect pipeline files (.pkl, .joblib, .sav) and meta jsons
    pkl_files = [f for f in files if f.endswith(".pkl") or f.endswith(".joblib") or f.endswith(".sav") or f.endswith(".gz")]
    json_files = [f for f in files if f.endswith(".json")]
    for p in pkl_files:
        label = os.path.splitext(os.path.basename(p))[0]
        candidate_meta = None
        # try to find matching meta by common prefix
        for j in json_files:
            if label.split("_")[0].lower() in j.lower():
                candidate_meta = j
                break
        found[label] = {"pipeline": os.path.join(models_dir, p), "meta": os.path.join(models_dir, candidate_meta) if candidate_meta else None}
    return found

AVAILABLE = discover_models(MODELS_DIR)

# ---------------------------
# Load pipelines (lazy)
# ---------------------------
LOADED = {}
METAS = {}

def load_pipeline(label: str):
    if label in LOADED:
        return LOADED[label]
    info = AVAILABLE.get(label)
    if not info:
        raise FileNotFoundError(f"No pipeline info for '{label}'")
    ppath = info["pipeline"]
    if not os.path.exists(ppath):
        raise FileNotFoundError(f"Pipeline file not found: {ppath}")
    pipeline = joblib.load(ppath)
    LOADED[label] = pipeline
    # load meta if present
    meta_path = info.get("meta")
    if meta_path and os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                METAS[label] = json.load(f)
        except Exception:
            METAS[label] = {}
    else:
        METAS[label] = {}
    return pipeline

# ---------------------------
# Minimal feature engineering helper
# ---------------------------
day_charge_cols = ['totaldaycharge','totalevecharge','totalnightcharge','totalintlcharge']
min_cols = ['totaldayminutes','totaleveminutes','totalnightminutes','totalintlminutes']
call_cols = ['totaldaycalls','totalevecalls','totalnightcalls','totalintlcalls']

def complete_customer_features_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # ensure expected numeric columns exist
    for c in set(day_charge_cols + min_cols + call_cols + ['numbercustomerservicecalls','accountlength']):
        if c not in df.columns:
            df[c] = 0
    # derived
    df['total_charge'] = df[day_charge_cols].sum(axis=1)
    df['total_minutes'] = df[min_cols].sum(axis=1)
    df['total_calls'] = df[call_cols].sum(axis=1)
    # per-call features (guard divide by zero)
    df['day_minutes_per_call'] = df['totaldayminutes'] / (df['totaldaycalls'].replace(0, np.nan) + 1e-5)
    df['eve_minutes_per_call'] = df['totaleveminutes'] / (df['totalevecalls'].replace(0, np.nan) + 1e-5)
    df['night_minutes_per_call'] = df['totalnightminutes'] / (df['totalnightcalls'].replace(0, np.nan) + 1e-5)
    df['intl_minutes_per_call'] = df['totalintlminutes'] / (df['totalintlcalls'].replace(0, np.nan) + 1e-5)
    df['service_call_frequency'] = df['numbercustomerservicecalls'] / (df['accountlength'] + 1e-5)
    df['charge_per_minute'] = df['total_charge'] / (df['total_minutes'] + 1e-5)
    # fill NaNs
    df = df.fillna(0)
    return df

# ---------------------------
# Prediction / CBA logic
# ---------------------------
DEFAULT_PROFIT = 100
DEFAULT_COST = 10
DEFAULT_LOSS = 80

def predict_single(inputs: Dict[str, Any], model_label: str, profit=DEFAULT_PROFIT, cost=DEFAULT_COST, loss=DEFAULT_LOSS):
    pipeline = load_pipeline(model_label)
    df = pd.DataFrame([inputs])
    df = complete_customer_features_df(df)

    prob = float(pipeline.predict_proba(df)[:, 1][0])
    pred = int(prob >= METAS.get(model_label, {}).get("best_threshold", 0.5))

    # --- Risk classification ---
    if prob > 0.7:
        risk = "HIGH"
        actions = ["Assign retention team", "Offer personalized package", "Immediate follow-up"]
    elif prob > 0.4:
        risk = "MEDIUM"
        actions = ["Watchlist", "Satisfaction survey", "Small incentive"]
    else:
        risk = "LOW"
        actions = ["No urgent action", "Loyalty program"]

    # --- CBA scaled by probability ---
    expected_retention = (1 - prob) * profit
    expected_loss = prob * loss
    contact_cost = cost
    net_profit = expected_retention - contact_cost - expected_loss
    roi = (net_profit / contact_cost * 100) if contact_cost > 0 else 0

    return {
        "prob": prob,
        "pred": pred,
        "risk": risk,
        "actions": actions,
        "net_profit": round(net_profit, 2),
        "roi": round(roi, 2),
        "benefit": round(expected_retention, 2),
        "cost": round(contact_cost, 2),
        "missed_loss": round(expected_loss, 2)
    }

# ---------------------------
# Gradio UI components & callbacks
# ---------------------------
def list_model_labels():
    if not AVAILABLE:
        return []
    return sorted(AVAILABLE.keys())

def single_predict_ui(
    model_label: str,
    accountlength: int,
    internationalplan: str,
    voicemailplan: str,
    numbervmailmessages: int,
    totaldayminutes: float,
    totaldaycalls: int,
    totaldaycharge: float,
    totaleveminutes: float,
    totalevecalls: int,
    totalevecharge: float,
    totalnightminutes: float,
    totalnightcalls: int,
    totalnightcharge: float,
    totalintlminutes: float,
    totalintlcalls: int,
    totalintlcharge: float,
    numbercustomerservicecalls: int,
    profit: float,
    cost: float,
    loss: float
):
    if model_label is None or model_label == "":
        return "No model available. Upload models.zip or check models directory."
    # map yes/no to 1/0
    inputs = {
        "accountlength": accountlength,
        "internationalplan": 1 if internationalplan=="Yes" else 0,
        "voicemailplan": 1 if voicemailplan=="Yes" else 0,
        "numbervmailmessages": numbervmailmessages,
        "totaldayminutes": totaldayminutes,
        "totaldaycalls": totaldaycalls,
        "totaldaycharge": totaldaycharge,
        "totaleveminutes": totaleveminutes,
        "totalevecalls": totalevecalls,
        "totalevecharge": totalevecharge,
        "totalnightminutes": totalnightminutes,
        "totalnightcalls": totalnightcalls,
        "totalnightcharge": totalnightcharge,
        "totalintlminutes": totalintlminutes,
        "totalintlcalls": totalintlcalls,
        "totalintlcharge": totalintlcharge,
        "numbercustomerservicecalls": numbercustomerservicecalls
    }
    try:
        res = predict_single(inputs, model_label, profit=profit, cost=cost, loss=loss)
    except Exception as e:
        return f"Error during prediction: {e}"
    txt_lines = [
        f"Churn probability: {res['prob']:.3f}",
        f"Predicted churn (binary): {res['pred']}",
        f"Risk level: {res['risk']}",
        "Recommended actions:",
    ] + [f"- {a}" for a in res['actions']] + [
        "",
        "Cost-benefit summary",
        f"Net profit: ${res['net_profit']:.2f}",
        f"ROI: {res['roi']:.2f}%",
        f"Benefit: ${res['benefit']:.2f}  Cost: ${res['cost']:.2f}  Missed loss: ${res['missed_loss']:.2f}"
    ]
    # RETURN A STRING (not dict) so Gradio displays cleanly
    return "\n".join(txt_lines)

# ---------------------------
# Build Gradio UI
# ---------------------------
model_choices = list_model_labels()
if not model_choices:
    model_choices = [""]  # empty placeholder

with gr.Blocks(title="Churn Prediction (Gradio)") as demo:
    gr.Markdown("## 💼 Churn Prediction App (Gradio)\nFull version: single prediction")
    with gr.Row():
        with gr.Column(scale=1):
            model_dropdown = gr.Dropdown(label="Choose model (from models.zip)", choices=model_choices, value=model_choices[0])
            gr.Markdown("**Business params**")
            profit_in = gr.Number(value=DEFAULT_PROFIT, label="Profit per retained customer ($)")
            cost_in = gr.Number(value=DEFAULT_COST, label="Cost per contact ($)")
            loss_in = gr.Number(value=DEFAULT_LOSS, label="Loss per missed churner ($)")
        with gr.Column(scale=2):
            gr.Markdown("### Single customer prediction (use the form)")
            with gr.Tabs():
                with gr.TabItem("Form"):
                    accountlength = gr.Slider(0, 300, value=100, label="Account Length")
                    internationalplan = gr.Radio(["No", "Yes"], value="No", label="International Plan")
                    voicemailplan = gr.Radio(["No", "Yes"], value="No", label="Voicemail Plan")
                    numbervmailmessages = gr.Number(value=5, label="Voicemail Messages")
                    totaldayminutes = gr.Number(value=200.0, label="Total Day Minutes")
                    totaldaycalls = gr.Number(value=80, label="Total Day Calls")
                    totaldaycharge = gr.Number(value=30.0, label="Total Day Charge")
                    totaleveminutes = gr.Number(value=180.0, label="Total Eve Minutes")
                    totalevecalls = gr.Number(value=90, label="Total Eve Calls")
                    totalevecharge = gr.Number(value=20.0, label="Total Eve Charge")
                    totalnightminutes = gr.Number(value=150.0, label="Total Night Minutes")
                    totalnightcalls = gr.Number(value=70, label="Total Night Calls")
                    totalnightcharge = gr.Number(value=15.0, label="Total Night Charge")
                    totalintlminutes = gr.Number(value=15.0, label="Total Intl Minutes")
                    totalintlcalls = gr.Number(value=10, label="Total Intl Calls")
                    totalintlcharge = gr.Number(value=5.0, label="Total Intl Charge")
                    numbercustomerservicecalls = gr.Number(value=1, label="Customer Service Calls")
                    single_predict_btn = gr.Button("Predict Single Customer")
                    single_output = gr.Textbox(label="Prediction & CBA", lines=12)


    gr.Markdown("---")
    # Wiring single predict form
    single_predict_btn.click(
        fn=single_predict_ui,
        inputs=[model_dropdown,
                accountlength, internationalplan, voicemailplan, numbervmailmessages,
                totaldayminutes, totaldaycalls, totaldaycharge,
                totaleveminutes, totalevecalls, totalevecharge,
                totalnightminutes, totalnightcalls, totalnightcharge,
                totalintlminutes, totalintlcalls, totalintlcharge,
                numbercustomerservicecalls,
                profit_in, cost_in, loss_in],
        outputs=[single_output]
    )

    
# Start server when run directly
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

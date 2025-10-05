# app_utils.py
import plotly.io as pio
from datetime import datetime
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, Image
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
import matplotlib.pyplot as plt
import shap
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow import keras
import streamlit as st
import plotly.graph_objects as go
import os
from typing import Iterable
import google.generativeai as genai
# ---------- Paths ----------
BUNDLE_DIR = Path("./model_bundle_merged")
NB5_CSV = Path("data_reports/nb5_all_configs_agg_clean.csv")
NB6_CSV = Path("data_reports/all_nb6_results.csv")

# ---------- Safe numerics ----------


def _to_logit(p, eps=1e-7):
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))



# ---------- Load artifacts ----------


def load_bundle():
    """
    Loads:
      - features.json            -> base feature names (list[str])
      - flags.json               -> missing flag names (list[str])
      - interaction.json         -> interaction feature names (list[str], e.g., "age__x__bmi")
      - train_medians.json       -> dict medians for base+flags+interactions (at least for base)
      - imputer.pkl, scaler.pkl, rf.pkl, meta.pkl
      - deep_ann.h5
      - threshold.json           -> tau
      - bg.npy (optional)        -> LIME background (already standardized) 
      - interaction_pairs.json   -> optional explicit pairs
    """
    with open(BUNDLE_DIR / "features.json") as f:
        use_cols = json.load(f)
    # with open(BUNDLE_DIR / "flags.json") as f:
    #     flag_cols = json.load(f)
    # with open(BUNDLE_DIR / "interaction.json") as f:
    #     interaction_cols = json.load(f)
    with open(BUNDLE_DIR / "train_medians.json") as f:
        train_medians = json.load(f)
    with open(BUNDLE_DIR / "threshold.json") as f:
        tau = json.load(f)["tau"]

    imputer = joblib.load(BUNDLE_DIR / "imputer.pkl")
    scaler = joblib.load(BUNDLE_DIR / "scaler.pkl")
    rf = joblib.load(BUNDLE_DIR / "rf.pkl")
    meta = joblib.load(BUNDLE_DIR / "meta.pkl")

    deep_ann = keras.models.load_model(
        BUNDLE_DIR / "deep_ann.h5", compile=False)
    # compile for predict; optimizer/loss won't be used further
    deep_ann.compile(optimizer="adam", loss="binary_crossentropy")

    # Final columns order used for training (base + flags + interactions)
    final_cols = list(use_cols)

    # background for LIME (std space)
    X_bg = None
    bg_path = BUNDLE_DIR / "X_bg.npy"
    if bg_path.exists():
        X_bg = np.load(bg_path)

    # # optional explicit pairs
    # pairs_path = BUNDLE_DIR / "interaction_pairs.json"
    # if pairs_path.exists():
    #     with open(pairs_path) as f:
    #         interaction_pairs = [tuple(x) for x in json.load(f)]
    # else:
    #     # derive pairs from interaction_cols by splitting
    #     interaction_pairs = []
    #     for c in interaction_cols:
    #         if "__x__" in c:
    #             a, b = c.split("__x__", 1)
    #             interaction_pairs.append((a, b))
    #     # ensure uniqueness in case of duplicates
    #     interaction_pairs = sorted(set(interaction_pairs))

    return {
        "use_cols": use_cols,
        # "flag_cols": flag_cols,
        # "interaction_cols": interaction_cols,
        # "interaction_pairs": interaction_pairs,
        "final_cols": final_cols,
        "imputer": imputer,
        "scaler": scaler,
        "rf": rf,
        "deep_ann": deep_ann,
        "meta": meta,
        "tau": tau,
        "train_medians": train_medians,
        "X_bg": X_bg
    }

# ---------- NB5 / NB6 loaders ----------


def load_nb5():
    if NB5_CSV.exists():
        df = pd.read_csv(NB5_CSV)
        df.columns = [c.strip().lower() for c in df.columns]
        return df
    return pd.DataFrame()


def load_nb6():
    if NB6_CSV.exists():
        df = pd.read_csv(NB6_CSV)
        df.columns = [c.strip().lower() for c in df.columns]
        if "scenario" in df.columns:
            sc = df["scenario"].astype(str)
            is_pair = sc.str.startswith("pair_")
            is_lodo = sc.str.startswith("lodo_")
            is_merged = sc.str.startswith("merged_cv_fold")
            df["scheme"] = np.where(is_pair, "pair",
                                    np.where(is_lodo, "lodo",
                                             np.where(is_merged, "merged_cv", "unknown")))
            tr, te = [], []
            for s in sc:
                if s.startswith("pair_") and "_to_" in s:
                    body = s.replace("pair_", "")
                    a, b = body.split("_to_", 1)
                    tr.append(a)
                    te.append(b)
                elif s.startswith("lodo_") and "_to_" in s:
                    body = s.replace("lodo_", "")
                    a, b = body.split("_to_", 1)
                    tr.append(a.replace("_", "+"))
                    te.append(b)
                elif s.startswith("merged_cv_fold"):
                    tr.append("ALL")
                    te.append("ALL")
                else:
                    tr.append("UNK")
                    te.append("UNK")
            df["train"] = tr
            df["test"] = te
        return df
    return pd.DataFrame()

# ---------- Random input generator ----------


def generate_random_user(bundle, jitter=0.10, seed=None):
    """
    Generate a more randomized, yet realistic synthetic patient profile.
    - Uses medians as baseline.
    - Adds controlled noise and random feature scaling per feature group.
    - Supports stochastic variation across calls (no fixed seed unless given).
    """
    import random

    if seed is not None:
        rng = np.random.RandomState(seed)
        random.seed(seed)
    else:
        # Use system randomness
        rng = np.random.RandomState(np.random.randint(0, 10_000_000))
        random.seed()

    use_cols = bundle["use_cols"]
    med = bundle["train_medians"]
    user = {}

    # Define rough feature ranges (factor of median)
    scale_up = 1.2 + rng.rand() * 0.6   # random 1.2–1.8
    scale_down = 0.8 - rng.rand() * 0.3  # random 0.5–0.8

    for c in use_cols:
        m = float(med.get(c, 0.0))
        if np.isnan(m):
            m = 1.0

        # random direction: up/down median
        direction = rng.choice([-1, 1])
        # random jitter scale: 5–30% base variation
        local_jitter = rng.uniform(0.05, jitter * 3)
        scale = abs(m) if m != 0.0 else 1.0
        v = m + direction * local_jitter * scale

        # occasional global outlier (3% chance)
        if rng.rand() < 0.03:
            v *= rng.uniform(1.5, 2.0)

        # enforce logical ranges
        if c in ("age",):
            v = np.clip(v, 18, 90)
        elif c in ("preg_count",):
            v = np.clip(round(v), 0, 15)
        elif c in ("glucose",):
            v = np.clip(v, 50, 250)
        elif c in ("bmi",):
            v = np.clip(v, 15, 60)
        elif c in ("insulin",):
            v = np.clip(v, 0, 400)
        elif c in ("dbp",):
            v = np.clip(v, 40, 120)
        elif c in ("SkinThickness",):
            v = np.clip(v, 5, 80)

        user[c] = round(float(v),2) if c in ("bmi", "SkinThickness") else int(v)

    return user


# ---------- Single-input preparation ----------


def prepare_single_input(user_dict, bundle):
    """
    Builds a 1-row DF for prediction:
      1) base features (use_cols) from inputs, blanks->NaN
      2) missing flags (only those in training)
      3) impute with trained imputer for base features
      4) compute interactions actually used in training
      5) align to final_cols
      6) standardize
    Returns: (df_one_raw, X_std)
      - df_one_raw = post-imputation raw row with flags+interactions (not standardized)
      - X_std      = standardized matrix (1, n_features)
    """
    use_cols = bundle["use_cols"]
    # flag_cols = bundle["flag_cols"]
    # interaction_cols = bundle["interaction_cols"]
    # interaction_pairs = bundle["interaction_pairs"]
    final_cols = bundle["final_cols"]

    # 1) Base row
    base = {}
    for c in use_cols:
        v = user_dict.get(c, None)
        if v is None or (isinstance(v, str) and v.strip() == ""):
            base[c] = np.nan
        else:
            base[c] = float(v)
    df_one = pd.DataFrame([base])

    # 2) Missing flags (only those model used)
    # for c in use_cols:
    #     f = f"{c}_na"
    #     if f in flag_cols:
    #         df_one[f] = df_one[c].isna().astype(int)
    #     else:
    #         df_one[f] = 0

    # 3) Impute base features
    X_base = df_one[use_cols].values.astype(float)
    X_imp = bundle["imputer"].transform(X_base)
    df_one[use_cols] = X_imp

    # # 4) Interactions: only compute those present in training
    # for (a, b) in interaction_pairs:
    #     colname = f"{a}__x__{b}"
    #     if a in df_one.columns and b in df_one.columns:
    #         df_one[colname] = df_one[a] * df_one[b]
    #     else:
    #         df_one[colname] = 0.0

    # # 5) Ensure all training-time flag/interaction columns exist
    # for f in flag_cols:
    #     if f not in df_one.columns:
    #         df_one[f] = 0.0
    # for ic in interaction_cols:
    #     if ic not in df_one.columns:
    #         df_one[ic] = 0.0

    # 6) Final ordering + standardize
    for c in final_cols:
        if c not in df_one.columns:
            df_one[c] = 0.0
    df_one = df_one[final_cols].astype(float)
    X_std = bundle["scaler"].transform(df_one.values)
    return df_one, X_std

# ---------- Meta predict ----------


def meta_predict_proba(X_std, bundle):
    """
    Returns p_meta (n,), p_rf(n,), p_nn(n,)
    """
    rf = bundle["rf"]
    nn = bundle["deep_ann"]
    meta = bundle["meta"]

    p_rf = rf.predict_proba(X_std)[:, 1]
    p_nn = nn.predict(X_std, verbose=0).ravel()
    Z = np.column_stack([_to_logit(p_rf), _to_logit(p_nn)])
    p_meta = meta.predict_proba(Z)[:, 1]
    return p_meta, p_rf, p_nn

# ---------- SHAP for the stack (meta over RF/NN) ----------
@st.cache_resource
def get_shap_explainer_cached(_bundle):
    """
    Streamlit will ignore hashing for args prefixed with '_' so you can
    pass the whole bundle safely. Returns a cached KernelExplainer.
    """
    return build_shap_explainer(_bundle)

# ---------------- NEW: Plotly waterfall for local SHAP ----------------


def _plotly_waterfall_from_explanation(explanation, top_k=None, originals_only=False):
    """
    Build a Plotly waterfall figure from a shap.Explanation (single row).
    - top_k: keep only top-k |contrib|
    - originals_only: drop *_na flags and __x__ interactions
    """
    vals = np.array(explanation.values).reshape(-1)
    feats = np.array(explanation.feature_names)
    data = np.array(explanation.data).reshape(-1)
    base = float(np.array(explanation.base_values).ravel()[0])

    # Filter original-only if requested
    keep = np.ones_like(vals, dtype=bool)
    if originals_only:
        keep &= ~np.char.endswith(feats.astype(str), "_na")
        keep &= ~np.core.defchararray.find(
            feats.astype(str), "__x__").astype(int) == 0
        keep = ~np.char.endswith(feats.astype(str), "_na") & (
            np.char.find(feats.astype(str), "__x__") == -1)
    vals = vals[keep]
    feats = feats[keep]
    data = data[keep]

    # Top-k by absolute contribution
    if top_k is not None and top_k > 0:
        order = np.argsort(np.abs(vals))[::-1][:top_k]
        feats, vals, data = feats[order], vals[order], data[order]

    # Prepare Plotly waterfall
    measures = ["relative"] * len(vals) + ["total"]
    x_labels = list(feats) + ["prediction"]
    y_vals = list(vals) + [base + np.sum(vals)]

    fig = go.Figure(go.Waterfall(
        measure=measures,
        x=x_labels,
        text=[f"{v:+.3f}" for v in y_vals],
        y=y_vals,
        connector={"line": {"color": "rgba(150,150,150,0.4)"}},
        increasing={"marker": {"color": "#2ca02c"}},
        decreasing={"marker": {"color": "#d62728"}},
        totals={"marker": {"color": "#9467bd"}}
    ))
    fig.update_layout(
        title="SHAP Waterfall (probability space)",
        yaxis_title="Contribution",
        xaxis_title="Features",
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def _meta_p1_fn(bundle):
    """
    Returns a function f(X_std) -> p_meta (1D) in the standardized feature space.
    This exactly mirrors the stack LR used in meta_predict_proba().
    """
    rf = bundle["rf"]
    nn = bundle["deep_ann"]
    meta = bundle["meta"]

    def f(X_std):
        X_std = np.asarray(X_std, dtype=float)
        p_rf = rf.predict_proba(X_std)[:, 1]
        p_nn = nn.predict(X_std, verbose=0).ravel()
        z_rf = _to_logit(p_rf)
        z_nn = _to_logit(p_nn)
        Z = np.column_stack([z_rf, z_nn])  # order = [rf, deep_ann]
        return meta.predict_proba(Z)[:, 1]  # 1D
    return f


def build_shap_explainer(bundle):
    """
    Build a SHAP KernelExplainer over the meta (stack) probability in the same
    standardized feature space used during training/inference.
    Uses bundle['X_bg'] if available; otherwise a small synthetic background.
    """
    X_bg = bundle.get("X_bg", None)
    if X_bg is None:
        n_features = len(bundle["final_cols"])
        # small synthetic background in standardized space
        X_bg = np.random.normal(0.0, 1.0, size=(200, n_features))
        st.warning("SHAP: Using synthetic background; consider saving X_bg.npy.")

    f_meta = _meta_p1_fn(bundle)
    explainer = shap.KernelExplainer(f_meta, X_bg)
    return explainer

def explain_with_shap(bundle, X_std, max_display=12, nsamples="auto",
                      explainer=None, originals_only=False, top_k=None):
    """
    Compute SHAP local explanation for the single row X_std (1,n_features).
    Returns (explanation, plotly_fig).
    """
    if explainer is None:
        explainer = build_shap_explainer(bundle)

    xi = np.asarray(X_std[:1], dtype=float)

    shap_vals = explainer.shap_values(xi, nsamples=nsamples)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]
    shap_vals = np.array(shap_vals).reshape(-1)

    base = explainer.expected_value
    if isinstance(base, (list, tuple, np.ndarray)):
        base = np.array(base).ravel()[0]

    explanation = shap.Explanation(
        values=shap_vals,
        base_values=base,
        data=xi[0],
        feature_names=bundle["final_cols"]
    )

    # Interactive (Plotly) waterfall; respect originals_only/top_k
    fig = _plotly_waterfall_from_explanation(
        explanation, top_k=top_k, originals_only=originals_only
    )
    return explanation, fig


def is_original_feature(name: str) -> bool:
    """Original feature = not a missing flag and not an interaction."""
    return (not name.endswith("_na")) and ("__x__" not in name)


def subset_explanation(explanation: shap.Explanation, keep_idx: Iterable[int]) -> shap.Explanation:
    """Return a new shap.Explanation with only indices in keep_idx (no recomputation)."""
    keep_idx = np.asarray(list(keep_idx), dtype=int)
    return shap.Explanation(
        values=np.array(explanation.values)[keep_idx],
        base_values=explanation.base_values,
        data=np.array(explanation.data)[keep_idx],
        feature_names=[explanation.feature_names[i] for i in keep_idx]
    )

# ---- Cache a single explainer instance safely (Streamlit-friendly) ----


def _shap_key_from_bundle(bundle: dict) -> str:
    # something stable & hashable: number of features + RF trees + NN params
    rf = bundle["rf"]
    nn = bundle["deep_ann"]
    nfeat = len(bundle["final_cols"])
    ntrees = getattr(rf, "n_estimators", 0)
    nn_params = int(np.sum([np.prod(w.shape) for w in nn.get_weights()]))
    return f"nfeat={nfeat}|ntrees={ntrees}|nnparams={nn_params}"


def get_shap_explainer(bundle: dict):
    """Singleton per session & model signature (no @cache on dict)."""
    key = "_shap_explainer_" + _shap_key_from_bundle(bundle)
    if key not in st.session_state:
        st.session_state[key] = build_shap_explainer(bundle)
    return st.session_state[key]


def explain_with_shap(bundle, X_std, nsamples="auto"):
    """
    Compute a SHAP local explanation for the single standardized row (1, n_features).
    Returns shap.Explanation for ALL features (final_cols).
    """
    explainer = get_shap_explainer(bundle)
    xi = np.asarray(X_std[:1], dtype=float)
    shap_vals = explainer.shap_values(xi, nsamples=nsamples)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]
    shap_vals = np.array(shap_vals).reshape(-1)

    base = explainer.expected_value
    if isinstance(base, (list, tuple, np.ndarray)):
        base = np.array(base).ravel()[0]

    return shap.Explanation(
        values=shap_vals,
        base_values=base,
        data=xi[0],
        feature_names=bundle["final_cols"]
    )


def fig_waterfall_matplotlib(explanation: shap.Explanation, max_display=12):
    """Return a Matplotlib waterfall figure for the provided (possibly subset) explanation."""
    plt.figure(figsize=(8.5, 6))
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    return plt.gcf()


def fig_waterfall_plotly(explanation: shap.Explanation, top_k=12):
    """
    Interactive bar-style 'waterfall' (signed contribution) using Plotly.
    We sort by |contrib|, take top_k, and show horizontal bars.
    """
    vals = np.array(explanation.values)
    feats = np.array(explanation.feature_names)
    data_vals = np.array(explanation.data)

    order = np.argsort(np.abs(vals))[::-1][:int(top_k)]
    sel_feats = feats[order]
    sel_vals = vals[order]
    sel_data = data_vals[order]

    colors = np.where(sel_vals >= 0, "#1f77b4", "#d62728")
    fig = go.Figure(go.Bar(
        x=sel_vals, y=sel_feats,
        orientation="h",
        marker_color=colors,
        hovertemplate="<b>%{y}</b><br>Contribution=%{x:.4f}<extra></extra>"
    ))
    fig.update_layout(
        height=450, margin=dict(l=120, r=20, t=40, b=40),
        title="Contribution to diagnosis",
    )
    fig.update_yaxes(autorange="reversed")
    return fig

# ====== NEW: Plain-language LLM summary ======


def llm_summary_from_shap(
    explanation,
    prob_pos: float,
    tau: float,
    top_k: int = 10,
    model_name: str = "gemini-2.5-pro"
) -> str:
    """
    Summarize the SHAP explanation in plain language with Gemini.
    Requires GOOGLE_API_KEY (st.secrets or env). Graceful fallback if not set.
    """
    api_key = st.secrets.get(
        "GOOGLE_API_KEY", os.environ.get("GOOGLE_API_KEY", "")
    )
    if not api_key:
        return "[LLM summary unavailable] Set GOOGLE_API_KEY in Streamlit secrets or environment."

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
    except Exception:
        return "[LLM summary unavailable] gemini package not available."
    FEATURE_INFO = {"age": {"label": "Age", "unit": "years", "desc": "Age of the individual"}, "bmi": {"label": "Body Mass Index (BMI)", "unit": "kg/m²", "desc": "Weight/height²"}, "dbp": {"label": "Blood Pressure", "unit": "mmHg", "desc": "Resting BP"}, "SkinThickness": {"label": "Skinfold Thickness", "unit": "mm", "desc": "Triceps skinfold"}, "glucose": {
        "label": "Plasma Glucose (OGTT)", "unit": "mg/dL", "desc": "Fasting glucose"}, "insulin": {"label": "Fasting Insulin", "unit": "µU/mL", "desc": "Serum insulin"}, "preg_count": {"label": "Pregnancy Count", "unit": "count", "desc": "Number of pregnancies"}, }

    # vals = np.array(explanation.values)


    # feats = np.array(explanation.feature_names)
    # data_vals = raw_value
    vals = explanation['contrib']

    feats = explanation['feature']
    data_vals = explanation['value']


    # Get top_k features by absolute SHAP value
    order = np.argsort(np.abs(vals))[::-1][:int(top_k)]

    rows = []
    for i in order:
        f = feats[i]
        label = FEATURE_INFO.get(f, {}).get("label", f)
        unit = FEATURE_INFO.get(f, {}).get("unit", "")
        contrib = vals[i]
        value = data_vals[i]
        # No hardcoded range — LLM will infer
        rows.append(f"{label} = {value:.2f} {unit} (contrib {contrib:+.4f})")

    bullet = "\n".join(f"- {r}" for r in rows)

    prompt = f"""
You are assisting a clinician. A stacked logistic model predicted diabetes probability p = {prob_pos:.3f} 
with decision threshold τ = {tau:.2f}. The goal is to generate both an interpretability summary and realistic counterfactual reasoning.

Feature values, units, and SHAP contributions:
{bullet}

Tasks:
1. Identify which features most strongly increased or decreased predicted risk.
2. For each, infer the healthy or normal range from your clinical knowledge.
3. Provide concise reasoning (4-6 bullet points) explaining:
   - which risk factors are elevated or abnormal
   - why they increase risk physiologically
   - which features are modifiable (e.g., glucose, BMI, blood pressure, insulin)
   - **what small realistic adjustments (e.g., lower glucose, reduce BMI)** could potentially reduce the predicted probability below τ.
4. Avoid discussing immutable features (e.g., age, pregnancy count).
5. Write in plain language suitable for clinicians and patients.
6. Do not include any introductory filler or disclaimers; start directly with the bullet points.
"""

    print(prompt)

    try:
        resp = model.generate_content(prompt)
        return resp.text.strip()
    except Exception as e:
        return f"[LLM summary unavailable] {e}"


def generate_prediction_report_pdf(pred_info, shap_df=None, llm_text=None, shap_fig=None):
    """
    Generate a clean, kaleido-free PDF report:
    - Feature scales drawn with matplotlib
    - SHAP chart rendered via Plotly's internal renderer (no kaleido)
    - LLM explanation included
    - Contains disclaimer: for experimental use only
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=36, bottomMargin=36, leftMargin=40, rightMargin=40
    )

    styles = getSampleStyleSheet()
    story = []

    # --- Styles ---
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Title'],
        fontSize=18,
        textColor=colors.HexColor("#1f4e79"),
        alignment=1,
        spaceAfter=12
    )
    heading_style = ParagraphStyle(
        'HeadingStyle',
        parent=styles['Heading2'],
        textColor=colors.HexColor("#0b5394"),
        spaceBefore=8, spaceAfter=4
    )
    note_style = ParagraphStyle(
        'NoteStyle',
        parent=styles['Normal'],
        textColor=colors.grey,
        fontSize=9,
        alignment=1,
        spaceBefore=10,
    )

    # --- Header ---
    story.append(Paragraph("Diabetes Prediction Report", title_style))
    story.append(
        Paragraph("Explainable Stacked Ensemble System", styles["Normal"]))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#1f4e79")))
    story.append(Spacer(1, 12))

    # --- Metadata & disclaimer ---
    story.append(Paragraph(
        f"<b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        styles["Normal"]
    ))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "<b>Disclaimer:</b> This report is generated for <b>research and experimental purposes only</b>. "
        "It is <b>not intended for clinical diagnosis or treatment decisions.</b>",
        styles["Normal"]
    ))
    story.append(Spacer(1, 12))

    # --- Prediction Summary ---
    story.append(Paragraph("Prediction Summary", heading_style))
    preds = [
        # ["Meta (Stack) P(+)", f"{pred_info['p']:.3f}"],
        # ["RF P(+)", f"{pred_info['p_rf']:.3f}"],
        # ["DeepANN P(+)", f"{pred_info['p_nn']:.3f}"],
        # ["Threshold (τ)", f"{pred_info['tau']:.2f}"],
        ["Final Decision", "POSITIVE" if pred_info["yhat"] == 1 else "NEGATIVE"],
    ]
    table = Table(preds, hAlign="LEFT", colWidths=[200, 120])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ]))
    story.append(table)
    story.append(Spacer(1, 12))

    # --- Input Features with Scale Bars ---
    story.append(Paragraph("Diagnostic Parameters", heading_style))
    feature_ranges = {
        "age": (18, 90),
        "bmi": (15, 60),
        "glucose": (50, 250),
        "insulin": (0, 400),
        "dbp": (40, 120),
        "SkinThickness": (5, 80),
        "preg_count": (0, 15)
    }

    def make_scale_bar(value, min_val, max_val):
        fig, ax = plt.subplots(figsize=(3, 0.25))
        ax.barh(0, max_val - min_val, color="#ddd", height=0.3)
        ax.barh(0, value - min_val, color="#2b78e4", height=0.3)
        ax.axvline(value, color="red", linewidth=2)
        ax.set_xlim(min_val, max_val)
        ax.axis("off")
        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        return buf

    for feat, val in pred_info["inputs"].items():
        minv, maxv = feature_ranges.get(feat, (0, 1))
        story.append(Paragraph(f"<b>{feat}</b>: {val:.2f}", styles["Normal"]))
        bar_img = make_scale_bar(val, minv, maxv)
        story.append(Image(bar_img, width=220, height=10))
        story.append(Spacer(1, 3))

    story.append(Spacer(1, 10))

    # --- SHAP Waterfall Chart (without kaleido) ---
    if shap_fig is not None:
        # story.append(
        #     Paragraph("Contribution to the Diagnosis", heading_style))
        img_bytes = pio.to_image(shap_fig, format="png")
        img_buf = BytesIO(img_bytes)
        story.append(Image(img_buf, width=440, height=300))
        story.append(Spacer(1, 15))

    # --- LLM Explanation ---
    if llm_text:
        story.append(
            Paragraph("LLM-Generated Clinical Explanation", heading_style))
        formatted_text = llm_text.replace("\n", "<br/>")
        story.append(Paragraph(formatted_text, styles["Normal"]))
        story.append(Spacer(1, 12))

    # --- Footer / Disclaimer ---
    story.append(HRFlowable(width="100%", color=colors.lightgrey))
    story.append(Paragraph(
        "This report is automatically generated using an experimental Explainable AI system for diabetes risk prediction. "
        "The outputs are for demonstration and educational purposes only.",
        note_style
    ))
    # story.append(Spacer(1, 8))
    # story.append(Paragraph(
    #     "© 2025 Tauhidul Islam | Built with Streamlit & ReportLab",
    #     note_style
    # ))

    doc.build(story)
    buffer.seek(0)
    return buffer

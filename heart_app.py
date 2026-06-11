import pickle
import json
import warnings
import logging
from pathlib import Path
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from typing import Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)

BASE = Path(__file__).resolve().parent

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="wide")

# ── Theme ─────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

/* Basic App Layout */
html, body, .stApp {
    background-color: #FFFFFF !important;
    color: #1C1C1A !important;
    font-family: 'DM Sans', sans-serif !important;
}
* { box-shadow: none !important; text-shadow: none !important; }
[data-testid="stSidebar"] { display: none !important; }
section[data-testid="stMain"] { background: #FFFFFF !important; }

h1, h2, h3, h4 { color: #1C1C1A !important; font-weight: 700 !important; }

/* ── Force Labels to be Visible (Dark) ── */
label, [data-testid="stWidgetLabel"] p, label p {
    color: #1C1C1A !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    margin-bottom: 0.15rem !important;
}

/* ── Selectbox ── */
[data-baseweb="select"] { background: #FAFAF7 !important; }
[data-baseweb="select"] > div {
    background: #FAFAF7 !important;
    border: 1.5px solid #D4D9BE !important;
    border-radius: 6px !important;
    color: #1C1C1A !important;
}
[data-baseweb="select"] span { color: #1C1C1A !important; }
[data-baseweb="select"] svg { fill: #4A5A2A !important; }

/* Dropdown list */
[data-baseweb="popover"],
[data-baseweb="menu"],
ul[role="listbox"] {
    background: #FFFFFF !important;
    border: 1.5px solid #D4D9BE !important;
    border-radius: 6px !important;
}
li[role="option"] {
    background: #FFFFFF !important;
    color: #1C1C1A !important;
}
li[role="option"]:hover,
li[aria-selected="true"] {
    background: #F0F4E8 !important;
    color: #1C1C1A !important;
}

/* ── Sliders ── */
[data-testid="stSlider"] > div > div > div > div {
    background: #4A5A2A !important;
}
[data-testid="stSlider"] [role="slider"] {
    background: #4A5A2A !important;
    border: 2px solid #FFFFFF !important;
}

/* Slider label (the thumb value number) */
[data-testid="stThumbValue"] {
    color: #4A5A2A !important;
    font-weight: 600 !important;
}

/* ── Number input ── */
input[type="number"] {
    background: #FAFAF7 !important;
    border: 1.5px solid #D4D9BE !important;
    border-radius: 6px !important;
    color: #1C1C1A !important;
}

/* ── Buttons ── */
.stButton > button[kind="primary"] {
    background-color: #4A5A2A !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.65rem 2.5rem !important;
}
.stButton > button[kind="primary"]:hover { background-color: #3A4820 !important; }
.stButton > button {
    background: #F7F7F2 !important;
    color: #4A5A2A !important;
    border: 1.5px solid #D4D9BE !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
}

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: #F7F7F2 !important;
    border: 1.5px solid #E4E8D4 !important;
    border-radius: 8px !important;
    padding: 1rem !important;
}
[data-testid="stMetricValue"] {
    color: #4A5A2A !important;
    font-family: 'DM Mono', monospace !important;
    font-weight: 600 !important;
}
[data-testid="stMetricLabel"] { color: #6B6B65 !important; font-size: 0.82rem !important; }

/* ── Misc ── */
hr { border: none !important; border-top: 1.5px solid #E4E8D4 !important; margin: 1.2rem 0 !important; }
.stCaption, small, [data-testid="stCaptionContainer"] { color: #6B6B65 !important; }
.stSuccess { background: #F0F4E8 !important; border-left: 4px solid #4A5A2A !important; }
.stSuccess * { color: #2A3A12 !important; }
.stError   { background: #FAF0EE !important; border-left: 4px solid #B85C3A !important; }
.stError *   { color: #7A2A10 !important; }
.stWarning { background: #FDF6EC !important; border-left: 4px solid #C4882A !important; }
.stWarning * { color: #7A4A10 !important; }
.stInfo    { background: #F4F4EE !important; border-left: 4px solid #6B7A3E !important; }
.stInfo *    { color: #2A3A12 !important; }

/* ── Tabs ── */
[data-baseweb="tab-list"] {
    background: #F7F7F2 !important;
    border-bottom: 2px solid #D4D9BE !important;
    gap: 0 !important;
}
[data-baseweb="tab"] {
    background: transparent !important;
    color: #6B6B65 !important;
    border: none !important;
    padding: 0.6rem 1.4rem !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
}
[aria-selected="true"] {
    background: #FFFFFF !important;
    color: #4A5A2A !important;
    border-bottom: 2px solid #4A5A2A !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border: 1.5px solid #E4E8D4 !important; border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)

# ── Colors ────────────────────────────────────────────────────────────────────

OLIVE = "#4A5A2A"; OLIVE_MID = "#6B7A3E"; OLIVE_PALE = "#D4D9BE"
RUST  = "#B85C3A"; BG = "#FFFFFF"; TEXT = "#1C1C1A"; MUTED = "#6B6B65"

def chart_style(ax: Axes, fig: Figure) -> None:
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(OLIVE_PALE)
    ax.spines["bottom"].set_color(OLIVE_PALE)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)

# ── Load / train ──────────────────────────────────────────────────────────────

@st.cache_resource
def load_or_train() -> tuple[
    ClassifierMixin | None,
    StandardScaler | None,
    pd.DataFrame | None,
    dict | None,
    str | None,
    str | None     # target column name
]:
    data_path = BASE / "heart.csv"
    if not data_path.exists():
        return None, None, None, None, "heart.csv not found.", None

    df = pd.read_csv(data_path)

    # Auto-detect target column
    target_col = "condition" if "condition" in df.columns else "target"
    if target_col not in df.columns:
        return None, None, None, None, f"Could not find target column. Columns: {list(df.columns)}", None

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    model_path  = BASE / "model.pkl"
    scaler_path = BASE / "scaler.pkl"

    if model_path.exists() and scaler_path.exists():
        with open(model_path,  "rb") as f: model  = pickle.load(f)
        with open(scaler_path, "rb") as f: scaler = pickle.load(f)
    else:
        X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        scaler = StandardScaler()
        model  = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(scaler.fit_transform(X_tr), y_tr)

    results_path = BASE / "results.json"
    if results_path.exists():
        with open(results_path) as f: results = json.load(f)
    else:
        X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        sc2 = StandardScaler()
        results = {}
        for name, m in {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
            "Gradient Boosting":   GradientBoostingClassifier(random_state=42),
        }.items():
            m.fit(sc2.fit_transform(X_tr2), y_tr2)
            yp    = m.predict(sc2.transform(X_te2))
            yprob = m.predict_proba(sc2.transform(X_te2))[:, 1]
            results[name] = {
                "Accuracy":  round(float(accuracy_score(y_te2, yp)),  4),
                "Precision": round(float(precision_score(y_te2, yp)), 4),
                "Recall":    round(float(recall_score(y_te2, yp)),    4),
                "F1 Score":  round(float(f1_score(y_te2, yp)),        4),
                "ROC-AUC":   round(float(roc_auc_score(y_te2, yprob)),4),
            }

    return model, scaler, df, results, None, target_col


with st.spinner("Loading model..."):
    model, scaler, df, results, err, target_col = load_or_train()

if err:
    st.error(err)
    st.caption("Download heart.csv from https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset")
    st.stop()

assert model      is not None
assert scaler     is not None
assert df         is not None
assert results    is not None
assert target_col is not None

# ── Feature metadata ──────────────────────────────────────────────────────────

FEATURES: dict[str, dict] = {
    "age":      {"label": "Age (years)",                   "type": "slider", "min": 20,  "max": 80,  "default": 52,  "step": 1},
    "sex":      {"label": "Sex",                           "type": "select", "options": {0:"Female", 1:"Male"}},
    "cp":       {"label": "Chest Pain Type",               "type": "select", "options": {0:"Typical Angina", 1:"Atypical Angina", 2:"Non-anginal", 3:"Asymptomatic"}},
    "trestbps": {"label": "Resting Blood Pressure (mmHg)", "type": "slider", "min": 80,  "max": 200, "default": 125, "step": 1},
    "chol":     {"label": "Cholesterol (mg/dl)",           "type": "slider", "min": 100, "max": 600, "default": 240, "step": 1},
    "fbs":      {"label": "Fasting Blood Sugar > 120",     "type": "select", "options": {0:"No", 1:"Yes"}},
    "restecg":  {"label": "Resting ECG",                   "type": "select", "options": {0:"Normal", 1:"ST-T Abnormality", 2:"LV Hypertrophy"}},
    "thalach":  {"label": "Max Heart Rate Achieved",       "type": "slider", "min": 60,  "max": 220, "default": 150, "step": 1},
    "exang":    {"label": "Exercise-Induced Angina",       "type": "select", "options": {0:"No", 1:"Yes"}},
    "oldpeak":  {"label": "ST Depression (exercise)",      "type": "slider", "min": 0.0, "max": 6.0, "default": 1.0, "step": 0.1},
    "slope":    {"label": "ST Slope",                      "type": "select", "options": {0:"Upsloping", 1:"Flat", 2:"Downsloping"}},
    "ca":       {"label": "Major Vessels Colored",         "type": "select", "options": {0:"0", 1:"1", 2:"2", 3:"3"}},
    "thal":     {"label": "Thalassemia",                   "type": "select", "options": {1:"Normal", 2:"Fixed Defect", 3:"Reversible Defect"}},
}

def render_input(key: str, cfg: dict, col: Any) -> float | int:
    with col:
        if cfg["type"] == "slider":
            return st.slider(cfg["label"], cfg["min"], cfg["max"], cfg["default"], step=cfg["step"])
        opts = cfg["options"]
        return st.selectbox(cfg["label"], list(opts.keys()), format_func=lambda x: opts[x])  # type: ignore[return-value]

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="text-align:center;padding:2rem 0 0.2rem 0;">
  <h1 style="font-size:2.8rem;letter-spacing:-1px;margin-bottom:0;">❤️ Heart Disease Predictor</h1>
  <p style="font-size:1rem;color:#6B6B65;margin-top:0.4rem;">
    End-to-end ML pipeline — EDA → preprocessing → model comparison → deployed inference
  </p>
</div>
<hr>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔍 Predict", "📊 Model Performance", "ℹ️ About"])

# ═══════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ═══════════════════════════════════════════════════════════

with tab1:
    st.markdown("#### Patient data")
    st.caption("Adjust all 13 clinical values then click Predict.")
    st.write("")

    keys    = list(FEATURES.keys())
    c1, c2, c3 = st.columns(3, gap="large")
    col_map = [c1]*5 + [c2]*4 + [c3]*4
    values: dict[str, float | int] = {}
    for key, col in zip(keys, col_map):
        values[key] = render_input(key, FEATURES[key], col)

    st.write("")
    _, btn_col, _ = st.columns([3, 2, 3])
    with btn_col:
        predict_btn = st.button("Predict risk →", type="primary")

    if predict_btn:
        input_arr    = np.array([[values[k] for k in keys]])
        input_scaled = scaler.transform(input_arr)
        pred  = int(model.predict(input_scaled)[0])             # type: ignore[union-attr]
        prob  = float(model.predict_proba(input_scaled)[0][1])  # type: ignore[union-attr]
        ham   = 1.0 - prob

        st.divider()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Prediction",           "High Risk ⚠️" if pred == 1 else "Low Risk ✅")
        c2.metric("Disease Probability",  f"{prob*100:.1f}%")
        c3.metric("Confidence",           f"{max(prob, ham)*100:.1f}%")
        c4.metric("Model",                "Random Forest")

        st.write("")

        # ── HTML/CSS risk bar — no matplotlib ─────────────────────────────
        bar_color  = RUST  if pred == 1 else OLIVE
        rust_pct   = f"{prob*100:.1f}%"
        olive_pct  = f"{ham*100:.1f}%"

        st.markdown(f"""
        <div style="margin:0.5rem 0 1.2rem 0;">
          <!-- tick labels -->
          <div style="display:flex;justify-content:space-between;
                      font-size:0.78rem;color:{MUTED};margin-bottom:5px;">
            <span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span>
          </div>
          <!-- bar track -->
          <div style="position:relative;background:#E8EAD8;border-radius:6px;
                      height:16px;overflow:hidden;">
            <!-- disease fill -->
            <div style="position:absolute;left:0;top:0;
                        width:{rust_pct};height:100%;
                        background:{RUST};border-radius:6px 0 0 6px;"></div>
            <!-- mid line -->
            <div style="position:absolute;left:50%;top:0;
                        width:2px;height:100%;background:{MUTED};opacity:0.35;"></div>
          </div>
          <!-- legend -->
          <div style="display:flex;justify-content:space-between;margin-top:6px;">
            <span style="font-size:0.85rem;font-weight:600;color:{RUST};">
              Disease risk — {rust_pct}
            </span>
            <span style="font-size:0.85rem;font-weight:600;color:{OLIVE};">
              Healthy — {olive_pct}
            </span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Verdict
        if   prob > 0.75: st.error("⚠️  High probability of heart disease. Recommend clinical evaluation.")
        elif prob > 0.50: st.warning("Elevated risk. Consider further testing.")
        elif prob > 0.30: st.info("Moderate risk. Monitor key indicators.")
        else:             st.success("✅  Low risk based on the provided data.")

        st.caption("This tool is for educational purposes only and is not a medical diagnosis.")

# ═══════════════════════════════════════════════════════════
# TAB 2 — PERFORMANCE
# ═══════════════════════════════════════════════════════════

with tab2:
    st.markdown("#### Model comparison")

    results_df = (pd.DataFrame(results).T * 100).round(2)
    results_df.index.name = "Model"
    st.dataframe(
        results_df.style.highlight_max(axis=0, color="#D4F1C4").format("{:.2f}%"),
        use_container_width=True
    )
    st.caption("Green = best per column. 80/20 stratified split.")

    st.divider()
    pc1, pc2 = st.columns(2, gap="large")

    # ── Feature importance ─────────────────────────────────
    with pc1:
        st.markdown("#### Feature importance")
        fi_path = BASE / "feature_importance.png"
        
        if fi_path.exists():
            st.image(str(fi_path), use_container_width=True)
            
        elif hasattr(model, "feature_importances_") or hasattr(model, "coef_"):
            feat_cols = df.drop(target_col, axis=1).columns.tolist()
            
            # Use safe getattr bypassing typing flags
            if hasattr(model, "feature_importances_"):
                importances = getattr(model, "feature_importances_")
                x_label = "Importance"
            else:
                coefs = getattr(model, "coef_")
                importances = np.abs(coefs[0] if len(coefs.shape) > 1 else coefs)
                x_label = "Absolute Coefficient (Importance)"

            fi = pd.Series(importances, index=feat_cols).sort_values()
            
            fig, ax = plt.subplots(figsize=(5, 4.5))
            ax.barh(np.asarray(fi.index), np.asarray(fi.values), color=OLIVE_MID, height=0.65)
            ax.set_xlabel(x_label)
            chart_style(ax, fig)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close()
            
        else:
            st.info("Feature importance not available for this model type.")

    # ── Confusion matrix ───────────────────────────────────
    with pc2:
        st.markdown("#### Confusion matrix")
        cm_path = BASE / "confusion_matrix.png"
        if cm_path.exists():
            st.image(str(cm_path), use_container_width=True)
        else:
            X_f = df.drop(target_col, axis=1)
            y_f = df[target_col]
            _, X_te, _, y_te = train_test_split(X_f, y_f, test_size=0.2, random_state=42, stratify=y_f)
            y_pred2 = model.predict(scaler.transform(X_te))   # type: ignore[union-attr]
            cm = confusion_matrix(y_te, y_pred2)
            fig, ax = plt.subplots(figsize=(4.5, 4))
            cell_colors = [[OLIVE_PALE, RUST+"44"], [RUST+"44", OLIVE+"CC"]]
            cell_labels = [["TN","FP"], ["FN","TP"]]
            for i in range(2):
                for j in range(2):
                    ax.add_patch(Rectangle((j-0.5,i-0.5),1,1,color=cell_colors[i][j],zorder=1))
                    ax.text(j, i, f"{cell_labels[i][j]}\n{cm[i,j]}",
                            ha="center", va="center", fontsize=14, fontweight="600",
                            color=TEXT, zorder=2)
            ax.set_xticks([0,1]); ax.set_xticklabels(["No Disease","Disease"], color=TEXT, fontsize=10)
            ax.set_yticks([0,1]); ax.set_yticklabels(["No Disease","Disease"], color=TEXT, fontsize=10)
            ax.set_xlabel("Predicted", color=MUTED, fontsize=10)
            ax.set_ylabel("Actual",    color=MUTED, fontsize=10)
            ax.set_xlim(-0.5,1.5); ax.set_ylim(-0.5,1.5)
            for sp in ax.spines.values(): sp.set_color(OLIVE_PALE)
            chart_style(ax, fig); fig.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close()
        st.caption("**TN** correct healthy · **TP** correct disease · **FP** false alarm · **FN** missed disease")

    # ── ROC curves ─────────────────────────────────────────
    st.divider()
    st.markdown("#### ROC curves — all models")
    roc_path = BASE / "roc_curve.png"
    if roc_path.exists():
        _, roc_col, _ = st.columns([1, 4, 1])
        with roc_col:
            st.image(str(roc_path), use_container_width=True)
    else:
        X_f  = df.drop(target_col, axis=1)
        y_f  = df[target_col]
        X_tr, X_te, y_tr, y_te = train_test_split(X_f, y_f, test_size=0.2, random_state=42, stratify=y_f)
        sc2 = StandardScaler()
        X_tr_sc = sc2.fit_transform(X_tr)
        X_te_sc = sc2.transform(X_te)
        fig, ax = plt.subplots(figsize=(7, 5))
        roc_cols_list = [OLIVE, RUST, OLIVE_MID]
        for (name, res), c in zip(results.items(), roc_cols_list):
            m_cls = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
                "Gradient Boosting":   GradientBoostingClassifier(random_state=42),
            }[name]
            m_cls.fit(X_tr_sc, y_tr)
            fpr, tpr, _ = roc_curve(y_te, m_cls.predict_proba(X_te_sc)[:, 1])
            ax.plot(fpr, tpr, color=c, linewidth=2.5, label=f"{name}  (AUC = {res['ROC-AUC']:.3f})")
        ax.plot([0,1],[0,1],"--",color=MUTED,alpha=0.4,linewidth=1)
        ax.set_xlabel("False Positive Rate", fontsize=10)
        ax.set_ylabel("True Positive Rate",  fontsize=10)
        ax.legend(fontsize=9, frameon=False, loc="lower right")
        chart_style(ax, fig); fig.tight_layout()
        _, roc_col, _ = st.columns([1, 4, 1])
        with roc_col:
            st.pyplot(fig, use_container_width=True); plt.close()

    # ── Dataset stats ───────────────────────────────────────
    st.divider()
    st.markdown("#### Dataset")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Rows",     f"{len(df):,}")
    d2.metric("Features", f"{len(df.columns)-1}")
    d3.metric("Disease",  f"{int(df[target_col].sum()):,}")
    d4.metric("Healthy",  f"{int((df[target_col]==0).sum()):,}")

# ═══════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ═══════════════════════════════════════════════════════════

with tab3:
    st.markdown("#### Pipeline overview")
    st.markdown("""
    **Dataset:** Heart Disease UCI — 303 patients, 13 clinical features, binary target (disease / healthy).

    **Pipeline steps:**
    1. **EDA** — class balance, null check, feature correlations
    2. **Split** — 80/20 stratified (preserves disease/healthy ratio in both sets)
    3. **Scale** — StandardScaler fit on train only, applied to test (prevents data leakage)
    4. **Train** — Logistic Regression, Random Forest, Gradient Boosting compared
    5. **Evaluate** — Accuracy, Precision, Recall, F1, ROC-AUC on held-out test
    6. **Cross-validate** — 5-fold CV to confirm results aren't a lucky split
    7. **Deploy** — best model served via Streamlit

    **Why stratified split?**
    The dataset is ~54% disease / ~46% healthy. Stratify ensures this ratio
    is preserved in both train and test sets, giving honest evaluation metrics.

    **Why fit scaler on train only?**
    Fitting on the full dataset leaks test set statistics (mean, std) into training.
    The scaler sees only train data and applies the same transform to test.

    **Key predictors:** `cp` (chest pain type), `thalach` (max heart rate),
    `ca` (blocked vessels), `thal` (blood disorder), `oldpeak` (ST depression).

    ---
    **Stack:** Scikit-learn · Streamlit · Matplotlib · Pandas · NumPy

    Garv Rana · EE Undergrad · [DTU](https://dtu.ac.in) ·
    [GitHub](https://github.com/garvranaaa) ·
    [LinkedIn](https://linkedin.com/in/garvsanjeevrana)
    """)
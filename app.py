import os, json, joblib, shap, numpy as np, pandas as pd, matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Loan Approval Dashboard", layout="wide")

DATA            = "data/processed/loan_approval_clean.csv"
MODEL_FILE      = "models/random_forest.pkl"          
FEATURE_CFG     = "models/feature_config.json"        
THRESHOLD_TXT   = "models/threshold.txt"              
FAIRNESS_JSON   = "reports/fairness.json"
FIG_DIR         = "reports/figures"
SHAP_SUMMARY    = "reports/shap_summary.json"

@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(DATA)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    if "loan_status" in df.columns:
        df["loan_status"] = df["loan_status"].astype(str).str.strip().str.lower()
    return df

@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_FILE)

@st.cache_data(show_spinner=False)
def load_configs():
    cfg = {"numeric": [], "categorical": []}
    if os.path.exists(FEATURE_CFG):
        with open(FEATURE_CFG) as f:
            cfg = json.load(f)

    thr = 0.50
    if os.path.exists(THRESHOLD_TXT):
        try:
            thr = float(open(THRESHOLD_TXT).read().strip())
        except Exception:
            pass

    fairness = None
    if os.path.exists(FAIRNESS_JSON):
        with open(FAIRNESS_JSON) as f:
            fairness = json.load(f)

    shap_summary = None
    if os.path.exists(SHAP_SUMMARY):
        with open(SHAP_SUMMARY) as f:
            shap_summary = json.load(f)

    return cfg, thr, fairness, shap_summary

def transformed_feature_names(pipe):
    prep = pipe.named_steps["prep"]
    num_cols = prep.transformers_[0][2] if len(prep.transformers_) > 0 else []
    cat_cols = prep.transformers_[1][2] if len(prep.transformers_) > 1 else []
    num_names = prep.named_transformers_.get("num").get_feature_names_out(num_cols).tolist() if prep.named_transformers_.get("num") else []
    cat_names = prep.named_transformers_.get("cat").get_feature_names_out(cat_cols).tolist() if prep.named_transformers_.get("cat") else []
    return num_names + cat_names

@st.cache_resource(show_spinner=False)
def make_explainer(_pipe, _background_df):
    pipe = _pipe
    background_df = _background_df

    prep = pipe.named_steps["prep"]
    clf  = pipe.named_steps["clf"]

    Xt_bg = prep.transform(background_df)
    try:
        Xt_bg = Xt_bg.toarray()
    except Exception:
        pass

    name = clf.__class__.__name__.lower()
    if any(k in name for k in ["randomforest","gradientboost","xgb","lightgbm","extratrees"]):
        return shap.TreeExplainer(
            clf,
            data=Xt_bg,
            feature_perturbation="interventional",
            model_output="probability"
        )
    if "logisticregression" in name:
        return shap.LinearExplainer(clf, Xt_bg)
    return shap.Explainer(clf, Xt_bg)

st.title("Loan Approval - Model Dashboard")

df = load_data()
pipe = load_model()
cfg, default_threshold, fairness, shap_sum = load_configs()

numeric     = [c for c in cfg.get("numeric", []) if c in df.columns]
categorical = [c for c in cfg.get("categorical", []) if c in df.columns]
feature_cols = numeric + categorical

with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Decision threshold (approve if ≥)", 0.0, 1.0, float(default_threshold), 0.01)
    st.caption("Adjust threshold to see decision flips.")

tab_predict, tab_global, tab_fair, tab_batch = st.tabs(["Predict", "Global SHAP", "Fairness", "Batch Scoring"])

with tab_predict:
    st.subheader("Enter application details")
    cols = st.columns(2)
    inputs = {}

    med = df[numeric].median(numeric_only=True) if numeric else pd.Series(dtype=float)
    for i, col in enumerate(numeric):
        with cols[i % 2]:
            lo = float(df[col].min())
            hi = float(df[col].max())
            default = float(med.get(col, (lo + hi) / 2))
            step = max((hi - lo) / 100, 1.0)
            inputs[col] = st.number_input(col, value=default, min_value=lo, max_value=hi, step=step)

    for j, col in enumerate(categorical):
        with cols[(j + len(numeric)) % 2]:
            options = sorted(df[col].dropna().astype(str).unique().tolist())
            default = options[0] if options else ""
            m = df[col].mode(dropna=True)
            if not m.empty and str(m.iloc[0]) in options:
                default = str(m.iloc[0])
            idx = options.index(default) if default in options else 0
            inputs[col] = st.selectbox(col, options or [default], index=idx)

    if st.button("Predict"):
        X_row = pd.DataFrame([inputs], columns=feature_cols)
        proba = float(pipe.predict_proba(X_row)[0, 1])
        pred  = int(proba >= threshold)
        st.markdown(f"### Result: {'Approved' if pred else 'Rejected'} · p(approve) = **{proba:.3f}** · threshold = {threshold:.2f}")

        st.markdown("#### Why? (local SHAP)")
        bg = df.sample(n=min(500, len(df)), random_state=42)[feature_cols].reset_index(drop=True)
        explainer = make_explainer(pipe, bg)

        prep = pipe.named_steps["prep"]
        Xt_row = prep.transform(X_row)
        try:
            Xt_row = Xt_row.toarray()
        except Exception:
            pass

        sv_all = explainer(Xt_row, check_additivity=False)
        if isinstance(sv_all, list):
            sv = sv_all[1].values if hasattr(sv_all[1], "values") else sv_all[1]
        else:
            sv = sv_all.values if hasattr(sv_all, "values") else sv_all
        if sv.ndim == 3 and sv.shape[-1] == 2:
            sv = sv[:, :, 1]
        sv = np.asarray(sv)

        feat_names_trans = transformed_feature_names(pipe)
        base = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value

        fig = plt.figure(figsize=(8, 5))
        shap.plots.waterfall(
            shap.Explanation(values=sv[0], base_values=base, data=np.array(Xt_row[0]), feature_names=feat_names_trans),
            max_display=12, show=False
        )
        st.pyplot(fig, clear_figure=True)

with tab_global:
    st.subheader("Global feature importance")
    col1, col2 = st.columns(2)
    with col1:
        p = os.path.join(FIG_DIR, "shap_global_bar.png")
        if os.path.exists(p): st.image(p, caption="Global importance (mean |SHAP|)")
        else: st.info("Run 04_explainability_shap.ipynb to generate global plots.")
    with col2:
        p = os.path.join(FIG_DIR, "shap_global_beeswarm.png")
        if os.path.exists(p): st.image(p, caption="SHAP summary (beeswarm)")

    if shap_sum:
        st.caption("Top features (from shap_summary.json):")
        st.write(shap_sum.get("top_features_by_mean_abs_shap"))

with tab_fair:
    st.subheader("Fairness summary")
    if fairness is None:
        st.info("Run 03_fairness_detector.ipynb to create reports/fairness.json.")
    else:
        st.write(f"**Model:** {fairness.get('model', 'unknown')}")
        for sens_col, res in fairness.get("fairness", {}).items():
            st.markdown(f"**{sens_col}**")
            st.json({
                "overall": res["overall"],
                "demographic_parity_difference": res["disparities"]["demographic_parity_difference"],
                "equalized_odds_difference": res["disparities"]["equalized_odds_difference"]
            })

with tab_batch:
    st.subheader("Batch scoring (CSV)")
    st.caption("Upload a CSV with the same columns as the training features.")
    upl = st.file_uploader("CSV file", type=["csv"])
    if upl is not None:
        data = pd.read_csv(upl)
        data.columns = data.columns.str.strip().str.lower().str.replace(" ", "_")
        missing = [c for c in feature_cols if c not in data.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            probs = pipe.predict_proba(data[feature_cols])[:, 1]
            preds = (probs >= threshold).astype(int)
            out = data.copy()
            out["prob_approve"] = probs
            out["prediction"] = preds
            st.dataframe(out.head(20))
            st.download_button(
                "Download predictions CSV",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv"
            )

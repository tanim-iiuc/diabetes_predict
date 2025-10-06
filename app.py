# app_unified.py
# Unified Streamlit app with dual-radar evaluation (within/cross) + prediction/explanation
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from app_utils import (
    generate_prediction_report_pdf, load_nb5, load_nb6, load_bundle, prepare_single_input,
    meta_predict_proba, explain_with_shap, is_original_feature,
    subset_explanation, fig_waterfall_plotly, llm_summary_from_shap,
    generate_random_user)

st.set_page_config(
    page_title="Diabetes Evaluation & Prediction", layout="wide")

# ============================================================
# Helpers
# ============================================================


def _radar_plot(df, metrics, title=""):
    fig = go.Figure()
    for _, r in df.iterrows():
        vals = [float(r[m]) for m in metrics]
        vals.append(vals[0])
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=metrics + [metrics[0]],
            name=str(r["model"]), fill="none",
            hovertemplate=f"{r['model']}<br>%{{theta}}: %{{r:.3f}}<extra></extra>"
        ))
    fig.update_layout(
        title=title,
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True, height=650
    )
    return fig


@st.cache_resource
def _get_bundle():
    return load_bundle()


# ============================================================
# Main UI
# ============================================================


st.markdown(
    """
    <h2 style='text-align: center;'>
        📈 <b>Diabetes Model Evaluation</b> &nbsp; &amp; &nbsp; 🩺 <b>Clinical Prediction</b>
    </h2>
    """,
    unsafe_allow_html=True
)



tab_eval, tab_pred = st.tabs(
    ["Performance Evaluation", "Prediction & Explanation"])

# ------------------------------------------------------------
# TAB 1: Evaluation (radio: within vs cross)
# ------------------------------------------------------------
with tab_eval:
    st.markdown(
        """
    <h3 style='text-align: center;'>
        📈 <b>Model Performance Evaluation Dashboard</b>
    </h3>
    """,
        unsafe_allow_html=True
    )
    mode = st.radio("Select Evaluation Type:", [
                    "Within-Dataset", "Cross-Dataset"], horizontal=True)
    # st.divider()

    # =============================================
    # WITHIN-DATASET (NB5)
    # =============================================
    if mode.startswith("Within"):
        st.markdown(
            """
    <h4 style='text-align: center;'>
        📈 <b>Within-dataset Model Performance Comparison</b>
    </h4>
    """,
            unsafe_allow_html=True
        )
        df = load_nb5()
        if df.empty:
            st.warning(
                "NB5 results not found at ./results_nb5/all_nb5_results.csv")
            st.stop()

        metrics_all = ["auroc", "auprc",
                       "accuracy", "f1", "recall", "precision"]
        metrics_mean = [m + "_mean" for m in metrics_all]
        theta_labels = ["AUROC", "AUPRC",
                        "Accuracy", "F1", "Recall", "Precision"]

        datasets = sorted(df["dataset"].dropna().unique())
        models = sorted(df["model"].dropna().unique())
        imputers = sorted(df["imputer"].dropna().unique())
        flags = sorted(df["with_flag"].dropna().unique())
        pairs = sorted(df["interaction"].dropna().unique())
        resamplings = sorted(df["resampling"].dropna().unique())

        # -------------------- Two-column selection --------------------
        # print(imputers)
        cols = st.columns(2)
        selections = []

        for i, c in enumerate(cols, start=1):
            with c:
                st.markdown(
                    f"""
    <h5 style='text-align: center;'>
        📈 <b>Plot {i}</b>
    </h5>
    """,
                    unsafe_allow_html=True
                )
                dset_sel = st.selectbox(f"Dataset ", datasets,
                                        key=f"dset{i}")
                imp_sel = st.selectbox(f"Imputer", imputers, key=f"imp{i}", )
                flag_sel = st.selectbox(f"Missing flag", flags, key=f"flag{i}")
                pair_sel = st.selectbox(
                    f"Interaction pair", pairs, key=f"pair{i}")
                resamp_sel = st.selectbox(
                    f"Resampling", resamplings, key=f"resamp{i}", index=i-1)
                selections.append(
                    (dset_sel, imp_sel, flag_sel, pair_sel, resamp_sel))

        # -------------------- Global model selection --------------------
        model_sel = st.multiselect(
            "Models (applies to both radars)", models, default=models)

        # -------------------- Two-column radar display --------------------
        radar_cols = st.columns(2)

        # Optional: find global min/max to scale both radars the same
        subset_all = df[
            (df["model"].isin(model_sel))
        ]
        if subset_all.empty:
            st.warning("No rows for selected models.")
            st.stop()

        global_min = 0  # metrics are [0,1] normalized
        global_max = 1
        work = pd.DataFrame()
        for i, (dset_sel, imp_sel, flag_sel, pair_sel, resamp_sel) in enumerate(selections, start=1):
            with radar_cols[i-1]:
                subset = df[
                    (df["dataset"] == dset_sel) &
                    (df["imputer"] == imp_sel) &
                    (df["with_flag"] == flag_sel) &
                    (df["model"].isin(model_sel)) &
                    (df["interaction"] == pair_sel) &
                    (df["resampling"] == resamp_sel)
                ].copy()

                if subset.empty:
                    st.info(f"No rows for Radar {i} selection.")
                    continue
                else:
                    work = pd.concat([work, subset])

                agg = subset.groupby("model")[
                    metrics_mean].mean().reset_index()
                if agg.empty:
                    st.info(f"No aggregated results for Radar {i}.")
                    continue

                fig = go.Figure()

                for _, r in agg.iterrows():
                    vals = [float(r[m]) for m in metrics_mean] + \
                        [float(r[metrics_mean[0]])]  # close polygon
                    theta = theta_labels + [theta_labels[0]]
                    hovertemplate = f"{r['model']}<br>%{{theta}}: %{{r:.3f}}<extra></extra>"

                    fig.add_trace(go.Scatterpolar(
                        r=vals,
                        theta=theta,
                        name=str(r["model"]),
                        fill='none',  # 'toself' for shaded polygon
                        hovertemplate=hovertemplate
                    ))

                fig.update_layout(
                    polar=dict(radialaxis=dict(
                        visible=True, range=[global_min, global_max])),
                    showlegend=True if i == 2 else False,  # only right radar shows legend
                    height=700,
                    # margin=dict(l=10, r=10, t=40, b=10),
                    title=f"{dset_sel} | {imp_sel} | Flag={bool(flag_sel)} | Interaction Pair: {bool(pair_sel)} | Resampling: {bool(resamp_sel)}"
                )
                st.plotly_chart(fig, use_container_width=True)

        # -------------------- Stability scatter — mean vs std (no legend, no labels) --------------------
        st.subheader("Stability scatter — mean vs std")

        avail_metrics = [m for m in metrics_all if (
            m+"_mean" in df.columns and m+"_std" in df.columns)]
        if not avail_metrics:
            st.info("Std columns not found.")
        else:
            metric_choice = st.selectbox("Metric", avail_metrics, index=0)
            m_mean, m_std = f"{metric_choice}_mean", f"{metric_choice}_std"

            work = work.groupby(["dataset", "imputer", "with_flag", "model", 'interaction', 'resampling'], as_index=False)[
                [m_mean, m_std]].mean()
            if work.empty:
                st.info("No rows for selected configurations.")
                st.stop()

            work["scenario"] = "dataset:" + work["dataset"].astype(
                str) + " | Model" + work["model"].astype(str) + " | Imputer:" + work["imputer"].astype(str) + " | Missing flag:" + work["with_flag"].astype(str) + " | Resampling:" + work['resampling'].astype(str) + " | Interaction Pairs:" + work['interaction'].astype(str)
            work = work.rename(columns={m_mean: "mean", m_std: "std"}).round(3)

            fig2 = px.scatter(
                work, x="std", y="mean",
                color="scenario",  # unique color per dot
                hover_name="model",
                hover_data={"scenario": True, "mean": True, "std": True},
                title=f"{metric_choice.upper()} — mean vs std"
            )
            # remove legend and on-point text
            fig2.update_traces(marker=dict(
                size=9), hovertemplate="<b>%{hovertext}</b><br>Scenario:%{customdata[0]}<br>Mean:%{y:.3f}<br>Std:%{x:.3f}<extra></extra>")
            fig2.update_layout(
                showlegend=False,
                xaxis_title="Std",
                yaxis_title="Mean",
                yaxis=dict(range=[0, 1]),
                height=520,
                margin=dict(l=10, r=10, t=50, b=10)
            )
            st.plotly_chart(fig2, use_container_width=True)

            # -------------------- Table used for the chart --------------------
            st.subheader("Values used in the stability scatter")
            st.dataframe(
                work[["dataset", "imputer", "with_flag", 'resampling',
                      'interaction', "model", "mean", "std"]]
                .round(3)
                .rename(columns={"with_flag": "flag"})
            )

    # =============================================
    # CROSS-DATASET (NB6)
    # =============================================
    else:
        st.markdown(
            """
    <h4 style='text-align: center;'>
        📈 <b>Cross-dataset Model Generalization Comparison</b>
    </h4>
    """,
            unsafe_allow_html=True
        )

        df = load_nb6()
        if df.empty:
            st.warning(
                "NB6 results not found at ./results_nb6/all_nb6_results.csv")
            st.stop()

        # Two columns for selection
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                """
    <h5 style='text-align: center;'>
        📈 <b>Plot 1</b>
    </h5>
    """,
                unsafe_allow_html=True
            )
            schemes = sorted(df['scheme'].dropna().unique())
            scheme1 = st.selectbox("Scheme", schemes, index=0)

            # Filter by scenario if applicable
            dd1 = df[df['scheme'] == scheme1].copy()
            scenes = sorted(dd1['scenario'].dropna().unique())
            scene_sel1 = st.selectbox("Scenario", scenes)
            dd1 = dd1[dd1['scenario'] == scene_sel1]

        with col2:
            st.markdown(
                """
    <h5 style='text-align: center;'>
        📈 <b>Plot 2</b>
    </h5>
    """,
                unsafe_allow_html=True
            )
            scheme2 = st.selectbox("Scheme", schemes, key="scheme2")

            dd2 = df[df['scheme'] == scheme2].copy()
            scenes = sorted(dd2['scenario'].dropna().unique())
            scene_sel2 = st.selectbox("Scenario", scenes, index=1)
            dd2 = dd2[dd2['scenario'] == scene_sel2]

        # Model and imputer selection (applied to both schemes)
        all_models = sorted(df['model'].dropna().unique())
        model_sel = st.multiselect(
            "Select Model(s) (both schemes)", all_models, default=all_models)
        dd1 = dd1[dd1['model'].isin(model_sel)]
        dd2 = dd2[dd2['model'].isin(model_sel)]

        all_imputers = sorted(df['imputer'].dropna().unique())

        # Radar plots side by side
        metrics = ['auroc', 'auprc', 'accuracy',
                   'recall', 'precision', 'f1', 'bss']
        agg1 = dd1.groupby('model')[metrics].mean().reset_index()
        agg2 = dd2.groupby('model')[metrics].mean().reset_index()

        st.subheader("Radar Plots Comparison")
        cols = st.columns(2)

        with cols[0]:
            st.markdown(f"**Scheme 1: {scheme1}**")
            st.markdown(f"**Train: {dd1['train'].values[0]}**")
            st.markdown(f"**Test: {dd1['test'].values[0]}**")
            if len(agg1) > 0:
                fig1 = go.Figure()

                for _, r in agg1.iterrows():
                    values = [r[m] for m in metrics]
                    values.append(values[0])
                    fig1.add_trace(go.Scatterpolar(
                        r=values,
                        theta=metrics + [metrics[0]],
                        fill='none',
                        name=r['model'],
                        hovertemplate=f"{r['model']}<br>%{{theta}}: %{{r:.3f}}<extra></extra>"
                    ))
                fig1.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                                   showlegend=False, height=700)
                st.plotly_chart(fig1, use_container_width=True)

        with cols[1]:
            st.markdown(f"**Scheme 2: {scheme2}**")
            st.markdown(f"**Train: {dd2['train'].values[0]}**")
            st.markdown(f"**Test: {dd2['test'].values[0]}**")
            if len(agg2) > 0:
                fig2 = go.Figure()
                for _, r in agg2.iterrows():
                    values = [r[m] for m in metrics]
                    values.append(values[0])
                    fig2.add_trace(go.Scatterpolar(
                        r=values,
                        theta=metrics + [metrics[0]],
                        fill='none',
                        name=r['model'],
                        hovertemplate=f"{r['model']}<br>%{{theta}}: %{{r:.3f}}<extra></extra>"
                    ))
                fig2.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                                   showlegend=True, height=700)
                st.plotly_chart(fig2, use_container_width=True)

        # Combined scatter plot
        dd_combined = pd.concat([dd1, dd2]).round(3)
        if set(['auroc', 'f1']).issubset(dd_combined.columns):
            fig_sc = px.scatter(dd_combined, x='auroc', y='f1', color='model',
                                hover_data=['imputer', 'train', 'test', 'model'])
            fig_sc.update_layout(title="AUROC vs F1 (combined schemes)")
            st.plotly_chart(fig_sc, use_container_width=True)

        # Combined table
        st.subheader("Metrics Table (Combined Schemes)")
        st.dataframe(dd_combined[['model', 'train', 'test', 'accuracy', 'precision', 'recall',
                                  'auroc', 'auprc', 'bss', 'f1', 'best_thresh']].style.format(precision=3))

# ------------------------------------------------------------
# TAB 2: Prediction & Explanation
# ------------------------------------------------------------
with tab_pred:
    st.warning(
        "⚠️ Disclaimer: This prediction and explanation are experimental and **not a medical diagnosis**. "
        "Consult a qualified clinician.")
    st.markdown(
        """
    <h4 style='text-align: center;'>
        📈 <b>Live Prediction & Explainability</b>
    </h4>
    """,
        unsafe_allow_html=True
    )

    bundle = _get_bundle()
    use_cols = bundle["use_cols"]
    tau = float(bundle["tau"])
    
    top_k = 7

    if "user_input" not in st.session_state:
        st.session_state["user_input"] = {c: "" for c in use_cols}

    # --- Random Profile Buttons ---
    colA, colB = st.columns([1, 1])
    with colA:
        if st.button("🎲 Generate Random Patient"):
            rand = generate_random_user(bundle, jitter=0.10)
            st.session_state["user_input"] = {
                c: rand.get(c, "") for c in use_cols}
    with colB:
        if st.button("🧹 Clear All"):
            st.session_state["user_input"] = {c: "" for c in use_cols}
            st.session_state.pop("pred", None)

    # --- Input grid ---
    cols = st.columns(3)
    for i, c in enumerate(use_cols):
        with cols[i % 3]:
            try:
                default_val = float(st.session_state["user_input"].get(c, 0.0))
            except (ValueError, TypeError):
                default_val = 0.0
            st.session_state["user_input"][c] = st.number_input(
                c, value=default_val,
                step=1.0 if c in ["age", "preg_count", "dbp"] else 0.1,
                min_value=0.0
            )

    # --- Predict button ---
    if st.button("🔮 Predict"):
        df_one, X_std = prepare_single_input(
            st.session_state["user_input"], bundle)
        p_meta, p_rf, p_nn = meta_predict_proba(X_std, bundle)
        p, p_rf, p_nn = float(p_meta[0]), float(p_rf[0]), float(p_nn[0])
        yhat = int(p >= tau)
        st.session_state["pred"] = dict(
            df_one=df_one, X_std=X_std, p=p, p_rf=p_rf, p_nn=p_nn, yhat=yhat)
        c1, c2, c3, c4 = st.columns(4)

        # --- Meta probability with red/green color based on threshold ---
        p_color = "red" if p >= tau else "green"
        c1.markdown(
            f"<div style='text-align:center; font-weight:600;'>"
            f"Meta (Stack) P(+): <span style='color:{p_color}; font-size:18px;'>{p:.3f}</span>"
            f"</div>",
            unsafe_allow_html=True
        )

        # --- Base model probabilities (neutral blue tone) ---
        c2.markdown(
            f"<div style='text-align:center; font-weight:600; color:#0073e6;'>"
            f"RF P(+): {p_rf:.3f}"
            f"</div>",
            unsafe_allow_html=True
        )
        c3.markdown(
            f"<div style='text-align:center; font-weight:600; color:#0073e6;'>"
            f"DeepANN P(+): {p_nn:.3f}"
            f"</div>",
            unsafe_allow_html=True
        )

        # --- Decision label + show τ as threshold reference ---
        decision_label = "POS" if yhat == 1 else "NEG"
        decision_color = "red" if yhat == 1 else "green"
        c4.markdown(
            f"<div style='text-align:center; font-weight:600;'>"
            f"Decision: <span style='color:{decision_color}; font-size:18px;'>{decision_label}</span><br>"
            f"<span style='color:gray; font-size:14px;'>(threshold τ = {tau:.2f})</span>"
            f"</div>",
            unsafe_allow_html=True
        )

# --- Generate PDF directly when user clicks Predict ---
        if "pred" in st.session_state:
            pr = st.session_state["pred"]
            df_one, X_std, p, p_rf, p_nn, yhat = pr["df_one"], pr["X_std"], pr["p"], pr["p_rf"], pr["p_nn"], pr["yhat"]

            # Compute SHAP explanation and plot
            explanation_full = explain_with_shap(bundle, X_std, nsamples="auto")
            keep_idx = [i for i, n in enumerate(
                bundle["final_cols"]) if is_original_feature(n)]
            explanation = subset_explanation(explanation_full, keep_idx)
            fig_plotly = fig_waterfall_plotly(
                explanation, top_k=len(bundle["use_cols"]))
            st.plotly_chart(fig_plotly, use_container_width=True)

            # Prepare LLM narrative
            order = np.argsort(np.abs(explanation.values))[::-1]
            local_df = pd.DataFrame({
                "feature": np.array(explanation.feature_names)[order],
                "value": [df_one.iloc[0].to_dict().get(f, np.nan) for f in np.array(explanation.feature_names)[order]],
                "contrib": np.array(explanation.values)[order]
            })
            with st.spinner("Generating explanation..."):
                summary = llm_summary_from_shap(
                    local_df, prob_pos=p, tau=tau, top_k=len(bundle["use_cols"]))

            st.markdown("**LLM Explanation**")
            st.write(summary)

            pdf_buffer = generate_prediction_report_pdf(
                pred_info={
                    "p": p, "p_rf": p_rf, "p_nn": p_nn,
                    "tau": tau, "yhat": yhat,
                    "inputs": st.session_state["user_input"]
                },
                shap_df=local_df,
                llm_text=summary,
                shap_fig=fig_plotly
            )

            st.download_button(
                label="⬇️ Download Full PDF Report",
                data=pdf_buffer,
                file_name="Diabetes_Prediction_Report.pdf",
                mime="application/pdf"
            )

    else:
        st.info(
            "Enter values (or generate a random profile) and click Predict to see results.")
# ---- Footer + Citation Block ----

# # 🧠 Your BibTeX entry
# bibtex_entry = r"""
# @article{YourName2025DiabetesXAI,
#   title   = {Explainable Stacked Ensemble for Cross-Dataset Diabetes Prediction and Clinical Interpretation},
#   author  = {Your Name and Coauthor, Another and Collaborator, Third},
#   journal = {Telematics and Informatics},
#   year    = {2025},
#   doi     = {10.xxxx/xxxxxx},
#   note    = {Available at: https://your-demo-url.streamlit.app}
# }
# """

# # ---- Citation block ----
# st.markdown("---")
# st.markdown("### 📝 Cite this work")
# st.info("If you use this prototype or findings in your research, please cite the paper below:")
# st.code(bibtex_entry, language="bibtex")

# ---- Footer ----
st.markdown("""
    <style>
    footer {visibility: hidden;}
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f8f9fa;
        color: #555;
        text-align: center;
        padding: 10px;
        font-size: 0.9em;
        border-top: 1px solid #ddd;
    }
    .footer a {
        color: #0073e6;
        text-decoration: none;
    }
    .footer a:hover {
        text-decoration: underline;
    }
    </style>

    <div class="footer">
        © 2025  Built by 
        <a href="https://scholar.google.com/citations?user=yflDgiMAAAAJ" target="_blank">
            <b>Tauhidul Islam</b>
        </a> with ❤️ using Streamlit
    </div>
""", unsafe_allow_html=True)

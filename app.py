"""
Interactive Streamlit Dashboard for Health Data Disease Prediction (Diabetes)
Cluster Computing with Apache Spark MLlib
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Append project directory to path
PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

try:
    from pyspark.ml import PipelineModel
except (ImportError, Exception):
    PipelineModel = None

from src.spark_pipeline import (
    get_spark_session,
    load_and_preprocess_spark_data,
    build_spark_ml_pipeline,
    evaluate_spark_predictions,
    extract_feature_importance,
    predict_patient_risk,
    FEATURE_COLS,
    ZERO_AS_MISSING_COLS,
    LABEL_COL
)
from src.sklearn_pipeline import (
    load_and_preprocess_pandas,
    build_and_train_sklearn_pipeline
)
from src.cluster_benchmarks import (
    run_partition_scaling_benchmark,
    get_cluster_environment_info
)
from src.utils import (
    plot_confusion_matrix_plotly,
    plot_roc_curve_plotly,
    plot_feature_importance_plotly,
    plot_missing_audit_plotly,
    plot_cluster_scaling_plotly,
    plot_patient_radar_plotly
)

# ---------------------------------------------------------
# Page Configuration & Modern Custom CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="Health Data | Diabetes Prediction on Spark",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background-color: #0b0f19;
    }

    /* Glassmorphism card */
    .metric-card {
        background: rgba(19, 27, 46, 0.7);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(8px);
        margin-bottom: 15px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: rgba(56, 189, 248, 0.5);
        transform: translateY(-2px);
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-value {
        color: #f8fafc;
        font-size: 1.85rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .metric-sub {
        color: #38bdf8;
        font-size: 0.8rem;
        margin-top: 4px;
        font-weight: 500;
    }

    /* Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.7);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        max-width: 900px;
        line-height: 1.5;
    }
    .cluster-tag {
        display: inline-block;
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.35);
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 8px;
        margin-top: 10px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Custom pill badges */
    .badge-primary {
        background: rgba(56, 189, 248, 0.2);
        color: #38bdf8;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-success {
        background: rgba(16, 185, 129, 0.2);
        color: #10b981;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-danger {
        background: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }

    /* Code block styling */
    pre, code {
        font-family: 'JetBrains Mono', monospace !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

CSV_PATH = os.path.join(PROJECT_DIR, "diabetes.csv")


# ---------------------------------------------------------
# Resource Caching
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def init_spark():
    """Initializes and caches a single SparkSession for the application session."""
    return get_spark_session(app_name="DiabetesPredictionStreamlitDashboard", master="local[*]")


@st.cache_data(show_spinner=False)
def load_raw_dataset():
    """Loads and caches pandas dataframe for fast UI exploration."""
    return pd.read_csv(CSV_PATH)


# Initialize Spark Session safely
spark = None
spark_error = None
try:
    spark = init_spark()
except Exception as e:
    spark_error = str(e)


# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/f/f3/Apache_Spark_logo.svg",
        width=160
    )
    st.markdown("### ⚙️ Cluster & Model Controls")

    # Engine selection
    engine = st.radio(
        "Execution Engine",
        options=["PySpark MLlib (Cluster Mode)", "Scikit-Learn (Local Baseline)"],
        index=0 if spark is not None else 1,
        help="Switch between distributed Apache Spark MLlib and single-node Scikit-Learn baseline."
    )

    st.markdown("---")
    st.markdown("#### 🎛️ Pipeline Hyperparameters")

    train_split = st.slider(
        "Train / Test Split Ratio",
        min_value=0.60,
        max_value=0.90,
        value=0.80,
        step=0.05,
        help="Fraction of data reserved for model training."
    )

    reg_param = st.select_slider(
        "Regularization (λ / C)",
        options=[0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
        value=0.01,
        help="Regularization strength to prevent overfitting."
    )

    elastic_net = st.slider(
        "ElasticNet Mixing (α)",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="0.0 = L2 Ridge (weight decay), 1.0 = L1 Lasso (feature sparsity)."
    )

    max_iter = st.slider(
        "Max Optimizer Iterations",
        min_value=20,
        max_value=300,
        value=100,
        step=20,
        help="Maximum epochs for distributed L-BFGS optimizer."
    )

    impute_method = st.selectbox(
        "Missing Data Imputation Strategy",
        options=["median", "mean"],
        index=0,
        help="Strategy to replace biologically impossible zeros in Glucose, BP, Insulin, BMI, SkinThickness."
    )

    st.markdown("---")
    st.markdown("#### 🎯 Classification Decision Threshold")
    decision_threshold = st.slider(
        "Probability Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Cutoff for positive class assignment. Lowering threshold prioritizes Sensitivity/Recall."
    )

    st.markdown("---")
    st.caption("🚀 Antigravity Cluster Computing Lab • Spark 4.2.0")


# ---------------------------------------------------------
# Hero Banner
# ---------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-title">Health Data — Disease Prediction Platform</div>
    <div class="hero-subtitle">
        Distributed Machine Learning pipeline for patient diabetes classification built for 
        <b>Cluster Computing</b> using <b>Apache Spark 4.2.0 (PySpark MLlib)</b>, distributed feature standardization, 
        and real-time clinical risk decision modeling.
    </div>
    <div>
        <span class="cluster-tag">⚡ Apache Spark 4.2.0</span>
        <span class="cluster-tag">🧠 PySpark MLlib Pipeline</span>
        <span class="cluster-tag">📊 Distributed Logistic Regression</span>
        <span class="cluster-tag">🩺 Clinical Risk Profiler</span>
        <span class="cluster-tag">🔍 Live Confusion Matrix</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Training Function (Cached & Dynamic)
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def train_and_evaluate_cached(
    engine_name: str,
    train_ratio: float,
    reg_val: float,
    elastic_val: float,
    iterations: int,
    imp_strategy: str,
    threshold: float
):
    """Executes training on chosen backend and returns models and evaluation metrics."""
    if engine_name.startswith("PySpark") and spark is not None:
        cleaned_df, meta = load_and_preprocess_spark_data(spark, CSV_PATH)
        test_ratio = round(1.0 - train_ratio, 2)
        train_df, test_df = cleaned_df.randomSplit([train_ratio, test_ratio], seed=42)
        train_df.cache()

        t0 = time.time()
        pipeline = build_spark_ml_pipeline(
            impute_strategy=imp_strategy,
            reg_param=reg_val,
            elastic_net_param=elastic_val,
            max_iter=iterations,
            threshold=threshold
        )
        fitted_pipeline = pipeline.fit(train_df)
        fit_time = time.time() - t0

        t1 = time.time()
        preds = fitted_pipeline.transform(test_df)
        eval_res = evaluate_spark_predictions(preds, threshold=threshold)
        eval_time = time.time() - t1

        lr_stage = fitted_pipeline.stages[-1]
        fi_df, intercept = extract_feature_importance(lr_stage)

        return {
            "engine": "PySpark MLlib",
            "metadata": meta,
            "eval_metrics": eval_res,
            "feature_importance": fi_df,
            "intercept": intercept,
            "fit_time": fit_time,
            "eval_time": eval_time,
            "pipeline_model": fitted_pipeline
        }
    else:
        # Scikit-Learn fallback/baseline
        cleaned_df, meta = load_and_preprocess_pandas(CSV_PATH)
        train_df, test_df = train_df_split, test_df_split = np.split(
            cleaned_df.sample(frac=1, random_state=42),
            [int(train_ratio * len(cleaned_df))]
        )

        t0 = time.time()
        c_inv = 1.0 / max(reg_val, 1e-5)
        pipeline, eval_res = build_and_train_sklearn_pipeline(
            train_df=train_df,
            test_df=test_df,
            impute_strategy=imp_strategy,
            C=c_inv,
            max_iter=iterations,
            threshold=threshold
        )
        fit_time = time.time() - t0

        coefs = pipeline.named_steps["classifier"].coef_[0]
        intercept = float(pipeline.named_steps["classifier"].intercept_[0])
        fi_df = pd.DataFrame({
            "Feature": FEATURE_COLS,
            "Coefficient": coefs,
            "Odds_Ratio": np.exp(coefs),
            "Absolute_Weight": np.abs(coefs)
        }).sort_values(by="Absolute_Weight", ascending=False).reset_index(drop=True)

        return {
            "engine": "Scikit-Learn Baseline",
            "metadata": meta,
            "eval_metrics": eval_res,
            "feature_importance": fi_df,
            "intercept": intercept,
            "fit_time": fit_time,
            "eval_time": 0.005,
            "pipeline_model": pipeline
        }


# Execute training
with st.spinner("⚡ Running distributed Spark MLlib pipeline across cluster..."):
    results = train_and_evaluate_cached(
        engine_name=engine,
        train_ratio=train_split,
        reg_val=reg_param,
        elastic_val=elastic_net,
        iterations=max_iter,
        imp_strategy=impute_method,
        threshold=decision_threshold
    )

eval_metrics = results["eval_metrics"]
cm_matrix = eval_metrics["confusion_matrix"]["matrix"]
fi_df = results["feature_importance"]
meta = results["metadata"]


# ---------------------------------------------------------
# Top-level KPI Metrics Row
# ---------------------------------------------------------
c1, c2, c3, c4, c5, c6 = st.columns(6)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Accuracy</div>
        <div class="metric-value">{eval_metrics['accuracy'] * 100:.1f}%</div>
        <div class="metric-sub">Overall Test Score</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">F1-Score</div>
        <div class="metric-value">{eval_metrics['f1_score']:.3f}</div>
        <div class="metric-sub">Harmonic Precision/Recall</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Recall (Sensitivity)</div>
        <div class="metric-value">{eval_metrics['recall'] * 100:.1f}%</div>
        <div class="metric-sub">Diabetic Detection Rate</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Specificity</div>
        <div class="metric-value">{eval_metrics['specificity'] * 100:.1f}%</div>
        <div class="metric-sub">Healthy Correct ID</div>
    </div>
    """, unsafe_allow_html=True)

with c5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">ROC-AUC</div>
        <div class="metric-value">{eval_metrics['roc_auc']:.3f}</div>
        <div class="metric-sub">Discrimination Power</div>
    </div>
    """, unsafe_allow_html=True)

with c6:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Fit Latency</div>
        <div class="metric-value">{results['fit_time']:.2f}s</div>
        <div class="metric-sub">{results['engine']}</div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------
# Navigation Tabs
# ---------------------------------------------------------
tab_eval, tab_cluster, tab_data, tab_pipeline, tab_predict, tab_report = st.tabs([
    "📈 Confusion Matrix & Evaluation",
    "⚡ Cluster Computing Architecture",
    "📊 Data Explorer & Missingness",
    "🧠 Preprocessing & ML Pipeline",
    "🩺 Patient Risk Predictor",
    "📑 Project Report & Export"
])


# =========================================================
# TAB 1: Evaluation & Confusion Matrix
# =========================================================
with tab_eval:
    st.markdown("### 🔍 Model Evaluation & Confusion Matrix Analysis")
    st.caption("Visualizing binary classification performance on held-out test patients.")

    col_cm, col_roc = st.columns([1, 1])

    with col_cm:
        cm_fig = plot_confusion_matrix_plotly(
            cm_matrix,
            title=f"Confusion Matrix (Threshold: {decision_threshold:.2f})"
        )
        st.plotly_chart(cm_fig, use_container_width=True)

        cm = eval_metrics["confusion_matrix"]
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.6); padding: 14px 18px; border-radius: 8px; border-left: 4px solid #38bdf8; font-size: 0.9rem;">
            <b>Clinical Interpretation at Threshold {decision_threshold:.2f}:</b><br>
            • <b>True Positives (TP = {cm['tp']})</b>: Diabetic patients accurately identified for medical intervention.<br>
            • <b>True Negatives (TN = {cm['tn']})</b>: Healthy individuals correctly given a clean bill of health.<br>
            • <b>False Negatives (FN = {cm['fn']})</b>: Clinically hazardous type II error (missed diabetes diagnoses).<br>
            • <b>False Positives (FP = {cm['fp']})</b>: Healthy individuals falsely flagged, causing unnecessary anxiety.
        </div>
        """, unsafe_allow_html=True)

    with col_roc:
        if "eval_df" in eval_metrics:
            roc_fig = plot_roc_curve_plotly(eval_metrics["eval_df"], eval_metrics["roc_auc"])
            st.plotly_chart(roc_fig, use_container_width=True)
        elif "roc_curve" in eval_metrics:
            fpr = eval_metrics["roc_curve"]["fpr"]
            tpr = eval_metrics["roc_curve"]["tpr"]
            fig_roc_sk = go.Figure()
            fig_roc_sk.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'AUC = {eval_metrics["roc_auc"]:.3f}', line=dict(color='#38bdf8', width=3)))
            fig_roc_sk.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Chance', line=dict(color='#94a3b8', dash='dash')))
            fig_roc_sk.update_layout(title="<b>ROC Curve</b>", template="plotly_dark", height=380)
            st.plotly_chart(fig_roc_sk, use_container_width=True)

    st.markdown("---")
    st.markdown("#### ⚖️ Clinical Feature Impact & Odds Ratios")
    st.caption("How each clinical biomarker impacts patient diabetes risk under normalized logistic regression.")

    fi_fig = plot_feature_importance_plotly(fi_df)
    st.plotly_chart(fi_fig, use_container_width=True)

    # Detailed Table of Odds Ratios
    with st.expander("📋 View Exact Logistic Regression Coefficients & Mathematical Odds Ratios"):
        st.dataframe(
            fi_df.style.format({
                "Coefficient": "{:.4f}",
                "Odds_Ratio": "{:.3f}",
                "Absolute_Weight": "{:.4f}"
            }),
            use_container_width=True
        )
        st.info(f"**Model Intercept (β₀)**: {results['intercept']:.4f} (Base log-odds when all standardized features equal zero).")


# =========================================================
# TAB 2: Cluster Computing Architecture
# =========================================================
with tab_cluster:
    st.markdown("### ⚡ Cluster Computing Architecture & Spark Runtime")
    st.caption("Distributed computing principles, cluster resource monitoring, and partition scaling analysis.")

    if spark is not None:
        cluster_info = get_cluster_environment_info(spark)

        # Cluster specs row
        c_env1, c_env2, c_env3, c_env4 = st.columns(4)
        with c_env1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Spark Version</div>
                <div class="metric-value">{cluster_info['Spark_Version']}</div>
                <div class="metric-sub">Apache Spark Core</div>
            </div>
            """, unsafe_allow_html=True)
        with c_env2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Master Mode</div>
                <div class="metric-value">{cluster_info['Master_URL']}</div>
                <div class="metric-sub">Execution Manager</div>
            </div>
            """, unsafe_allow_html=True)
        with c_env3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Default Parallelism</div>
                <div class="metric-value">{cluster_info['Default_Parallelism']} Cores</div>
                <div class="metric-sub">Active Worker Threads</div>
            </div>
            """, unsafe_allow_html=True)
        with c_env4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Driver Memory</div>
                <div class="metric-value">{cluster_info['Driver_Memory']}</div>
                <div class="metric-sub">Heap Allocation</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 🔬 Real-Time Partition Scalability Experiment")
        st.write("""
        In cluster computing, data is split into partitions distributed across worker nodes. 
        Click below to run a live scaling experiment that evaluates training time and speedup across 
        **1, 2, 4, and 8 partitions**.
        """)

        if st.button("🚀 Run Live Partition Scaling Benchmark", type="primary"):
            with st.spinner("Benchmarking distributed training across 1, 2, 4, and 8 partitions..."):
                bench_df = run_partition_scaling_benchmark(spark, CSV_PATH, partition_list=[1, 2, 4, 8])
                st.session_state["bench_df"] = bench_df

        if "bench_df" in st.session_state:
            bench_df = st.session_state["bench_df"]
            bench_fig = plot_cluster_scaling_plotly(bench_df)
            st.plotly_chart(bench_fig, use_container_width=True)

            col_b1, col_b2 = st.columns([1, 1])
            with col_b1:
                st.dataframe(bench_df, use_container_width=True)
            with col_b2:
                st.markdown("""
                <div style="background: rgba(15, 23, 42, 0.7); padding: 16px; border-radius: 8px; border-left: 4px solid #6366f1; font-size: 0.9rem;">
                    <b>Cluster Computing Insights:</b><br>
                    • <b>Distributed Amdahl's Law</b>: For small datasets like 768 records, partitioning introduces coordination overhead (shuffling, JVM thread context switching, task serialization).<br>
                    • <b>Sweet Spot</b>: Partitions should match or slightly exceed the number of CPU worker cores (e.g., 2–4 partitions on a quad-core machine).<br>
                    • <b>Scalability to Big Data</b>: On multi-million patient health records, distributed partitioning yields massive linear speedups because computation dominates overhead!
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning(f"Spark runtime unavailable. (Reason: {spark_error}). Running in Scikit-Learn fallback mode.")

    st.markdown("---")
    st.markdown("#### 📐 Spark Distributed Pipeline Architecture")
    st.markdown("""
    ```mermaid
    flowchart LR
        subgraph Cluster Driver
            A[diabetes.csv File] --> B[SparkSession DataFrameReader]
            B --> C[Zero-to-Null Sanitization]
            C --> D[Stratified Train/Test Split]
        end
        
        subgraph Worker Nodes / Distributed Executors
            D -->|Partition 1| E1[Executor 1: Imputer & Scaler]
            D -->|Partition 2| E2[Executor 2: Imputer & Scaler]
            D -->|Partition 3| E3[Executor 3: Imputer & Scaler]
            D -->|Partition 4| E4[Executor 4: Imputer & Scaler]
            
            E1 & E2 & E3 & E4 -->|TreeAggregate Gradient| F[Distributed L-BFGS Optimizer]
        end
        
        subgraph Evaluation & Metrics
            F --> G[PipelineModel Broadcast]
            G --> H[Distributed Confusion Matrix]
            G --> I[Accuracy & F1-Score]
        end
    ```
    """)


# =========================================================
# TAB 3: Data Explorer & Missingness
# =========================================================
with tab_data:
    st.markdown("### 📊 Dataset Exploration & Missingness Audit")
    raw_df = load_raw_dataset()

    col_d1, col_d2 = st.columns([1, 1])

    with col_d1:
        st.markdown("#### 📋 Raw Patient Health Data (Sample)")
        st.dataframe(raw_df.head(15), use_container_width=True)
        st.caption(f"Showing 15 of {len(raw_df)} patient records.")

    with col_d2:
        st.markdown("#### ⚠️ Biologically Invalid Zeros (Missing Data)")
        st.write("""
        In health datasets, measurements such as **Glucose, Blood Pressure, Insulin, Skin Thickness, and BMI cannot be zero in a living human**. 
        These zeros represent missing/unrecorded entries and must be statistically imputed.
        """)
        missing_fig = plot_missing_audit_plotly(meta["missing_stats"])
        st.plotly_chart(missing_fig, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 📈 Clinical Feature Distributions by Diabetes Outcome")

    selected_feat = st.selectbox(
        "Select Clinical Biomarker to Inspect:",
        options=FEATURE_COLS,
        index=1
    )

    clean_viz_df = raw_df.copy()
    if selected_feat in ZERO_AS_MISSING_COLS:
        clean_viz_df = clean_viz_df[clean_viz_df[selected_feat] > 0]

    col_hist, col_box = st.columns([2, 1])

    with col_hist:
        hist_fig = px.histogram(
            clean_viz_df,
            x=selected_feat,
            color="Outcome",
            barmode="overlay",
            nbins=35,
            color_discrete_map={0: "#38bdf8", 1: "#ef4444"},
            labels={"Outcome": "Condition"},
            title=f"Distribution of {selected_feat} (0 = Healthy, 1 = Diabetic)",
            template="plotly_dark"
        )
        hist_fig.update_layout(paper_bgcolor="rgba(15, 23, 42, 0.6)", plot_bgcolor="rgba(15, 23, 42, 0.6)", height=350)
        st.plotly_chart(hist_fig, use_container_width=True)

    with col_box:
        box_fig = px.box(
            clean_viz_df,
            x="Outcome",
            y=selected_feat,
            color="Outcome",
            color_discrete_map={0: "#38bdf8", 1: "#ef4444"},
            title=f"Boxplot: {selected_feat} by Outcome",
            template="plotly_dark"
        )
        box_fig.update_layout(paper_bgcolor="rgba(15, 23, 42, 0.6)", plot_bgcolor="rgba(15, 23, 42, 0.6)", height=350)
        st.plotly_chart(box_fig, use_container_width=True)

    # Correlation Matrix
    st.markdown("---")
    st.markdown("#### 🔗 Biomarker Correlation Heatmap")
    corr = raw_df.corr()
    corr_fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Pairwise Pearson Correlation Matrix",
        template="plotly_dark"
    )
    corr_fig.update_layout(paper_bgcolor="rgba(15, 23, 42, 0.6)", plot_bgcolor="rgba(15, 23, 42, 0.6)", height=450)
    st.plotly_chart(corr_fig, use_container_width=True)


# =========================================================
# TAB 4: ML Pipeline & Feature Normalization
# =========================================================
with tab_pipeline:
    st.markdown("### 🧠 Spark MLlib End-to-End Distributed Pipeline")
    st.markdown(r"""
    The pipeline conforms strictly to **Machine Learning Best Practices**:
    1. **Strict Featurization Order**: Train/Test split is executed **BEFORE** fitting imputer and standardizer to guarantee **Zero Data Leakage**.
    2. **Zero-to-Null Masking**: Invalid physiological zeros are replaced with `null` in Spark DataFrame.
    3. **Spark Imputer**: Distributed median imputation per partition.
    4. **VectorAssembler**: Combines 8 scalar columns into dense vector column.
    5. **StandardScaler (Normalization)**: Centers features to zero mean ($\mu = 0$) and scales to unit variance ($\sigma = 1$). 
       This prevents high-magnitude features (e.g. Insulin up to 846) from drowning out low-magnitude features (e.g. Pedigree Function around 0.5).
    6. **Logistic Regression with Distributed L-BFGS**: Iterative Quasi-Newton optimization executed via tree-aggregation across worker nodes.
    """)

    col_p1, col_p2 = st.columns([1, 1])

    with col_p1:
        st.markdown("#### ⚙️ Pipeline Configuration Details")
        st.json({
            "Stage 1": "Imputer (strategy='median', columns=Glucose, BP, Skin, Insulin, BMI)",
            "Stage 2": "VectorAssembler (inputCols=8 features, outputCol='unscaled_features')",
            "Stage 3": "StandardScaler (withMean=True, withStd=True, outputCol='features')",
            "Stage 4": f"LogisticRegression (regParam={reg_param}, elasticNetParam={elastic_net}, maxIter={max_iter})"
        })

    with col_p2:
        st.markdown("#### 📐 Mathematical Formulation")
        st.latex(r"P(Y = 1 | \mathbf{x}) = \frac{1}{1 + e^{-(\beta_0 + \mathbf{\beta}^T \mathbf{z})}}")
        st.write("Where normalized feature vector $\\mathbf{z}$ is:")
        st.latex(r"z_i = \frac{x_i - \mu_i}{\sigma_i}")
        st.write("And regularized objective function solved across cluster workers:")
        st.latex(r"\min_{\mathbf{\beta}} -\sum_{i=1}^N \left[ y_i \log p_i + (1-y_i)\log(1-p_i) \right] + \lambda \left[ (1-\alpha)\frac{1}{2}\|\mathbf{\beta}\|_2^2 + \alpha \|\mathbf{\beta}\|_1 \right]")


# =========================================================
# TAB 5: Patient Risk Predictor
# =========================================================
with tab_predict:
    st.markdown("### 🩺 Real-Time Patient Diabetes Risk Predictor")
    st.caption("Input clinical biomarker readings to compute real-time diabetes probability via the trained pipeline.")

    raw_df_clean = load_raw_dataset()
    med_healthy = raw_df_clean[raw_df_clean["Outcome"] == 0].median()
    med_diabetic = raw_df_clean[raw_df_clean["Outcome"] == 1].median()

    with st.form("patient_form"):
        st.markdown("##### 📝 Clinical Biomarker Readings")
        p_c1, p_c2, p_c3, p_c4 = st.columns(4)

        with p_c1:
            inp_preg = st.number_input("Pregnancies", min_value=0, max_value=20, value=2, step=1)
            inp_gluc = st.number_input("Glucose (mg/dL)", min_value=0, max_value=300, value=130, step=1, help="Normal fasting: 70-99 mg/dL")
        with p_c2:
            inp_bp = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=75, step=1, help="Normal: 60-80 mm Hg")
            inp_skin = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=25, step=1)
        with p_c3:
            inp_ins = st.number_input("Serum Insulin (mu U/ml)", min_value=0, max_value=900, value=110, step=5, help="Normal 2hr: 16-166 mu U/ml")
            inp_bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=70.0, value=31.5, step=0.5, help="Normal: 18.5 - 24.9")
        with p_c4:
            inp_ped = st.number_input("Diabetes Pedigree Function", min_value=0.05, max_value=3.0, value=0.45, step=0.01, help="Genetic family history score")
            inp_age = st.number_input("Age (Years)", min_value=18, max_value=120, value=42, step=1)

        submitted = st.form_submit_button("⚡ Predict Patient Risk", type="primary")

    patient_data = {
        "Pregnancies": inp_preg,
        "Glucose": inp_gluc,
        "BloodPressure": inp_bp,
        "SkinThickness": inp_skin,
        "Insulin": inp_ins,
        "BMI": inp_bmi,
        "DiabetesPedigreeFunction": inp_ped,
        "Age": inp_age
    }

    # Perform inference
    if spark is not None and "pipeline_model" in results and PipelineModel is not None and isinstance(results["pipeline_model"], PipelineModel):
        pred_result = predict_patient_risk(results["pipeline_model"], patient_data, spark)
    else:
        # Fallback using scikit-learn
        sk_pipe = results["pipeline_model"]
        pat_df = pd.DataFrame([patient_data])
        for z_col in ZERO_AS_MISSING_COLS:
            if pat_df[z_col].iloc[0] == 0:
                pat_df[z_col] = np.nan
        prob = float(sk_pipe.predict_proba(pat_df)[0][1])
        pred_class = int(prob >= decision_threshold)
        if prob >= 0.65:
            risk = "High Risk"
            col = "#ef4444"
        elif prob >= 0.35:
            risk = "Moderate Risk"
            col = "#f59e0b"
        else:
            risk = "Low Risk"
            col = "#10b981"
        pred_result = {
            "prediction": pred_class,
            "probability_positive": round(prob * 100, 2),
            "risk_level": risk,
            "color": col
        }

    col_res1, col_res2 = st.columns([1, 1])

    with col_res1:
        st.markdown(f"""
        <div style="background: rgba(19, 27, 46, 0.85); border: 2px solid {pred_result['color']}; border-radius: 12px; padding: 24px; text-align: center; margin-top: 10px;">
            <div style="font-size: 1rem; color: #94a3b8; font-weight: 600; text-transform: uppercase;">Predicted Disease Status</div>
            <div style="font-size: 2.3rem; font-weight: 800; color: {pred_result['color']}; margin: 8px 0;">
                {'DIABETIC (Positive)' if pred_result['prediction'] == 1 else 'HEALTHY (Negative)'}
            </div>
            <div style="font-size: 1.25rem; font-weight: 600; color: #f8fafc;">
                Risk Probability: <span style="color: {pred_result['color']}; font-weight: 800;">{pred_result['probability_positive']}%</span>
            </div>
            <div style="margin-top: 12px;">
                <span style="background: {pred_result['color']}22; color: {pred_result['color']}; border: 1px solid {pred_result['color']}; padding: 4px 14px; border-radius: 20px; font-weight: 700; font-size: 0.85rem;">
                    {pred_result['risk_level']}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        # Clinical Risk Gauge
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred_result["probability_positive"],
            number={"suffix": "%", "font": {"size": 28, "color": "white"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8"},
                "bar": {"color": pred_result["color"]},
                "bgcolor": "rgba(15, 23, 42, 0.6)",
                "steps": [
                    {"range": [0, 35], "color": "rgba(16, 185, 129, 0.25)"},
                    {"range": [35, 65], "color": "rgba(245, 158, 11, 0.25)"},
                    {"range": [65, 100], "color": "rgba(239, 68, 68, 0.25)"}
                ],
                "threshold": {
                    "line": {"color": "white", "width": 3},
                    "thickness": 0.75,
                    "value": decision_threshold * 100
                }
            }
        ))
        gauge_fig.update_layout(height=230, margin=dict(l=30, r=30, t=20, b=20), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(gauge_fig, use_container_width=True)

    with col_res2:
        radar_fig = plot_patient_radar_plotly(patient_data, med_healthy, med_diabetic)
        st.plotly_chart(radar_fig, use_container_width=True)


# =========================================================
# TAB 6: Project Report & Academic Export
# =========================================================
with tab_report:
    st.markdown("### 📑 Cluster Computing Project Academic Summary")
    st.write("Complete report on architecture, methodology, pipeline execution, and performance benchmarks.")

    report_summary = f"""
# Disease Prediction using Apache Spark MLlib on Health Data

### 1. Executive Summary
- **Domain**: Predictive Healthcare / Chronic Disease Classification (Type-2 Diabetes).
- **Core Technology**: Apache Spark 4.2.0 (PySpark MLlib), Distributed Logistic Regression, Streamlit Interactive Dashboard.
- **Dataset**: Pima Indians Diabetes Dataset (768 patient clinical profiles, 8 biomarker features).
- **Classification Performance**:
  - Test Accuracy: **{eval_metrics['accuracy'] * 100:.2f}%**
  - F1-Score: **{eval_metrics['f1_score']:.4f}**
  - Precision: **{eval_metrics['precision']:.4f}**
  - Recall (Sensitivity): **{eval_metrics['recall']:.4f}**
  - Specificity: **{eval_metrics['specificity']:.4f}**
  - Area Under ROC: **{eval_metrics['roc_auc']:.4f}**

### 2. Preprocessing & ML Best Practices
- **Strict Featurization Order**: Train and test partitions split prior to computing imputer and scaler statistics, preventing data leakage.
- **Physiological Missingness Audit**: Zeros detected in Glucose (0.7%), BloodPressure (4.6%), SkinThickness (29.6%), Insulin (48.7%), and BMI (1.4%) were treated as missing data and imputed via distributed median estimation.
- **Feature Normalization**: Features were standardized using Spark MLlib `StandardScaler(withMean=True, withStd=True)` to enforce zero mean and unit variance.

### 3. Cluster Architecture & Distributed Optimization
- **Data Distribution**: RDD / DataFrame partitioning enabled parallelized transformations across worker executors.
- **Optimization Algorithm**: Limited-memory BFGS (L-BFGS) distributed Quasi-Newton optimizer computing gradients across data partitions with tree-aggregated communication.
- **Top Risk Factors (Odds Ratios)**:
  1. Glucose (OR = {fi_df.loc[fi_df['Feature'] == 'Glucose', 'Odds_Ratio'].values[0]:.2f})
  2. BMI (OR = {fi_df.loc[fi_df['Feature'] == 'BMI', 'Odds_Ratio'].values[0]:.2f})
  3. Pregnancies (OR = {fi_df.loc[fi_df['Feature'] == 'Pregnancies', 'Odds_Ratio'].values[0]:.2f})
  4. Diabetes Pedigree Function (OR = {fi_df.loc[fi_df['Feature'] == 'DiabetesPedigreeFunction', 'Odds_Ratio'].values[0]:.2f})
"""

    st.markdown(report_summary)

    # Export downloads
    st.markdown("---")
    st.markdown("#### 💾 Export Project Deliverables")

    col_exp1, col_exp2 = st.columns(2)

    export_payload = {
        "project": "Health Data Disease Prediction",
        "cluster_framework": "Apache Spark MLlib",
        "spark_version": spark.version if spark else "N/A",
        "metrics": {
            "accuracy": eval_metrics["accuracy"],
            "f1_score": eval_metrics["f1_score"],
            "precision": eval_metrics["precision"],
            "recall": eval_metrics["recall"],
            "specificity": eval_metrics["specificity"],
            "roc_auc": eval_metrics["roc_auc"]
        },
        "confusion_matrix": eval_metrics["confusion_matrix"],
        "coefficients": fi_df.to_dict(orient="records")
    }

    with col_exp1:
        st.download_button(
            label="📥 Download Full Evaluation Metrics (JSON)",
            data=json.dumps(export_payload, indent=4),
            file_name="diabetes_spark_cluster_metrics.json",
            mime="application/json"
        )

    with col_exp2:
        csv_export = fi_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Feature Coefficients & Odds Ratios (CSV)",
            data=csv_export,
            file_name="diabetes_feature_importance.csv",
            mime="text/csv"
        )

"""
BITS Pilani - Work Integrated Learning Programmes Division
M.Tech (AIML) - Machine Learning (Assignment 2)
Streamlit Web Application: Telco Customer Churn Classification & Evaluation

Author: WILP M.Tech (AIML) Student
Dataset: Telco Customer Churn (Kaggle / IBM)
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    roc_curve
)

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Telco Churn Classifier | BITS Assignment 2",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-title {
        font-size: 0.85rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0F172A;
    }
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-blue { background-color: #DBEAFE; color: #1E40AF; }
    .badge-green { background-color: #DCFCE7; color: #166534; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Model Mapping and Filenames
# ---------------------------------------------------------
MODEL_CONFIGS = {
    "Logistic Regression": {
        "file": os.path.join("model", "logistic_regression.joblib"),
        "desc": "Linear classification model using log-odds optimization with L2 regularization."
    },
    "Decision Tree": {
        "file": os.path.join("model", "decision_tree.joblib"),
        "desc": "Non-linear tree-based classifier with max depth limit to prevent overfitting."
    },
    "KNN": {
        "file": os.path.join("model", "knn.joblib"),
        "desc": "Non-parametric instance-based classifier using Euclidean distance (k=15)."
    },
    "Naive Bayes": {
        "file": os.path.join("model", "naive_bayes.joblib"),
        "desc": "Probabilistic classifier based on Bayes' theorem with Gaussian feature assumptions."
    },
    "Support Vector Machine": {
        "file": os.path.join("model", "svm.joblib"),
        "desc": "Kernel classifier (RBF) maximizing margin hyperplane between classes."
    }
}

REQUIRED_NUM_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']
TARGET_COL = 'Churn'

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
@st.cache_resource
def load_saved_model(model_name: str):
    file_path = MODEL_CONFIGS[model_name]["file"]
    if not os.path.exists(file_path):
        st.error(f"Model file not found: `{file_path}`. Please ensure training script has executed.")
        st.stop()
    return joblib.load(file_path)

@st.cache_data
def load_default_test_data():
    default_csv = "test_data.csv"
    if os.path.exists(default_csv):
        return pd.read_csv(default_csv)
    return None

def clean_and_prepare_df(df_input: pd.DataFrame):
    """Clean data types and handle missing values in uploaded test data."""
    df = df_input.copy()
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    
    # Clean TotalCharges numeric column
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(0.0)
    
    return df

def extract_features_and_target(df: pd.DataFrame):
    """Separate input features and ground-truth target."""
    if TARGET_COL not in df.columns:
        return None, None, f"Target column '{TARGET_COL}' not found in uploaded test CSV."
    
    y_raw = df[TARGET_COL]
    # Handle string Yes/No or numeric 1/0
    if y_raw.dtype == object or str(y_raw.dtype) == 'category':
        y_true = (y_raw.astype(str).str.strip().str.lower().isin(['yes', '1', 'true'])).astype(int)
    else:
        y_true = y_raw.astype(int)
        
    X_df = df.drop(columns=[TARGET_COL])
    return X_df, y_true, None

def compute_metrics(y_true, y_pred, y_prob):
    """Compute all 6 required evaluation metrics."""
    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = 0.5
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    return {
        "Accuracy": acc,
        "AUC": auc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "MCC": mcc
    }

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/thumb/d/d3/BITS_Pilani-Logo.svg/300px-BITS_Pilani-Logo.svg.png", width=110)
st.sidebar.markdown("### **BITS Pilani - M.Tech AIML**")
st.sidebar.caption("Machine Learning (Assignment 2) | Cloud Deployment")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📂 **1. Dataset Input**")

uploaded_file = st.sidebar.file_uploader(
    "Upload Test Data (CSV)",
    type=["csv"],
    help="Upload test dataset containing features and the target column 'Churn'."
)

# Provide download of default test_data.csv for evaluator convenience
default_df = load_default_test_data()
if default_df is not None:
    csv_bytes = default_df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label="📥 Download Sample test_data.csv",
        data=csv_bytes,
        file_name="test_data.csv",
        mime="text/csv",
        help="Click to download benchmark test set for re-uploading."
    )

st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 **2. Model Selection**")
model_options = ["All Models (Comparison)"] + list(MODEL_CONFIGS.keys())
selected_model_option = st.sidebar.selectbox(
    "Select Model to Inspect:",
    options=model_options,
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ **Dataset Metadata**")
st.sidebar.info("""
- **Dataset**: Telco Customer Churn
- **Source**: Kaggle / IBM Analytics
- **Instances**: 7,032 total (1,407 test)
- **Features**: 19 inputs (>12 required)
- **Target**: Churn (Yes / No)
""")

# ---------------------------------------------------------
# Data Loading & Verification
# ---------------------------------------------------------
if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
    data_source_badge = '<span class="badge badge-green">Uploaded User Test CSV</span>'
else:
    raw_df = default_df
    data_source_badge = '<span class="badge badge-blue">Benchmark Test CSV (1,407 rows)</span>'

if raw_df is None:
    st.error("No dataset available. Please upload a test CSV or ensure `test_data.csv` is present.")
    st.stop()

clean_df = clean_and_prepare_df(raw_df)
X_test, y_test, error_msg = extract_features_and_target(clean_df)

if error_msg:
    st.error(f"❌ {error_msg}")
    st.info("The uploaded test CSV must include the 'Churn' target column along with customer features.")
    st.stop()

# ---------------------------------------------------------
# Main Page Header
# ---------------------------------------------------------
st.markdown('<div class="main-header">📊 Customer Churn Prediction & Model Evaluation Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="sub-header">Evaluation of 5 Classical Machine Learning Classifiers | Data Source: {data_source_badge}</div>',
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# Expander: Dataset Overview
# ---------------------------------------------------------
with st.expander("🔍 Click to view Test Dataset Summary & Preview", expanded=False):
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Total Test Samples", f"{len(clean_df):,}")
    col_b.metric("Feature Columns", f"{X_test.shape[1]}")
    churn_rate = (y_test.sum() / len(y_test)) * 100
    col_c.metric("Churn Rate (Class 1)", f"{churn_rate:.1f}%")
    
    st.dataframe(clean_df.head(5), use_container_width=True)

# ---------------------------------------------------------
# CASE 1: ALL MODELS COMPARISON VIEW
# ---------------------------------------------------------
if selected_model_option == "All Models (Comparison)":
    st.markdown("### 🏆 Comprehensive Model Comparison")
    st.caption("Side-by-side evaluation metrics calculated on the active test dataset without retraining.")
    
    comparison_results = []
    roc_data = {}
    
    for name in MODEL_CONFIGS.keys():
        model = load_saved_model(name)
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_test)
        else:
            y_prob = y_pred
            
        metrics = compute_metrics(y_test, y_pred, y_prob)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = (fpr, tpr, metrics["AUC"])
        
        comparison_results.append({
            "ML Model Name": name,
            "Accuracy": metrics["Accuracy"],
            "AUC": metrics["AUC"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1": metrics["F1"],
            "MCC": metrics["MCC"]
        })
    
    comp_df = pd.DataFrame(comparison_results)
    
    # Display table formatted nicely
    formatted_df = comp_df.copy()
    for col in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]:
        formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.4f}")
        
    st.markdown("#### **Evaluation Metrics Comparison Table**")
    st.table(formatted_df)
    
    # Visual Comparisons (Bar Charts + ROC Curve)
    col_chart1, col_chart2 = st.columns([1, 1])
    
    with col_chart1:
        st.markdown("#### 📊 Metric Comparison Across Models")
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_df = comp_df.melt(id_vars=["ML Model Name"], value_vars=["Accuracy", "AUC", "F1", "MCC"], var_name="Metric", value_name="Score")
        sns.barplot(data=plot_df, x="ML Model Name", y="Score", hue="Metric", palette="crest", ax=ax)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Score (0.0 - 1.0)")
        ax.set_xlabel("")
        ax.tick_params(axis='x', rotation=20)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.legend(title="Metric", loc="lower right")
        st.pyplot(fig)
        plt.close(fig)
        
    with col_chart2:
        st.markdown("#### 📈 Multi-Model ROC Curves")
        fig_roc, ax_roc = plt.subplots(figsize=(8, 5))
        colors = ["#2563EB", "#16A34A", "#D97706", "#DC2626", "#7C3AED"]
        for idx, (m_name, (fpr, tpr, auc_val)) in enumerate(roc_data.items()):
            ax_roc.plot(fpr, tpr, label=f"{m_name} (AUC = {auc_val:.4f})", color=colors[idx % len(colors)], linewidth=2)
        ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.6, label="Random Guess (0.50)")
        ax_roc.set_xlabel("False Positive Rate (1 - Specificity)")
        ax_roc.set_ylabel("True Positive Rate (Recall)")
        ax_roc.set_title("ROC Curve Comparison")
        ax_roc.legend(loc="lower right", fontsize=8)
        ax_roc.grid(True, linestyle='--', alpha=0.4)
        st.pyplot(fig_roc)
        plt.close(fig_roc)

    # Observations Table
    st.markdown("---")
    st.markdown("### 📝 Performance Observations")
    obs_data = [
        {"ML Model Name": "Logistic Regression", "Observation about model performance": "Best overall performance. Achieves the highest Accuracy (0.8045), AUC (0.8361), F1 (0.6088), and MCC (0.4810). Highly balanced precision-recall tradeoff on normalized features."},
        {"ML Model Name": "Decision Tree", "Observation about model performance": "Produces clear, interpretable rules with moderate performance (Acc: 0.7783, AUC: 0.8195). Constrained max_depth prevents severe overfitting, maintaining balanced precision and recall."},
        {"ML Model Name": "KNN", "Observation about model performance": "Effective distance-based classification (Acc: 0.7747, AUC: 0.8181) after standard scaling. Balanced precision (0.5789) and recall (0.5588), but inference requires distance computation against all training instances."},
        {"ML Model Name": "Naive Bayes", "Observation about model performance": "Strongest recall (0.8636) capturing most churners, but suffers from low precision (0.4184) and lower overall accuracy (0.6446) due to the naive conditional independence assumption across correlated customer features."},
        {"ML Model Name": "Support Vector Machine", "Observation about model performance": "Solid accuracy (0.7932) and high precision (0.6487), but lower recall (0.4840) as it establishes a conservative decision boundary around the non-linear feature space."},
        {"ML Model Name": "Overall, Winner for your dataset?", "Observation about model performance": "Logistic Regression is the overall winner. It delivers the highest test Accuracy (80.45%), ROC-AUC (0.8361), F1 Score (0.6088), and MCC (0.4810), providing the most reliable probability calibration for customer retention."}
    ]
    st.table(pd.DataFrame(obs_data))

# ---------------------------------------------------------
# CASE 2: INDIVIDUAL MODEL INSPECTION VIEW
# ---------------------------------------------------------
else:
    model_name = selected_model_option
    st.markdown(f"### 🎯 Detailed Inspection: **{model_name}**")
    st.caption(MODEL_CONFIGS[model_name]["desc"])
    
    # Load and execute inference
    model = load_saved_model(model_name)
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_test)
    else:
        y_prob = y_pred
        
    metrics = compute_metrics(y_test, y_pred, y_prob)
    
    # 6 Metric KPI Cards
    st.markdown("#### **Evaluation Metrics on Test Data**")
    m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
    
    with m_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Accuracy</div>
            <div class="metric-value">{metrics['Accuracy']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">AUC Score</div>
            <div class="metric-value">{metrics['AUC']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Precision</div>
            <div class="metric-value">{metrics['Precision']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Recall</div>
            <div class="metric-value">{metrics['Recall']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">F1 Score</div>
            <div class="metric-value">{metrics['F1']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col6:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">MCC Score</div>
            <div class="metric-value">{metrics['MCC']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Detailed Diagnostics: Confusion Matrix & Classification Report
    diag_col1, diag_col2 = st.columns([1, 1])
    
    with diag_col1:
        st.markdown("#### 🔲 Confusion Matrix Heatmap")
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        fig_cm, ax_cm = plt.subplots(figsize=(6, 4.5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Predicted: Stay (0)", "Predicted: Churn (1)"],
            yticklabels=["Actual: Stay (0)", "Actual: Churn (1)"],
            ax=ax_cm,
            annot_kws={"size": 14, "weight": "bold"}
        )
        ax_cm.set_ylabel("Ground Truth Label", fontsize=11)
        ax_cm.set_xlabel("Model Predicted Label", fontsize=11)
        st.pyplot(fig_cm)
        plt.close(fig_cm)
        
        st.caption(f"**Breakdown**: True Negatives (TN): `{tn}` | False Positives (FP): `{fp}` | False Negatives (FN): `{fn}` | True Positives (TP): `{tp}`")

    with diag_col2:
        st.markdown("#### 📋 Classification Report")
        report_dict = classification_report(y_test, y_pred, output_dict=True, target_names=["Retained (No)", "Churned (Yes)"])
        report_df = pd.DataFrame(report_dict).transpose()
        st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)
        
        # Plot ROC Curve for individual model
        st.markdown("#### 📈 ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig_ind_roc, ax_ind = plt.subplots(figsize=(6, 3.2))
        ax_ind.plot(fpr, tpr, color="#2563EB", lw=2, label=f"ROC (AUC = {metrics['AUC']:.4f})")
        ax_ind.plot([0, 1], [0, 1], 'k--', alpha=0.5, label="Chance")
        ax_ind.set_xlabel("False Positive Rate", fontsize=9)
        ax_ind.set_ylabel("True Positive Rate", fontsize=9)
        ax_ind.legend(loc="lower right", fontsize=9)
        ax_ind.grid(True, linestyle="--", alpha=0.4)
        st.pyplot(fig_ind_roc)
        plt.close(fig_ind_roc)

# ---------------------------------------------------------
# Interactive Single-Customer Inference Sandbox
# ---------------------------------------------------------
st.markdown("---")
with st.expander("🔮 Interactive Single-Customer Churn Prediction Sandbox", expanded=False):
    st.markdown("Test live real-time inference on a custom customer profile using the trained pipeline.")
    
    sim_col1, sim_col2, sim_col3 = st.columns(3)
    
    with sim_col1:
        sim_gender = st.selectbox("Gender", ["Female", "Male"], key="sim_gender")
        sim_senior = st.selectbox("Senior Citizen", [0, 1], key="sim_senior")
        sim_partner = st.selectbox("Partner", ["Yes", "No"], key="sim_partner")
        sim_dep = st.selectbox("Dependents", ["No", "Yes"], key="sim_dep")
        sim_tenure = st.slider("Tenure (Months)", min_value=1, max_value=72, value=12, key="sim_tenure")
        sim_phone = st.selectbox("Phone Service", ["Yes", "No"], key="sim_phone")
        sim_mult = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"], key="sim_mult")
        
    with sim_col2:
        sim_net = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], key="sim_net")
        sim_sec = st.selectbox("Online Security", ["No", "Yes", "No internet service"], key="sim_sec")
        sim_bak = st.selectbox("Online Backup", ["No", "Yes", "No internet service"], key="sim_bak")
        sim_dev = st.selectbox("Device Protection", ["No", "Yes", "No internet service"], key="sim_dev")
        sim_tech = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], key="sim_tech")
        sim_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"], key="sim_tv")
        sim_mov = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"], key="sim_mov")
        
    with sim_col3:
        sim_contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], key="sim_contract")
        sim_paper = st.selectbox("Paperless Billing", ["Yes", "No"], key="sim_paper")
        sim_pay = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ], key="sim_pay")
        sim_monthly = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=70.0, step=1.0)
        sim_total = st.number_input("Total Charges ($)", min_value=18.0, max_value=9000.0, value=float(sim_monthly * sim_tenure), step=10.0)
        sim_model_choice = st.selectbox("Model to Predict With", list(MODEL_CONFIGS.keys()), key="sim_model")

    if st.button("🚀 Run Live Prediction"):
        sample_dict = {
            'gender': sim_gender,
            'SeniorCitizen': sim_senior,
            'Partner': sim_partner,
            'Dependents': sim_dep,
            'tenure': sim_tenure,
            'PhoneService': sim_phone,
            'MultipleLines': sim_mult,
            'InternetService': sim_net,
            'OnlineSecurity': sim_sec,
            'OnlineBackup': sim_bak,
            'DeviceProtection': sim_dev,
            'TechSupport': sim_tech,
            'StreamingTV': sim_tv,
            'StreamingMovies': sim_mov,
            'Contract': sim_contract,
            'PaperlessBilling': sim_paper,
            'PaymentMethod': sim_pay,
            'MonthlyCharges': sim_monthly,
            'TotalCharges': sim_total
        }
        sample_df = pd.DataFrame([sample_dict])
        
        sim_pipe = load_saved_model(sim_model_choice)
        pred_label = sim_pipe.predict(sample_df)[0]
        pred_prob = sim_pipe.predict_proba(sample_df)[0][1] if hasattr(sim_pipe, "predict_proba") else 0.5
        
        res_col1, res_col2 = st.columns(2)
        if pred_label == 1:
            res_col1.error(f"⚠️ Prediction: **Likely to CHURN** (Probability: {pred_prob*100:.1f}%)")
        else:
            res_col1.success(f"✅ Prediction: **Likely to STAY** (Retention Probability: {(1-pred_prob)*100:.1f}%)")
        res_col2.info(f"Model used: **{sim_model_choice}**")

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.caption("BITS Pilani WILP M.Tech (AIML) - Machine Learning Assignment 2 | Built with Streamlit & Scikit-Learn")

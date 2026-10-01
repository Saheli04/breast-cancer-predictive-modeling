
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

# =========================================================
# LOAD DASHBOARD DATA
# =========================================================

data = joblib.load("dashboard_data.pkl")

df = data["df"]
X = data["X"]
y = data["y"]
X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]

logistic_accuracy = data["logistic_accuracy"]
tree_accuracy = data["tree_accuracy"]
forest_accuracy = data["forest_accuracy"]

logistic_auc = data["logistic_auc"]
tree_auc = data["tree_auc"]
forest_auc = data["forest_auc"]

y_pred_logistic = data["y_pred_logistic"]
y_pred_tree = data["y_pred_tree"]
y_pred_forest = data["y_pred_forest"]

y_prob_logistic = data["y_prob_logistic"]
y_prob_tree = data["y_prob_tree"]
y_prob_forest = data["y_prob_forest"]

cm_logistic = data["cm_logistic"]
cm_tree = data["cm_tree"]
cm_forest = data["cm_forest"]

fpr_logistic = data["fpr_logistic"]
tpr_logistic = data["tpr_logistic"]

fpr_tree = data["fpr_tree"]
tpr_tree = data["tpr_tree"]

fpr_forest = data["fpr_forest"]
tpr_forest = data["tpr_forest"]

feature_importance = data["feature_importance"]
metrics_comparison = data["metrics_comparison"]




# =========================================================
# PROFESSIONAL CHART STYLE
# =========================================================

plt.rcParams.update({
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#ffffff",
    "axes.edgecolor": "#dbe4f0",
    "axes.labelcolor": "#334155",
    "axes.titlecolor": "#0f172a",
    "xtick.color": "#475569",
    "ytick.color": "#475569",
    "text.color": "#0f172a",
    "font.size": 10,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.color": "#e2e8f0",
    "grid.alpha": 0.7
})


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Breast Cancer ML Dashboard",
    page_icon="🩺",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

/* =========================================
   GLOBAL PAGE
   ========================================= */

.stApp {
    background: #f4f7fb;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}


/* =========================================
   MAIN TITLE
   ========================================= */

h1 {
    color: #0f172a;
    font-size: 2.5rem !important;
    font-weight: 800 !important;
    letter-spacing: -1px;
}

h2, h3 {
    color: #172554;
    font-weight: 700 !important;
}


/* =========================================
   SIDEBAR
   ========================================= */

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #0f172a 0%,
        #172554 100%
    );
}

section[data-testid="stSidebar"] * {
    color: #f8fafc !important;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
}

section[data-testid="stSidebar"] .stSelectbox label {
    color: #cbd5e1 !important;
    font-weight: 600;
}


/* =========================================
   KPI CARDS
   ========================================= */

[data-testid="stMetric"] {
    background: #ffffff;
    padding: 22px 24px;
    border-radius: 14px;
    border: 1px solid #dbe4f0;
    box-shadow: 0 4px 15px rgba(15, 23, 42, 0.07);
    transition: all 0.2s ease;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 22px rgba(15, 23, 42, 0.12);
}

[data-testid="stMetricLabel"] {
    color: #64748b !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
}

[data-testid="stMetricValue"] {
    color: #0f172a !important;
    font-weight: 800 !important;
}


/* =========================================
   CHART / CONTENT CONTAINERS
   ========================================= */

div[data-testid="stVerticalBlock"] {
    border-radius: 12px;
}


/* =========================================
   DATAFRAME
   ========================================= */

[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #dbe4f0;
    box-shadow: 0 3px 12px rgba(15, 23, 42, 0.05);
}


/* =========================================
   SELECT BOX
   ========================================= */

div[data-baseweb="select"] > div {
    border-radius: 10px !important;
    border: 1px solid #475569 !important;
    background: #1e293b !important;
}


/* =========================================
   SUCCESS MESSAGE
   ========================================= */

div[data-testid="stAlert"] {
    border-radius: 12px;
}


/* =========================================
   DIVIDERS
   ========================================= */

hr {
    border: none;
    height: 1px;
    background: #dbe4f0;
    margin: 25px 0;
}


/* =========================================
   SCROLLBAR
   ========================================= */

::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: #e2e8f0;
}

::-webkit-scrollbar-thumb {
    background: #64748b;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div style="
    background: linear-gradient(135deg, #0f172a, #1e3a8a);
    padding: 28px 35px;
    border-radius: 18px;
    margin-bottom: 25px;
    box-shadow: 0 8px 25px rgba(15, 23, 42, 0.18);
">

<div style="
    color: #ffffff;
    font-size: 32px;
    font-weight: 800;
    letter-spacing: -0.5px;
">
🩺 Breast Cancer Predictive Modeling
</div>

<div style="
    color: #cbd5e1;
    font-size: 16px;
    margin-top: 8px;
">
Machine Learning Dashboard • Logistic Regression • Decision Tree • Random Forest
</div>

</div>
""", unsafe_allow_html=True)

st.markdown("---")


# =========================================================
# MODEL SELECTION
# =========================================================

st.sidebar.title("Dashboard Controls")

selected_model = st.sidebar.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ]
)


# =========================================================
# MODEL INFORMATION
# =========================================================

models = {

    "Logistic Regression": {
        "accuracy": logistic_accuracy,
        "auc": logistic_auc,
        "pred": y_pred_logistic,
        "cm": cm_logistic,
        "fpr": fpr_logistic,
        "tpr": tpr_logistic
    },

    "Decision Tree": {
        "accuracy": tree_accuracy,
        "auc": tree_auc,
        "pred": y_pred_tree,
        "cm": cm_tree,
        "fpr": fpr_tree,
        "tpr": tpr_tree
    },

    "Random Forest": {
        "accuracy": forest_accuracy,
        "auc": forest_auc,
        "pred": y_pred_forest,
        "cm": cm_forest,
        "fpr": fpr_forest,
        "tpr": tpr_forest
    }
}

model_data = models[selected_model]


# =========================================================
# KPI CARDS
# =========================================================

st.subheader("📌 Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Cases",
        len(df)
    )

with col2:
    st.metric(
        "Features",
        X.shape[1]
    )

with col3:
    st.metric(
        "Model Accuracy",
        f"{model_data['accuracy'] * 100:.2f}%"
    )

with col4:
    st.metric(
        "ROC-AUC",
        f"{model_data['auc']:.4f}"
    )


st.markdown("---")


# =========================================================
# ROW 1
# =========================================================

col1, col2 = st.columns(2)


# ---------------------------------------------------------
# DIAGNOSIS DISTRIBUTION
# ---------------------------------------------------------

with col1:

    st.subheader("Diagnosis Distribution")

    diagnosis_counts = y.value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.bar(
        ["Benign", "Malignant"],
        diagnosis_counts.values
    )

    ax.set_ylabel("Number of Cases")
    ax.set_xlabel("Diagnosis")

    for i, value in enumerate(diagnosis_counts.values):

        ax.text(
            i,
            value + 5,
            str(value),
            ha="center"
        )

    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------
# MODEL ACCURACY
# ---------------------------------------------------------

with col2:

    st.subheader("Model Accuracy Comparison")

    model_names = list(models.keys())

    accuracy_values = [
        models[m]["accuracy"]
        for m in model_names
    ]

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.bar(
        model_names,
        accuracy_values
    )

    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")

    ax.tick_params(
        axis="x",
        rotation=20
    )

    for i, value in enumerate(accuracy_values):

        ax.text(
            i,
            value + 0.02,
            f"{value:.3f}",
            ha="center"
        )

    st.pyplot(fig)
    plt.close(fig)


# =========================================================
# ROW 2
# =========================================================

col1, col2 = st.columns(2)


# ---------------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------------

with col1:

    st.subheader(
        f"{selected_model} - Confusion Matrix"
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        model_data["cm"],
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[
            "Benign",
            "Malignant"
        ],
        yticklabels=[
            "Benign",
            "Malignant"
        ],
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------
# ROC CURVE
# ---------------------------------------------------------

with col2:

    st.subheader("ROC Curve Comparison")

    fig, ax = plt.subplots(figsize=(7, 5))

    for name, model in models.items():

        ax.plot(
            model["fpr"],
            model["tpr"],
            label=f"{name} (AUC={model['auc']:.3f})"
        )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    ax.legend()

    st.pyplot(fig)
    plt.close(fig)


# =========================================================
# ROW 3
# =========================================================

col1, col2 = st.columns(2)


# ---------------------------------------------------------
# PRECISION RECALL F1
# ---------------------------------------------------------

with col1:

    st.subheader(
        f"{selected_model} - Classification Metrics"
    )

    predictions = model_data["pred"]

    precision = precision_score(
        y_test,
        predictions
    )

    recall = recall_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions
    )

    metric_names = [
        "Precision",
        "Recall",
        "F1-Score"
    ]

    metric_values = [
        precision,
        recall,
        f1
    ]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.bar(
        metric_names,
        metric_values
    )

    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")

    for i, value in enumerate(metric_values):

        ax.text(
            i,
            value + 0.02,
            f"{value:.3f}",
            ha="center"
        )

    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------
# FEATURE IMPORTANCE
# ---------------------------------------------------------

with col2:

    st.subheader("Top 10 Feature Importance")

    top_features = feature_importance.head(10)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.barh(
        top_features["Feature"][::-1],
        top_features["Importance"][::-1]
    )

    ax.set_xlabel("Importance")

    st.pyplot(fig)
    plt.close(fig)


# =========================================================
# MODEL PERFORMANCE TABLE
# =========================================================

st.markdown("---")

st.subheader("📊 Complete Model Performance")

display_table = metrics_comparison.copy()

numeric_columns = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score",
    "ROC-AUC"
]

display_table[numeric_columns] = (
    display_table[numeric_columns].round(4)
)

st.dataframe(
    display_table,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# DATASET INFORMATION
# =========================================================

st.markdown("---")

st.subheader("📁 Dataset Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.write(
        "**Dataset:** Breast Cancer Wisconsin (Diagnostic)"
    )

with col2:
    st.write(
        f"**Training Samples:** {len(X_train)}"
    )

with col3:
    st.write(
        f"**Testing Samples:** {len(X_test)}"
    )


st.markdown("---")

st.success(
    "Interactive dashboard generated successfully using Python + Streamlit."
)

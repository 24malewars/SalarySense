import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(APP_DIR, "salary_model.joblib")
METRICS_FILE = os.path.join(APP_DIR, "model_metrics.json")
DATASET_URL = "https://huggingface.co/datasets/hugginglearners/data-science-job-salaries/resolve/main/ds_salaries.csv"

# Fixed illustrative exchange rate matching training labels
DEFAULT_USD_TO_INR = 85.0

# -----------------------------------------------------------------------------
# Streamlit Page Setup & Custom CSS (Clean White Background + Vibrant Colorful Theme)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SalarySense — Data Science Salary Analytics & Prediction",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    /* Global Base: Crisp White Theme */
    .stApp {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        background-color: #FFFFFF !important;
    }

    /* Headings & Text Color Overrides for White Theme */
    h1, h2, h3, h4, h5, h6 {
        color: #0F172A !important;
        font-weight: 800;
    }
    p, span, label, div {
        color: #334155;
    }

    /* Sidebar Clean Light Background */
    [data-testid="stSidebar"] {
        background-color: #F8FAFC !important;
        border-right: 1.5px solid #E2E8F0 !important;
    }
    [data-testid="stSidebar"] * {
        color: #1E293B !important;
    }

    /* Top Hero Header: Vibrant Contrast Card */
    .hero-banner {
        background: linear-gradient(135deg, #0A0F1D 0%, #1E1B4B 35%, #0F3E48 70%, #064E3B 100%);
        border: 1.5px solid #0D9488;
        padding: 28px 34px;
        border-radius: 18px;
        margin-bottom: 22px;
        box-shadow: 0 10px 25px rgba(15, 23, 42, 0.12);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        margin-top: 8px;
        color: #CBD5E1 !important;
        line-height: 1.6;
    }

    /* Colorful Badge Pills */
    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 14px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.3px;
        text-transform: uppercase;
        transition: transform 0.15s ease;
    }
    .badge-pill:hover {
        transform: scale(1.04);
    }
    .pill-cyan {
        background: rgba(14, 165, 233, 0.2);
        color: #38BDF8 !important;
        border: 1px solid rgba(56, 189, 248, 0.5);
    }
    .pill-emerald {
        background: rgba(16, 185, 129, 0.2);
        color: #34D399 !important;
        border: 1px solid rgba(52, 211, 153, 0.5);
    }
    .pill-purple {
        background: rgba(139, 92, 246, 0.2);
        color: #C084FC !important;
        border: 1px solid rgba(168, 85, 247, 0.5);
    }
    .pill-amber {
        background: rgba(245, 158, 11, 0.2);
        color: #FBBF24 !important;
        border: 1px solid rgba(251, 191, 36, 0.5);
    }
    .pill-rose {
        background: rgba(244, 63, 94, 0.2);
        color: #FB7185 !important;
        border: 1px solid rgba(251, 113, 133, 0.5);
    }

    /* Vibrant Golden-Teal Exchange Rate Notice for White Theme */
    .rate-banner {
        background: linear-gradient(90deg, #FEF3C7 0%, #ECFDF5 100%);
        border-left: 5px solid #D97706;
        border-top: 1px solid #FDE68A;
        border-right: 1px solid #A7F3D0;
        border-bottom: 1px solid #A7F3D0;
        padding: 14px 20px;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
    }
    .rate-banner * {
        color: #1E293B !important;
    }

    /* Colorful Glassmorphic KPI Cards on White Background */
    .metric-card {
        border-radius: 16px;
        padding: 20px 22px;
        display: flex;
        align-items: center;
        gap: 16px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 12px;
    }
    .metric-card:hover {
        transform: translateY(-4px);
    }

    .card-cyan {
        background: #F0F9FF;
        border: 1.5px solid #7DD3FC;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.08);
    }
    .card-cyan:hover {
        box-shadow: 0 8px 24px rgba(2, 132, 199, 0.2);
    }

    .card-emerald {
        background: #ECFDF5;
        border: 1.5px solid #6EE7B7;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.08);
    }
    .card-emerald:hover {
        box-shadow: 0 8px 24px rgba(16, 185, 129, 0.2);
    }

    .card-purple {
        background: #F5F3FF;
        border: 1.5px solid #C4B5FD;
        box-shadow: 0 4px 14px rgba(139, 92, 246, 0.08);
    }
    .card-purple:hover {
        box-shadow: 0 8px 24px rgba(139, 92, 246, 0.2);
    }

    .card-coral {
        background: #FFF1F2;
        border: 1.5px solid #FDA4AF;
        box-shadow: 0 4px 14px rgba(244, 63, 94, 0.08);
    }
    .card-coral:hover {
        box-shadow: 0 8px 24px rgba(244, 63, 94, 0.2);
    }

    .metric-icon-wrap {
        width: 50px;
        height: 50px;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.6rem;
        flex-shrink: 0;
    }
    .icon-cyan { background: #E0F2FE; border: 1px solid #38BDF8; color: #0284C7; }
    .icon-emerald { background: #D1FAE5; border: 1px solid #34D399; color: #059669; }
    .icon-purple { background: #EDE9FE; border: 1px solid #A855F7; color: #7C3AED; }
    .icon-coral { background: #FFE4E6; border: 1px solid #FB7185; color: #E11D48; }

    .text-cyan { color: #0284C7 !important; }
    .text-emerald { color: #059669 !important; }
    .text-purple { color: #7C3AED !important; }
    .text-coral { color: #E11D48 !important; }

    .metric-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        color: #64748B !important;
        font-weight: 700;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 800;
        margin-top: 3px;
        letter-spacing: -0.5px;
    }
    .metric-sub {
        font-size: 0.83rem;
        color: #475569 !important;
        font-weight: 600;
        margin-top: 2px;
    }

    /* Prediction Result Card */
    .prediction-card {
        background: linear-gradient(135deg, #0A0F1D 0%, #1E1B4B 40%, #064E3B 100%);
        border: 2px solid #2DD4BF;
        border-radius: 18px;
        padding: 30px 34px;
        color: white;
        margin-top: 24px;
        margin-bottom: 26px;
        box-shadow: 0 12px 35px rgba(15, 23, 42, 0.15);
    }
    .prediction-card * {
        color: white;
    }
    .prediction-value-gradient {
        font-size: 3.1rem;
        font-weight: 900;
        background: linear-gradient(90deg, #38BDF8 0%, #34D399 50%, #FBBF24 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1px;
    }

    /* Form Submit & Action Buttons */
    .stFormSubmitButton > button {
        background: linear-gradient(135deg, #10B981 0%, #06B6D4 50%, #6366F1 100%) !important;
        color: white !important;
        font-weight: 800 !important;
        font-size: 1.05rem !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 28px !important;
        box-shadow: 0 6px 20px rgba(6, 182, 212, 0.3) !important;
        transition: all 0.2s ease !important;
    }
    .stFormSubmitButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.45) !important;
    }

    /* Missing Artifact Callout Box */
    .info-callout {
        background: #FEF3C7;
        border-left: 5px solid #F59E0B;
        border: 1px solid #FCD34D;
        padding: 16px 20px;
        border-radius: 12px;
        color: #92400E;
        margin-bottom: 22px;
    }
    .info-callout * {
        color: #92400E !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Formatting Helpers (Indian Numbering System & LPA)
# -----------------------------------------------------------------------------
def format_inr(number):
    """
    Format a number in the Indian numbering style (e.g. ₹12,50,000).
    """
    if number is None or (isinstance(number, float) and np.isnan(number)):
        return "₹0"
    n = int(round(number))
    is_neg = n < 0
    s = str(abs(n))
    if len(s) <= 3:
        formatted = s
    else:
        last3 = s[-3:]
        remaining = s[:-3]
        pairs = []
        while len(remaining) > 2:
            pairs.append(remaining[-2:])
            remaining = remaining[:-2]
        if remaining:
            pairs.append(remaining)
        formatted = ",".join(reversed(pairs)) + "," + last3
    return f"{'-' if is_neg else ''}₹{formatted}"

def format_lpa(number):
    """
    Format a number into Lakhs Per Annum (e.g. ₹12.50 LPA).
    1 Lakh = 1,00,000 INR
    """
    if number is None or (isinstance(number, float) and np.isnan(number)):
        return "₹0.00 LPA"
    lpa = number / 100000.0
    return f"₹{lpa:,.2f} LPA"

def format_monthly_inr(number):
    """
    Format monthly equivalent of an annual salary in INR.
    """
    if number is None or (isinstance(number, float) and np.isnan(number)):
        return "₹0 / mo"
    monthly = number / 12.0
    return f"{format_inr(monthly)} / mo"

# -----------------------------------------------------------------------------
# Data & Model Artifact Loading
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_dataset():
    """
    Loads and cleans the Data Science Job Salaries dataset.
    Converts salary_in_usd to salary_in_inr using the configured exchange rate.
    """
    try:
        df = pd.read_csv(DATASET_URL)
    except Exception:
        local_csv = os.path.join(APP_DIR, "ds_salaries.csv")
        if os.path.exists(local_csv):
            df = pd.read_csv(local_csv)
        else:
            return None

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df = df.drop_duplicates().reset_index(drop=True)

    df["salary_in_inr"] = df["salary_in_usd"] * DEFAULT_USD_TO_INR
    df["salary_in_lpa"] = df["salary_in_inr"] / 100000.0
    return df

@st.cache_resource(show_spinner=False)
def load_trained_model():
    """
    Loads the trained pipeline from salary_model.joblib.
    """
    if os.path.exists(MODEL_FILE):
        try:
            return joblib.load(MODEL_FILE)
        except Exception:
            return None
    return None

@st.cache_data(show_spinner=False)
def load_model_metrics():
    """
    Loads model evaluation metrics and metadata from model_metrics.json.
    """
    if os.path.exists(METRICS_FILE):
        try:
            with open(METRICS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def train_model_locally():
    """
    Trains the champion Gradient Boosting pipeline locally and exports salary_model.joblib.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.ensemble import GradientBoostingRegressor

    df = load_dataset()
    if df is None:
        return False

    num_cols = ["work_year", "remote_ratio"]
    cat_cols = ["experience_level", "employment_type", "job_title", "employee_residence", "company_location", "company_size"]
    features = num_cols + cat_cols

    X = df[features].copy()
    y = df["salary_in_inr"].copy()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        ]
    )
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, max_depth=4, random_state=42))
    ])
    pipeline.fit(X, y)
    joblib.dump(pipeline, MODEL_FILE)
    return True

df_data = load_dataset()
model_pipeline = load_trained_model()
metrics_data = load_model_metrics()

# -----------------------------------------------------------------------------
# UI Header & Currency Notice
# -----------------------------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px; flex-wrap: wrap;">
        <span class="badge-pill pill-emerald">🇮🇳 Target: Indian Rupees (₹)</span>
        <span class="badge-pill pill-purple">⚡ Benchmark: ₹85 / USD</span>
        <span class="badge-pill pill-amber">🏆 Gradient Boosting Champion</span>
    </div>
    <h1 class="hero-title">💼 SalarySense — Data Science Salary Intelligence</h1>
    <p class="hero-subtitle">
        Explore global data science compensation dynamics and predict annual packages converted into Indian Rupees (₹ and LPA) using trained Scikit-learn regressors.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="rate-banner">
    <div style="display: flex; align-items: flex-start; gap: 12px;">
        <span style="font-size: 1.4rem;">💡</span>
        <div>
            <strong style="color: #B45309; font-size: 1rem;">Exchange Rate Benchmark: 1 USD = ₹{DEFAULT_USD_TO_INR:.2f} INR (Illustrative Rate)</strong><br>
            <span style="color: #334155; font-size: 0.9rem;">
                This conversion provides an intuitive financial reference to evaluate global packages in the Indian numbering system (₹ and LPA). Note: direct currency conversion does not imply purchasing power parity (PPP) or domestic Indian hiring budgets.
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Navigation Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧭 Navigation")
    section = st.radio(
        "Go to Section:",
        ["📊 Overview", "🔮 Predict Salary", "🔍 Salary Explorer", "📈 Model Insights"],
        index=0
    )

# -----------------------------------------------------------------------------
# SECTION 1: OVERVIEW
# -----------------------------------------------------------------------------
if section == "📊 Overview":
    st.markdown("### 📌 Executive Summary & Key Indicators")
    
    if df_data is not None:
        total_records = len(df_data)
        median_salary_inr = df_data["salary_in_inr"].median()
        median_salary_lpa = df_data["salary_in_lpa"].median()
    else:
        total_records = metrics_data["dataset_stats"]["total_records"] if metrics_data else 565
        median_salary_inr = metrics_data["dataset_stats"]["median_salary_inr"] if metrics_data else 8627500.0
        median_salary_lpa = metrics_data["dataset_stats"]["median_salary_lpa"] if metrics_data else 86.28

    best_r2 = metrics_data["best_model_metrics"]["Test R²"] if metrics_data else "0.4612"
    best_model_name = metrics_data["best_model_name"] if metrics_data else "Gradient Boosting"
    best_mae_lpa = metrics_data["best_model_metrics"]["Test MAE (LPA)"] if metrics_data else "27.49"

    # Display Vibrant Colorful KPI Cards (Light theme)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card card-cyan">
            <div class="metric-icon-wrap icon-cyan">📁</div>
            <div>
                <div class="metric-title">Dataset Records</div>
                <div class="metric-value text-cyan">{total_records:,}</div>
                <div class="metric-sub">Cleaned Roles Analyzed</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card card-emerald">
            <div class="metric-icon-wrap icon-emerald">💰</div>
            <div>
                <div class="metric-title">Median Salary (INR)</div>
                <div class="metric-value text-emerald">{format_lpa(median_salary_inr)}</div>
                <div class="metric-sub">{format_inr(median_salary_inr)} / yr</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card card-purple">
            <div class="metric-icon-wrap icon-purple">⚡</div>
            <div>
                <div class="metric-title">Champion Model R²</div>
                <div class="metric-value text-purple">{best_r2}</div>
                <div class="metric-sub">{best_model_name}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card card-coral">
            <div class="metric-icon-wrap icon-coral">🎯</div>
            <div>
                <div class="metric-title">Test MAE (Error)</div>
                <div class="metric-value text-coral">₹{best_mae_lpa} LPA</div>
                <div class="metric-sub">Mean Absolute Error</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Overview Visualizations with Light Theme Charts
    if df_data is not None:
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.markdown("#### 📊 Salary Distribution (in LPA at ₹85/USD)")
            fig_dist = px.histogram(
                df_data,
                x="salary_in_lpa",
                nbins=35,
                color_discrete_sequence=["#06B6D4"],
                opacity=0.9,
                labels={"salary_in_lpa": "Annual Salary (₹ Lakhs Per Annum)"}
            )
            fig_dist.add_vline(
                x=median_salary_lpa,
                line_dash="dash",
                line_color="#E11D48",
                line_width=3,
                annotation_text=f"Median: ₹{median_salary_lpa:.1f} LPA",
                annotation_position="top right",
                annotation_font_color="#E11D48",
                annotation_font_size=12
            )
            fig_dist.update_layout(
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                margin=dict(l=20, r=20, t=30, b=20),
                height=340,
                xaxis=dict(gridcolor="#E2E8F0", tickfont=dict(color="#334155")),
                yaxis=dict(gridcolor="#E2E8F0", tickfont=dict(color="#334155"))
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        with chart_col2:
            st.markdown("#### 🎯 Salary Range by Experience Level")
            exp_mapping = {
                "EN": "Entry-level (EN)",
                "MI": "Mid-level (MI)",
                "SE": "Senior-level (SE)",
                "EX": "Executive (EX)"
            }
            df_plot = df_data.copy()
            df_plot["exp_label"] = df_plot["experience_level"].map(exp_mapping)
            order_exp = ["Entry-level (EN)", "Mid-level (MI)", "Senior-level (SE)", "Executive (EX)"]
            
            exp_colors = {
                "Entry-level (EN)": "#10B981",  # Vibrant Emerald
                "Mid-level (MI)": "#06B6D4",    # Electric Cyan
                "Senior-level (SE)": "#8B5CF6", # Royal Violet
                "Executive (EX)": "#F59E0B"     # Vibrant Amber
            }
            
            fig_box = px.box(
                df_plot,
                x="exp_label",
                y="salary_in_lpa",
                category_orders={"exp_label": order_exp},
                color="exp_label",
                color_discrete_map=exp_colors,
                labels={"exp_label": "Experience Level", "salary_in_lpa": "Salary (₹ LPA)"}
            )
            fig_box.update_layout(
                showlegend=False,
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                margin=dict(l=20, r=20, t=30, b=20),
                height=340,
                xaxis=dict(gridcolor="#E2E8F0", tickfont=dict(color="#334155")),
                yaxis=dict(gridcolor="#E2E8F0", tickfont=dict(color="#334155"))
            )
            st.plotly_chart(fig_box, use_container_width=True)

        # Top 10 High Paying Roles with Chromatic Vibrant Palette
        st.markdown("#### 🏆 Top 10 Data Science Roles by Median Salary (₹ LPA)")
        top_roles = df_data.groupby("job_title")["salary_in_lpa"].agg(["median", "count"]).reset_index()
        top_roles = top_roles[top_roles["count"] >= 2].sort_values(by="median", ascending=False).head(10)
        top_roles["median_formatted"] = top_roles["median"].apply(lambda x: f" ₹{x:.1f} LPA")

        vibrant_palette = [
            "#F43F5E", "#EC4899", "#8B5CF6", "#6366F1", "#3B82F6",
            "#0EA5E9", "#06B6D4", "#10B981", "#84CC16", "#F59E0B"
        ]

        fig_bar = px.bar(
            top_roles,
            x="median",
            y="job_title",
            orientation="h",
            color="job_title",
            color_discrete_sequence=vibrant_palette,
            labels={"median": "Median Salary (₹ LPA)", "job_title": "Job Role"},
            text="median_formatted"
        )
        fig_bar.update_traces(
            textposition="outside",
            textfont=dict(size=11, color="#0F172A", weight=700)
        )
        fig_bar.update_layout(
            yaxis=dict(autorange="reversed", gridcolor="#E2E8F0", tickfont=dict(color="#334155")),
            xaxis=dict(gridcolor="#E2E8F0", tickfont=dict(color="#334155")),
            showlegend=False,
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            margin=dict(l=20, r=65, t=20, b=20),
            height=430
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Dataset preview is not currently loaded. Place `ds_salaries.csv` or model artifacts in the app folder.")

# -----------------------------------------------------------------------------
# SECTION 2: PREDICT SALARY
# -----------------------------------------------------------------------------
elif section == "🔮 Predict Salary":
    st.markdown("### 🔮 Predict Data Science Compensation")
    st.write("Configure candidate profile details matching the trained Scikit-learn pipeline features.")

    # Graceful handling for missing model file
    if model_pipeline is None:
        st.markdown("""
        <div class="info-callout">
            <strong style="color: #92400E; font-size: 1.05rem;">⚠️ Model Artifact Missing (<code>salary_model.joblib</code>)</strong><br>
            The machine learning model has not been detected in the application directory. You can either:
            <ol style="margin-top: 8px; margin-bottom: 6px;">
                <li>Click the <strong>Train & Generate Model Locally</strong> button below to train immediately in ~3 seconds.</li>
                <li>Or run <code>SalarySense_Training.ipynb</code> in Google Colab and place the downloaded file here.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Train & Generate salary_model.joblib Locally Now", type="primary"):
            with st.spinner("Training Scikit-learn pipeline on Data Science Job Salaries..."):
                success = train_model_locally()
                if success:
                    st.success("✅ Model trained and saved as `salary_model.joblib`! Reloading dashboard...")
                    st.cache_resource.clear()
                    st.rerun()
                else:
                    st.error("Failed to load dataset. Please ensure an active internet connection or download the file.")

    # Input Form
    with st.form("salary_prediction_form"):
        st.markdown('<h5 style="color: #0284C7;">1. Role & Seniority</h5>', unsafe_allow_html=True)
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            work_year = st.selectbox(
                "Work Year",
                options=[2022, 2021, 2020, 2023, 2024],
                index=0,
                help="Year of employment record."
            )
        with col_b:
            exp_level = st.selectbox(
                "Experience Level",
                options=["EN", "MI", "SE", "EX"],
                format_func=lambda x: {
                    "EN": "🟢 Entry-level / Junior (EN)",
                    "MI": "🔵 Mid-level / Intermediate (MI)",
                    "SE": "🟣 Senior-level / Expert (SE)",
                    "EX": "🟡 Executive / Director (EX)"
                }[x],
                index=2
            )
        with col_c:
            emp_type = st.selectbox(
                "Employment Type",
                options=["FT", "PT", "CT", "FL"],
                format_func=lambda x: {
                    "FT": "Full-Time (FT)",
                    "PT": "Part-Time (PT)",
                    "CT": "Contractor (CT)",
                    "FL": "Freelancer (FL)"
                }[x],
                index=0
            )

        st.markdown('<h5 style="color: #7C3AED; margin-top: 14px;">2. Specialization & Work Mode</h5>', unsafe_allow_html=True)
        col_d, col_e = st.columns(2)
        
        available_titles = [
            "Data Scientist",
            "Data Engineer",
            "Data Analyst",
            "Machine Learning Engineer",
            "Research Scientist",
            "Data Science Manager",
            "Data Architect",
            "Machine Learning Scientist",
            "Big Data Engineer",
            "Director of Data Science",
            "Principal Data Scientist",
            "AI Scientist",
            "Lead Data Engineer",
            "BI Data Analyst",
            "ML Engineer",
            "Computer Vision Engineer",
            "Data Analytics Manager"
        ]
        if metrics_data and "features" in metrics_data and "categorical_options" in metrics_data["features"]:
            available_titles = metrics_data["features"]["categorical_options"].get("job_title", available_titles)
        elif df_data is not None:
            available_titles = sorted(df_data["job_title"].unique().tolist())

        with col_d:
            job_title = st.selectbox(
                "Job Title / Role",
                options=available_titles,
                index=available_titles.index("Data Scientist") if "Data Scientist" in available_titles else 0
            )
        with col_e:
            remote_ratio = st.select_slider(
                "Remote Work Arrangement",
                options=[0, 50, 100],
                value=100,
                format_func=lambda x: {
                    0: "🏢 0% (On-site / Office)",
                    50: "🌐 50% (Hybrid)",
                    100: "🏠 100% (Fully Remote)"
                }[x]
            )

        st.markdown('<h5 style="color: #059669; margin-top: 14px;">3. Geography & Organization Size</h5>', unsafe_allow_html=True)
        col_f, col_g, col_h = st.columns(3)

        country_options = ["US", "IN", "GB", "CA", "DE", "FR", "ES", "NL", "AU", "SG"]
        if df_data is not None:
            country_options = sorted(list(set(df_data["employee_residence"].unique().tolist() + ["IN", "US"])))

        with col_f:
            emp_residence = st.selectbox(
                "Employee Residence (Country Code)",
                options=country_options,
                index=country_options.index("US") if "US" in country_options else 0,
                help="ISO 2-letter country code (e.g., IN for India, US for United States)"
            )
        with col_g:
            comp_location = st.selectbox(
                "Company Location (Country Code)",
                options=country_options,
                index=country_options.index("US") if "US" in country_options else 0
            )
        with col_h:
            comp_size = st.selectbox(
                "Company Size",
                options=["S", "M", "L"],
                format_func=lambda x: {
                    "S": "🌱 Small (< 50 employees)",
                    "M": "🏢 Medium (50 - 250 employees)",
                    "L": "🏰 Large (> 250 employees)"
                }[x],
                index=1
            )

        submit_btn = st.form_submit_button("⚡ Calculate Predicted Salary (INR & LPA)", use_container_width=True)

    if submit_btn:
        input_data = pd.DataFrame([{
            "work_year": int(work_year),
            "remote_ratio": int(remote_ratio),
            "experience_level": str(exp_level),
            "employment_type": str(emp_type),
            "job_title": str(job_title),
            "employee_residence": str(emp_residence),
            "company_location": str(comp_location),
            "company_size": str(comp_size)
        }])

        if model_pipeline is not None:
            try:
                predicted_inr = float(model_pipeline.predict(input_data)[0])
                predicted_inr = max(500000.0, predicted_inr)
                predicted_lpa = predicted_inr / 100000.0
                predicted_usd = predicted_inr / DEFAULT_USD_TO_INR

                st.markdown(f"""
                <div class="prediction-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="font-size: 0.95rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.8px; font-weight: 700;">
                            ✨ Estimated Annual Package
                        </span>
                        <span class="badge-pill pill-emerald" style="font-size: 0.82rem;">⚡ Real-time ML Inference</span>
                    </div>
                    <div class="prediction-value-gradient">{format_lpa(predicted_inr)}</div>
                    <div style="display: flex; gap: 10px; margin-top: 14px; margin-bottom: 20px; flex-wrap: wrap;">
                        <span class="badge-pill pill-emerald">🇮🇳 Annual: {format_inr(predicted_inr)}</span>
                        <span class="badge-pill pill-cyan">📅 Monthly: {format_monthly_inr(predicted_inr)}</span>
                        <span class="badge-pill pill-amber">💵 USD Eqv: ${predicted_usd:,.0f}</span>
                        <span class="badge-pill pill-purple">🎯 Role: {job_title}</span>
                    </div>
                    <div style="font-size: 0.85rem; color: #CBD5E1; border-top: 1px solid rgba(255,255,255,0.15); padding-top: 14px;">
                        💡 <strong>Illustrative Conversion (₹85/USD):</strong> Represents a global benchmark translated to INR. Domestic Indian tech hiring budgets follow regional market pay scales.
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Context Gauge / Comparison
                st.markdown("##### 📊 Benchmark Context Comparison")
                median_ref = df_data["salary_in_lpa"].median() if df_data is not None else 86.28
                diff_pct = ((predicted_lpa - median_ref) / median_ref) * 100
                diff_sign = "+" if diff_pct >= 0 else ""
                diff_color = "#059669" if diff_pct >= 0 else "#E11D48"

                st.markdown(
                    f"This prediction is <strong style='color: {diff_color};'>{diff_sign}{diff_pct:.1f}%</strong> "
                    f"relative to the overall dataset median of <strong style='color: #0284C7;'>₹{median_ref:.2f} LPA</strong>.",
                    unsafe_allow_html=True
                )

            except Exception as e:
                st.error(f"Error executing prediction pipeline: {str(e)}")
        else:
            st.error("Cannot predict: `salary_model.joblib` was not found. Please train and export the model using the button above or Google Colab notebook.")

# -----------------------------------------------------------------------------
# SECTION 3: SALARY EXPLORER
# -----------------------------------------------------------------------------
elif section == "🔍 Salary Explorer":
    st.markdown("### 🔍 Interactive Salary Explorer")
    st.write("Filter, compare, and analyze international salary records converted into Indian Rupees (₹ and LPA).")

    if df_data is not None:
        # Filter controls
        st.markdown("##### 🎯 Filter Records")
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)

        all_roles = sorted(df_data["job_title"].unique().tolist())
        all_exp = ["EN", "MI", "SE", "EX"]
        all_remote = [0, 50, 100]
        all_sizes = ["S", "M", "L"]

        with f_col1:
            sel_roles = st.multiselect("Job Title(s)", options=all_roles, default=[])
        with f_col2:
            sel_exp = st.multiselect("Experience Level", options=all_exp, default=[])
        with f_col3:
            sel_remote = st.multiselect(
                "Remote Ratio (%)",
                options=all_remote,
                format_func=lambda x: f"{x}%",
                default=[]
            )
        with f_col4:
            sel_sizes = st.multiselect("Company Size", options=all_sizes, default=[])

        filtered_df = df_data.copy()
        if sel_roles:
            filtered_df = filtered_df[filtered_df["job_title"].isin(sel_roles)]
        if sel_exp:
            filtered_df = filtered_df[filtered_df["experience_level"].isin(sel_exp)]
        if sel_remote:
            filtered_df = filtered_df[filtered_df["remote_ratio"].isin(sel_remote)]
        if sel_sizes:
            filtered_df = filtered_df[filtered_df["company_size"].isin(sel_sizes)]

        st.markdown(f"**Found <span style='color: #0284C7;'>{len(filtered_df)}</span> matching records out of {len(df_data)} total.**", unsafe_allow_html=True)

        if len(filtered_df) > 0:
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f"""
                <div class="metric-card card-cyan">
                    <div>
                        <div class="metric-title">Filtered Count</div>
                        <div class="metric-value text-cyan">{len(filtered_df)}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="metric-card card-emerald">
                    <div>
                        <div class="metric-title">Median Salary</div>
                        <div class="metric-value text-emerald">{format_lpa(filtered_df["salary_in_inr"].median())}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div class="metric-card card-purple">
                    <div>
                        <div class="metric-title">Average Salary</div>
                        <div class="metric-value text-purple">{format_lpa(filtered_df["salary_in_inr"].mean())}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with m4:
                st.markdown(f"""
                <div class="metric-card card-coral">
                    <div>
                        <div class="metric-title">Maximum Salary</div>
                        <div class="metric-value text-coral">{format_lpa(filtered_df["salary_in_inr"].max())}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            exp_c1, exp_c2 = st.columns(2)
            with exp_c1:
                st.markdown("###### Salary Distribution in Selection (LPA)")
                fig_f_dist = px.histogram(
                    filtered_df,
                    x="salary_in_lpa",
                    nbins=25,
                    color_discrete_sequence=["#8B5CF6"],
                    labels={"salary_in_lpa": "Salary (₹ LPA)"}
                )
                fig_f_dist.update_layout(
                    paper_bgcolor="#FFFFFF",
                    plot_bgcolor="#FFFFFF",
                    margin=dict(l=20, r=20, t=20, b=20),
                    height=310,
                    xaxis=dict(gridcolor="#E2E8F0", tickfont=dict(color="#334155")),
                    yaxis=dict(gridcolor="#E2E8F0", tickfont=dict(color="#334155"))
                )
                st.plotly_chart(fig_f_dist, use_container_width=True)

            with exp_c2:
                st.markdown("###### Compensation by Remote Arrangement")
                remote_color_map = {
                    0: "#3B82F6",   # Electric Blue
                    50: "#8B5CF6",  # Purple
                    100: "#10B981"  # Emerald
                }
                fig_f_remote = px.box(
                    filtered_df,
                    x="remote_ratio",
                    y="salary_in_lpa",
                    color="remote_ratio",
                    labels={"remote_ratio": "Remote Ratio (%)", "salary_in_lpa": "Salary (₹ LPA)"},
                    color_discrete_map=remote_color_map
                )
                fig_f_remote.update_layout(
                    showlegend=False,
                    paper_bgcolor="#FFFFFF",
                    plot_bgcolor="#FFFFFF",
                    margin=dict(l=20, r=20, t=20, b=20),
                    height=310,
                    xaxis=dict(gridcolor="#E2E8F0", tickfont=dict(color="#334155")),
                    yaxis=dict(gridcolor="#E2E8F0", tickfont=dict(color="#334155"))
                )
                st.plotly_chart(fig_f_remote, use_container_width=True)

            # Data Table with Indian Formatting
            st.markdown("##### 📋 Filtered Records Data Table")
            display_df = filtered_df[[
                "job_title", "experience_level", "employment_type",
                "remote_ratio", "employee_residence", "company_location", "company_size",
                "salary_in_usd", "salary_in_inr", "salary_in_lpa"
            ]].copy()

            display_df["salary_in_inr_fmt"] = display_df["salary_in_inr"].apply(format_inr)
            display_df["salary_in_lpa_fmt"] = display_df["salary_in_inr"].apply(format_lpa)
            display_df["salary_in_usd_fmt"] = display_df["salary_in_usd"].apply(lambda x: f"${x:,.0f}")

            st.dataframe(
                display_df[[
                    "job_title", "experience_level", "employment_type", "remote_ratio",
                    "company_location", "company_size", "salary_in_usd_fmt", "salary_in_inr_fmt", "salary_in_lpa_fmt"
                ]].rename(columns={
                    "job_title": "Job Title",
                    "experience_level": "Exp",
                    "employment_type": "Type",
                    "remote_ratio": "Remote %",
                    "company_location": "Location",
                    "company_size": "Size",
                    "salary_in_usd_fmt": "Salary (USD)",
                    "salary_in_inr_fmt": "Salary (INR ₹)",
                    "salary_in_lpa_fmt": "Salary (LPA)"
                }),
                use_container_width=True,
                height=350
            )
        else:
            st.warning("No records matched the selected combination of filters. Please broaden your selection.")
    else:
        st.info("The original dataset is not currently loaded. Ensure `ds_salaries.csv` is accessible.")

# -----------------------------------------------------------------------------
# SECTION 4: MODEL INSIGHTS
# -----------------------------------------------------------------------------
elif section == "📈 Model Insights":
    st.markdown("### 📈 Model Evaluation, Comparisons & Caveats")
    st.write("Examine regression algorithm performance, actual vs predicted evaluations, and key methodology details.")

    # Model Comparison Table & Visuals with 4 distinct algorithm colors
    if metrics_data and "model_comparison" in metrics_data:
        comp_df = pd.DataFrame(metrics_data["model_comparison"])
        
        st.markdown("##### 🏆 Algorithm Performance Comparison (Held-Out Test Set)")
        st.dataframe(
            comp_df.rename(columns={
                "Model": "Algorithm",
                "Test R²": "R² Score",
                "Test MAE (₹)": "MAE (₹)",
                "Test MAE (LPA)": "MAE (₹ LPA)",
                "Test RMSE (₹)": "RMSE (₹)",
                "Test RMSE (LPA)": "RMSE (₹ LPA)"
            }),
            use_container_width=True
        )

        algo_colors = {
            "Gradient Boosting": "#10B981", # Emerald Champion
            "Random Forest": "#6366F1",     # Indigo
            "Linear Regression": "#0EA5E9", # Cyan
            "Decision Tree": "#F59E0B"      # Amber
        }

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("###### Test R² Score Comparison")
            fig_r2 = px.bar(
                comp_df,
                x="Model",
                y="Test R²",
                color="Model",
                color_discrete_map=algo_colors,
                text_auto=".3f"
            )
            fig_r2.update_layout(
                showlegend=False,
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                margin=dict(l=20, r=20, t=20, b=20),
                height=310,
                xaxis=dict(gridcolor="#E2E8F0", tickfont=dict(color="#334155")),
                yaxis=dict(gridcolor="#E2E8F0", tickfont=dict(color="#334155"))
            )
            st.plotly_chart(fig_r2, use_container_width=True)

        with c2:
            st.markdown("###### Test Mean Absolute Error (MAE in LPA)")
            fig_mae = px.bar(
                comp_df,
                x="Model",
                y="Test MAE (LPA)",
                color="Model",
                color_discrete_map=algo_colors,
                text_auto=".2f"
            )
            fig_mae.update_layout(
                showlegend=False,
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                margin=dict(l=20, r=20, t=20, b=20),
                height=310,
                xaxis=dict(gridcolor="#E2E8F0", tickfont=dict(color="#334155")),
                yaxis=dict(gridcolor="#E2E8F0", tickfont=dict(color="#334155"))
            )
            st.plotly_chart(fig_mae, use_container_width=True)
    else:
        st.info("Exported metrics JSON (`model_metrics.json`) is not present. Run `SalarySense_Training.ipynb` to generate full comparison metrics.")

    # Actual vs. Predicted Visual Inspection
    if metrics_data and "actual_vs_predicted_samples" in metrics_data:
        st.markdown("##### 🎯 Actual vs. Predicted Salaries on Held-Out Test Set")
        samples_df = pd.DataFrame(metrics_data["actual_vs_predicted_samples"])

        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=samples_df["actual_lpa"],
            y=samples_df["predicted_lpa"],
            mode="markers",
            marker=dict(
                size=10,
                color=samples_df["actual_lpa"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="LPA", tickfont=dict(color="#334155")),
                line=dict(width=1.5, color="white")
            ),
            name="Test Predictions"
        ))
        
        min_v = min(samples_df["actual_lpa"].min(), samples_df["predicted_lpa"].min())
        max_v = max(samples_df["actual_lpa"].max(), samples_df["predicted_lpa"].max())
        fig_scatter.add_trace(go.Scatter(
            x=[min_v, max_v],
            y=[min_v, max_v],
            mode="lines",
            line=dict(color="#E11D48", dash="dash", width=2.5),
            name="Ideal Fit (y = x)"
        ))
        fig_scatter.update_layout(
            xaxis_title="Actual Salary (₹ LPA)",
            yaxis_title="Predicted Salary (₹ LPA)",
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            margin=dict(l=20, r=20, t=20, b=20),
            height=390,
            xaxis=dict(gridcolor="#E2E8F0", tickfont=dict(color="#334155")),
            yaxis=dict(gridcolor="#E2E8F0", tickfont=dict(color="#334155"))
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Model Architecture, Preprocessing & Market Nuances
    st.markdown("##### 🔬 Preprocessing, Leakage Prevention & Key Limitations")
    with st.expander("1. Leakage-Free Preprocessing Architecture", expanded=True):
        st.markdown("""
        - **Pipeline Encapsulation:** Numerical features (`work_year`, `remote_ratio`) are scaled via `StandardScaler`, and categorical features are encoded via `OneHotEncoder(handle_unknown='ignore')`.
        - **Train-Test Isolation:** Preprocessors and scalers are fitted strictly on training data within cross-validation folds and evaluated on held-out test data, preventing any data leakage.
        - **Redundant Column Exclusion:** Raw local salaries (`salary`, `salary_currency`) and indices are explicitly dropped before pipeline training.
        """)

    with st.expander("2. Robust Handling of Unseen Categorical Values"):
        st.markdown("""
        - Real-world applications frequently encounter rare or novel job titles and country codes.
        - By configuring `OneHotEncoder(handle_unknown='ignore')`, any unseen categorical combination is safely ignored without crashing or corrupting predictions.
        """)

    with st.expander("3. Crucial Limitations of Currency-Converted Salaries in the Indian Context"):
        st.markdown("""
        - **Dataset Coverage:** The source dataset predominately reflects international compensation (US, Europe, UK, Canada).
        - **Nominal FX vs. Purchasing Power Parity (PPP):** Direct conversion using ₹85/USD provides an illustrative financial equivalent, but **does not imply that an identical role in India commands that dollar-converted salary**. Domestic Indian tech salaries reflect local market dynamics, cost of living, and domestic hiring budgets.
        """)

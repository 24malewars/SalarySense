# 💼 SalarySense — Data Science Salary Prediction (INR)

**SalarySense** is a machine learning web application built with **Python, Scikit-learn, and Streamlit**. It predicts and visualizes Data Science job compensation converted into **Indian Rupees (₹ INR)** and **Lakhs Per Annum (₹ LPA)** using a standard illustrative benchmark rate of **₹85 per USD**.

---

## 🚀 Key Features

- **End-to-End ML Pipeline**: Preprocessing with `ColumnTransformer` (`StandardScaler` + `OneHotEncoder` with `handle_unknown='ignore'`) preventing data leakage.
- **Model Benchmarking**: Trains and compares 4 regressors: Linear Regression, Decision Tree, Random Forest, and Gradient Boosting on MAE, RMSE, and $R^2$.
- **Indian Salary Representation**: Formats salaries using the Indian numbering system (`₹12,50,000`), Lakhs Per Annum (`₹12.50 LPA`), and monthly equivalents (`₹1,04,167 / mo`).
- **Interactive Streamlit Web Dashboard**:
  - **📊 Overview**: Dataset statistics, key performance metrics, and salary distribution by experience level.
  - **🔮 Predict Salary**: Instant salary estimation with comprehensive feature inputs (seniority, job title, remote ratio, location, company size).
  - **🔍 Salary Explorer**: Dynamic multi-filter analytics to inspect compensation patterns across roles and work arrangements.
  - **📈 Model Insights**: Algorithm performance comparisons, held-out test evaluations, actual vs. predicted charts, and technical caveats.
- **Illustrative Benchmark Caveat**: Clearly explains that international salaries converted at ₹85/USD provide a global benchmark and do not represent domestic Indian hiring budgets or purchasing power parity (PPP).

---

## 📁 Project Structure

```text
SalarySense/
│
├── SalarySense_Training.ipynb   # Complete Google Colab notebook for EDA, training & exporting
├── app.py                      # Modern navy-and-teal Streamlit dashboard (4 sections)
├── requirements.txt            # Minimal, production dependencies
└── README.md                   # Setup guide and instructions
```

*Note: Once trained, the exported artifacts `salary_model.joblib` and `model_metrics.json` reside directly in this directory alongside `app.py`.*

---

## 🛠️ Step-by-Step Guide

### Step 1: Run the Training Notebook in Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Click **Upload** and select `SalarySense_Training.ipynb` from your computer.
3. In Colab, navigate to the top menu and select **Runtime** → **Run all** (or press `Ctrl + F9`).
4. The notebook will:
   - Download the free Data Science Job Salaries dataset (`ds_salaries.csv`).
   - Clean the data and inspect distributions.
   - Convert USD salaries to INR targets (`salary_in_inr = salary_in_usd * 85.0`).
   - Train and cross-validate Linear Regression, Decision Tree, Random Forest, and Gradient Boosting models.
   - Select the champion model on held-out test data.
   - Automatically trigger browser downloads for:
     - `salary_model.joblib` (the fitted Scikit-Learn pipeline)
     - `model_metrics.json` (evaluation metrics and metadata)

---

### Step 2: Download Model Artifacts from Colab

If the automatic browser download doesn't trigger automatically:
1. In the left sidebar of Google Colab, click the **Folder icon** 📁 to open the file browser.
2. Locate `salary_model.joblib` and `model_metrics.json`.
3. Right-click each file and select **Download**.

---

### Step 3: Place Artifacts Beside `app.py`

Move or copy the two downloaded files directly into your local `SalarySense/` directory:

```text
SalarySense/
├── SalarySense_Training.ipynb
├── app.py
├── requirements.txt
├── README.md
├── salary_model.joblib          <-- Place here
└── model_metrics.json           <-- Place here
```

---

### Step 4: Install Dependencies (Windows)

Open **PowerShell** or **Command Prompt** in Windows, navigate to your project directory, and install the required dependencies:

```powershell
cd C:\Users\sgmal\.gemini\antigravity\scratch\SalarySense
pip install -r requirements.txt
```

*(Optional best practice: use a virtual environment)*
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

### Step 5: Launch the Streamlit Web Application

Run the following command in your terminal:

```powershell
streamlit run app.py
```

Your default web browser will open `http://localhost:8501` showing the SalarySense dashboard.

---

## 📊 Dataset & Conversion Details

- **Dataset**: [Hugging Face Data Science Job Salaries](https://huggingface.co/datasets/hugginglearners/data-science-job-salaries/resolve/main/ds_salaries.csv)
- **Target Column**: `salary_in_usd` converted to `salary_in_inr`
- **Exchange Rate**: Fixed illustrative benchmark of **1 USD = ₹85.00 INR**
- **Excluded Features**: Raw redundant salary columns (`salary`, `salary_currency`) and indices to prevent target leakage.
- **Model Features**:
  - `work_year` (Numeric)
  - `remote_ratio` (Numeric: 0, 50, 100)
  - `experience_level` (Categorical: EN, MI, SE, EX)
  - `employment_type` (Categorical: FT, PT, CT, FL)
  - `job_title` (Categorical: Data Scientist, ML Engineer, etc.)
  - `employee_residence` (Categorical: ISO country codes)
  - `company_location` (Categorical: ISO country codes)
  - `company_size` (Categorical: S, M, L)

---

## ⚖️ Indian Market & PPP Disclaimer

Salaries in this dataset originate primarily from international markets (United States, Europe, United Kingdom, Canada). The conversion rate of **₹85 / USD** is applied consistently as an illustrative financial metric to express values in Indian numbering style (₹ and LPA). 

This conversion serves as an **illustrative global compensation benchmark** and does **not imply** that local domestic employers in India offer direct dollar-equivalent salaries, due to differences in Purchasing Power Parity (PPP), cost of living, and domestic market compensation structures.

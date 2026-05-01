# 🏥 Dialysis Desert Risk Assessment Tracker

An interactive health policy and geospatial dashboard designed to provide early-warning signals for dialysis facility closures across the United States. 

By ingesting raw CMS Healthcare Cost Report Information System (HCRIS) data, this tool utilizes unsupervised machine learning to identify financial outliers that deviate significantly from industry norms, helping policymakers predict and prevent the formation of "Dialysis Deserts."

## 🌟 Key Features
* **Predictive Insolvency Modeling:** Uses an Isolation Forest algorithm to flag anomalous financial profiles (negative margins, massive cost-inefficiencies).
* **Automated Geocoding:** Translates raw CMS zip codes into geospatial coordinates on-the-fly using `pgeocode`.
* **Interactive Heatmap:** Visualizes high-risk clinics across the US using Plotly Scatter Mapbox.
* **Longitudinal Tracking:** Seamlessly hot-swap between fiscal years (2022-2025) using highly compressed Parquet data files.
* **Local Vulnerability Search:** Drill down into specific states, cities, or zip codes to find at-risk centers in your community.

## 🛠️ Tech Stack
* **Frontend:** Streamlit
* **Data Processing:** Pandas, NumPy, PyArrow
* **Machine Learning:** Scikit-Learn (`IsolationForest`, `RobustScaler`, `Pipeline`)
* **Visualization:** Plotly Express
* **Geospatial Analytics:** pgeocode

## 📊 The Methodology

**The Clinical Reality:** Patients with End-Stage Renal Disease (ESRD) rely on hemodialysis multiple times a week. When a clinic closes, creating a "Dialysis Desert," vulnerable populations are forced to travel unsustainable distances, drastically increasing mortality rates and hospitalization. 

**The Financial Math:**
This tool defines "At Risk" strictly as *financial insolvency*, which is the primary leading indicator of a physical clinic closure.
* **Operating Margin:** `(Net Patient Revenue - Operating Expenses) / Net Patient Revenue`
* **Efficiency Ratio:** `Total Dialysis Costs / Total Treatments`

**The Data Science:**
Because the dialysis market is heavily consolidated among a few mega-chains (creating a skewed "false normal"), standard standard-deviation math fails. We use an **Isolation Forest** to build random decision trees. Clinics with extreme, failing financial profiles require far fewer partitions to be "isolated" from the rest of the industry, resulting in a higher Risk Score.

## 🚀 How to Run Locally

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/dialysis-desert-tracker.git](https://github.com/YOUR_USERNAME/dialysis-desert-tracker.git)
cd dialysis-desert-tracker

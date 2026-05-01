import streamlit as st
import pandas as pd
import numpy as np
import os
import pgeocode  # Requirement: pip install pgeocode
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import plotly.express as px

# --- PAGE CONFIG ---
st.set_page_config(page_title="Dialysis Desert Risk Tracker", layout="wide")

# --- SIDEBAR: ABOUT THE AUTHOR & CONTROLS ---
with st.sidebar:
    st.title("🛡️ Policy Control Center")
    st.markdown("---")
    
    # NEW FEATURE: Year Selector
    st.subheader("🗓️ Temporal Analysis")
    selected_year = st.selectbox("Select Fiscal Year", ["2022", "2023", "2024", "2025"], index=2)
    
    st.markdown("---")
    st.markdown("""
    ### About the Author
    **Neel Agarwak** | M3 at OSU COM.  
    *neel.agarwal@osumc.edu*
    
    This tool is designed to provide early-warning signals for dialysis facility 
    closures by identifying financial outliers that deviate from industry norms.
    """)
    st.markdown("---")
    st.info(f"📊 **Source:** CMS Form 265-11 ({selected_year} Fiscal Year)")

# --- METHODOLOGY EXPLANATIONS ---
def render_methodology():
    with st.expander("📖 Methodology: The Science, Medicine, & Math"):
        st.markdown(r"""
        #### The Medicine: Why "Dialysis Deserts" Matter
        Patients with End-Stage Renal Disease (ESRD) rely on renal replacement therapy (hemodialysis) multiple times a week to manage fluid volume and clear uremic toxins. Missing even a single session drastically increases the risk of fatal complications like hyperkalemia or pulmonary edema. When a clinic closes, creating a "Dialysis Desert," vulnerable patient populations are forced to travel unsustainable distances, directly impacting mortality rates and care continuity.

        #### What "At Risk" Means
        In this context, "At Risk" refers strictly to **financial insolvency**. Facilities flagged by this tool are operating with severe margin deficits or extreme cost-inefficiencies compared to their peers. Sustained financial distress is the primary leading indicator of sudden facility closure.

        #### The Financial Math
        1. **Operating Margin:** The primary indicator of viability. 
        $$Operating\ Margin = \frac{Net\ Patient\ Revenue - Operating\ Expenses}{Net\ Patient\ Revenue}$$
        2. **Efficiency Ratio:** Measures the cost-effectiveness of treatment delivery.
        $$Efficiency = \frac{Total\ Dialysis\ Costs}{Total\ Treatments}$$
        
        #### The Data Science: Isolation Forests
        Instead of assuming financial data follows a normal distribution (bell curve), this tool uses an unsupervised Machine Learning algorithm called an **Isolation Forest**. It builds random decision trees to partition the data. Anomalous clinics (those with extreme financial profiles) require far fewer partitions to be "isolated" from the rest of the industry. The shorter the path length to isolate a clinic, the higher its resulting Risk Score.
        """)

# --- CORE LOGIC: DATA PROCESSING ---
# Adding 'year' as an argument so Streamlit caches each year separately
@st.cache_data
def load_and_process_local_data(year):
    # Dynamically select the files based on the chosen year
    f_alpha = f"RNL11_{year}_alpha.parquet"
    f_nmrc = f"RNL11_{year}_nmrc.parquet"
    f_rpt = f"RNL11_{year}_rpt.parquet"

    if not all(os.path.exists(f) for f in [f_alpha, f_nmrc, f_rpt]):
        st.error(f"Missing {year} HCRIS files in the local directory. Please ensure RNL11_{year} files are present.")
        return None

    nmrc_cols = ['rpt_rec_num', 'wksht_cd', 'line_num', 'clmn_num', 'itm_val_num']
    alpha_cols = ['rpt_rec_num', 'wksht_cd', 'line_num', 'clmn_num', 'itm_val_alphnmrc']
    rpt_cols = ['rpt_rec_num', 'prvdr_ctrl_type_cd', 'prvdr_num'] + [f'v{i}' for i in range(3, 18)]

    # Load Parquet files (Headers and dtypes are already baked in!)
    nmr = pd.read_parquet(f_nmrc)
    alpha = pd.read_parquet(f_alpha)
    rpt = pd.read_parquet(f_rpt)

    # Enforce Numeric types
    nmr['itm_val_num'] = pd.to_numeric(nmr['itm_val_num'], errors='coerce')

    for df in [nmr, alpha]:
        df['wksht_cd'] = df['wksht_cd'].str.strip()
        df['line_num'] = df['line_num'].str.strip()
        df['clmn_num'] = df['clmn_num'].str.strip()

    # --- FINANCIAL MAPPING ---
    metric_mapping = {
        'F100000_00100_00100': 'net_patient_revenue',    
        'F100000_00400_00100': 'operating_expenses',     
        'C000000_01800_00200': 'total_costs',            
        'C000000_01800_00100': 'total_treatments',       
    }

    nmr['metric_code'] = nmr['wksht_cd'] + '_' + nmr['line_num'] + '_' + nmr['clmn_num']
    nmr_filtered = nmr[nmr['metric_code'].isin(metric_mapping.keys())].copy()
    nmr_filtered['metric_name'] = nmr_filtered['metric_code'].map(metric_mapping)

    financials = nmr_filtered.pivot_table(
        index='rpt_rec_num', 
        columns='metric_name', 
        values='itm_val_num', 
        aggfunc='first'
    ).reset_index()

    expected_metrics = ['net_patient_revenue', 'operating_expenses', 'total_costs', 'total_treatments']
    for col in expected_metrics:
        if col not in financials.columns:
            financials[col] = np.nan

    # --- FINAL FIXED ALPHA MAPPING (S000002) ---
    s_wksht = alpha[alpha['wksht_cd'] == 'S000002'].copy()
    
    names = s_wksht[(s_wksht['line_num'] == '00100') & (s_wksht['clmn_num'] == '00100')].rename(columns={'itm_val_alphnmrc': 'facility_name'})[['rpt_rec_num', 'facility_name']].drop_duplicates('rpt_rec_num')
    cities = s_wksht[(s_wksht['line_num'] == '00300') & (s_wksht['clmn_num'] == '00100')].rename(columns={'itm_val_alphnmrc': 'city'})[['rpt_rec_num', 'city']].drop_duplicates('rpt_rec_num')
    states = s_wksht[(s_wksht['line_num'] == '00300') & (s_wksht['clmn_num'] == '00200')].rename(columns={'itm_val_alphnmrc': 'state_cd'})[['rpt_rec_num', 'state_cd']].drop_duplicates('rpt_rec_num')
    zips = s_wksht[(s_wksht['line_num'] == '00300') & (s_wksht['clmn_num'] == '00300')].rename(columns={'itm_val_alphnmrc': 'zip_cd'})[['rpt_rec_num', 'zip_cd']].drop_duplicates('rpt_rec_num')

    # --- SAFE MERGING ---
    df = rpt.merge(financials, on='rpt_rec_num', how='left')
    df = df.merge(names, on='rpt_rec_num', how='left')
    df = df.merge(cities, on='rpt_rec_num', how='left')
    df = df.merge(states, on='rpt_rec_num', how='left')
    df = df.merge(zips, on='rpt_rec_num', how='left')

    df['facility_name'] = df['facility_name'].fillna("Unknown Facility")

    # Feature Engineering
    df['operating_margin'] = (df['net_patient_revenue'] - df['operating_expenses']) / df['net_patient_revenue'].replace(0, np.nan)
    df['efficiency_ratio'] = df['total_costs'] / df['total_treatments'].replace(0, np.nan)

    # --- AUTOMATED GEOCODING ---
    with st.spinner(f"Geocoding {year} facilities via Zip Codes..."):
        nomi = pgeocode.Nominatim('us')
        
        # Ruthless extraction: strictly pull 5-digit zip codes
        df['zip_clean'] = df['zip_cd'].astype(str).str.extract(r'(\d{5})')[0]
        unique_zips = df['zip_clean'].dropna().unique()
        
        if len(unique_zips) > 0:
            geo_lookup = nomi.query_postal_code(unique_zips)
            geo_lookup = geo_lookup[['postal_code', 'latitude', 'longitude']].rename(columns={'latitude': 'lat', 'longitude': 'lon'})
            df = df.merge(geo_lookup, left_on='zip_clean', right_on='postal_code', how='left')
        else:
            df['lat'] = np.nan
            df['lon'] = np.nan
            
    df_clean = df.dropna(subset=['operating_margin', 'efficiency_ratio', 'total_treatments']).copy()
    
    # Clip insane outliers for realistic dashboard averages
    df_clean = df_clean[(df_clean['operating_margin'] > -5) & (df_clean['operating_margin'] < 5)]
    
    return df_clean

# --- MACHINE LEARNING PIPELINE ---
def apply_ml_scoring(df):
    features = ['operating_margin', 'efficiency_ratio', 'total_treatments']
    
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler()),
        ('model', IsolationForest(contamination=0.1, random_state=42))
    ])
    
    df['anomaly_label'] = pipeline.fit_predict(df[features])
    
    raw_scores = pipeline.named_steps['model'].decision_function(
        pipeline.named_steps['scaler'].transform(
            pipeline.named_steps['imputer'].transform(df[features])
        )
    )
    
    df['risk_score'] = (1 - (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())) * 100
    return df

# --- DASHBOARD UI ---
st.title(f"🏥 Dialysis Desert Risk Assessment ({selected_year})")
render_methodology()

df_processed = load_and_process_local_data(selected_year)

if df_processed is not None:
    if df_processed.empty:
        st.error(f"🚨 **Critical Data Missing for {selected_year}:** One or more of your financial metrics is reading '0'. CMS likely used different Worksheet/Line codes in {selected_year} than they did in 2024. Check Form 265-11 documentation for that specific year.")
    else:
        final_df = apply_ml_scoring(df_processed)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Clinics Analyzed", len(final_df))
        m2.metric("High-Risk Outliers", len(final_df[final_df['anomaly_label'] == -1]))
        m3.metric("Avg Margin", f"{final_df['operating_margin'].mean():.1%}")
        m4.metric("Avg Cost/Treatment", f"${final_df['efficiency_ratio'].mean():.2f}")

        st.subheader("Geospatial Insolvency Risk Heatmap")
        
        if not final_df['lat'].isnull().all():
            fig = px.scatter_mapbox(
                final_df, 
                lat="lat", 
                lon="lon", 
                color="risk_score",
                size="total_treatments", 
                color_continuous_scale="RdYlGn_r",
                hover_name="facility_name", 
                hover_data={
                    "operating_margin": ":.2%",
                    "efficiency_ratio": ":$.2f",
                    "risk_score": ":.1f",
                    "lat": False,
                    "lon": False
                },
                mapbox_style="carto-positron", 
                zoom=3.5,
                height=700,
                center={"lat": 39.8283, "lon": -98.5795} # Centers map on US
            )
            fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Could not generate map: Geocoding failed. Ensure Zip Codes are valid.")
            state_fig = px.box(final_df, x="state_cd", y="risk_score", color="state_cd")
            st.plotly_chart(state_fig, use_container_width=True)

        st.markdown("---")
        st.subheader("🔍 Local Search: Find At-Risk Centers")
        st.write("Filter the risk watchlist by region to identify localized vulnerabilities.")
        
        col_st, col_city, col_zip = st.columns(3)
        search_state = col_st.text_input("State (e.g., OH, CA)")
        search_city = col_city.text_input("City")
        search_zip = col_zip.text_input("Zip Code")

        # Apply Filters
        display_df = final_df.copy()
        if search_state:
            display_df = display_df[display_df['state_cd'].astype(str).str.contains(search_state, case=False, na=False)]
        if search_city:
            display_df = display_df[display_df['city'].astype(str).str.contains(search_city, case=False, na=False)]
        if search_zip:
            display_df = display_df[display_df['zip_cd'].astype(str).str.contains(search_zip, case=False, na=False)]

        st.subheader("Critical Priority Watchlist")
        watchlist = display_df[['facility_name', 'city', 'state_cd', 'zip_cd', 'operating_margin', 'efficiency_ratio', 'total_treatments', 'risk_score']]
        st.dataframe(watchlist.sort_values('risk_score', ascending=False), use_container_width=True)
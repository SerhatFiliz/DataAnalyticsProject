"""
=============================================================================
BIG DATA ANALYTICS CAPSTONE PROJECT - REAL-TIME DASHBOARD
=============================================================================
Author: Serhat Filiz (ID: B211202031)
System: Hybrid Machine Learning Pipeline (Kafka + Spark + MongoDB + Streamlit)
Version: 3.0 (Production Release)

Description:
This dashboard serves as the presentation layer for a real-time big data pipeline.
It visualizes:
1. Live Sales Forecasts (Streaming from Apache Spark)
2. Contextual Segmentation (K-Means Clustering)
3. Model Diagnostics (R2, MAE, Feature Importance)
4. Data Exploratory Analysis (EDA) on raw data.
=============================================================================
"""

import streamlit as st
import pandas as pd
import time
import numpy as np
from pymongo import MongoClient
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & LAYOUT SETTINGS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Big Data Analytics Project", 
    layout="wide", 
    page_icon="📊",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------------------------
# 2. SESSION STATE MANAGEMENT
# ---------------------------------------------------------------------------
# Initialize global variables to persist data across Streamlit re-runs.
# 'cluster_map' & 'cluster_table': Used to lock segmentation logic to prevent flickering.
# 'anomaly_log': Stores detected anomalies in memory.
if 'cluster_map' not in st.session_state:
    st.session_state['cluster_map'] = None
if 'cluster_table' not in st.session_state:
    st.session_state['cluster_table'] = None
if 'anomaly_log' not in st.session_state:
    st.session_state['anomaly_log'] = []

# ---------------------------------------------------------------------------
# 3. DATABASE CONNECTION LAYER (MongoDB)
# ---------------------------------------------------------------------------
@st.cache_resource
def init_connection():
    """
    Establishes a singleton connection to the MongoDB service.
    Service Name: 'mongodb' (defined in Kubernetes)
    Port: 27017
    """
    return MongoClient("mongodb://mongodb:27017/")

try:
    client = init_connection()
    db = client['sales_db'] # Database: sales_db
except Exception as e:
    st.error(f"Critical Database Error: {e}")
    st.stop()

# ---------------------------------------------------------------------------
# 4. DASHBOARD HEADER & METADATA
# ---------------------------------------------------------------------------
# Using a 3-column layout for a professional header.
top_c1, top_c2, top_c3 = st.columns([1, 4, 1])

with top_c1:
    # Analytics Icon (Hosted asset)
    st.image("https://cdn-icons-png.flaticon.com/512/2920/2920326.png", width=60)

with top_c2:
    st.markdown("<h1 style='text-align: center; color: #00CC96;'>Big Data Analytics Project</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Real-Time Hybrid Machine Learning Pipeline Architecture</p>", unsafe_allow_html=True)

with top_c3:
    st.info("**Student:** Serhat Filiz\n**ID:** B211202031")

st.divider()

# ---------------------------------------------------------------------------
# 5. SYSTEM ARCHITECTURE VISUALIZATION
# ---------------------------------------------------------------------------
with st.expander("📌 System Architecture & Data Flow Pipeline", expanded=False):
    st.markdown("""
    **End-to-End Data Pipeline Specification:**
    
    `📄 CSV Data Source` ➡ `🐍 Producer` ➡ `📨 Apache Kafka` ➡ `⚡ Spark Streaming (ML Engine)` ➡ `💾 MongoDB` ➡ `📊 Dashboard`
    
    * **1. Data Ingestion:** High-throughput buffering with **Apache Kafka**.
    * **2. Stream Processing:** Micro-batch processing with **Spark Structured Streaming**.
    * **3. Feature Engineering:** Date Extraction (Month, Day), One-Hot Encoding.
    * **4. Machine Learning:** Hybrid Model -> **K-Means** (Unsupervised Context) + **Random Forest** (Supervised Prediction).
    * **5. Visualization:** Real-time monitoring via **Streamlit**.
    """)

# ---------------------------------------------------------------------------
# 6. TAB CONFIGURATION
# ---------------------------------------------------------------------------
# Segregating the dashboard into logical sections.
tab_live, tab_eval, tab_eda = st.tabs([
    "⚡ Live Operations", 
    "📈 Model Diagnostics",
    "📊 Data Exploration (EDA)"
])

# ---------------------------------------------------------------------------
# 7. TAB 3: DATASET ANALYSIS (EDA) - STATIC ANALYSIS
# ---------------------------------------------------------------------------
with tab_eda:
    st.header("📊 Exploratory Data Analysis")
    st.markdown("Statistical profile of the raw input data (Sampled 25,000 Records).")
    
    # Fetch a large static batch for statistical significance
    raw_static = list(db.raw_data.find().limit(25000))
    
    if raw_static:
        df_raw = pd.DataFrame(raw_static)
        if '_id' in df_raw.columns: df_raw.drop('_id', axis=1, inplace=True)
        
        # A. High-Level Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Analyzed Records", f"{len(df_raw):,}", help="Total raw records fetched from MongoDB")
        c2.metric("Mean Sales", f"{df_raw['sales'].mean():.2f}", help="Average sales value")
        c3.metric("Zero Sales count", f"{(df_raw['sales']==0).sum()}", help="Inactive transactions filtered out during training")
        c4.metric("Unique Stores", f"{df_raw['store_nbr'].nunique()}", help="Distinct store locations")
        
        st.divider()
        
        # B. 2x2 GRAPH GRID START (ROW 1: Top 10 & Time Series)
        row1_col1, row1_col2 = st.columns(2)
        
        with row1_col1:
            st.markdown("##### 🏆 Top 10 Product Families")
            top_fam = df_raw['family'].value_counts().head(10)
            fig_bar = px.bar(top_fam, orientation='h', color_discrete_sequence=['#EF553B'])
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with row1_col2:
            st.markdown("##### 📅 Time Series Trend")
            df_raw['date'] = pd.to_datetime(df_raw['date'])
            daily_sales = df_raw.groupby('date')['sales'].sum().reset_index()
            fig_line_eda = px.line(daily_sales, x='date', y='sales', color_discrete_sequence=['#00CC96'])
            st.plotly_chart(fig_line_eda, use_container_width=True)
            
        st.divider() # ROW 1 ve ROW 2 arasını ayırır

        # C. 2x2 GRAPH GRID (ROW 2: Box Plot & Day of Week)
        row2_col1, row2_col2 = st.columns(2)

        with row2_col1:
            st.markdown("##### 📦 Outlier Detection (Box Plot)")
            # Box plots are crucial for detecting anomalies in statistical distribution
            fig_box = px.box(df_raw, x='family', y='sales', log_y=True, color_discrete_sequence=['#AB63FA'])
            fig_box.update_layout(showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)

        with row2_col2:
            st.markdown("##### 📆 Sales by Day of Week")
            df_raw['day_name'] = df_raw['date'].dt.day_name()
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_counts = df_raw.groupby('day_name')['sales'].mean().reindex(day_order).reset_index()
            fig_day = px.bar(day_counts, x='day_name', y='sales', title="Average Sales by Day", color='sales', color_continuous_scale='Viridis')
            st.plotly_chart(fig_day, use_container_width=True)
            
    else:
        st.warning("Waiting for data ingestion into MongoDB...")

# ---------------------------------------------------------------------------
# 8. PREPARE LIVE PLACEHOLDERS
# ---------------------------------------------------------------------------
# Initializing UI containers for live updates to prevent layout shifts.
with tab_live:
    # Row 1: KPI Metrics & Anomaly Alert
    l_c1, l_c2, l_c3 = st.columns([1, 1, 1])
    ph_live_metric = l_c1.empty()
    ph_live_anomaly = l_c2.empty() 
    ph_live_box = l_c3.empty()     
    
    st.write("---")
    st.markdown("#### 📉 Forecast vs Actual Stream")
    ph_live_chart = st.empty()
    
    st.write("---")
    l_c4, l_c5, l_c6 = st.columns([1.5, 1.5, 2])
    with l_c4:
        st.markdown("#### 🧩 Context Segments")
        ph_live_table = st.empty()
    with l_c5:
        st.markdown("#### 🥧 Cluster Ratio")
        ph_live_pie = st.empty()
    with l_c6:
        st.markdown("#### 📍 Store Clustering Map")
        ph_live_scatter = st.empty()
    
    st.write("---")
    st.subheader("🚨 Detected Anomalies Log (Last 5)")
    ph_anomaly_table = st.empty()

    st.write("---")
    st.subheader("📥 Incoming Raw Data Stream") 
    ph_raw_table_live = st.empty()

# ---------------------------------------------------------------------------
# 9. PREPARE MODEL DIAGNOSTICS PLACEHOLDERS
# ---------------------------------------------------------------------------
with tab_eval:
    st.header("📈 Real-Time Model Evaluation")
    st.caption("Metrics calculated on the sliding window of the last 600 predictions.")
    
    # Live Metrics
    e_c1, e_c2, e_c3 = st.columns(3)
    ph_eval_r2 = e_c1.empty()
    ph_eval_mae = e_c2.empty()
    ph_eval_rmse = e_c3.empty()
    
    st.divider()
    
    # Feature Impact & Fit
    col_feat, col_dist = st.columns(2)
    with col_feat:
        st.markdown("### 🧠 Feature Impact Analysis")
        st.caption("Correlation of features (Date, Store, Promo, Cluster) with Actual Sales.")
        ph_feat_imp = st.empty()
        
    with col_dist:
        st.markdown("### 🎯 Prediction Fit")
        st.caption("Actual vs Predicted Scatter Plot (Ideal: Diagonal Line).")
        ph_pred_fit = st.empty()

    st.divider()
    
    # Drift Analysis
    st.markdown("### 📉 Error Trend Over Time")
    st.caption("Monitoring Model Drift: Rolling mean of Absolute Error")
    ph_error_trend = st.empty()

# ---------------------------------------------------------------------------
# 10. BUSINESS LOGIC: SMART CLUSTERING
# ---------------------------------------------------------------------------
def calculate_smart_stats(df_in):
    """
    Calculates cluster statistics and ensures ranges do not visually overlap.
    Returns: Mapping dictionary, Stats DataFrame.
    """
    valid = df_in[df_in['predicted_sales'] > 0]
    
    # Stability Check: Wait for at least 3 clusters
    if len(valid) < 50 or valid['cluster_id'].nunique() < 3: 
        return None
    
    # Group by Cluster ID
    stats = valid.groupby('cluster_id')['predicted_sales'].agg(
        mean='mean', count='count',
        p10=lambda x: x.quantile(0.10),
        p90=lambda x: x.quantile(0.90)
    ).reset_index().sort_values('mean')
    
    labels = ["Low Volume", "Standard Volume", "High Volume"]
    mapping = {}
    
    for idx, row in enumerate(stats.itertuples()):
        label = labels[idx] if idx < len(labels) else f"Cluster {row.cluster_id}"
        mapping[row.cluster_id] = label
        
    stats['Segment'] = stats['cluster_id'].map(mapping)
    stats['Avg Sales'] = stats['mean'].apply(lambda x: f"{x:.0f}")
    
    # Smart Range Logic (No Overlap)
    ranges = []
    prev_limit = 0
    for i, row in enumerate(stats.itertuples()):
        if i == 0: 
            start = row.p10
        else: 
            start = max(row.p10, prev_limit + 1)
        
        if i == len(stats) - 1: 
            ranges.append(f"{start:.0f}+")
        else:
            end = max(row.p90, start + 10)
            ranges.append(f"{start:.0f} - {end:.0f}")
            prev_limit = row.mean
            
    stats['Volume Range'] = ranges
    return mapping, stats[['Segment', 'Avg Sales', 'Volume Range']]

# ---------------------------------------------------------------------------
# 11. MAIN EXECUTION LOOP (REAL-TIME STREAM)
# ---------------------------------------------------------------------------
while True:
    try:
        # Fetching latest 600 predictions for live visualization
        preds = list(db.predictions.find().sort('_id', -1).limit(600))
        
        if preds:
            df = pd.DataFrame(preds)
            df_disp = df.iloc[::-1].reset_index(drop=True)
            df_disp['seq'] = df_disp.index
            
            # --- CRITICAL: FORCE NUMERIC TYPES ---
            # Ensures all columns used in correlation/metrics are numeric.
            # Fixes "Blank Chart" issues.
            numeric_cols = ['sales', 'actual_sales', 'predicted_sales', 'onpromotion', 'store_nbr', 'cluster_id', 'month', 'day_of_week']
            for col in numeric_cols:
                if col in df_disp.columns:
                    df_disp[col] = pd.to_numeric(df_disp[col], errors='coerce').fillna(0)
            
            df_disp['diff'] = df_disp['predicted_sales'] - df_disp['actual_sales']
            
            # --- CLUSTER LOCK ---
            if st.session_state['cluster_map'] is None:
                res = calculate_smart_stats(df)
                if res:
                    st.session_state['cluster_map'] = res[0]
                    st.session_state['cluster_table'] = res[1]
            
            # --- TAB 1: LIVE UPDATES ---
            if st.session_state['cluster_table'] is not None:
                # Update Segment Table
                ph_live_table.dataframe(st.session_state['cluster_table'], hide_index=True, use_container_width=True)
                
                fixed_map = st.session_state['cluster_map']
                df_disp['cluster_name'] = df_disp['cluster_id'].map(fixed_map)
                
                # Custom Color Palette
                colors = {"Low Volume": "#3366CC", "Standard Volume": "#FFD700", "High Volume": "#FF4B4B"}

                if 'cluster_name' in df_disp.columns:
                    # Update Pie Chart
                    pie_data = df_disp['cluster_name'].value_counts().reset_index()
                    pie_data.columns = ['Segment', 'Count']
                    fig_pie = px.pie(pie_data, values='Count', names='Segment', 
                                     color='Segment', color_discrete_map=colors, hole=0.4)
                    fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=200, showlegend=False)
                    ph_live_pie.plotly_chart(fig_pie, use_container_width=True, key=f"pie_{time.time()}")
                    
                    # Update Scatter Map
                    fig_scat = px.scatter(df_disp, x='store_nbr', y='predicted_sales', color='cluster_name',
                                          labels={'store_nbr': 'Store ID', 'predicted_sales': 'Predicted'},
                                          color_discrete_map=colors)
                    fig_scat.update_layout(margin=dict(t=20, b=0, l=0, r=0), height=250, showlegend=False)
                    ph_live_scatter.plotly_chart(fig_scat, use_container_width=True, key=f"scat_{time.time()}")

            # Get Latest Transaction
            last = df_disp.iloc[-1]
            
            # Forecast Metric Card
            ph_live_metric.metric(f"Forecast ({last['family']})", f"{last['predicted_sales']:.0f}", 
                                  f"{last['predicted_sales'] - last['actual_sales']:.0f} Diff", delta_color="inverse")
            
            # --- ANOMALY DETECTION LOGIC ---
            # Threshold: Deviation > 500 units
            is_anomaly = abs(last['diff']) > 500
            if is_anomaly:
                ph_live_anomaly.error(f"🚨 **Anomaly Detected!**\n\nDeviation: {last['diff']:.0f} units")
                
                # Record Anomaly
                anomaly_record = {
                    "Time": time.strftime("%H:%M:%S"),
                    "Trans. ID": str(last.get('id', 'N/A')),
                    "Store ID": str(last.get('store_nbr', 'N/A')),
                    "Family": last['family'],
                    "Pred": f"{last['predicted_sales']:.0f}",
                    "Act": f"{last['actual_sales']:.0f}",
                    "Diff": f"{last['diff']:.0f}"
                }
                # Add to log (Prevent Duplicates)
                if not st.session_state['anomaly_log'] or st.session_state['anomaly_log'][0]['Trans. ID'] != str(last.get('id')):
                    st.session_state['anomaly_log'].insert(0, anomaly_record)
                    st.session_state['anomaly_log'] = st.session_state['anomaly_log'][:5] # Keep last 5
            else:
                ph_live_anomaly.success(f"✅ **System Normal**\n\nStable operation")

            # Update Anomaly Table
            if st.session_state['anomaly_log']:
                ph_anomaly_table.dataframe(pd.DataFrame(st.session_state['anomaly_log']), use_container_width=True)
            else:
                ph_anomaly_table.info("No anomalies detected in this session yet.")

            # Update Raw Data Table (Live)
            raw_d = list(db.raw_data.find().sort('_id', -1).limit(5))
            if raw_d:
                r_df = pd.DataFrame(raw_d)
                if '_id' in r_df.columns: r_df.drop('_id', axis=1, inplace=True)
                ph_raw_table_live.dataframe(r_df, hide_index=True, use_container_width=True)

            # Context Box (Explaining 0 sales)
            act_txt = f"{last['actual_sales']:.0f}"
            if last['actual_sales'] == 0: act_txt += " (No Sales)"
            seg_txt = last.get('cluster_name', 'Unknown')
            ph_live_box.info(f"**Context Profile:** {seg_txt}\n\nPred: **{last['predicted_sales']:.0f}** | Act: **{act_txt}**")
            
            # Forecast Chart (Line Plot)
            fig_line = px.line(df_disp.tail(100), x='seq', y='predicted_sales', 
                               color_discrete_sequence=["#00CC96"],
                               hover_data=['actual_sales', 'diff', 'family', 'store_nbr', 'cluster_name'])
            # Add Actual Sales as Reference Line
            fig_line.add_scatter(x=df_disp.tail(100)['seq'], y=df_disp.tail(100)['actual_sales'], 
                                 mode='lines', name='Actual', line=dict(color='white', width=1, dash='dot'), opacity=0.5)
            fig_line.update_layout(showlegend=True, margin=dict(t=10, b=10, l=10, r=10), legend=dict(y=1, x=0))
            ph_live_chart.plotly_chart(fig_line, use_container_width=True, key=f"main_{time.time()}")

            # --- TAB 2: MODEL DIAGNOSTICS (UPDATES) ---
            if len(df_disp) > 10:
                y_true = df_disp['actual_sales']
                y_pred = df_disp['predicted_sales']
                
                # Metrics
                live_r2 = r2_score(y_true, y_pred)
                live_mae = mean_absolute_error(y_true, y_pred)
                live_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                
                ph_eval_r2.metric("Live R² (Accuracy)", f"{live_r2:.2f}", help="1.0 is perfect. <0 is poor.")
                ph_eval_mae.metric("Live MAE (Error)", f"{live_mae:.1f}", help="Average absolute error.")
                ph_eval_rmse.metric("Live RMSE (Penalty)", f"{live_rmse:.1f}", help="Penalizes large errors.")
                
                # 1. Feature Impact Analysis (With All Features)
                # Prepare data for correlation
                corr_prep = df_disp[['actual_sales', 'onpromotion', 'store_nbr', 'cluster_id', 'month', 'day_of_week', 'family']].copy()
                
                # Encode Family (Text -> Code)
                corr_prep['family_code'] = corr_prep['family'].astype('category').cat.codes
                corr_prep = corr_prep.drop(columns=['family'])
                
                # Ensure Numeric
                for col in corr_prep.columns:
                    corr_prep[col] = pd.to_numeric(corr_prep[col], errors='coerce').fillna(0)

                # Calculate Correlation (If variance exists)
                if corr_prep.std().min() > 0:
                    corr_series = corr_prep.corr()['actual_sales'].drop('actual_sales').sort_values()
                    
                    fig_imp = px.bar(
                        x=corr_series.values, y=corr_series.index, orientation='h',
                        labels={'x': 'Impact (Correlation)', 'y': 'Feature'},
                        color=corr_series.values, color_continuous_scale='RdBu', range_color=[-1, 1]
                    )
                    fig_imp.update_layout(height=300, margin=dict(t=20))
                    ph_feat_imp.plotly_chart(fig_imp, use_container_width=True, key=f"feat_{time.time()}")
                else:
                    ph_feat_imp.info("Gathering data variance...")

                # 2. Prediction Fit
                fig_fit = px.scatter(df_disp, x='actual_sales', y='predicted_sales', opacity=0.5,
                                     labels={'actual_sales': 'Actual', 'predicted_sales': 'Predicted'},
                                     color_discrete_sequence=['#AB63FA'])
                fig_fit.add_shape(type="line", line=dict(dash='dash', color='red'), x0=0, y0=0, x1=max(y_true), y1=max(y_true))
                fig_fit.update_layout(height=300, margin=dict(t=20))
                ph_pred_fit.plotly_chart(fig_fit, use_container_width=True, key=f"fit_{time.time()}")

                # 3. Error Trend
                df_disp['abs_error'] = df_disp['diff'].abs()
                df_disp['rolling_mae'] = df_disp['abs_error'].rolling(window=20).mean().fillna(0)
                fig_trend = px.line(df_disp.tail(100), x='seq', y='rolling_mae', title="Rolling MAE (Error Trend)",
                                    color_discrete_sequence=['#FFA15A'])
                fig_trend.update_layout(height=250, margin=dict(t=30))
                ph_error_trend.plotly_chart(fig_trend, use_container_width=True, key=f"trend_{time.time()}")

        else:
            ph_live_table.warning("Waiting for data stream...")
            
    except Exception as e:
        print(f"Error: {e}")
        
    time.sleep(1.5)
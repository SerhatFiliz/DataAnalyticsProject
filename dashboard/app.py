import streamlit as st
import pandas as pd
import time
import numpy as np
from pymongo import MongoClient
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. CONFIGURATION
st.set_page_config(page_title="Sales Analytics Dashboard", layout="wide", page_icon="📈")

# 2. DB CONNECTION
@st.cache_resource
def init_connection():
    return MongoClient("mongodb://mongodb:27017/")

try:
    client = init_connection()
    db = client['sales_db']
except Exception as e:
    st.error(f"DB Error: {e}")
    st.stop()

# ==========================================
# SECTION 1: PROJECT HEADER & INFO
# ==========================================
st.title("🎓 Big Data Analytics Capstone Project")

# Student Info Box
st.info("**Student:** Serhat Filiz | **ID:** B211202031")

with st.expander("ℹ️ Project Architecture & Pipeline Details", expanded=False):
    st.markdown("""
    ### 🏗️ System Architecture
    This project implements a **Real-Time Hybrid Machine Learning Pipeline** for sales forecasting.
    
    **🚀 Data Pipeline:**
    1.  **Ingestion:** `Producer` reads CSV data, shuffles it, and streams to **Apache Kafka**.
    2.  **Processing:** **Apache Spark (Structured Streaming)** consumes data from Kafka.
    3.  **Hybrid ML Model:** * *Step 1 (Unsupervised):* **K-Means Clustering** segments stores based on volume context (Store + Family + Promotion).
        * *Step 2 (Supervised):* **Random Forest Regressor** predicts sales using the cluster ID as a key feature.
    4.  **Storage:** Results are stored in **MongoDB** (NoSQL).
    5.  **Visualization:** **Streamlit** dashboard monitors the pipeline in real-time.
    
    **🛠️ Technologies:** Kubernetes, Docker, Python, PySpark, Kafka, Zookeeper, MongoDB.
    """)

st.divider()

# ==========================================
# SECTION 2: OPERATIONAL DASHBOARD
# ==========================================
st.subheader("📊 Live Operations")

col_metrics, col_charts = st.columns([1, 3])

with col_metrics:
    st.markdown("### 📡 Live Status")
    status_ph = st.empty()
    
    st.write("---")
    # --- BLUE BOX AREA (ANALYSIS PANEL) ---
    st.markdown("### 📝 Analysis Result")
    metric_ph = st.empty()
    analysis_box_ph = st.empty()

    st.write("---")
    st.markdown("### 🧩 Cluster Profiles (Locked)")
    cluster_info_ph = st.empty() 

with col_charts:
    tab1, tab2 = st.tabs(["📈 Forecast Stream", "🧩 Cluster Analysis"])
    with tab1: line_chart_ph = st.empty() 
    with tab2: scatter_chart_ph = st.empty() 

# ==========================================
# SECTION 3: RAW DATA & PERFORMANCE
# ==========================================
st.write("---")
col_raw, col_perf = st.columns([1, 1])

# Placeholder for Raw Data (Will be at the bottom as requested)
st.subheader("📥 Raw Data Stream")
raw_data_ph = st.empty()

st.write("---")
st.subheader("🧠 Model Performance & Health Metrics")
st.markdown("*Evaluated on the last 500 records streaming in real-time.*")

# Performance Containers
perf_kpi_col1, perf_kpi_col2, perf_kpi_col3 = st.columns(3)
perf_chart_col1, perf_chart_col2 = st.columns(2)

kpi_mae_ph = perf_kpi_col1.empty()
kpi_rmse_ph = perf_kpi_col2.empty()
kpi_r2_ph = perf_kpi_col3.empty()

chart_resid_ph = perf_chart_col1.empty()
chart_hist_ph = perf_chart_col2.empty()

# --- HELPER FUNCTIONS ---
def try_define_clusters(df_hist):
    """
    Analyzes historical data to define stable clusters using Quantiles.
    """
    valid_df = df_hist[df_hist['predicted_sales'] > 0]
    unique_clusters = valid_df['cluster_id'].unique()
    
    if len(valid_df) < 50 or len(unique_clusters) < 2:
        return False
    
    # Stats with Quantiles to remove outliers from range display
    stats = valid_df.groupby('cluster_id')['predicted_sales'].agg(
        mean='mean',
        p10=lambda x: x.quantile(0.10),
        p90=lambda x: x.quantile(0.90)
    ).reset_index()
    
    stats = stats.sort_values('mean') 
    
    labels = ["Low Volume", "Standard Volume", "High Volume"]
    mapping = {}
    
    for idx, row in enumerate(stats.itertuples()):
        label = labels[idx] if idx < len(labels) else f"Cluster {row.cluster_id}"
        mapping[row.cluster_id] = label
        
    stats['Segment Name'] = stats['cluster_id'].map(mapping)
    stats['Average Sales'] = stats['mean'].apply(lambda x: f"{x:.0f}")
    
    def format_range(row):
        if row.name == stats.index[-1]: 
             return f"{row['p10']:.0f}+"
        return f"{row['p10']:.0f} - {row['p90']:.0f}"

    stats['Range (10%-90%)'] = stats.apply(format_range, axis=1)
    
    st.session_state['cluster_map'] = mapping
    st.session_state['cluster_table'] = stats[['Segment Name', 'Average Sales', 'Range (10%-90%)']]
    return True

# --- INITIALIZE SESSION STATE ---
if 'cluster_map' not in st.session_state:
    st.session_state['cluster_map'] = None
if 'cluster_table' not in st.session_state:
    st.session_state['cluster_table'] = None

# ==========================================
# MAIN LOOP
# ==========================================
while True:
    try:
        # Fetch Data (More history for better metrics)
        predictions = list(db.predictions.find().sort('_id', -1).limit(500))
        raw_data = list(db.raw_data.find().sort('_id', -1).limit(5))

        if predictions:
            df = pd.DataFrame(predictions)
            df_display = df.iloc[::-1].copy() # Reverse for time series
            df_display = df_display.reset_index(drop=True)
            df_display['sequence'] = df_display.index
            df_display['diff'] = df_display['predicted_sales'] - df_display['actual_sales']

            # --- 1. CLUSTER LOCK CHECK ---
            if st.session_state['cluster_map'] is None:
                is_ready = try_define_clusters(df)
                if not is_ready:
                    status_ph.warning(f"⏳ Calibrating Segments... ({len(df)} records)")
                    df_display['cluster_name'] = "Analyzing..."
                else:
                    status_ph.success("✅ Clusters Locked & System Active")
            else:
                status_ph.success("✅ System Online")
            
            fixed_map = st.session_state['cluster_map']
            if fixed_map:
                with cluster_info_ph.container():
                    st.dataframe(st.session_state['cluster_table'], hide_index=True, use_container_width=True)
                df_display['cluster_name'] = df_display['cluster_id'].map(fixed_map)

            # --- 2. OPERATIONAL METRICS ---
            last_row = df_display.iloc[-1]
            pred_val = last_row['predicted_sales']
            act_val = last_row['actual_sales']
            diff_val = pred_val - act_val
            seg_name = last_row.get('cluster_name', 'Unknown')
            family_name = last_row['family']

            metric_ph.metric(
                label=f"Forecast ({family_name})",
                value=f"{pred_val:.0f} Units",
                delta=f"{diff_val:+.0f} Diff",
                delta_color="inverse"
            )

            analysis_box_ph.info(
                f"**Analysis Result:** Transaction belongs to **{seg_name}**.\n\n"
                f"Predicted: **{pred_val:.0f}** | Actual: **{act_val:.0f}**"
            )

            # --- 3. CHARTS ---
            # Line Chart
            fig_line = px.line(
                df_display, x='sequence', y='predicted_sales', 
                title='Real-Time Forecast Stream (Predicted Only)',
                labels={'sequence': 'Sequence', 'predicted_sales': 'Forecast'},
                color_discrete_sequence=["#00CC96"],
                hover_data=['actual_sales', 'diff', 'cluster_name', 'family']
            )
            fig_line.update_traces(hovertemplate="<b>Seq:</b> %{x}<br><b>Pred:</b> %{y:.0f}<br><b>Act:</b> %{customdata[0]:.0f}<br><b>Diff:</b> %{customdata[1]:.0f}<br><b>Seg:</b> %{customdata[2]}")
            line_chart_ph.plotly_chart(fig_line, use_container_width=True, key=f"line_{time.time()}")

            # Scatter Plot
            if 'cluster_name' in df_display.columns:
                fig_scatter = px.scatter(
                    df_display, x='store_nbr', y='predicted_sales',
                    color='cluster_name',
                    title='Cluster Analysis (Context-Based)',
                    category_orders={"cluster_name": ["Low Volume", "Standard Volume", "High Volume"]}
                )
                scatter_chart_ph.plotly_chart(fig_scatter, use_container_width=True, key=f"scatter_{time.time()}")

            # --- 4. MODEL PERFORMANCE METRICS (NEW SECTION) ---
            if len(df_display) > 2:
                # Calculations
                y_true = df_display['actual_sales']
                y_pred = df_display['predicted_sales']
                
                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                r2 = r2_score(y_true, y_pred)

                # KPI Cards
                kpi_mae_ph.metric("MAE (Avg Error)", f"{mae:.1f}", help="Lower is better")
                kpi_rmse_ph.metric("RMSE (Penalty for Large Errors)", f"{rmse:.1f}", help="Lower is better")
                kpi_r2_ph.metric("R² Score (Accuracy)", f"{r2:.2f}", help="1.0 is perfect, 0.0 is poor")

                # Residual Plot (Error Distribution)
                # Ideally, errors should be randomly distributed around 0
                fig_resid = px.scatter(
                    df_display, x='predicted_sales', y='diff',
                    title='Residual Plot (Prediction vs Error)',
                    labels={'predicted_sales': 'Predicted Value', 'diff': 'Error (Diff)'},
                    opacity=0.6
                )
                fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
                chart_resid_ph.plotly_chart(fig_resid, use_container_width=True, key=f"resid_{time.time()}")

                # Distribution Histogram (Actual vs Predicted)
                # Shows if the model learned the "shape" of the data
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(x=y_true, name='Actual', opacity=0.5, marker_color='#EF553B'))
                fig_hist.add_trace(go.Histogram(x=y_pred, name='Predicted', opacity=0.5, marker_color='#00CC96'))
                fig_hist.update_layout(
                    title='Distribution: Actual vs Predicted',
                    barmode='overlay',
                    xaxis_title='Sales Volume',
                    yaxis_title='Count'
                )
                chart_hist_ph.plotly_chart(fig_hist, use_container_width=True, key=f"hist_{time.time()}")

        else:
            status_ph.info("⏳ Waiting for data stream...")

        # --- 5. RAW DATA TABLE ---
        if raw_data:
            raw_df = pd.DataFrame(raw_data)
            if '_id' in raw_df.columns: raw_df = raw_df.drop('_id', axis=1)
            raw_data_ph.dataframe(raw_df, hide_index=True, use_container_width=True)
        
    except Exception as e:
        status_ph.error(f"Error: {e}")

    time.sleep(2)
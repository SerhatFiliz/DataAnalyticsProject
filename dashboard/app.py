import streamlit as st
import pandas as pd
import time
from pymongo import MongoClient
import plotly.express as px

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

# 3. HEADER
st.title("📊 Hybrid Sales Analytics System")
st.markdown("Visualizing **Clustering (Segmentation)** and **Regression (Forecast)** pipeline.")
st.divider()

# 4. LAYOUT
col_metrics, col_charts = st.columns([1, 3])

with col_metrics:
    st.subheader("📡 Live Status")
    status_ph = st.empty()
    
    st.write("---")
    # --- BLUE BOX AREA (ANALYSIS PANEL) ---
    st.subheader("📝 Analysis Result")
    metric_ph = st.empty()
    analysis_box_ph = st.empty()

    st.write("---")
    st.subheader("🧩 Cluster Profiles (Locked)")
    cluster_info_ph = st.empty() 

with col_charts:
    tab1, tab2 = st.tabs(["📈 Forecast Stream", "🧩 Cluster Analysis"])
    with tab1: line_chart_ph = st.empty() 
    with tab2: scatter_chart_ph = st.empty() 

st.subheader("📥 Raw Data Stream")
raw_data_ph = st.empty()

# --- CLUSTER LOCKING ---
if 'cluster_map' not in st.session_state:
    st.session_state['cluster_map'] = None
if 'cluster_table' not in st.session_state:
    st.session_state['cluster_table'] = None

def try_define_clusters(df_hist):
    """
    Analyzes historical data to define stable clusters.
    Uses Quantiles (10th - 90th percentile) to avoid outliers in range display.
    """
    valid_df = df_hist[df_hist['predicted_sales'] > 0]
    unique_clusters = valid_df['cluster_id'].unique()
    
    if len(valid_df) < 50 or len(unique_clusters) < 2:
        return False
    
    # 1. Calculate Stats (Mean for sorting, Quantiles for display)
    stats = valid_df.groupby('cluster_id')['predicted_sales'].agg(
        mean='mean',
        p10=lambda x: x.quantile(0.10), # 10th percentile (Alt sınır)
        p90=lambda x: x.quantile(0.90)  # 90th percentile (Üst sınır)
    ).reset_index()
    
    stats = stats.sort_values('mean') # Sort Low -> High
    
    # 2. Assign Labels
    labels = ["Low Volume", "Standard Volume", "High Volume"]
    mapping = {}
    
    for idx, row in enumerate(stats.itertuples()):
        label = labels[idx] if idx < len(labels) else f"Cluster {row.cluster_id}"
        mapping[row.cluster_id] = label
        
    # 3. Format Table
    stats['Segment Name'] = stats['cluster_id'].map(mapping)
    stats['Average Sales'] = stats['mean'].apply(lambda x: f"{x:.0f}")
    
    # Custom Range Formatting with Quantiles
    def format_range(row):
        # If it's the High Volume cluster, show as "X+"
        if row.name == stats.index[-1]: 
             return f"{row['p10']:.0f}+"
        return f"{row['p10']:.0f} - {row['p90']:.0f}"

    stats['Range (10%-90%)'] = stats.apply(format_range, axis=1)
    
    # LOCK
    st.session_state['cluster_map'] = mapping
    st.session_state['cluster_table'] = stats[['Segment Name', 'Average Sales', 'Range (10%-90%)']]
    return True

# 5. MAIN LOOP
while True:
    try:
        # Fetch Data
        predictions = list(db.predictions.find().sort('_id', -1).limit(500))
        raw_data = list(db.raw_data.find().sort('_id', -1).limit(5))

        if predictions:
            df = pd.DataFrame(predictions)
            df_display = df.iloc[::-1].copy()
            
            # Sequence Column
            df_display = df_display.reset_index(drop=True)
            df_display['sequence'] = df_display.index
            
            # Diff Calculation
            df_display['diff'] = df_display['predicted_sales'] - df_display['actual_sales']

            # --- LOCK CHECK ---
            if st.session_state['cluster_map'] is None:
                is_ready = try_define_clusters(df)
                if not is_ready:
                    status_ph.warning(f"⏳ Calibrating... ({len(df)} records)")
                    df_display['cluster_name'] = "Analyzing..."
                else:
                    status_ph.success("✅ Clusters Locked & System Active")
            else:
                status_ph.success("✅ System Online")
            
            # Use Locked Definitions
            fixed_map = st.session_state['cluster_map']
            
            if fixed_map:
                with cluster_info_ph.container():
                    st.dataframe(st.session_state['cluster_table'], hide_index=True, use_container_width=True)
                
                df_display['cluster_name'] = df_display['cluster_id'].map(fixed_map)

            # --- METRICS ---
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

            # --- VISUALS ---
            fig_line = px.line(
                df_display, 
                x='sequence', 
                y='predicted_sales', 
                title='Real-Time Forecast Stream (Predicted Only)',
                labels={'sequence': 'Sequence', 'predicted_sales': 'Forecast'},
                color_discrete_sequence=["#00CC96"],
                hover_data=['actual_sales', 'diff', 'cluster_name', 'family']
            )
            
            fig_line.update_traces(
                hovertemplate="<br>".join([
                    "<b>Seq:</b> %{x}",
                    "<b>Forecast:</b> %{y:.0f}",
                    "<b>Actual:</b> %{customdata[0]:.0f}",
                    "<b>Diff:</b> %{customdata[1]:.0f}",
                    "<b>Segment:</b> %{customdata[2]}",
                    "<b>Family:</b> %{customdata[3]}"
                ])
            )
            line_chart_ph.plotly_chart(fig_line, use_container_width=True, key=f"line_{time.time()}")

            if 'cluster_name' in df_display.columns:
                fig_scatter = px.scatter(
                    df_display, x='store_nbr', y='predicted_sales',
                    color='cluster_name',
                    title='Cluster Analysis (Context-Based)',
                    category_orders={"cluster_name": ["Low Volume", "Standard Volume", "High Volume"]}
                )
                scatter_chart_ph.plotly_chart(fig_scatter, use_container_width=True, key=f"scatter_{time.time()}")
            
        else:
            status_ph.info("⏳ Waiting for data stream...")

        if raw_data:
            raw_df = pd.DataFrame(raw_data)
            if '_id' in raw_df.columns: raw_df = raw_df.drop('_id', axis=1)
            raw_data_ph.dataframe(raw_df, hide_index=True, use_container_width=True)
        
    except Exception as e:
        status_ph.error(f"Error: {e}")

    time.sleep(2)
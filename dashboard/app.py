import streamlit as st
import pandas as pd
import time
import numpy as np
from pymongo import MongoClient
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. PAGE CONFIGURATION (Wide Layout)
st.set_page_config(page_title="Big Data Analytics Capstone", layout="wide", page_icon="🎓")

# 2. SESSION STATE INITIALIZATION
if 'cluster_map' not in st.session_state: st.session_state['cluster_map'] = None
if 'cluster_table' not in st.session_state: st.session_state['cluster_table'] = None

# 3. DB CONNECTION
@st.cache_resource
def init_connection():
    return MongoClient("mongodb://mongodb:27017/")

try:
    client = init_connection()
    db = client['sales_db']
except Exception as e:
    st.error(f"Database Connection Error: {e}")
    st.stop()

# ==========================================
# NAVBAR & HEADER (SIDEBAR YOK, TOPBAR VAR)
# ==========================================
# Üst kısmı 3 kolona böldük: Logo/Başlık - Açıklama - Öğrenci Bilgisi
top_c1, top_c2, top_c3 = st.columns([1.5, 4, 1.5])

with top_c1:
    st.markdown("### 🚀 Analytics Suite")
    st.caption("v2.0 | Real-Time Pipeline")

with top_c2:
    st.markdown("<h1 style='text-align: center; color: #00CC96;'>Big Data Analytics Capstone</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Hybrid Machine Learning Pipeline (Kafka + Spark + MongoDB)</p>", unsafe_allow_html=True)

with top_c3:
    st.info("**Student:** Serhat Filiz\n**ID:** B211202031")

st.divider()

# ==========================================
# TABS
# ==========================================
tab_live, tab_eda, tab_model = st.tabs([
    "⚡ Live Operations Center", 
    "📊 Data Exploration (EDA)",
    "🧠 Model Diagnostics"
])

# ==========================================
# TAB 2: EDA (STATIC)
# ==========================================
with tab_eda:
    st.header("Exploratory Data Analysis")
    raw_static = list(db.raw_data.find().limit(20000))
    if raw_static:
        df_raw = pd.DataFrame(raw_static)
        if '_id' in df_raw.columns: df_raw.drop('_id', axis=1, inplace=True)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Dataset Size", f"{len(df_raw):,}")
        c2.metric("Zero Sales", f"{(df_raw['sales']==0).sum()}")
        c3.metric("Product Families", f"{df_raw['family'].nunique()}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(px.histogram(df_raw, x="sales", log_y=True, title="Sales Distribution (Log)", color_discrete_sequence=['#636EFA']), use_container_width=True)
        with col2:
            st.plotly_chart(px.bar(df_raw['family'].value_counts().head(10), orientation='h', title="Top Products", color_discrete_sequence=['#EF553B']), use_container_width=True)
    else:
        st.warning("Waiting for data...")

# ==========================================
# TAB 3: MODEL CONFIG (STATIC TEXT)
# ==========================================
with tab_model:
    st.header("Model Configuration")
    c1, c2, c3 = st.columns(3)
    c1.info("**K-Means**\n\nK=3\nFeatures: Context")
    c2.info("**Random Forest**\n\nTrees=20\nDepth=5")
    c3.info("**Streaming**\n\nTrigger=0s\nMode=Append")

# ==========================================
# TAB 1: LIVE MONITOR (MAIN LOOP)
# ==========================================
with tab_live:
    # --- LAYOUT GRID ---
    # Row 1: Metrics & Analysis Box (Grafik Üstüne Taşındı)
    row1_c1, row1_c2, row1_c3, row1_c4, row1_c5 = st.columns(5)
    
    # Placeholders for Live Metrics
    ph_metric_forecast = row1_c1.empty()
    ph_box_analysis = row1_c2.empty() # Mavi Kutu artık burada
    
    # Dynamic Performance Metrics (Canlı değişecekler)
    ph_metric_r2 = row1_c3.empty()
    ph_metric_mae = row1_c4.empty()
    ph_metric_rmse = row1_c5.empty()

    st.write("---")

    # Row 2: Main Line Chart
    ph_chart_line = st.empty()

    st.write("---")

    # Row 3: Cluster Analysis (Table + Pie + Scatter)
    row3_c1, row3_c2 = st.columns([1, 2])
    
    with row3_c1:
        st.markdown("#### 🧩 Segmentation Engine")
        ph_table = st.empty()
        st.markdown("#### 🥧 Volume Distribution")
        ph_pie = st.empty()
        
    with row3_c2:
        st.markdown("#### 📍 Cluster Scatter Map")
        ph_scatter = st.empty() # İstenilen noktalı grafik geri geldi

    st.subheader("📥 Incoming Stream Data")
    ph_raw_table = st.empty()

    # --- CLUSTER LOGIC ---
    def calculate_smart_stats(df_in):
        valid = df_in[df_in['predicted_sales'] > 0]
        if len(valid) < 50 or valid['cluster_id'].nunique() < 2: return None
        
        stats = valid.groupby('cluster_id')['predicted_sales'].agg(
            mean='mean', count='count',
            p10=lambda x: x.quantile(0.10),
            p90=lambda x: x.quantile(0.90)
        ).reset_index().sort_values('mean')
        
        labels = ["Low Volume", "Standard Volume", "High Volume"]
        mapping = {}
        for idx, row in enumerate(stats.itertuples()):
            label = labels[idx] if idx < len(labels) else f"C{row.cluster_id}"
            mapping[row.cluster_id] = label
            
        stats['Segment'] = stats['cluster_id'].map(mapping)
        stats['Avg Sales'] = stats['mean'].apply(lambda x: f"{x:.0f}")
        
        ranges = []
        prev_limit = 0
        for i, row in enumerate(stats.itertuples()):
            if i == 0: start = row.p10
            else: start = max(row.p10, prev_limit + 1)
            
            if i == len(stats) - 1: ranges.append(f"{start:.0f}+")
            else:
                end = max(row.p90, start + 10)
                ranges.append(f"{start:.0f} - {end:.0f}")
                prev_limit = row.mean
                
        stats['Volume Range'] = ranges
        return mapping, stats[['Segment', 'Avg Sales', 'Volume Range']]

    # --- MAIN LOOP ---
    while True:
        try:
            # 600 satır veri çekiyoruz (Hem grafik hem metrik hesaplamak için yeterli)
            preds = list(db.predictions.find().sort('_id', -1).limit(600))
            raw_d = list(db.raw_data.find().sort('_id', -1).limit(5))
            
            if preds:
                df = pd.DataFrame(preds)
                df_disp = df.iloc[::-1].reset_index(drop=True)
                df_disp['seq'] = df_disp.index
                df_disp['diff'] = df_disp['predicted_sales'] - df_disp['actual_sales']
                
                # 1. CLUSTER CALCULATION (If not locked)
                if st.session_state['cluster_map'] is None:
                    res = calculate_smart_stats(df)
                    if res:
                        st.session_state['cluster_map'] = res[0]
                        st.session_state['cluster_table'] = res[1]
                
                # Apply Lock
                fixed_map = st.session_state['cluster_map']
                if fixed_map:
                    df_disp['cluster_name'] = df_disp['cluster_id'].map(fixed_map)
                    
                    # UPDATE TABLE
                    if st.session_state['cluster_table'] is not None:
                        ph_table.dataframe(st.session_state['cluster_table'], hide_index=True, use_container_width=True)
                    
                    # UPDATE PIE CHART (Live)
                    pie_data = df_disp['cluster_name'].value_counts().reset_index()
                    pie_data.columns = ['Segment', 'Count']
                    fig_pie = px.pie(pie_data, values='Count', names='Segment', 
                                     color='Segment',
                                     color_discrete_map={"Low Volume": "#636EFA", "Standard Volume": "#EF553B", "High Volume": "#00CC96"},
                                     hole=0.4)
                    fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=200, showlegend=False)
                    ph_pie.plotly_chart(fig_pie, use_container_width=True, key=f"pie_{time.time()}")

                    # UPDATE SCATTER PLOT (Live) - İSTENİLEN NOKTALI GRAFİK
                    fig_scat = px.scatter(df_disp, x='store_nbr', y='predicted_sales', 
                                          color='cluster_name',
                                          labels={'store_nbr': 'Store ID', 'predicted_sales': 'Predicted Sales'},
                                          color_discrete_map={"Low Volume": "#636EFA", "Standard Volume": "#EF553B", "High Volume": "#00CC96"})
                    fig_scat.update_layout(margin=dict(t=30, b=0, l=0, r=0), height=350)
                    ph_scatter.plotly_chart(fig_scat, use_container_width=True, key=f"scat_{time.time()}")

                # 2. UPDATE METRICS & BLUE BOX (TOP ROW)
                last = df_disp.iloc[-1]
                
                # Forecast Metric
                ph_metric_forecast.metric(
                    label=f"Forecast ({last['family']})",
                    value=f"{last['predicted_sales']:.0f}",
                    delta=f"{last['predicted_sales'] - last['actual_sales']:.0f} Diff",
                    delta_color="inverse"
                )
                
                # Blue Analysis Box
                act_txt = f"{last['actual_sales']:.0f}"
                if last['actual_sales'] == 0: act_txt += " (Closed)"
                seg_txt = last.get('cluster_name', 'Unknown')
                ph_box_analysis.info(f"**Analysis Result**\n\nSegment: **{seg_txt}**\n\nPred: **{last['predicted_sales']:.0f}** | Act: **{act_txt}**")
                
                # Live Performance Metrics (Dynamic R2, MAE, RMSE)
                # Son 600 veri üzerinden anlık hesaplanır
                if len(df_disp) > 10:
                    y_true = df_disp['actual_sales']
                    y_pred = df_disp['predicted_sales']
                    live_r2 = r2_score(y_true, y_pred)
                    live_mae = mean_absolute_error(y_true, y_pred)
                    live_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                    
                    ph_metric_r2.metric("Live R² (Accuracy)", f"{live_r2:.2f}")
                    ph_metric_mae.metric("Live MAE (Error)", f"{live_mae:.1f}")
                    ph_metric_rmse.metric("Live RMSE (Penalty)", f"{live_rmse:.1f}")

                # 3. MAIN FORECAST CHART (HOVER DETAILS EKLENDİ)
                fig_line = px.line(
                    df_disp.tail(150), x='seq', y='predicted_sales', 
                    title="Real-Time Forecast Stream (Predicted Only)",
                    color_discrete_sequence=["#00CC96"],
                    # Hover Data geri geldi
                    hover_data=['actual_sales', 'diff', 'cluster_name', 'family', 'store_nbr']
                )
                
                # Custom Tooltip Template
                fig_line.update_traces(
                    hovertemplate="<br>".join([
                        "<b>Seq:</b> %{x}",
                        "<b>Forecast:</b> %{y:.0f}",
                        "<b>Actual:</b> %{customdata[0]:.0f}",
                        "<b>Diff:</b> %{customdata[1]:.0f}",
                        "<b>Segment:</b> %{customdata[2]}",
                        "<b>Family:</b> %{customdata[3]}",
                        "<b>Store:</b> %{customdata[4]}"
                    ])
                )
                fig_line.update_layout(height=400)
                ph_chart_line.plotly_chart(fig_line, use_container_width=True, key=f"line_{time.time()}")

            # Raw Data Update
            if raw_d:
                r_df = pd.DataFrame(raw_d)
                if '_id' in r_df.columns: r_df.drop('_id', axis=1, inplace=True)
                ph_raw_table.dataframe(r_df, hide_index=True, use_container_width=True)
                
        except Exception as e:
            # Hata olursa sessizce geç veya konsola yaz
            print(f"Loop Error: {e}")
            
        time.sleep(1.2) # Biraz hızlandırdım akıcı olsun
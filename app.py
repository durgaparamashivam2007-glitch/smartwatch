import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Import custom src modules
from src.preprocessing import load_dataset, preprocess_data, scale_features, detect_columns
from src.clustering import compute_elbow, compute_pca, train_kmeans, save_ml_artifacts, load_ml_artifacts
from src.analysis import profile_clusters, select_important_features
from src.visualization import (
    plot_elbow_curve,
    plot_pca_2d,
    plot_pca_comparison,
    plot_activity_distribution,
    plot_feature_distribution,
    plot_correlation_heatmap,
    plot_cluster_distribution,
    plot_feature_comparison
)

# ------------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Smartwatch Activity Analysis",
    page_icon="⌚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern glassmorphism aesthetic
st.markdown("""
<style>
    /* Dark glassmorphism theme styling */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        color: #f8fafc;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Header card */
    .main-header {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    
    .main-header h1 {
        color: #38bdf8;
        font-weight: 800;
        font-size: 2.2rem;
        margin: 0;
    }
    
    .main-header p {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-top: 6px;
        margin-bottom: 0;
    }

    /* Metric card styling */
    .metric-card {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 18px 20px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: rgba(56, 189, 248, 0.4);
    }
    
    .metric-title {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .metric-value {
        color: #f8fafc;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 4px;
    }
    
    .metric-sub {
        color: #38bdf8;
        font-size: 0.8rem;
        font-weight: 500;
        margin-top: 2px;
    }
    
    /* Cluster Group Card */
    .cluster-card {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .cluster-card-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #38bdf8;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    
    .badge-high { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }
    .badge-moderate { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }
    .badge-light { background: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.4); }
    .badge-low { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }
    
    /* Prediction output card */
    .prediction-card {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.2) 0%, rgba(99, 102, 241, 0.2) 100%);
        border: 1px solid rgba(56, 189, 248, 0.5);
        border-radius: 16px;
        padding: 24px;
        margin-top: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# 2. DATA CACHING & PIPELINE FUNCTIONS
# ------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_cached_data(filepath):
    if not os.path.exists(filepath):
        return None, None, None
    df = load_dataset(filepath)
    df_clean, X, summary = preprocess_data(df)
    return df_clean, X, summary


@st.cache_data(show_spinner=False)
def get_cached_elbow(X_scaled_values):
    return compute_elbow(X_scaled_values, min_k=2, max_k=8)


# ------------------------------------------------------------------------------
# 3. SIDEBAR CONTROLS & NAVIGATION
# ------------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/isometric-folders/100/smart-watch.png", width=64)
st.sidebar.title("⌚ Activity Analysis")
st.sidebar.markdown("---")

# Navigation menu
navigation = st.sidebar.radio(
    "📍 Navigation",
    [
        "📊 Dashboard",
        "📁 Dataset Analysis",
        "🔍 Exploratory Data Analysis (EDA)",
        "🤖 Clustering & PCA",
        "🏷️ Behavior Groups",
        "⚡ Activity Prediction"
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Model Controls")

# Dataset File selection / status
data_path = os.path.join("data", "dataset.csv")
if not os.path.exists(data_path):
    st.error(f"Dataset file not found at `{data_path}`. Please make sure `data/dataset.csv` exists.")
    st.stop()

# Cluster K Slider (Default = 4)
selected_k = st.sidebar.slider(
    "Number of Clusters (K):",
    min_value=2,
    max_value=8,
    value=4,
    step=1,
    help="Select the number of activity clusters for K-Means."
)

# Dynamic feature subset selection option in sidebar
feature_mode = st.sidebar.selectbox(
    "Feature Space:",
    ["All Sensor Features", "Top Variance Sensor Features"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Project Info**\nUses Unsupervised K-Means & PCA to profile smartwatch activity behavior without using activity labels during training.")


# ------------------------------------------------------------------------------
# 4. LOAD & PREPROCESS DATA
# ------------------------------------------------------------------------------
with st.spinner("Loading smartwatch dataset..."):
    df_clean, X, summary = get_cached_data(data_path)

if df_clean is None:
    st.error("Error loading dataset.")
    st.stop()

sensor_cols = summary['sensor_cols']
target_col = summary['target_col']

# Filter X if user chooses Top Variance Features
if feature_mode == "Top Variance Sensor Features":
    top_var_cols = select_important_features(X, n_features=30)
    X_active = X[top_var_cols].copy()
else:
    X_active = X.copy()

# Scale active features
X_scaled, scaler = scale_features(X_active)

# Perform PCA
pca_df, pca_model, var_ratio, cum_var = compute_pca(X_scaled, n_components=2)

# Train K-Means with user selected K
kmeans_model, cluster_labels = train_kmeans(X_scaled, n_clusters=selected_k, random_state=42)
df_clean['Cluster'] = cluster_labels

# Generate Cluster Profile
profile_df = profile_clusters(df_clean, 'Cluster', sensor_cols, target_col=target_col)

# Save artifacts to models/ directory
save_ml_artifacts(
    models_dir="models",
    kmeans_model=kmeans_model,
    scaler=scaler,
    pca_model=pca_model,
    feature_names=X_active.columns.tolist(),
    cluster_profiles=profile_df
)

# Top 6 features for prediction interface
top_pred_features = select_important_features(X_active, n_features=6)


# ------------------------------------------------------------------------------
# 5. RENDER HEADER
# ------------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>⌚ Smartwatch Activity Analysis</h1>
    <p>Identify Activity Behavior Groups Using Unsupervised Machine Learning (K-Means & PCA)</p>
</div>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# PAGE 1: DASHBOARD
# ------------------------------------------------------------------------------
if navigation == "📊 Dashboard":
    st.subheader("📌 Executive Summary & Key Metrics")
    
    # Calculate key metrics
    total_records = len(df_clean)
    num_features = X_active.shape[1]
    
    # Find most common behavior group
    most_common_row = profile_df.loc[profile_df['Records'].idxmax()]
    most_common_behavior = most_common_row['Behavior Group']
    dominant_overall_act = df_clean[target_col].mode()[0] if target_col and target_col in df_clean.columns else "N/A"

    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Records</div>
            <div class="metric-value">{total_records:,}</div>
            <div class="metric-sub">Smartwatch Samples</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Sensor Features</div>
            <div class="metric-value">{num_features}</div>
            <div class="metric-sub">Input Signals</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Clusters (K)</div>
            <div class="metric-value">{selected_k}</div>
            <div class="metric-sub">Selected Groups</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Primary Behavior</div>
            <div class="metric-value" style="font-size: 1.1rem; margin-top: 10px; color: #38bdf8;">{most_common_behavior.split('(')[0].strip()}</div>
            <div class="metric-sub">{most_common_row['Records']:,} records</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Dominant Activity</div>
            <div class="metric-value" style="font-size: 1.1rem; margin-top: 10px; color: #4ade80;">{dominant_overall_act}</div>
            <div class="metric-sub">Ground Truth Label</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Overview Charts Layout
    c1, c2 = st.columns([1.2, 1])
    
    with c1:
        st.plotly_chart(plot_pca_2d(pca_df, cluster_labels, hover_df=df_clean), use_container_width=True)
        
    with c2:
        st.plotly_chart(plot_cluster_distribution(profile_df), use_container_width=True)

    st.markdown("### 📋 Activity Behavior Summary Table")
    st.dataframe(
        profile_df[['Cluster', 'Behavior Group', 'Records', 'Percentage (%)', 'Dominant Activity', 'Activity Level']],
        use_container_width=True,
        hide_index=True
    )


# ------------------------------------------------------------------------------
# PAGE 2: DATASET ANALYSIS
# ------------------------------------------------------------------------------
elif navigation == "📁 Dataset Analysis":
    st.subheader("📋 Dataset Inspection & Preprocessing Summary")
    
    # Metadata Overview
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Raw Dataset Rows", f"{summary['initial_shape'][0]:,}")
    col2.metric("Total Columns", summary['initial_shape'][1])
    col3.metric("Duplicates Removed", summary['duplicate_count'])
    col4.metric("Imputed Null Values", summary['missing_values_imputed'])

    st.markdown("---")
    
    st.markdown("### 🔍 Sample Dataset Rows (First 10 Rows)")
    st.dataframe(df_clean.head(10), use_container_width=True)

    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("### 📊 Column Types & Roles")
        col_info = pd.DataFrame({
            "Column Category": ["Target / Activity Label", "ID / Subject Columns", "Sensor Numerical Features"],
            "Detected Name / Count": [
                summary['target_col'] if summary['target_col'] else "None",
                ", ".join(summary['id_cols']) if summary['id_cols'] else "None",
                f"{len(summary['sensor_cols'])} features"
            ]
        })
        st.table(col_info)

    with c2:
        st.markdown("### 📈 Statistical Summary (First 6 Features)")
        st.dataframe(df_clean[sensor_cols[:6]].describe().round(4), use_container_width=True)


# ------------------------------------------------------------------------------
# PAGE 3: EDA
# ------------------------------------------------------------------------------
elif navigation == "🔍 Exploratory Data Analysis (EDA)":
    st.subheader("🔍 Exploratory Data Analysis")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🏷️ Activity Distribution",
        "📈 Sensor Feature Distributions",
        "🔥 Correlation Heatmap",
        "📊 Cluster Feature Comparisons"
    ])

    with tab1:
        if target_col and target_col in df_clean.columns:
            st.plotly_chart(plot_activity_distribution(df_clean, target_col), use_container_width=True)
        else:
            st.info("Ground-truth target activity label not available in dataset.")

    with tab2:
        selected_feature = st.selectbox("Select Sensor Feature to Inspect:", sensor_cols, index=0)
        st.plotly_chart(plot_feature_distribution(df_clean, selected_feature), use_container_width=True)

    with tab3:
        st.markdown("Heatmap showing correlations between sensor signals.")
        st.plotly_chart(plot_correlation_heatmap(df_clean, sensor_cols, max_features=12), use_container_width=True)

    with tab4:
        st.plotly_chart(plot_feature_comparison(df_clean, 'Cluster', sensor_cols, profile_df=profile_df), use_container_width=True)


# ------------------------------------------------------------------------------
# PAGE 4: CLUSTERING & PCA
# ------------------------------------------------------------------------------
elif navigation == "🤖 Clustering & PCA":
    st.subheader("🤖 Unsupervised Learning & Dimensionality Reduction")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📉 Elbow Method Curve")
        elbow_df = get_cached_elbow(X_scaled)
        st.plotly_chart(plot_elbow_curve(elbow_df, selected_k=selected_k), use_container_width=True)
        st.caption(f"Default K = 4. Currently selected K = **{selected_k}**.")

    with col2:
        st.markdown("### 🌌 PCA Explained Variance")
        var_df = pd.DataFrame({
            "Principal Component": ["PC1", "PC2"],
            "Explained Variance Ratio": [f"{var_ratio[0]*100:.2f}%", f"{var_ratio[1]*100:.2f}%"],
            "Cumulative Variance": [f"{cum_var[0]*100:.2f}%", f"{cum_var[1]*100:.2f}%"]
        })
        st.table(var_df)
        st.info(f"The 2 principal components capture **{(cum_var[1]*100):.2f}%** of the total feature variance.")

    st.markdown("---")

    st.markdown("### 📍 2D PCA Cluster Scatter Plot")
    show_comparison = st.checkbox("Compare with Ground-Truth Activity Labels", value=False)

    if show_comparison and target_col and target_col in df_clean.columns:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_pca_2d(pca_df, cluster_labels, hover_df=df_clean), use_container_width=True)
        with c2:
            st.plotly_chart(plot_pca_comparison(pca_df, cluster_labels, df_clean[target_col]), use_container_width=True)
    else:
        st.plotly_chart(plot_pca_2d(pca_df, cluster_labels, hover_df=df_clean), use_container_width=True)


# ------------------------------------------------------------------------------
# PAGE 5: BEHAVIOR GROUPS
# ------------------------------------------------------------------------------
elif navigation == "🏷️ Behavior Groups":
    st.subheader("🏷️ Identified Activity Behavior Groups")
    st.markdown("Detailed profiling and characterization of each discovered cluster.")

    for idx, row in profile_df.iterrows():
        c_id = row['Cluster_ID']
        group_name = row['Behavior Group']
        act_level = row['Activity Level']
        records = row['Records']
        pct = row['Percentage (%)']
        dom_act = row['Dominant Activity']
        dom_share = row['Dominant Activity Share (%)']

        # Determine badge class based on activity level
        if 'High' in act_level:
            badge_cls = 'badge-high'
        elif 'Moderate' in act_level:
            badge_cls = 'badge-moderate'
        elif 'Light' in act_level:
            badge_cls = 'badge-light'
        else:
            badge_cls = 'badge-low'

        st.markdown(f"""
        <div class="cluster-card">
            <div class="cluster-card-title">
                📌 Cluster {c_id}: {group_name}
                <span class="badge {badge_cls}">Intensity Level: {act_level}</span>
            </div>
            <div style="margin-top: 12px; display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;">
                <div><strong>Records:</strong> {records:,} ({pct}%)</div>
                <div><strong>Dominant Activity:</strong> {dom_act}</div>
                <div><strong>Dominant Share:</strong> {dom_share}%</div>
                <div><strong>Signal Intensity:</strong> {row['Mean Signal Intensity']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Expandable top feature means
        with st.expander(f"View Average Sensor Measurements for Cluster {c_id}"):
            feat_means = {k.replace('Mean_', ''): v for k, v in row.items() if k.startswith('Mean_')}
            st.dataframe(pd.DataFrame([feat_means]), use_container_width=True)


# ------------------------------------------------------------------------------
# PAGE 6: ACTIVITY PREDICTION
# ------------------------------------------------------------------------------
elif navigation == "⚡ Activity Prediction":
    st.subheader("⚡ Predict Activity Behavior Group")
    st.markdown("Input smartwatch sensor values to predict the activity behavior group using the trained K-Means model.")

    st.info("Input key sensor measurements below. Default values are set to the dataset averages.")

    user_inputs = {}
    
    # Form layout for key features
    with st.form("prediction_form"):
        st.markdown("### 🎛️ Sensor Input Parameters")
        cols = st.columns(2)
        
        for i, feat in enumerate(top_pred_features):
            col = cols[i % 2]
            f_min = float(X_active[feat].min())
            f_max = float(X_active[feat].max())
            f_mean = float(X_active[feat].mean())
            
            user_inputs[feat] = col.number_input(
                label=f"**{feat}**",
                min_value=f_min,
                max_value=f_max,
                value=f_mean,
                step=float((f_max - f_min) / 100.0) if f_max != f_min else 0.01,
                help=f"Range: [{f_min:.4f}, {f_max:.4f}]"
            )

        submit = st.form_submit_button("🚀 Predict Activity Behavior", use_container_width=True)

    if submit:
        # Build full input vector with dataset means for non-selected features
        input_dict = {f: X_active[f].mean() for f in X_active.columns}
        input_dict.update(user_inputs)
        
        input_df = pd.DataFrame([input_dict])[X_active.columns]
        
        # Scale input using exact saved StandardScaler
        input_scaled = scaler.transform(input_df)
        
        # Predict cluster using K-Means model
        pred_cluster = kmeans_model.predict(input_scaled)[0]
        
        # Retrieve cluster behavior group profile
        match_profile = profile_df[profile_df['Cluster_ID'] == pred_cluster].iloc[0]
        pred_group = match_profile['Behavior Group']
        pred_level = match_profile['Activity Level']
        pred_dominant = match_profile['Dominant Activity']

        st.markdown(f"""
        <div class="prediction-card">
            <h2 style="color: #38bdf8; margin-top: 0;">🎯 Prediction Result</h2>
            <div style="font-size: 1.5rem; font-weight: 700; color: #f8fafc; margin-bottom: 8px;">
                Behavior Group: <span style="color: #4ade80;">{pred_group}</span>
            </div>
            <div style="font-size: 1.1rem; color: #cbd5e1; margin-bottom: 12px;">
                <strong>Assigned Cluster:</strong> Cluster {pred_cluster} &nbsp;|&nbsp; 
                <strong>Activity Level:</strong> {pred_level} &nbsp;|&nbsp;
                <strong>Dominant Ground-Truth Activity:</strong> {pred_dominant}
            </div>
            <p style="color: #94a3b8; font-size: 0.95rem; line-height: 1.5; margin-bottom: 0;">
                <strong>Explanation:</strong> The input sensor reading vector was transformed using the exact StandardScaler feature space and mapped to Cluster {pred_cluster}. Based on the sensor signal magnitude and directional acceleration characteristics, this sample aligns with <strong>{pred_group}</strong> behavior.
            </p>
        </div>
        """, unsafe_allow_html=True)

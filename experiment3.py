import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Smart Grid Analytics", layout="wide", page_icon="📊")

# Simple header
st.title("📊 Smart Grid Big Data Analytics")
st.subheader("Machine Learning for Energy Data Analysis")

# Generate simple sample data
@st.cache_data
def create_sample_data():
    np.random.seed(42)
    n_samples = 800
    
    data = {
        'voltage': np.random.normal(230, 15, n_samples),
        'current': np.random.normal(45, 10, n_samples), 
        'power': np.random.normal(10000, 2500, n_samples),
        'frequency': np.random.normal(50, 1, n_samples),
        'temperature': np.random.normal(25, 8, n_samples),
        'load_demand': np.random.normal(8000, 2000, n_samples)
    }
    return pd.DataFrame(data)

# Sidebar controls
st.sidebar.header("🎛️ Controls")

use_sample = st.sidebar.checkbox("📊 Use Sample Data", value=True)

if use_sample:
    df = create_sample_data()
    st.sidebar.success("✅ Sample data loaded")
    
    # Sample data information (like previous experiments)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Sample Data Includes")
    st.sidebar.markdown("""
    **⚡ Electrical Parameters:**
    • Voltage: 230V ± 15V variations
    • Current: 45A ± 10A fluctuations
    • Frequency: 50Hz ± 1Hz stability
    
    **🏭 Load Characteristics:**
    • Power: 10kW ± 2.5kW consumption
    • Load Demand: 8kW ± 2kW requests
    • Temperature: 25°C ± 8°C ambient
    
    **📊 Dataset Size:** 800 smart meter readings
    """)
    
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        st.info("👆 Please upload a CSV file or use sample data")
        st.stop()

# Show basic data info
st.markdown("## 📋 Data Overview")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("📊 Total Rows", len(df))
with col2:
    st.metric("📈 Columns", len(df.columns))
with col3:
    st.metric("🔢 Numeric Features", len(df.select_dtypes(include=[np.number]).columns))

# Data preview section (NEW)
st.markdown("### 👀 Data Preview")
st.dataframe(df.head(10), use_container_width=True)

# Basic statistics (NEW)
st.markdown("### 📊 Data Statistics")
st.dataframe(df.describe().round(2), use_container_width=True)

# Feature selection
numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
selected_features = st.sidebar.multiselect(
    "🎯 Select Features:", 
    numeric_features, 
    default=numeric_features[:4],
    help="Choose features for analysis"
)

# ML parameters
num_clusters = st.sidebar.slider("🎯 Number of Clusters", 2, 8, 4)

# Additional sidebar info (like previous experiments)
st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 Machine Learning Pipeline")
st.sidebar.markdown("""
• **Data Preprocessing:** Clean and standardize
• **PCA Analysis:** Reduce to 2D visualization  
• **K-Means Clustering:** Group similar patterns
• **Results Visualization:** Interactive scatter plots
""")

# Main analysis
if st.sidebar.button("🚀 Run Analysis", type="primary") and len(selected_features) >= 2:
    
    # Prepare data
    data_clean = df[selected_features].dropna()
    
    # Step 1: Standardize data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_clean)
    
    # Step 2: PCA (reduce to 2D)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data_scaled)
    
    # Step 3: Clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(data_scaled)
    
    # Results
    results_df = pd.DataFrame({
        'PC1': pca_result[:, 0],
        'PC2': pca_result[:, 1], 
        'Cluster': clusters
    })
    
    # Show key metrics
    st.markdown("## 🎯 Analysis Results")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🔍 Features Analyzed", len(selected_features))
    with col2:
        variance_explained = sum(pca.explained_variance_ratio_) * 100
        st.metric("📊 Variance Explained", f"{variance_explained:.1f}%")
    with col3:
        st.metric("🎯 Clusters Found", num_clusters)
    
    # Main visualization - PCA Scatter Plot
    st.markdown("### 🎨 Customer Segmentation Results")
    
    fig = px.scatter(
        results_df, 
        x='PC1', y='PC2', 
        color='Cluster',
        title="🎯 Smart Grid Customer Clusters (PCA Visualization)",
        color_continuous_scale="viridis"
    )
    fig.update_layout(height=500, template="plotly_white")
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    st.plotly_chart(fig, use_container_width=True)
    
    # Cluster summary
    st.markdown("### 📊 Cluster Analysis")
    cluster_data = data_clean.copy()
    cluster_data['Cluster'] = clusters
    cluster_summary = cluster_data.groupby('Cluster').mean().round(2)
    
    st.dataframe(cluster_summary, use_container_width=True)
    
    # Show sample results
    st.markdown("### 📋 Sample Results with Cluster Assignments")
    sample_results = results_df.head(15).copy()
    sample_results['PC1'] = sample_results['PC1'].round(3)
    sample_results['PC2'] = sample_results['PC2'].round(3)
    st.dataframe(sample_results, use_container_width=True)

else:
    if len(selected_features) < 2:
        st.warning("⚠️ Please select at least 2 features")
    
    st.markdown("## 👈 Configure settings and click 'Run Analysis'")
    st.markdown("### 🎯 What this does:")
    st.markdown("""
    - **🔍 Data Analysis**: Analyze smart grid energy consumption patterns
    - **📊 PCA**: Reduce complex data to 2D for visualization  
    - **🎯 Clustering**: Group similar customers automatically
    - **📈 Insights**: Identify different types of energy users
    """)

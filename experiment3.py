import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Smart Grid Analytics Platform", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# Enhanced CSS styling
st.markdown("""
<style>
    /* Wider sidebar */
    .css-1d391kg, section[data-testid="stSidebar"] {
        min-width: 400px !important;
        max-width: 400px !important;
    }
    
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    .analytics-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        border: 2px solid rgba(255,255,255,0.1);
    }
    
    .feature-card {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 5px solid #ff6b6b;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .insight-box {
        background: linear-gradient(135deg, #d299c2 0%, #fef9d7 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #4CAF50;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .sidebar-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    .data-info {
        background: rgba(102, 126, 234, 0.1);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    
    .upload-zone {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: rgba(102, 126, 234, 0.05);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
<div class="main-header">
    <h1>📊 Smart Grid Big Data Analytics Platform</h1>
    <h3>Advanced Pattern Recognition & Clustering Analysis</h3>
    <p>Upload your data or use sample smart grid datasets for comprehensive analysis</p>
</div>
""", unsafe_allow_html=True)

# Generate enhanced sample data
def generate_sample_data():
    np.random.seed(42)
    n_samples = 1500
    
    # Create more realistic smart grid data with patterns
    hour = np.random.randint(0, 24, n_samples)
    season = np.random.choice(['Summer', 'Winter', 'Spring', 'Autumn'], n_samples)
    
    # Base patterns
    seasonal_factor = np.where(season == 'Summer', 1.3, 
                      np.where(season == 'Winter', 1.1, 1.0))
    
    daily_pattern = 1 + 0.3 * np.sin(2 * np.pi * hour / 24)
    
    data = {
        'voltage': np.random.normal(230, 10, n_samples) * daily_pattern,
        'current': np.random.normal(45, 8, n_samples) * seasonal_factor,
        'power': np.random.normal(10000, 2000, n_samples) * daily_pattern * seasonal_factor,
        'frequency': np.random.normal(50, 0.5, n_samples),
        'temperature': np.random.normal(25, 5, n_samples) * seasonal_factor,
        'power_factor': np.random.uniform(0.8, 0.95, n_samples),
        'harmonic_distortion': np.random.exponential(2, n_samples),
        'load_demand': np.random.normal(8000, 1500, n_samples) * daily_pattern,
        'hour': hour,
        'season': season
    }
    
    return pd.DataFrame(data)

# Enhanced Sidebar
st.sidebar.markdown("""
<div class="sidebar-header">
    <h2>🎛️ Analytics Control Panel</h2>
    <p>Configure your analysis parameters</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### 📁 Data Source")
use_sample = st.sidebar.checkbox("📊 Use Enhanced Sample Data", value=True)

if use_sample:
    df = generate_sample_data()
    st.sidebar.success("✅ Sample smart grid data loaded")
    
    st.sidebar.markdown("""
    <div class="data-info">
        <strong>📋 Sample Data Includes:</strong><br>
        • Voltage & Current measurements<br>
        • Power consumption patterns<br>
        • Seasonal variations<br>
        • Daily consumption cycles<br>
        • Grid quality metrics
    </div>
    """, unsafe_allow_html=True)
    
else:
    st.sidebar.markdown("""
    <div class="upload-zone">
        <h3>📤 Upload Your Data</h3>
        <p>Support CSV files with numerical data</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.sidebar.file_uploader(
        "Choose CSV file", 
        type=["csv"],
        help="Upload a CSV file with numerical columns for analysis"
    )
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("✅ File uploaded successfully!")
    else:
        st.info("👆 Please upload a CSV file or use sample data")
        st.stop()

# Data overview
st.markdown("## 📋 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="analytics-card">
        <h3>📊 Rows</h3>
        <h2>{df.shape[0]:,}</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="analytics-card">
        <h3>📈 Columns</h3>
        <h2>{df.shape[1]}</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    numeric_count = len(df.select_dtypes(include=[np.number]).columns)
    st.markdown(f"""
    <div class="analytics-card">
        <h3>🔢 Numeric</h3>
        <h2>{numeric_count}</h2>
    </div>
    """, unsafe_allow_html=True)

with col4:
    missing_count = df.isnull().sum().sum()
    st.markdown(f"""
    <div class="analytics-card">
        <h3>❓ Missing</h3>
        <h2>{missing_count}</h2>
    </div>
    """, unsafe_allow_html=True)

# Enhanced data preview
st.markdown("### 👀 Data Preview")
st.dataframe(df.head(10), use_container_width=True)

# Feature selection
numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()

st.sidebar.markdown("### ⚙️ Analysis Configuration")
selected_features = st.sidebar.multiselect(
    "🎯 Select Features for Analysis:",
    options=numeric_features,
    default=numeric_features[:4] if len(numeric_features) >= 4 else numeric_features,
    help="Choose numerical features for clustering and PCA analysis"
)

num_clusters = st.sidebar.slider("🎯 Number of Clusters", 2, 10, 4)

st.sidebar.markdown("### 📊 Analysis Options")
show_correlation = st.sidebar.checkbox("📈 Show Correlation Analysis", True)
show_distribution = st.sidebar.checkbox("📊 Show Feature Distributions", True)
show_time_series = st.sidebar.checkbox("⏰ Show Time Series (if available)", True)

if st.sidebar.button("🚀 Run Analytics", type="primary", use_container_width=True) and len(selected_features) >= 2:
    
    with st.spinner('🔄 Running advanced analytics...'):
        # Prepare data
        data_clean = df[selected_features].dropna()
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_clean)
        
        # PCA
        pca = PCA(n_components=min(3, len(selected_features)))
        pca_result = pca.fit_transform(data_scaled)
        
        # Clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(data_scaled)
        
        # Results DataFrame
        results_df = pd.DataFrame({
            'PC1': pca_result[:, 0],
            'PC2': pca_result[:, 1],
            'Cluster': clusters
        })
        
        if pca_result.shape[1] > 2:
            results_df['PC3'] = pca_result[:, 2]
        
        st.markdown("## 🎯 Analytics Results")
        
        # Key insights
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="insight-box">
                <h3>🔍 Features Analyzed</h3>
                <h2>{len(selected_features)}</h2>
                <p>Dimensional analysis completed</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            variance_explained = sum(pca.explained_variance_ratio_[:2]) * 100
            st.markdown(f"""
            <div class="insight-box">
                <h3>📊 Variance Explained</h3>
                <h2>{variance_explained:.1f}%</h2>
                <p>By first 2 components</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="insight-box">
                <h3>🎯 Clusters Found</h3>
                <h2>{num_clusters}</h2>
                <p>Distinct data patterns</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Enhanced PCA visualization
        st.markdown("### 🎨 Principal Component Analysis Visualization")
        
        if len(results_df.columns) > 3:  # 3D plot
            fig1 = px.scatter_3d(
                results_df, 
                x='PC1', y='PC2', z='PC3', 
                color='Cluster',
                title="🌐 3D PCA Cluster Visualization",
                color_continuous_scale="viridis"
            )
        else:  # 2D plot
            fig1 = px.scatter(
                results_df, 
                x='PC1', y='PC2', 
                color='Cluster',
                title="🎨 2D PCA Cluster Visualization",
                color_continuous_scale="viridis",
                size_max=10
            )
        
        fig1.update_layout(
            template="plotly_white",
            title_font_size=18,
            font=dict(size=14),
            height=500
        )
        fig1.update_traces(marker=dict(size=8, opacity=0.7))
        st.plotly_chart(fig1, use_container_width=True)
        
        # Correlation analysis
        if show_correlation:
            st.markdown("### 🔗 Feature Correlation Matrix")
            corr_matrix = data_clean.corr()
            
            fig2 = px.imshow(
                corr_matrix, 
                title="🔗 Feature Correlation Heatmap", 
                color_continuous_scale="RdBu",
                aspect="auto"
            )
            fig2.update_layout(
                template="plotly_white",
                title_font_size=18,
                font=dict(size=14)
            )
            st.plotly_chart(fig2, use_container_width=True)
            
            # Correlation insights
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
            
            if high_corr_pairs:
                st.markdown("#### 🔍 High Correlation Insights")
                for feat1, feat2, corr_val in high_corr_pairs[:3]:
                    st.markdown(f"""
                    <div class="feature-card">
                        <strong>{feat1}</strong> ↔ <strong>{feat2}</strong>: {corr_val:.2f} correlation
                    </div>
                    """, unsafe_allow_html=True)
        
        # Cluster profiles
        st.markdown("### 📊 Cluster Analysis")
        
        cluster_data = data_clean.copy()
        cluster_data['Cluster'] = clusters
        cluster_means = cluster_data.groupby('Cluster').mean()
        
        # Heatmap of cluster characteristics
        fig3 = px.imshow(
            cluster_means.T,
            title="🎯 Cluster Characteristics Heatmap",
            color_continuous_scale="viridis",
            aspect="auto"
        )
        fig3.update_layout(
            template="plotly_white",
            title_font_size=18,
            font=dict(size=14),
            xaxis_title="Cluster",
            yaxis_title="Features"
        )
        st.plotly_chart(fig3, use_container_width=True)
        
        # Cluster summary table
        st.markdown("#### 📋 Cluster Summary Statistics")
        display_means = cluster_means.round(2)
        st.dataframe(display_means, use_container_width=True)
        
        # Feature distributions by cluster
        if show_distribution and len(selected_features) >= 1:
            st.markdown("### 📈 Feature Distribution Analysis")
            
            feature_to_plot = st.selectbox("Select feature for distribution analysis:", selected_features)
            
            fig4 = px.histogram(
                cluster_data, 
                x=feature_to_plot, 
                color='Cluster',
                title=f"📊 {feature_to_plot} Distribution by Cluster",
                marginal="box"
            )
            fig4.update_layout(
                template="plotly_white",
                title_font_size=18,
                font=dict(size=14)
            )
            st.plotly_chart(fig4, use_container_width=True)
        
        # Time series analysis if hour column exists
        if show_time_series and 'hour' in df.columns:
            st.markdown("### ⏰ Time Series Pattern Analysis")
            
            if len(selected_features) >= 1:
                hourly_avg = df.groupby('hour')[selected_features[0]].mean().reset_index()
                
                fig5 = px.line(
                    hourly_avg,
                    x='hour',
                    y=selected_features[0],
                    title=f"⏰ Daily Pattern: {selected_features[0]}",
                    markers=True
                )
                fig5.update_layout(
                    template="plotly_white",
                    title_font_size=18,
                    font=dict(size=14),
                    xaxis_title="Hour of Day",
                    yaxis_title=selected_features[0]
                )
                st.plotly_chart(fig5, use_container_width=True)

else:
    if len(selected_features) < 2:
        st.warning("⚠️ Please select at least 2 features for analysis")
    
    st.markdown("## 👈 Configure analysis settings and click 'Run Analytics'")
    st.markdown("### 🎯 Platform Capabilities:")
    st.markdown("""
    - **🔍 Advanced Clustering:** K-Means algorithm with optimized parameters
    - **📊 Dimensionality Reduction:** PCA for data visualization and insights
    - **🔗 Correlation Analysis:** Identify relationships between variables
    - **📈 Pattern Recognition:** Discover hidden trends in your data
    - **⏰ Time Series Analysis:** Understand temporal patterns
    - **🎨 Interactive Visualizations:** Explore data with dynamic charts
    """)
    
    # Demo visualization
    if 'voltage' in df.columns and 'current' in df.columns:
        st.markdown("### 📊 Sample Data Visualization")
        demo_fig = px.scatter(
            df.sample(200), 
            x='voltage', 
            y='current',
            title="🔌 Voltage vs Current Relationship (Sample)",
            opacity=0.6
        )
        demo_fig.update_layout(template="plotly_white", title_font_size=16)
        st.plotly_chart(demo_fig, use_container_width=True)


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration with custom styling
st.set_page_config(
    page_title="Maharashtra Smart Grid", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="⚡"
)

# Custom CSS for better aesthetics
st.markdown("""
<style>
     .css-1d391kg {
        min-width: 300px !important;
        max-width: 300px !important;
    }
       section[data-testid="stSidebar"] {
        width: 300px !important;
        min-width: 300px !important;
        max-width: 300px !important;
    }
           
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .info-box {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #ff7b7b;
    }
    
    .success-box {
        background: linear-gradient(135deg, #d299c2 0%, #fef9d7 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #4CAF50;
    }
    
    .sidebar-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Main header with gradient background
st.markdown("""
<div class="main-header">
    <h1>⚡ Maharashtra Smart Grid Analytics</h1>
    <h3>Dynamic Electricity Pricing Dashboard</h3>
    <p>Real-time pricing algorithms for MSEDCL grid optimization</p>
</div>
""", unsafe_allow_html=True)

class MaharashtraPricingEngine:
    def __init__(self):
        self.base_price = 4.50
        
    def generate_profiles(self):
        hours = np.arange(24)
        
        # Enhanced demand profile with more realistic patterns
        demand = 8000 + 3000 * np.sin(2 * np.pi * (hours - 6) / 24)
        demand += 2000 * np.exp(-0.5 * ((hours - 9) / 2) ** 2)
        demand += 4000 * np.exp(-0.5 * ((hours - 20) / 3) ** 2)
        demand += np.random.normal(0, 300, 24)
        demand = np.maximum(demand, 5000)
        
        # Supply profile
        thermal = 12000
        solar = 3000 * np.maximum(0, np.sin(np.pi * (hours - 6) / 12))
        solar[(hours < 6) | (hours > 18)] = 0
        wind = 2000 + 1500 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 500, 24)
        wind = np.maximum(wind, 500)
        hydro = 1500 + 500 * np.sin(2 * np.pi * (hours - 12) / 24)
        hydro = np.maximum(hydro, 1000)
        
        supply = thermal + solar + wind + hydro
        
        return pd.DataFrame({
            'hour': hours,
            'demand': demand,
            'supply': supply,
            'thermal': thermal,
            'solar': solar,
            'wind': wind,
            'hydro': hydro
        })
    
    def calculate_prices(self, df):
        prices = []
        categories = []
        
        for _, row in df.iterrows():
            ratio = row['supply'] / row['demand']
            hour = row['hour']
            
            if 6 <= hour <= 9 or 18 <= hour <= 23:
                multiplier = 1.5
            elif 10 <= hour <= 17:
                multiplier = 1.2
            else:
                multiplier = 0.8
            
            if ratio >= 1.3:
                dynamic_factor = 0.7
                category = "🟢 Low Demand"
            elif ratio >= 1.1:
                dynamic_factor = 0.9
                category = "🟡 Normal"
            elif ratio >= 0.95:
                dynamic_factor = 1.3
                category = "🟠 High Demand"
            else:
                dynamic_factor = 1.8
                category = "🔴 Peak Demand"
            
            final_price = self.base_price * multiplier * dynamic_factor
            fuel_adjustment = 0.25
            final_price += fuel_adjustment
            
            prices.append(final_price)
            categories.append(category)
        
        df['price'] = prices
        df['category'] = categories
        df['time_period'] = df['hour'].apply(self.get_time_period)
        return df
    
    def get_time_period(self, hour):
        if 6 <= hour <= 9 or 18 <= hour <= 23:
            return "⚡ Peak Hours"
        elif 10 <= hour <= 17:
            return "☀️ Day Hours"
        else:
            return "🌙 Off-Peak Hours"

# Enhanced Sidebar with custom styling
st.sidebar.markdown("""
<div class="sidebar-header">
    <h2>🎛️ Control Panel</h2>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### ⚙️ Grid Parameters")
base_price = st.sidebar.slider("💰 Base Tariff (₹/kWh)", 3.0, 8.0, 4.5, 0.1)
fuel_surcharge = st.sidebar.slider("⛽ Fuel Surcharge (₹/kWh)", 0.0, 1.0, 0.25, 0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 MSEDCL Tariff Structure")
st.sidebar.markdown("""
- ⚡**Peak Hours:** 6-9 AM, 6-11 PM
- ☀️ **Day Hours:** 10 AM - 5 PM  
- 🌙 **Off-Peak:** 11 PM - 6 AM
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🏭 Maharashtra Power Sources")
st.sidebar.markdown("""
- 🔥 **Thermal:** Koradi, Chandrapur
- 💨 **Wind:** 3rd largest in India
- ☀️ **Solar:** Rapidly expanding
- 💧 **Hydro:** Western Ghats
""")

engine = MaharashtraPricingEngine()
engine.base_price = base_price

# Enhanced button styling
if st.sidebar.button("🚀 Generate Analysis", type="primary"):
    df = engine.generate_profiles()
    df = engine.calculate_prices(df)
    
    # Key metrics with custom styling
    st.markdown("## 📊 Grid Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <h3>📈 Avg Demand</h3>
            <h2>{df['demand'].mean():.0f} MW</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <h3>⚡ Avg Supply</h3>
            <h2>{df['supply'].mean():.0f} MW</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <h3>💰 Avg Price</h3>
            <h2>₹{df['price'].mean():.2f}/kWh</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-container">
            <h3>🔥 Peak Price</h3>
            <h2>₹{df['price'].max():.2f}/kWh</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Enhanced info boxes
    st.markdown("## 💼 Financial & Environmental Impact")
    col1, col2 = st.columns(2)
    
    with col1:
        daily_revenue = (df['demand'] * df['price']).sum()
        st.markdown(f"""
        <div class="info-box">
            <h3>💵 Daily Revenue</h3>
            <h2>₹{daily_revenue:.0f} Crores</h2>
            <p>Estimated grid revenue for 24 hours</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        renewable_share = ((df['solar'] + df['wind'] + df['hydro']) / df['supply'] * 100).mean()
        st.markdown(f"""
        <div class="success-box">
            <h3>🌱 Renewable Share</h3>
            <h2>{renewable_share:.1f}%</h2>
            <p>Clean energy contribution to grid</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Enhanced charts with better color schemes
    st.markdown("## 📈 Real-Time Analytics")
    
    # Supply vs Demand chart
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df['hour'], y=df['demand'], 
                             mode='lines+markers',
                             name='Demand',
                             line=dict(color='#ff7f0e', width=3),
                             marker=dict(size=8)))
    fig1.add_trace(go.Scatter(x=df['hour'], y=df['supply'], 
                             mode='lines+markers',
                             name='Supply',
                             line=dict(color='#2ca02c', width=3),
                             marker=dict(size=8)))
    
    fig1.update_layout(
        title="⚡ 24-Hour Supply vs Demand Profile",
        xaxis_title="Hour of Day",
        yaxis_title="Power (MW)",
        template="plotly_white",
        title_font_size=20,
        font=dict(size=14),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    # Dynamic pricing chart
    color_map = {
        "⚡ Peak Hours": "#ff4444",
        "☀️ Day Hours": "#ffaa00", 
        "🌙 Off-Peak Hours": "#4444ff"
    }
    
    fig2 = px.line(df, x='hour', y='price', color='time_period',
                   title="💰 Dynamic Pricing Throughout Day",
                   color_discrete_map=color_map)
    
    fig2.update_layout(
        template="plotly_white",
        title_font_size=20,
        font=dict(size=14),
        xaxis_title="Hour of Day",
        yaxis_title="Price (₹/kWh)"
    )
    fig2.update_traces(line=dict(width=4), marker=dict(size=10))
    st.plotly_chart(fig2, use_container_width=True)
    
    # Energy sources breakdown
    fig3 = go.Figure()
    
    fig3.add_trace(go.Bar(x=df['hour'], y=df['thermal'], name='🔥 Thermal',
                         marker_color='#8B4513'))
    fig3.add_trace(go.Bar(x=df['hour'], y=df['solar'], name='☀️ Solar',
                         marker_color='#FFD700'))
    fig3.add_trace(go.Bar(x=df['hour'], y=df['wind'], name='💨 Wind',
                         marker_color='#87CEEB'))
    fig3.add_trace(go.Bar(x=df['hour'], y=df['hydro'], name='💧 Hydro',
                         marker_color='#4169E1'))
    
    fig3.update_layout(
        title="🏭 Maharashtra Energy Sources Mix",
        xaxis_title="Hour of Day",
        yaxis_title="Power (MW)",
        template="plotly_white",
        title_font_size=20,
        font=dict(size=14),
        barmode='stack'
    )
    st.plotly_chart(fig3, use_container_width=True)
    
    # Price category pie chart
    fig4 = px.pie(df, names='category', 
                  title="📊 Price Category Distribution",
                  color_discrete_sequence=['#ff4444', '#ffaa00', '#44ff44', '#4444ff'])
    
    fig4.update_layout(
        template="plotly_white",
        title_font_size=20,
        font=dict(size=14)
    )
    st.plotly_chart(fig4, use_container_width=True)
    
    # Enhanced data table
    st.markdown("## 📋 Detailed Hourly Analysis")
    
    display_df = df[['hour', 'demand', 'supply', 'price', 'category', 'time_period']].copy()
    display_df['demand'] = display_df['demand'].round(0).astype(str) + ' MW'
    display_df['supply'] = display_df['supply'].round(0).astype(str) + ' MW'
    display_df['price'] = display_df['price'].apply(lambda x: f"₹{x:.2f}")
    display_df.columns = ['🕐 Hour', '📈 Demand', '⚡ Supply', '💰 Price/kWh', '📊 Status', '⏰ Period']
    
    # Style the dataframe
    st.dataframe(
        display_df, 
        use_container_width=True,
        height=400
    )

else:
    st.markdown("## 👈 Click 'Generate Analysis' to start the simulation")
    st.markdown("### 🎯 What this dashboard shows:")
    st.markdown("""
    - **Real-time pricing** based on supply-demand dynamics
    - **Maharashtra-specific** energy mix and consumption patterns  
    - **MSEDCL tariff structure** with time-of-use pricing
    - **Financial impact** analysis with revenue calculations
    - **Environmental metrics** showing renewable energy contribution
    """)

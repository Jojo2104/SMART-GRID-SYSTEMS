import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Dynamic Pricing Maharashtra", layout="wide", page_icon="💰")

# Simplified CSS styling - NO sidebar customization
st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem; border-radius: 8px; text-align: center; color: white; margin: 0.5rem 0;
}
.main-header {
    background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
    padding: 2rem; border-radius: 10px; text-align: center; color: white; margin-bottom: 2rem;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>💰 Maharashtra Smart Grid Analytics</h1>
    <h3>Dynamic Electricity Pricing Dashboard</h3>
    <p>Real-time pricing algorithms for MSEDCL grid optimization</p>
</div>
""", unsafe_allow_html=True)

class MaharashtraPricingEngine:
    def __init__(self):
        self.base_price = 4.50
        
    def generate_profiles(self):
        hours = np.arange(24)
        
        # Enhanced demand profile for Maharashtra
        demand = 8000 + 3000 * np.sin(2 * np.pi * (hours - 6) / 24)
        demand += 2000 * np.exp(-0.5 * ((hours - 9) / 2) ** 2)    # Morning peak
        demand += 4000 * np.exp(-0.5 * ((hours - 20) / 3) ** 2)   # Evening peak
        demand += np.random.normal(0, 300, 24)
        demand = np.maximum(demand, 5000)
        
        # Supply profile for Maharashtra
        thermal = 12000  # Constant baseload
        
        # Solar profile
        solar = 3000 * np.maximum(0, np.sin(np.pi * (hours - 6) / 12))
        solar[(hours < 6) | (hours > 18)] = 0
        
        # Wind profile with variations
        wind = 2000 + 1500 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 500, 24)
        wind = np.maximum(wind, 500)
        
        # Hydro profile
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
        time_periods = []
        
        for _, row in df.iterrows():
            ratio = row['supply'] / row['demand']
            hour = row['hour']
            
            # Time-of-use multipliers
            if 6 <= hour <= 9 or 18 <= hour <= 23:  # Peak hours
                multiplier = 1.5
                period = "⚡ Peak Hours"
            elif 10 <= hour <= 17:  # Day hours
                multiplier = 1.2
                period = "☀️ Day Hours"
            else:  # Off-peak hours
                multiplier = 0.8
                period = "🌙 Off-Peak Hours"
            
            # Dynamic adjustment based on supply-demand ratio
            if ratio >= 1.2:
                dynamic_factor = 0.7
            elif ratio >= 1.0:
                dynamic_factor = 0.9
            elif ratio >= 0.9:
                dynamic_factor = 1.3
            else:
                dynamic_factor = 1.8
            
            final_price = self.base_price * multiplier * dynamic_factor
            final_price += 0.25  # Fuel adjustment
            
            prices.append(final_price)
            time_periods.append(period)
        
        df['price'] = prices
        df['time_period'] = time_periods
        return df

# Sidebar - using Streamlit's default width
st.sidebar.header("🎛️ Control Panel")

st.sidebar.markdown("### ⚙️ Grid Parameters")
base_price = st.sidebar.slider(
    "💰 Base Tariff (₹/kWh)", 
    3.0, 8.0, 4.5, 0.1,
    help="Base electricity rate before time-of-use and dynamic adjustments"
)

fuel_surcharge = st.sidebar.slider(
    "⛽ Fuel Surcharge (₹/kWh)", 
    0.0, 1.0, 0.25, 0.05,
    help="Additional fuel cost adjustment charge"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 MSEDCL Tariff Structure")
st.sidebar.markdown("""
- ⚡ **Peak Hours:** 6-9 AM, 6-11 PM  
- ☀️ **Day Hours:** 10 AM - 5 PM  
- 🌙 **Off-Peak:** 11 PM - 6 AM
""")

st.sidebar.markdown("### 🏭 Maharashtra Power Sources")
st.sidebar.markdown("""
- 🔥 **Thermal:** Koradi, Chandrapur  
- 💨 **Wind:** 3rd largest in India  
- ☀️ **Solar:** Rapidly expanding  
- 💧 **Hydro:** Western Ghats
""")

engine = MaharashtraPricingEngine()
engine.base_price = base_price

if st.sidebar.button("🚀 Generate Analysis", type="primary", use_container_width=True):
    df = engine.generate_profiles()
    df = engine.calculate_prices(df)
    
    # Key metrics
    st.markdown("## 📊 Grid Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📈 Avg Demand</h3>
            <h2>{df['demand'].mean():.0f} MW</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>⚡ Avg Supply</h3>
            <h2>{df['supply'].mean():.0f} MW</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>💰 Avg Price</h3>
            <h2>₹{df['price'].mean():.2f}/kWh</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🔥 Peak Price</h3>
            <h2>₹{df['price'].max():.2f}/kWh</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Financial & Environmental Impact
    st.markdown("## 💼 Financial & Environmental Impact")
    col1, col2 = st.columns(2)
    
    with col1:
        daily_revenue = (df['demand'] * df['price']).sum()
        st.success(f"""
        **💵 Daily Revenue**  
        **₹{daily_revenue:.0f} Crores**  
        Estimated grid revenue for 24 hours
        """)
    
    with col2:
        renewable_share = ((df['solar'] + df['wind'] + df['hydro']) / df['supply'] * 100).mean()
        st.info(f"""
        **🌱 Renewable Share**  
        **{renewable_share:.1f}%**  
        Clean energy contribution to grid
        """)
    
    # Charts - Only 2 essential charts now
    st.markdown("## 📈 Real-Time Analytics")
    
    # Supply vs Demand
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df['hour'], y=df['demand'], 
                             mode='lines+markers', name='Demand',
                             line=dict(color='#ff7f0e', width=3), marker=dict(size=8)))
    fig1.add_trace(go.Scatter(x=df['hour'], y=df['supply'], 
                             mode='lines+markers', name='Supply',
                             line=dict(color='#2ca02c', width=3), marker=dict(size=8)))
    
    fig1.update_layout(title="⚡ 24-Hour Supply vs Demand Profile",
                      xaxis_title="Hour of Day", yaxis_title="Power (MW)",
                      template="plotly_white", height=400)
    st.plotly_chart(fig1, use_container_width=True)
    
    # Energy sources breakdown
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=df['hour'], y=df['thermal'], name='🔥 Thermal', marker_color='#8B4513'))
    fig2.add_trace(go.Bar(x=df['hour'], y=df['solar'], name='☀️ Solar', marker_color='#FFD700'))
    fig2.add_trace(go.Bar(x=df['hour'], y=df['wind'], name='💨 Wind', marker_color='#87CEEB'))
    fig2.add_trace(go.Bar(x=df['hour'], y=df['hydro'], name='💧 Hydro', marker_color='#4169E1'))
    
    fig2.update_layout(title="🏭 Maharashtra Energy Sources Mix",
                      xaxis_title="Hour of Day", yaxis_title="Power (MW)",
                      template="plotly_white", height=400, barmode='stack')
    st.plotly_chart(fig2, use_container_width=True)
    
    # Hourly details table
    st.markdown("## 📋 Detailed Hourly Analysis")
    
    display_df = df[['hour', 'demand', 'supply', 'price', 'time_period']].copy()
    display_df['demand'] = display_df['demand'].round(0).astype(str) + ' MW'
    display_df['supply'] = display_df['supply'].round(0).astype(str) + ' MW'
    display_df['price'] = display_df['price'].apply(lambda x: f"₹{x:.2f}")
    display_df.columns = ['🕐 Hour', '📈 Demand', '⚡ Supply', '💰 Price/kWh', '⏰ Period']
    
    st.dataframe(display_df, use_container_width=True, height=400)

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

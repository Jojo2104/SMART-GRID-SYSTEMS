import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import random

st.set_page_config(
    page_title="Smart Grid Protocol Analyzer", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🔗"
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
    
    .protocol-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        border: 2px solid rgba(255,255,255,0.1);
    }
    
    .protocol-card h3 {
        margin-bottom: 0.5rem;
        font-size: 1.2rem;
    }
    
    .protocol-card h2 {
        margin: 0;
        font-size: 2rem;
        font-weight: bold;
    }
    
    .network-status {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #ff6b6b;
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
    
    .protocol-info {
        background: rgba(102, 126, 234, 0.1);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
<div class="main-header">
    <h1>🔗 Smart Grid Communication Protocol Analyzer</h1>
    <h3>Performance Testing & Network Optimization</h3>
    <p>Compare MODBUS, DNP3 & IEC 61850 protocols under various network conditions</p>
</div>
""", unsafe_allow_html=True)

class ProtocolSimulator:
    def __init__(self):
        self.protocols = {
            'MODBUS': {'overhead': 12, 'reliability': 0.85, 'icon': '🔧', 'color': '#ff7f0e'},
            'DNP3': {'overhead': 20, 'reliability': 0.95, 'icon': '⚡', 'color': '#2ca02c'},
            'IEC61850': {'overhead': 28, 'reliability': 0.98, 'icon': '🏭', 'color': '#1f77b4'}
        }
        
    def simulate(self, protocol, messages, condition):
        config = self.protocols[protocol]
        delays = {'🟢 Excellent': 10, '🟡 Good': 25, '🟠 Fair': 50, '🔴 Poor': 100}
        losses = {'🟢 Excellent': 0.01, '🟡 Good': 0.05, '🟠 Fair': 0.10, '🔴 Poor': 0.20}
        
        base_delay = delays[condition] + config['overhead']
        loss_rate = losses[condition]
        
        successful = 0
        total_delay = 0
        
        for _ in range(messages):
            if random.random() > loss_rate:
                delay = base_delay + random.uniform(-5, 5)
                total_delay += delay
                successful += 1
        
        avg_delay = total_delay / successful if successful > 0 else 0
        success_rate = successful / messages
        throughput = (successful * 64) / 1000  # KB/s assuming 64-byte messages
        
        return {
            'protocol': protocol,
            'condition': condition,
            'avg_delay': avg_delay,
            'success_rate': success_rate,
            'successful': successful,
            'throughput': throughput,
            'icon': config['icon'],
            'color': config['color']
        }

# Enhanced Sidebar
st.sidebar.markdown("""
<div class="sidebar-header">
    <h2>🎛️ Network Simulator</h2>
    <p>Configure test parameters</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### 📡 Protocol Selection")
protocols = st.sidebar.multiselect(
    "Choose protocols to test:", 
    ['MODBUS', 'DNP3', 'IEC61850'], 
    ['MODBUS', 'DNP3'],
    help="Select one or more protocols for comparison"
)

st.sidebar.markdown("### 📊 Test Configuration")
messages = st.sidebar.slider("📨 Number of Messages", 100, 2000, 500, 50)
condition = st.sidebar.selectbox(
    "🌐 Network Condition", 
    ['🟢 Excellent', '🟡 Good', '🟠 Fair', '🔴 Poor'],
    help="Select network quality for testing"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Protocol Information")

protocol_info = {
    'MODBUS': "Simple, lightweight protocol with minimal overhead",
    'DNP3': "Robust protocol designed for SCADA systems",
    'IEC61850': "Advanced standard for electrical substation automation"
}

for protocol in ['MODBUS', 'DNP3', 'IEC61850']:
    st.sidebar.markdown(f"""
    <div class="protocol-info">
        <strong>🔧 {protocol}</strong><br>
        <small>{protocol_info[protocol]}</small>
    </div>
    """, unsafe_allow_html=True)

simulator = ProtocolSimulator()

if st.sidebar.button("🚀 Run Analysis", type="primary", use_container_width=True):
    with st.spinner('🔄 Running protocol analysis...'):
        results = []
        for protocol in protocols:
            for cond in ['🟢 Excellent', '🟡 Good', '🟠 Fair', '🔴 Poor']:
                result = simulator.simulate(protocol, messages, cond)
                results.append(result)
        
        df = pd.DataFrame(results)
        
        # Current condition metrics
        st.markdown("## 📊 Real-Time Performance Metrics")
        current = df[df['condition'] == condition]
        
        if len(protocols) > 0:
            cols = st.columns(len(protocols))
            
            for i, protocol in enumerate(protocols):
                if protocol in current['protocol'].values:
                    data = current[current['protocol'] == protocol].iloc[0]
                    with cols[i]:
                        st.markdown(f"""
                        <div class="protocol-card">
                            <h3>{data['icon']} {protocol}</h3>
                            <h2>{data['success_rate']:.1%}</h2>
                            <p>Success Rate</p>
                            <hr style="border-color: rgba(255,255,255,0.3);">
                            <p><strong>{data['avg_delay']:.1f} ms</strong><br>Avg Delay</p>
                            <p><strong>{data['throughput']:.1f} KB/s</strong><br>Throughput</p>
                        </div>
                        """, unsafe_allow_html=True)
        
        # Network condition overview
        st.markdown("## 🌐 Network Condition Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            condition_colors = {
                '🟢 Excellent': '#4CAF50',
                '🟡 Good': '#FFC107', 
                '🟠 Fair': '#FF9800',
                '🔴 Poor': '#F44336'
            }
            
            st.markdown(f"""
            <div class="network-status">
                <h3>Current Network: {condition}</h3>
                <p>Testing with {messages:,} messages per protocol</p>
                <p><strong>Expected characteristics:</strong></p>
                <ul>
                    <li>Base latency: {10 if 'Excellent' in condition else 25 if 'Good' in condition else 50 if 'Fair' in condition else 100} ms</li>
                    <li>Packet loss: {1 if 'Excellent' in condition else 5 if 'Good' in condition else 10 if 'Fair' in condition else 20}%</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Quick stats
            best_protocol = df[df['condition'] == condition].loc[df[df['condition'] == condition]['success_rate'].idxmax()]
            worst_delay = df[df['condition'] == condition]['avg_delay'].max()
            
            st.markdown(f"""
            <div style="background: linear-gradient(135d, #d4edda 0%, #c3e6cb 100%); padding: 1rem; border-radius: 10px; border-left: 5px solid #28a745;">
                <h3>🏆 Best Performer</h3>
                <h2>{best_protocol['icon']} {best_protocol['protocol']}</h2>
                <p>{best_protocol['success_rate']:.1%} success rate</p>
                <p>Max delay observed: {worst_delay:.1f} ms</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Enhanced Charts
        st.markdown("## 📈 Performance Analysis Charts")
        
        # Delay comparison
        fig1 = go.Figure()
        
        for protocol in protocols:
            protocol_data = df[df['protocol'] == protocol]
            protocol_config = simulator.protocols[protocol]
            
            fig1.add_trace(go.Scatter(
                x=[cond.split(' ')[1] for cond in protocol_data['condition']],
                y=protocol_data['avg_delay'],
                mode='lines+markers',
                name=f"{protocol_config['icon']} {protocol}",
                line=dict(color=protocol_config['color'], width=4),
                marker=dict(size=12, symbol='circle')
            ))
        
        fig1.update_layout(
            title="⏱️ Average Delay Comparison Across Network Conditions",
            xaxis_title="Network Condition",
            yaxis_title="Average Delay (ms)",
            template="plotly_white",
            title_font_size=18,
            font=dict(size=14),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Success rate heatmap
        pivot_success = df.pivot(index='protocol', columns='condition', values='success_rate')
        fig2 = px.imshow(
            pivot_success,
            title="📊 Success Rate Heatmap",
            color_continuous_scale="RdYlGn",
            aspect="auto"
        )
        fig2.update_layout(
            title_font_size=18,
            font=dict(size=14),
            template="plotly_white"
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Throughput comparison
        fig3 = px.bar(
            df, 
            x='condition', 
            y='throughput', 
            color='protocol',
            title="🚀 Throughput Performance by Protocol",
            color_discrete_map={p: simulator.protocols[p]['color'] for p in protocols}
        )
        fig3.update_layout(
            template="plotly_white",
            title_font_size=18,
            font=dict(size=14),
            xaxis_title="Network Condition",
            yaxis_title="Throughput (KB/s)"
        )
        st.plotly_chart(fig3, use_container_width=True)
        
        # Detailed results table
        st.markdown("## 📋 Detailed Results")
        
        display_df = df[['protocol', 'condition', 'avg_delay', 'success_rate', 'throughput']].copy()
        display_df['avg_delay'] = display_df['avg_delay'].apply(lambda x: f"{x:.1f} ms")
        display_df['success_rate'] = display_df['success_rate'].apply(lambda x: f"{x:.1%}")
        display_df['throughput'] = display_df['throughput'].apply(lambda x: f"{x:.1f} KB/s")
        display_df.columns = ['🔗 Protocol', '🌐 Network', '⏱️ Avg Delay', '✅ Success Rate', '🚀 Throughput']
        
        st.dataframe(display_df, use_container_width=True, height=400)

else:
    st.markdown("## 👈 Configure settings and click 'Run Analysis'")
    st.markdown("### 🎯 What this analyzer does:")
    st.markdown("""
    - **Protocol Comparison:** Test MODBUS, DNP3, and IEC 61850 side-by-side
    - **Network Simulation:** Simulate different network conditions and their impact
    - **Performance Metrics:** Measure delay, success rate, and throughput
    - **Visual Analytics:** Interactive charts and heatmaps for easy comparison
    - **Real-world Scenarios:** Based on actual smart grid communication requirements
    """)
    
    # Demo chart when no analysis is running
    demo_data = pd.DataFrame({
        'Protocol': ['MODBUS', 'DNP3', 'IEC61850'],
        'Typical_Delay': [15, 25, 35],
        'Reliability': [85, 95, 98]
    })
    
    fig_demo = px.bar(demo_data, x='Protocol', y=['Typical_Delay', 'Reliability'],
                     title="📊 Protocol Characteristics (Demo)",
                     barmode='group')
    fig_demo.update_layout(template="plotly_white", title_font_size=18)
    st.plotly_chart(fig_demo, use_container_width=True)


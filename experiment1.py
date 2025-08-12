import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import random

st.set_page_config(page_title="Protocol Analyzer", layout="wide", page_icon="🔗")

# CSS styling
st.markdown("""
<style>
.css-1d391kg, section[data-testid="stSidebar"] {
    min-width: 420px !important;
    max-width: 420px !important;
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem; border-radius: 8px; text-align: center; color: white; margin: 0.5rem 0;
}
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem; border-radius: 10px; text-align: center; color: white; margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h2>🔗 Smart Grid Protocol Performance Analyzer</h2>
    <p>Compare MODBUS, DNP3 & IEC 61850 protocols</p>
</div>
""", unsafe_allow_html=True)

class ProtocolSimulator:
    def __init__(self):
        self.protocols = {
            'MODBUS': {'overhead': 12, 'reliability': 0.85, 'icon': '🔧'},
            'DNP3': {'overhead': 20, 'reliability': 0.95, 'icon': '⚡'},
            'IEC61850': {'overhead': 28, 'reliability': 0.98, 'icon': '🏭'}
        }
        
    def simulate(self, protocol, messages, condition):
        # REMOVED the seed setting from here - that was the problem!
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
        throughput = (successful * 64) / 1000
        
        return {
            'protocol': protocol,
            'condition': condition,
            'avg_delay': avg_delay,
            'success_rate': success_rate,
            'throughput': throughput,
            'icon': config['icon']
        }

# Sidebar
st.sidebar.header("🎛️ Configuration")

protocols = st.sidebar.multiselect(
    "Protocols:", 
    ['MODBUS', 'DNP3', 'IEC61850'], 
    ['MODBUS', 'DNP3'],
    help="Select one or more protocols for comparison"
)

messages = st.sidebar.slider(
    "Messages:", 
    100, 2000, 500, 50,
    help="Number of messages to simulate for each protocol"
)

condition = st.sidebar.selectbox(
    "Network:", 
    ['🟢 Excellent', '🟡 Good', '🟠 Fair', '🔴 Poor'],
    help="Network quality: Excellent(1% loss) to Poor(20% loss)"
)

reproducible = st.sidebar.checkbox(
    "🔒 Reproducible Results", 
    value=True,
    help="Use fixed random seed for consistent results"
)

# Protocol Specifications
st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Protocol Specifications")

st.sidebar.markdown("**🔧 MODBUS**")
st.sidebar.markdown("• Overhead: 12 bytes")
st.sidebar.markdown("• Reliability: 85%")
st.sidebar.markdown("• Use: Basic industrial automation")

st.sidebar.markdown("**⚡ DNP3**")
st.sidebar.markdown("• Overhead: 20 bytes")
st.sidebar.markdown("• Reliability: 95%")
st.sidebar.markdown("• Use: SCADA systems in utilities")

st.sidebar.markdown("**🏭 IEC61850**")
st.sidebar.markdown("• Overhead: 28 bytes")
st.sidebar.markdown("• Reliability: 98%")
st.sidebar.markdown("• Use: Advanced substation automation")

simulator = ProtocolSimulator()

if st.sidebar.button("🚀 Run Analysis", type="primary"):
    # SET SEED ONLY ONCE HERE - not for each protocol!
    if reproducible:
        random.seed(42)
        np.random.seed(42)
        st.success("🔒 Using fixed seed for consistent results")
    
    with st.spinner('🔄 Analyzing protocols...'):
        results = []
        for protocol in protocols:
            for cond in ['🟢 Excellent', '🟡 Good', '🟠 Fair', '🔴 Poor']:
                result = simulator.simulate(protocol, messages, cond)  # No seed parameter needed
                results.append(result)
        
        df = pd.DataFrame(results)
        
        # Current condition metrics
        st.markdown("## 📊 Performance Metrics")
        current = df[df['condition'] == condition]
        
        if len(protocols) > 0:
            cols = st.columns(len(protocols))
            for i, protocol in enumerate(protocols):
                if protocol in current['protocol'].values:
                    data = current[current['protocol'] == protocol].iloc[0]
                    with cols[i]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>{data['icon']} {protocol}</h4>
                            <h3>{data['success_rate']:.1%}</h3>
                            <p>Success Rate</p>
                            <p><strong>{data['avg_delay']:.1f} ms</strong> delay</p>
                            <p><strong>{data['throughput']:.1f} KB/s</strong> throughput</p>
                        </div>
                        """, unsafe_allow_html=True)
        
        # Network condition summary
        st.markdown("## 🌐 Network Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            **Current Network**: {condition}  
            **Test Messages**: {messages:,}  
            **Expected Loss**: {1 if 'Excellent' in condition else 5 if 'Good' in condition else 10 if 'Fair' in condition else 20}%
            """)
        
        with col2:
            best_protocol = current.loc[current['success_rate'].idxmax()]
            st.success(f"""
            **Best Performer**: {best_protocol['icon']} {best_protocol['protocol']}  
            **Success Rate**: {best_protocol['success_rate']:.1%}  
            **Avg Delay**: {best_protocol['avg_delay']:.1f} ms
            """)
        
        # Charts
        st.markdown("## 📈 Performance Charts")
        
        # Delay comparison
        fig1 = px.line(df, x='condition', y='avg_delay', color='protocol',
                      title="⏱️ Average Delay by Network Condition",
                      markers=True, line_shape='spline')
        fig1.update_layout(height=400)
        st.plotly_chart(fig1, use_container_width=True)
        
        # Success rate heatmap
        pivot_success = df.pivot(index='protocol', columns='condition', values='success_rate')
        fig2 = px.imshow(pivot_success, title="📊 Success Rate Heatmap",
                        color_continuous_scale="RdYlGn", text_auto=".1%")
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)
        
        # Throughput bar chart
        fig3 = px.bar(df, x='condition', y='throughput', color='protocol',
                     title="🚀 Throughput Performance", barmode='group')
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, use_container_width=True)
        
        # Results table
        st.markdown("## 📋 Detailed Results")
        display_df = df[['protocol', 'condition', 'avg_delay', 'success_rate', 'throughput']].copy()
        display_df['avg_delay'] = display_df['avg_delay'].apply(lambda x: f"{x:.1f} ms")
        display_df['success_rate'] = display_df['success_rate'].apply(lambda x: f"{x:.1%}")
        display_df['throughput'] = display_df['throughput'].apply(lambda x: f"{x:.1f} KB/s")
        display_df.columns = ['Protocol', 'Network', 'Avg Delay', 'Success Rate', 'Throughput']
        st.dataframe(display_df, use_container_width=True)

else:
    st.markdown("## 👈 Configure settings and click 'Run Analysis'")
    st.markdown("### 🎯 What this analyzer demonstrates:")
    st.markdown("""
    - **📡 Protocol Comparison**: Side-by-side testing of MODBUS, DNP3, and IEC 61850
    - **🌐 Network Simulation**: Performance under various network quality conditions
    - **📊 Performance Metrics**: Comprehensive measurement of delay, success rate, and throughput
    - **📈 Visual Analytics**: Interactive charts and heatmaps for easy interpretation
    - **🔒 Reproducible Testing**: Consistent results for academic and professional use
    """)
    
    st.info("💡 **Tip**: Start with 'Excellent' network conditions to understand baseline performance, then compare with other conditions to see protocol robustness.")

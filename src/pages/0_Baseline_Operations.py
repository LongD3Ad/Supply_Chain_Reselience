# src/pages/0_Baseline_Operations.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Baseline Operations")

st.title("📊 Baseline Supply Chain Operations")

st.markdown("""
Under normal conditions, supply chains focus on optimizing efficiency, minimizing defects, and maintaining service levels.
These baseline metrics serve as crucial reference points for evaluating performance deviations during disruptions and setting
continuous improvement targets, often guided by methodologies like Lean Six Sigma.
""")

st.divider()

# --- Key Performance Indicators (KPIs) ---
st.subheader("Example Baseline Key Performance Indicators (KPIs)")
col1, col2, col3 = st.columns(3)
col1.metric("Manufacturing Defect Rate", "10 / 10k units", help="Typical defects per 10,000 units produced.")
col2.metric("Avg. Order Processing Time", "1 hour", help="Average time from order placement to shipment initiation.")
col3.metric("Support Call Resolution", "5 minutes", help="Average time to resolve customer support inquiries.")

st.markdown("_(Note: These are illustrative examples. Actual baseline KPIs vary significantly by industry and company.)_")

st.divider()

# --- Lean Six Sigma Integration ---
st.subheader("Lean Six Sigma Context")
st.markdown("""
Baseline performance data is fundamental to Lean Six Sigma initiatives, which aim to reduce waste and process variability.
Organizations use these benchmarks to:
*   **Measure Current State:** Understand existing process capabilities (e.g., current sigma level).
*   **Set Improvement Goals:** Define specific, measurable targets (e.g., reduce processing time by 50%).
*   **Track Progress:** Continuously monitor KPIs against the baseline to gauge the effectiveness of improvement efforts.
*   **Strive for Excellence:** Aim for high sigma levels, such as Six Sigma's target of 3.4 Defects Per Million Opportunities (DPMO).
""")

# --- Illustrative Chart ---
st.subheader("Visualizing Baseline vs. Target")
# Example: Order Processing Time
baseline_time = 60 # minutes
target_time = 30  # minutes

fig = go.Figure(data=[
    go.Bar(name='Baseline', x=['Order Processing'], y=[baseline_time], text=[f"{baseline_time} min"], textposition='auto'),
    go.Bar(name='Target', x=['Order Processing'], y=[target_time], text=[f"{target_time} min"], textposition='auto')
])
fig.update_layout(
    title='Example: Baseline vs. Target Order Processing Time',
    yaxis_title='Time (Minutes)',
    barmode='group'
)
st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption("Source Information [1]: General industry benchmarks and Six Sigma principles.")
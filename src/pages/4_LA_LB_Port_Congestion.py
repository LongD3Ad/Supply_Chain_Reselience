# src/pages/4_LA_LB_Port_Congestion.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="LA/LB Port Congestion")

st.title("⚓ Port of Los Angeles/Long Beach Congestion (2024 Focus)")

st.markdown("""
The Ports of Los Angeles and Long Beach, the busiest container port complex in the U.S., have faced recurring
congestion challenges. While improvements have been made since the peak backlogs of 2021-2022, pressures
in early 2024 highlighted ongoing vulnerabilities.
""")

st.divider()

# --- Operational Metrics (Q1 2024) ---
st.subheader("Operational Metrics (Q1 2024 & Context)")
col1, col2, col3 = st.columns(3)
col1.metric("Q1 2024 TEU Volume", "1.6M TEUs", "+35% YoY", help="Total Twenty-Foot Equivalent Units handled.")
col2.metric("Avg. Rail Dwell Time", "5-8 Days", delta="> Target", delta_color="inverse", help="Time containers wait for rail transport (Target: 2-4 Days).")
col3.metric("Terminal Utilization", "75-80%", help="Percentage of container terminal capacity being used.")

# --- Dwell Time Visualization ---
dwell_data = pd.DataFrame({
    'Category': ['Rail Dwell Time'],
    'Actual (Avg)': [6.5], # Midpoint of 5-8
    'Target (Max)': [4.0]
})
fig_dwell = go.Figure()
fig_dwell.add_trace(go.Bar(name='Actual Avg (Q1 2024)', x=dwell_data['Category'], y=dwell_data['Actual (Avg)'], text=dwell_data['Actual (Avg)'], textposition='auto'))
fig_dwell.add_trace(go.Bar(name='Target Max', x=dwell_data['Category'], y=dwell_data['Target (Max)'], text=dwell_data['Target (Max)'], textposition='auto'))
fig_dwell.update_layout(title='Rail Dwell Time vs. Target (Days)', barmode='group', yaxis_title='Days')
st.plotly_chart(fig_dwell, use_container_width=True)


st.divider()

# --- Contributing Factors & Pressures ---
st.subheader("Contributing Factors & Pressures (Early 2024)")
st.markdown("""
*   **Strong Consumer Demand:** Sustained U.S. consumer spending (GDP growth ~3%) drove high import volumes, particularly from Asia.
*   **Panama Canal Restrictions:** Drought conditions significantly reduced canal transits (~40%), diverting an estimated **220,000 TEUs** to West Coast ports, adding pressure.
*   **Labor Challenges:** Ongoing negotiations and union resistance to terminal automation initiatives impacted throughput efficiency, even with available physical capacity (75-80% utilization suggests bottlenecks elsewhere).
*   **Inland Logistics Strain:** High volumes strained rail and trucking capacity, contributing to longer dwell times for containers leaving the port complex.
*   **Inventory Management:** Retailers potentially rebuilding inventories after post-pandemic drawdowns may have contributed to import surges.
""")

st.divider()

# --- Projected Risks ---
st.subheader("Projected Risks (Mid-2024 Outlook)")
st.warning("""
Analysts highlighted risks of significant congestion returning during the summer 2024 peak season, potentially mirroring
the severe backlogs of 2021-2022 if labor negotiations remained unresolved and import volumes stayed high, further
straining infrastructure capacity.
""", icon="⚠️")

st.divider()
st.caption("Source Information [5]: Based on reports covering Q1 2024 port performance and contributing factors.")
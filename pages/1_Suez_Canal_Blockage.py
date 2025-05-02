# src/pages/1_Suez_Canal_Blockage.py
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide", page_title="Suez Canal Blockage")

st.title("🚢 Suez Canal Blockage (March 2021)")

st.markdown("""
The grounding of the container ship *Ever Given* in the Suez Canal from March 23 to March 29, 2021,
caused an unprecedented blockage of one of the world's most critical maritime trade routes. This event
sent shockwaves through global supply chains, highlighting their vulnerability to chokepoint disruptions.
""")

st.divider()

# --- Incident Overview ---
st.subheader("Incident Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Date", "Mar 23-29, 2021")
col2.metric("Duration", "6 Days, 7 Hrs")
col3.metric("Vessel", "Ever Given (20k TEU)")
col4.metric("Location", "Suez Canal, Egypt")

# --- Economic and Operational Impact ---
st.subheader("Economic & Operational Impact")
col_a, col_b, col_c = st.columns(3)
col_a.metric("Est. Daily Trade Loss", "$9.6 Billion", help="Estimated value of global trade blocked per day.")
col_b.metric("Vessels Delayed", "~369", help="Number of ships queued by the end of the blockage.")
col_c.metric("Post-Incident Claim", "$1 Billion+", help="Initial compensation demanded by Egyptian authorities.")

st.divider()

# --- Timeline ---
st.subheader("Event Timeline")
# Data extracted from provided text [2]
timeline_data = pd.DataFrame([
    dict(Task="Ever Given Runs Aground", Start="2021-03-23", Finish="2021-03-23", Resource="Incident"),
    dict(Task="Blockage Persists", Start="2021-03-24", Finish="2021-03-28", Resource="Incident"),
    dict(Task="Initial Salvage Attempts", Start="2021-03-24", Finish="2021-03-28", Resource="Response"),
    dict(Task="Vessel Partially Refloated", Start="2021-03-29", Finish="2021-03-29", Resource="Response"),
    dict(Task="Ever Given Fully Refloated", Start="2021-03-29", Finish="2021-03-29", Resource="Response"),
    dict(Task="Canal Traffic Resumes", Start="2021-03-29", Finish="2021-03-29", Resource="Recovery"),
    dict(Task="Vessel Impounded", Start="2021-04-13", Finish="2021-07-07", Resource="Aftermath"), # Added impoundment period
    dict(Task="Backlog Cleared (Approx.)", Start="2021-03-30", Finish="2021-04-03", Resource="Recovery")
])
# Convert to datetime, add small duration for point events for visibility
timeline_data['Start'] = pd.to_datetime(timeline_data['Start'])
timeline_data['Finish'] = pd.to_datetime(timeline_data['Finish'])
timeline_data['Finish'] = timeline_data.apply(lambda row: row['Start'] + pd.Timedelta(hours=12) if row['Start'] == row['Finish'] else row['Finish'], axis=1)

fig_timeline = px.timeline(timeline_data, x_start="Start", x_end="Finish", y="Task",
                           color="Resource", title="Suez Canal Blockage: Timeline of Events")
fig_timeline.update_yaxes(autorange="reversed")
st.plotly_chart(fig_timeline, use_container_width=True)

st.divider()

# --- Supply Chain Consequences & Implications ---
st.subheader("Supply Chain Consequences")
st.markdown("""
*   **Shipping Delays & Backlogs:** Massive delays on the vital Asia-Europe lane, affecting goods ranging from consumer products to oil. Rerouting around Africa added ~1-2 weeks and significant fuel costs.
*   **Port Congestion:** Delayed vessels arrived at destination ports (e.g., Rotterdam, Felixstowe) in waves, overwhelming capacity and causing further delays.
*   **Increased Freight Costs:** Spot freight rates surged due to the effective reduction in shipping capacity and increased operational costs.
*   **Inventory Strain:** Disrupted Just-In-Time (JIT) systems and depleted buffer stocks for many industries relying on predictable lead times.
*   **Geopolitical & Strategic Reassessment:** Heightened focus on the vulnerability of maritime chokepoints and spurred discussions about alternative routes and supply chain diversification.
""")

st.divider()
st.caption("Source Information [2]: Based on reports regarding the 2021 Suez Canal blockage.")
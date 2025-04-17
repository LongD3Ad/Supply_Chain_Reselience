# src/pages/3_Russia_Ukraine_War.py
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide", page_title="Russia-Ukraine War Impacts")

st.title("⚔️ Russia-Ukraine War: Supply Chain Shocks")

st.markdown("""
The full-scale invasion of Ukraine by Russia in February 2022 triggered immediate and severe disruptions
across global supply chains, particularly impacting energy, agriculture, and fertilizer markets due to the
significant export roles of both countries.
""")

st.divider()

# --- Pre-War Export Significance ---
st.subheader("Pre-War Export Profiles (Share of National Exports)")
# Data for illustrative charts/metrics
ukraine_data = {'Category': ['Agriculture (Wheat, Corn etc.)', 'Manufactured Goods', 'Other'], 'Percentage': [46, 42, 12]}
russia_data = {'Category': ['Energy (Oil, Gas)', 'Fertilizers (Nitrogen)', 'Metals/Other', 'Other'], 'Percentage': [63, 5, 12, 20]} # Added rough split for fertilizer

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Ukraine:**")
    fig_ukr = px.pie(ukraine_data, values='Percentage', names='Category', title='Ukraine Exports (Pre-War)')
    st.plotly_chart(fig_ukr, use_container_width=True)
with col2:
    st.markdown("**Russia:**")
    fig_rus = px.pie(russia_data, values='Percentage', names='Category', title='Russia Exports (Pre-War)')
    st.plotly_chart(fig_rus, use_container_width=True)

st.markdown(f"- Russia accounted for ~**25%** of global nitrogen fertilizer production.")

st.divider()

# --- Post-Invasion Disruptions & Impacts ---
st.subheader("Post-Invasion Disruptions & Impacts")
col_a, col_b, col_c = st.columns(3)
col_a.metric("Global Wheat Supply Disrupted", "28%", help="Share of global wheat exports impacted.")
col_b.metric("Fertilizer Price Increase", "~300%", help="Peak increase in global fertilizer prices.")
col_c.metric("EU Gas Imports from Russia", "40% ➔ 17%", help="Reduction in EU reliance on Russian gas.")

st.markdown("""
*   **Food Security Crisis:** Disruption to Ukrainian grain exports (via Black Sea ports) and Russian fertilizer exports led to global food price spikes and shortages, severely impacting import-dependent nations in Africa and the Middle East.
*   **Energy Market Volatility:** Sanctions on Russia and weaponization of energy supplies caused extreme price volatility for oil and natural gas. Europe rapidly sought alternative suppliers (LNG from US, Qatar).
*   **Input Cost Inflation:** Soaring energy and fertilizer costs increased production expenses across agriculture and manufacturing globally.
*   **Logistics Rerouting:** Black Sea shipping became high-risk, and airspace closures over Russia/Ukraine forced costly rerouting for air cargo.
""")

st.divider()

# --- Long-Term Adjustments ---
st.subheader("Long-Term Supply Chain Adjustments")
st.markdown("""
The conflict accelerated trends towards supply chain regionalization and resilience:
*   **Energy Diversification:** Nations accelerated investments in renewable energy and sought non-Russian fossil fuel sources.
*   **Food Source Diversification:** Countries explored alternative grain suppliers and invested in domestic agriculture.
*   **Nearshoring/Friend-shoring:** Companies re-evaluated sourcing strategies, favoring geographically closer or geopolitically aligned partners (e.g., EU imports from Turkey/India increased **22%**).
*   **Increased Scrutiny:** Heightened awareness of geopolitical risks embedded within global supply chains.
""")

st.divider()
st.caption("Source Information [4]: Based on reports analyzing the economic and supply chain impacts of the Russia-Ukraine war.")
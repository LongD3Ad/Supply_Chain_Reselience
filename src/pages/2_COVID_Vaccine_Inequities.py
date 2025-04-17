# src/pages/2_COVID_Vaccine_Inequities.py
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide", page_title="COVID Vaccine Inequities")

st.title("💉 COVID-19 Vaccine Distribution Inequities")

st.markdown("""
The global rollout of COVID-19 vaccines, while a monumental scientific achievement, was marked by significant
distribution challenges and inequities, both between high- and low-income countries and within specific regions.
These disparities had profound public health and economic consequences.
""")

st.divider()

# --- Regional Case Study: St. Louis & Kansas City (Dec 2020 – Feb 2022) ---
st.subheader("Regional Case Study: St. Louis & Kansas City (Dec 2020 – Feb 2022)")
st.markdown("Data highlights disparities based on race/ethnicity and social vulnerability:")

col1, col2, col3 = st.columns(3)
col1.metric("Primary Series Completed", "1,763,036", help="Total individuals completing primary series in the study period.")
col2.metric("Booster Doses Administered", "872,324", help="Total booster doses administered.")
col3.metric("Cases in High SVI Areas", "25%", help="Percentage of COVID-19 cases located in high Social Vulnerability Index zip codes.")


st.markdown("**Inequity Metrics:**")
st.markdown("""
*   **Racial/Ethnic Disparity:** During early rollout phases, vaccination rates for Black and Hispanic individuals were less than half (<50%) those of their White counterparts in these areas.
*   **SVI Disparity:** High SVI zip codes received only **19.3%** of vaccinations despite accounting for **25%** of cases, indicating a mismatch between need and access.
""")
# Simple chart illustrating the SVI gap
svi_data = pd.DataFrame({
    'Metric': ['% of Cases', '% of Vaccinations'],
    'Percentage': [25.0, 19.3]
})
fig_svi = px.bar(svi_data, x='Metric', y='Percentage', title='Vaccination vs. Case Burden in High SVI Zip Codes',
                 text='Percentage', range_y=[0,30])
fig_svi.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
st.plotly_chart(fig_svi, use_container_width=True)


st.divider()

# --- Global Macroeconomic & Health Consequences ---
st.subheader("Global Consequences of Vaccine Inequity")
col_a, col_b = st.columns(2)
col_a.metric("Est. GDP Loss (Low-Income Countries, 2021)", "$38 Billion", help="Economic impact due to delayed vaccine access [6].")
col_b.metric("Childhood Immunization Drop", "30% (Measles/DTP3)", help="Decline in routine immunizations, risking secondary outbreaks [7].")

st.markdown("""
*   **Economic Impact:** Delayed access in low-income nations hampered economic recovery and exacerbated global inequalities.
*   **Public Health Setbacks:** Diversion of resources and healthcare system strain led to declines in other essential health services, like routine childhood immunizations.
*   **Prolonged Pandemic:** Uneven vaccination allowed the virus to continue spreading and mutating, potentially leading to new variants that challenged existing immunity.
""")

st.divider()
st.caption("Source Information [3, 6, 7]: Based on regional studies (St. Louis/KC) and global reports on vaccine equity impacts.")
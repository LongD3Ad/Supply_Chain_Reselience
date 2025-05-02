# src/tabs/risk_tab.py
import streamlit as st
import pandas as pd
import plotly.express as px
import logging

def render_tab(df_filtered, selected_run_file, start_date_filter, end_date_filter):
    """Renders the Risk Analysis tab."""
    st.header("Risk Analysis")

    if df_filtered.empty:
        if selected_run_file and start_date_filter <= end_date_filter: st.warning("No data available for the selected filters in this run.")
        elif not selected_run_file: st.info("Please select a simulation run file from the sidebar.")
        return # Stop rendering

    st.markdown("Analyzing the frequency and impact of simulated risk events across selected factories.")

    # Risk analysis code... (using 'active_risks' column)
    # (Copy risk analysis logic from original Tab 3 here)
    df_risk = df_filtered.copy()
    # Ensure 'active_risks' exists and handle potential non-string types robustly
    if 'active_risks' not in df_risk.columns:
         df_risk['active_risks'] = 'none' # Add dummy column if missing
    else:
         df_risk['active_risks'] = df_risk['active_risks'].fillna('none').astype(str)

    # Explode comma-separated risks
    df_risk['active_risks_list'] = df_risk['active_risks'].apply(lambda x: [r.strip() for r in x.split(',') if r.strip() and r.strip().lower() != 'none'])
    df_risk_exploded = df_risk.explode('active_risks_list')
    # Filter out rows that became NaN after exploding (originally 'none' or empty)
    df_risk_events = df_risk_exploded.dropna(subset=['active_risks_list']).copy()


    if df_risk_events.empty:
        st.info("No specific risk events identified in the data for the selected filters.")
        # Optionally, show impact analysis even without named risks
        df_impact_anon = df_risk[pd.to_numeric(df_risk.get('production_impact_factor'), errors='coerce').fillna(1.0) < 1.0]
        df_lt_anon = df_risk[pd.to_numeric(df_risk.get('lead_time_multiplier'), errors='coerce').fillna(1.0) > 1.0]
        if not df_impact_anon.empty or not df_lt_anon.empty:
             st.markdown("However, some weeks showed impacts:")
             if not df_impact_anon.empty:
                  st.metric("Weeks with Production Impact (<1.0)", f"{len(df_impact_anon)}")
             if not df_lt_anon.empty:
                  st.metric("Weeks with Lead Time Impact (>1.0)", f"{len(df_lt_anon)}")

    else:
        st.subheader("Risk Event Frequency (Active Weeks)")
        # Count occurrences of each *individual* risk type from the exploded list
        risk_counts = df_risk_events['active_risks_list'].value_counts().reset_index()
        risk_counts.columns = ['Risk Type', 'Frequency (Weeks Active)'] # Clarify this counts weeks where the risk was active
        fig_risk_freq = px.bar(risk_counts, x='Risk Type', y='Frequency (Weeks Active)', title="Frequency of Each Risk Type")
        st.plotly_chart(fig_risk_freq, use_container_width=True)

        # --- Impact Analysis ---
        st.subheader("Impact Analysis by Risk Type")
        # Ensure impact factor is numeric before filtering/plotting
        df_risk_events['production_impact_factor_numeric'] = pd.to_numeric(df_risk_events['production_impact_factor'], errors='coerce')
        df_impact = df_risk_events.dropna(subset=['production_impact_factor_numeric']).copy()
        # Filter for actual impact (<1.0) for production plot
        df_impact_filtered = df_impact[df_impact['production_impact_factor_numeric'] < 1.0]

        if not df_impact_filtered.empty:
             fig_impact_box = px.box(df_impact_filtered, x='active_risks_list', y='production_impact_factor_numeric',
                                    title="Production Impact Factor Distribution (Impact < 1.0)",
                                    labels={'active_risks_list': 'Risk Type', 'production_impact_factor_numeric': 'Impact Factor'},
                                    points="all")
             fig_impact_box.update_yaxes(range=[0, 1.1]) # Keep range 0-1.1
             st.plotly_chart(fig_impact_box, use_container_width=True)
        elif not df_impact.empty: # If risks exist but none had impact < 1.0
              st.info("Selected risk events did not result in a production impact factor less than 1.0.")
        else:
              st.info("No production impact data found for risk events in selection.")


        # Ensure lead time multiplier is numeric
        df_risk_events['lead_time_multiplier_numeric'] = pd.to_numeric(df_risk_events['lead_time_multiplier'], errors='coerce')
        df_lead_impact = df_risk_events.dropna(subset=['lead_time_multiplier_numeric']).copy()
        # Filter for actual impact (>1.0) for lead time plot
        df_lead_impact_filtered = df_lead_impact[df_lead_impact['lead_time_multiplier_numeric'] > 1.0]

        if not df_lead_impact_filtered.empty:
             fig_lead_box = px.box(df_lead_impact_filtered, x='active_risks_list', y='lead_time_multiplier_numeric',
                                 title="Lead Time Multiplier Distribution (Multiplier > 1.0)",
                                 labels={'active_risks_list': 'Risk Type', 'lead_time_multiplier_numeric': 'Lead Time Multiplier'},
                                 points="all")
             st.plotly_chart(fig_lead_box, use_container_width=True)
        elif not df_lead_impact.empty:
             st.info("Selected risk events did not result in a lead time multiplier greater than 1.0.")
        else:
             st.info("No lead time multiplier data found for risk events in selection.")


        st.subheader("Risk Events per Factory")
        # Group by original factory name and the exploded risk list
        try:
             # Count non-null risk types per factory
             risk_per_factory = df_risk_events.groupby('factory_name')['active_risks_list'].value_counts().unstack(fill_value=0)
             st.dataframe(risk_per_factory)
        except Exception as e:
             logging.warning(f"Could not generate risk per factory table: {e}")
             st.dataframe(df_risk_events[['factory_name', 'active_risks_list']].head()) # Show raw data instead
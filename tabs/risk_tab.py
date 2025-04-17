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
        elif not selected_run_file: st.info("Please select a simulation run file or upload data.")
        else: st.warning("Invalid date range selected.")
        return

    st.markdown("Analyzing the frequency and impact of simulated risk events across selected factories.")

    # --- Risk Analysis Logic (with checks) ---
    df_risk = df_filtered.copy()

    # Check for 'active_risks' before proceeding with risk-specific analysis
    if 'active_risks' not in df_risk.columns:
        st.warning("Risk analysis requires an 'active_risks' column, which is missing from the data.")
        # Optionally show fallback impact analysis here if desired
        # ... (code for df_impact_anon, df_lt_anon) ...
        return # Stop risk-specific analysis

    # Ensure 'active_risks' exists and handle NAs
    df_risk['active_risks'] = df_risk['active_risks'].fillna('none').astype(str)

    # Explode comma-separated risks
    try:
        df_risk['active_risks_list'] = df_risk['active_risks'].apply(lambda x: [r.strip() for r in x.split(',') if r.strip() and r.strip().lower() != 'none'])
        df_risk_exploded = df_risk.explode('active_risks_list')
        # Filter out rows that became NaN after exploding (originally 'none' or empty)
        df_risk_events = df_risk_exploded.dropna(subset=['active_risks_list']).copy()
    except Exception as e:
        st.error(f"Error processing 'active_risks' column: {e}")
        logging.error(f"Failed to process active_risks: {e}", exc_info=True)
        st.info("Could not analyze risk events due to an issue processing the 'active_risks' column.")
        df_risk_events = pd.DataFrame() # Ensure it's empty on error


    # --- Display Results ---
    if df_risk_events.empty:
        st.info("No specific risk events identified in the 'active_risks' column for the selected filters.")
        # Fallback impact analysis
        impact_col = 'production_impact_factor'
        lt_col = 'lead_time_multiplier'
        impact_data_exists = impact_col in df_risk.columns and df_risk[impact_col].notna().any()
        lt_data_exists = lt_col in df_risk.columns and df_risk[lt_col].notna().any()

        if impact_data_exists or lt_data_exists:
             st.markdown("However, analyzing overall weekly impacts:")
             if impact_data_exists:
                  df_impact_anon = df_risk[pd.to_numeric(df_risk[impact_col], errors='coerce').fillna(1.0) < 1.0]
                  if not df_impact_anon.empty:
                       st.metric("Weeks with Production Impact (<1.0)", f"{len(df_impact_anon)}")
             if lt_data_exists:
                  df_lt_anon = df_risk[pd.to_numeric(df_risk[lt_col], errors='coerce').fillna(1.0) > 1.0]
                  if not df_lt_anon.empty:
                       st.metric("Weeks with Lead Time Impact (>1.0)", f"{len(df_lt_anon)}")
    else:
        # Risk Frequency Plot
        st.subheader("Risk Event Frequency (Active Weeks)")
        risk_counts = df_risk_events['active_risks_list'].value_counts().reset_index()
        risk_counts.columns = ['Risk Type', 'Frequency (Weeks Active)']
        fig_risk_freq = px.bar(risk_counts, x='Risk Type', y='Frequency (Weeks Active)', title="Frequency of Each Risk Type")
        st.plotly_chart(fig_risk_freq, use_container_width=True)

        # Impact Analysis
        st.subheader("Impact Analysis by Risk Type")

        # Production Impact Plot
        impact_col = 'production_impact_factor'
        if impact_col in df_risk_events.columns:
            df_risk_events['production_impact_factor_numeric'] = pd.to_numeric(df_risk_events[impact_col], errors='coerce')
            df_impact = df_risk_events.dropna(subset=['production_impact_factor_numeric']).copy()
            df_impact_filtered = df_impact[df_impact['production_impact_factor_numeric'] < 1.0]
            if not df_impact_filtered.empty:
                 fig_impact_box = px.box(df_impact_filtered, x='active_risks_list', y='production_impact_factor_numeric', title="Production Impact Factor Distribution (Impact < 1.0)", labels={'active_risks_list': 'Risk Type', 'production_impact_factor_numeric': 'Impact Factor'}, points="all")
                 fig_impact_box.update_yaxes(range=[0, 1.1])
                 st.plotly_chart(fig_impact_box, use_container_width=True)
            elif not df_impact.empty:
                 st.info("Selected risk events did not result in a production impact factor less than 1.0.")
            else:
                 st.info("No valid production impact data found for the identified risk events.")
        else:
            st.info(f"Production impact analysis requires the '{impact_col}' column, which is missing.")

        # Lead Time Impact Plot
        lt_col = 'lead_time_multiplier'
        if lt_col in df_risk_events.columns:
            df_risk_events['lead_time_multiplier_numeric'] = pd.to_numeric(df_risk_events[lt_col], errors='coerce')
            df_lead_impact = df_risk_events.dropna(subset=['lead_time_multiplier_numeric']).copy()
            df_lead_impact_filtered = df_lead_impact[df_lead_impact['lead_time_multiplier_numeric'] > 1.0]
            if not df_lead_impact_filtered.empty:
                 fig_lead_box = px.box(df_lead_impact_filtered, x='active_risks_list', y='lead_time_multiplier_numeric', title="Lead Time Multiplier Distribution (Multiplier > 1.0)", labels={'active_risks_list': 'Risk Type', 'lead_time_multiplier_numeric': 'Lead Time Multiplier'}, points="all")
                 st.plotly_chart(fig_lead_box, use_container_width=True)
            elif not df_lead_impact.empty:
                 st.info("Selected risk events did not result in a lead time multiplier greater than 1.0.")
            else:
                 st.info("No valid lead time multiplier data found for the identified risk events.")
        else:
             st.info(f"Lead time impact analysis requires the '{lt_col}' column, which is missing.")


        # Risk Events Per Factory Table
        st.subheader("Risk Events per Factory")
        # Requires 'factory_name' which is essential, so we assume it exists if df_risk_events is not empty
        try:
             # --- Data Cleaning Step ---
             # Create a copy to avoid modifying the original df_risk_events if needed elsewhere
             df_factory_risk_count = df_risk_events[['factory_name', 'active_risks_list']].copy()

             # Clean the risk names: remove potential suffixes like (Active), (Recovery) and extra spaces
             # This regex removes space + parenthesis + anything inside + closing parenthesis
             df_factory_risk_count['cleaned_risk_type'] = df_factory_risk_count['active_risks_list'].str.replace(r'\s*\([^)]*\)', '', regex=True).str.strip()

             # --- Grouping and Counting ---
             # Now group by factory and the *cleaned* risk type
             risk_per_factory = df_factory_risk_count.groupby('factory_name')['cleaned_risk_type'].value_counts().unstack(fill_value=0)

             # Rename columns for better readability (optional, but nice)
             risk_per_factory.columns = [col.replace('_', ' ').title() for col in risk_per_factory.columns]

             st.dataframe(risk_per_factory, use_container_width=True)

        except Exception as e:
             logging.warning(f"Could not generate risk per factory table: {e}", exc_info=True)
             st.info("Could not generate risk per factory breakdown.")
             # Optionally show raw data as fallback
             # st.dataframe(df_risk_events[['factory_name', 'active_risks_list']].head())
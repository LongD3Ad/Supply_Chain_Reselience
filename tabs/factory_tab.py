# src/tabs/factory_tab.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import logging # Added for logging potential missing columns

def render_tab(df_filtered, selected_run_file, start_date_filter, end_date_filter):
    """Renders the Factory Deep Dive tab."""
    st.header("Factory Deep Dive")

    # Initial check for filtered data
    if df_filtered.empty:
        if selected_run_file and start_date_filter <= end_date_filter: st.warning("No data available for the selected filters in this run.")
        elif not selected_run_file: st.info("Please select a simulation run file or upload data.")
        else: st.warning("Invalid date range selected.") # Should be caught by app.py
        return # Stop rendering

    st.markdown("Select a specific factory below to see its detailed performance metrics.")
    factories_in_filtered_data = sorted(df_filtered['factory_name'].unique().tolist())

    if not factories_in_filtered_data:
         st.warning("No factories match the current filter selection.")
         return # Stop rendering

    # Factory selection dropdown
    current_selection = st.session_state.get("hist_factory_inspect", None)
    if current_selection not in factories_in_filtered_data:
         st.session_state.hist_factory_inspect = factories_in_filtered_data[0]
    factory_to_inspect = st.selectbox(
        "Select Factory for Details:",
        options=factories_in_filtered_data,
        key="hist_factory_inspect"
    )

    if factory_to_inspect:
        df_factory = df_filtered[df_filtered['factory_name'] == factory_to_inspect].copy()
        st.subheader(f"Details for: {factory_to_inspect}")

        if df_factory.empty:
            st.warning("No data available for this specific factory within the selected date range.")
            return

        # --- KPIs (Calculate safely) ---
        f_total_prod = df_factory['adjusted_production_units'].sum() if 'adjusted_production_units' in df_factory else 0
        f_avg_lead = df_factory['lead_time_days'].mean() if 'lead_time_days' in df_factory else None
        f_total_shortage = df_factory['shortage_units'].sum() if 'shortage_units' in df_factory else 0
        f_avg_inv = df_factory['inventory_level_units'].mean() if 'inventory_level_units' in df_factory else None

        fcol1, fcol2, fcol3, fcol4 = st.columns(4)
        fcol1.metric("Total Production", f"{f_total_prod:,.0f} units")
        fcol2.metric("Avg Lead Time", f"{f_avg_lead:.1f} days" if pd.notna(f_avg_lead) else "N/A")
        fcol3.metric("Total Shortage", f"{f_total_shortage:,.0f} units")
        fcol4.metric("Avg Inventory", f"{f_avg_inv:,.0f} units" if pd.notna(f_avg_inv) else "N/A")

        # --- Plotting (with checks) ---

        # Production Plot (Assumes adjusted_production_units exists based on core checks)
        if 'adjusted_production_units' in df_factory.columns:
            fig_f_prod = px.line(df_factory, x='date', y='adjusted_production_units', title='Adjusted Production', labels={'adjusted_production_units':'Units Produced'})
            fig_f_prod.update_layout(hovermode="x unified")
            # Add risk shading only if 'active_risks' column exists and has events
            if 'active_risks' in df_factory.columns and (df_factory['active_risks'] != 'none').any():
                risk_events = df_factory[df_factory['active_risks'] != 'none']
                for _, event_week in risk_events.iterrows():
                    risk_names = event_week['active_risks']
                    fig_f_prod.add_vrect(x0=event_week['date']-pd.Timedelta(days=3.5), x1=event_week['date']+pd.Timedelta(days=3.5), fillcolor="rgba(255,0,0,0.1)", layer="below", line_width=0, annotation_text=risk_names[:20], annotation_position="top left")
            st.plotly_chart(fig_f_prod, use_container_width=True)
        else:
            st.info("Production data ('adjusted_production_units') not available.")

        # Inventory Plot (Assumes inventory_level_units exists)
        if 'inventory_level_units' in df_factory.columns:
            fig_f_inv = px.line(df_factory, x='date', y='inventory_level_units', title='Inventory Level', labels={'inventory_level_units':'Units'})
            # Check for safety stock before adding trace
            if 'safety_stock_level' in df_factory.columns and df_factory['safety_stock_level'].notna().any():
                fig_f_inv.add_trace(go.Scatter(x=df_factory['date'], y=df_factory['safety_stock_level'], mode='lines', name='Safety Stock', line=dict(dash='dash', color='grey')))
            fig_f_inv.update_layout(hovermode="x unified")
            st.plotly_chart(fig_f_inv, use_container_width=True)
        else:
            st.info("Inventory data ('inventory_level_units') not available.")

        # Lead Time Plot
        if 'lead_time_days' in df_factory.columns and df_factory['lead_time_days'].notna().any():
            fig_f_lead = px.line(df_factory, x='date', y='lead_time_days', title='Lead Time', labels={'lead_time_days':'Days'})
            fig_f_lead.update_layout(hovermode="x unified")
            st.plotly_chart(fig_f_lead, use_container_width=True)
        else:
            st.info("Lead time data ('lead_time_days') not available.")

        # Shortage Plot
        if 'shortage_units' in df_factory.columns and df_factory['shortage_units'].notna().any():
            # Only plot if there are actual shortages to show, otherwise it's blank
            if df_factory['shortage_units'].sum() > 0:
                 fig_f_short = px.bar(df_factory, x='date', y='shortage_units', title='Shortage Events', labels={'shortage_units':'Unmet Demand (Units)'})
                 fig_f_short.update_layout(hovermode="x unified")
                 st.plotly_chart(fig_f_short, use_container_width=True)
            else:
                 st.info("No shortage events occurred for this factory in the selected period.")
        else:
            st.info("Shortage data ('shortage_units') not available.")

        # --- Risk summary table ---
        st.subheader("Weekly Risk Summary")
        required_risk_cols = ['date', 'active_risks', 'production_impact_factor', 'adjusted_production_units', 'lead_time_days', 'lead_time_multiplier']
        if all(col in df_factory.columns for col in required_risk_cols):
            # Filter only rows where risks are active
            risk_summary_df = df_factory[df_factory['active_risks'] != 'none'][required_risk_cols].copy()
            risk_summary_df.rename(columns={
                'active_risks': 'Active Risks', 'production_impact_factor': 'Avg Prod Impact',
                'lead_time_multiplier': 'Avg LT Multiplier', 'adjusted_production_units': 'Production',
                'lead_time_days': 'Lead Time'
                }, inplace=True)

            if not risk_summary_df.empty:
                st.dataframe(risk_summary_df, use_container_width=True)
            else:
                st.info("No weeks with active risks recorded for this factory in the selected period.")
        else:
            missing_cols = [col for col in required_risk_cols if col not in df_factory.columns]
            st.info(f"Cannot display weekly risk summary because required data is missing: {', '.join(missing_cols)}")
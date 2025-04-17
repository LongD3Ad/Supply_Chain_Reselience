# src/tabs/summary_tab.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import logging # Added for logging potential errors

def render_tab(df_filtered, selected_run_file, start_date_filter, end_date_filter, selected_factories):
    """Renders the Overall Summary tab."""
    st.header("Overall Supply Chain Performance")

    # Initial check for filtered data
    if df_filtered.empty:
        # More specific messages based on why it might be empty
        if selected_run_file and start_date_filter <= end_date_filter:
             st.warning("No data available for the selected filters in this run.")
        elif not selected_run_file:
             st.info("Please select a simulation run file or upload data.")
        else: # Handles case where dates might be invalid upstream
             st.warning("Invalid date range selected or no data source available.")
        return # Stop rendering if no data

    # Display context info
    source_desc = selected_run_file if selected_run_file else "selected data"
    st.markdown(f"Aggregated data for **{len(selected_factories)}** selected factories from **{start_date_filter}** to **{end_date_filter}** (Source: `{source_desc}`).")

    # --- KPIs (Calculate safely based on available columns) ---
    total_production = df_filtered['adjusted_production_units'].sum() if 'adjusted_production_units' in df_filtered.columns else 0
    avg_lead_time = df_filtered['lead_time_days'].mean() if 'lead_time_days' in df_filtered.columns else None
    total_shortage = df_filtered['shortage_units'].sum() if 'shortage_units' in df_filtered.columns else 0
    avg_inventory = df_filtered['inventory_level_units'].mean() if 'inventory_level_units' in df_filtered.columns else None

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Production", f"{total_production:,.0f} units")
    col2.metric("Average Lead Time", f"{avg_lead_time:.1f} days" if pd.notna(avg_lead_time) else "N/A")
    col3.metric("Total Shortage", f"{total_shortage:,.0f} units")
    col4.metric("Average Inventory", f"{avg_inventory:,.0f} units" if pd.notna(avg_inventory) else "N/A")


    # --- Charts ---
    st.subheader("Performance Over Time")

    # Define potential aggregations, mapping original columns to desired aggregated names and functions
    aggregations = {
        'total_adjusted_production': ('adjusted_production_units', 'sum'),
        'average_lead_time': ('lead_time_days', 'mean'),
        'total_inventory': ('inventory_level_units', 'sum'),
        'total_shortage': ('shortage_units', 'sum')
    }

    # Check which *original* columns are actually available in df_filtered
    available_original_cols = [
        orig_col for agg_name, (orig_col, _) in aggregations.items() if orig_col in df_filtered.columns
    ]

    # Build the final aggregation dictionary only using available columns
    final_agg_dict = {
        agg_name: (orig_col, agg_func)
        for agg_name, (orig_col, agg_func) in aggregations.items()
        if orig_col in available_original_cols
    }

    # Check if aggregation is possible
    if not final_agg_dict or 'date' not in df_filtered.columns:
        st.warning("Cannot generate summary charts because essential data (like 'date' or core metrics) is missing or could not be identified.")
        return # Stop rendering charts if no data to aggregate

    # Perform aggregation
    try:
        df_agg = df_filtered.groupby('date').agg(**final_agg_dict).reset_index()
    except Exception as e:
        st.error(f"Failed to aggregate data for charts: {e}")
        logging.error(f"Aggregation failed: {e}", exc_info=True)
        return # Cannot proceed if aggregation fails


    # --- Production Chart ---
    prod_col_agg = 'total_adjusted_production'
    # Check if the *aggregated* column exists and has valid data
    if prod_col_agg in df_agg.columns and df_agg[prod_col_agg].notna().any():
        fig_prod = px.line(df_agg, x='date', y=prod_col_agg, title='Total Production Over Time', labels={prod_col_agg:'Units Produced'})
        fig_prod.update_layout(hovermode="x unified")
        st.plotly_chart(fig_prod, use_container_width=True)
    # Only show info message if the original column was available but chart failed
    elif 'adjusted_production_units' in available_original_cols:
        st.info("Production data ('adjusted_production_units') not available or invalid for charting after aggregation.")


    # --- Inventory vs. Shortage Chart ---
    inv_col_agg = 'total_inventory'
    short_col_agg = 'total_shortage'
    # Check if the *aggregated* columns exist and have data
    inv_present = inv_col_agg in df_agg.columns and df_agg[inv_col_agg].notna().any()
    short_present = short_col_agg in df_agg.columns and df_agg[short_col_agg].notna().any()

    if inv_present or short_present:
        fig_inv_short = go.Figure()
        if inv_present:
            fig_inv_short.add_trace(go.Scatter(x=df_agg['date'], y=df_agg[inv_col_agg], mode='lines', name='Total Inventory'))
        if short_present:
            fig_inv_short.add_trace(go.Scatter(x=df_agg['date'], y=df_agg[short_col_agg], mode='lines', name='Total Shortage', line=dict(color='red')))
        fig_inv_short.update_layout(title='Inventory vs. Shortage Over Time', hovermode="x unified", yaxis_title="Units")
        st.plotly_chart(fig_inv_short, use_container_width=True)
    else:
        # Check if the original columns were available but charting failed
        missing_info = []
        if 'inventory_level_units' not in available_original_cols: missing_info.append("'inventory_level_units'")
        if 'shortage_units' not in available_original_cols: missing_info.append("'shortage_units'")
        # Only show message if *at least one* original column was missing
        if missing_info:
             st.info(f"Inventory/Shortage data ({', '.join(missing_info)}) not available in the source data.")
        # If original columns existed but aggregation failed/resulted in NaNs (unlikely with sum, but possible)
        elif 'inventory_level_units' in available_original_cols or 'shortage_units' in available_original_cols:
             st.info("Inventory and/or Shortage data could not be plotted after aggregation.")


    # --- Lead Time Chart (Corrected Check) ---
    lead_time_col_agg = 'average_lead_time'
    # Check if the *aggregated* column exists and has valid (non-NaN) data
    if lead_time_col_agg in df_agg.columns and df_agg[lead_time_col_agg].notna().any():
        fig_lead = px.line(df_agg, x='date', y=lead_time_col_agg, title='Average Lead Time Over Time', labels={lead_time_col_agg:'Days'})
        fig_lead.update_layout(hovermode="x unified")
        st.plotly_chart(fig_lead, use_container_width=True)
    # Display message only if the *original* 'lead_time_days' column was present in the input data
    elif 'lead_time_days' in available_original_cols:
        st.info("Lead time data ('lead_time_days') not available or invalid for charting after aggregation.")
    # Else: If 'lead_time_days' wasn't even in df_filtered, don't show a message for this specific chart.
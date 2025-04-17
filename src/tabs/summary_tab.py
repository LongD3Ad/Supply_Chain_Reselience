# src/tabs/summary_tab.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def render_tab(df_filtered, selected_run_file, start_date_filter, end_date_filter, selected_factories):
    """Renders the Overall Summary tab."""
    st.header("Overall Supply Chain Performance")

    if df_filtered.empty:
        # More specific messages based on why it might be empty
        if selected_run_file and start_date_filter <= end_date_filter:
             st.warning("No data available for the selected filters in this run.")
        elif not selected_run_file:
             st.info("Please select a simulation run file from the sidebar.")
        # Date range error is handled in app.py before calling this
        # else: st.warning("Invalid date range selected.") # Should not be reached if app.py handles it
        return # Stop rendering if no data

    st.markdown(f"Aggregated data for **{len(selected_factories)}** selected factories from **{start_date_filter}** to **{end_date_filter}** (Run: `{selected_run_file}`).")

    # --- KPIs ---
    # (Copy KPI calculation and st.metric calls from original Tab 1 here)
    total_production = df_filtered['adjusted_production_units'].sum()
    avg_lead_time = df_filtered['lead_time_days'].mean()
    total_shortage = df_filtered['shortage_units'].sum()
    avg_inventory = df_filtered['inventory_level_units'].mean()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Production", f"{total_production:,.0f} units")
    col2.metric("Average Lead Time", f"{avg_lead_time:.1f} days" if pd.notna(avg_lead_time) else "N/A")
    col3.metric("Total Shortage", f"{total_shortage:,.0f} units")
    col4.metric("Average Inventory", f"{avg_inventory:,.0f} units" if pd.notna(avg_inventory) else "N/A")


    # --- Charts ---
    st.subheader("Performance Over Time")
    # (Copy chart generation code from original Tab 1 here)
    # Aggregate data for charts
    df_agg = df_filtered.groupby('date').agg(
        total_adjusted_production=('adjusted_production_units', 'sum'),
        average_lead_time=('lead_time_days', 'mean'),
        total_inventory=('inventory_level_units', 'sum'),
        total_shortage=('shortage_units', 'sum')
    ).reset_index()

    # Plotting code...
    fig_prod = px.line(df_agg, x='date', y='total_adjusted_production', title='Total Production Over Time', labels={'total_adjusted_production':'Units Produced'})
    fig_prod.update_layout(hovermode="x unified")
    st.plotly_chart(fig_prod, use_container_width=True)

    fig_inv_short = go.Figure()
    fig_inv_short.add_trace(go.Scatter(x=df_agg['date'], y=df_agg['total_inventory'], mode='lines', name='Total Inventory'))
    fig_inv_short.add_trace(go.Scatter(x=df_agg['date'], y=df_agg['total_shortage'], mode='lines', name='Total Shortage', line=dict(color='red')))
    fig_inv_short.update_layout(title='Inventory vs. Shortage Over Time', hovermode="x unified", yaxis_title="Units")
    st.plotly_chart(fig_inv_short, use_container_width=True)

    fig_lead = px.line(df_agg, x='date', y='average_lead_time', title='Average Lead Time Over Time', labels={'average_lead_time':'Days'})
    fig_lead.update_layout(hovermode="x unified")
    st.plotly_chart(fig_lead, use_container_width=True)
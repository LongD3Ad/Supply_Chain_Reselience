# src/tabs/sim_results_modal.py
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Dict, Any, Optional
import logging
import numpy as np
# import json # json import was not used

def display_simulation_results_modal(
    timeseries_data: Optional[List[Dict[str, Any]]] = None,
    recommendations: Optional[List[Dict[str, Any]]] = None,
    risk_summary: Optional[Dict[str, Any]] = None,
    financial_summary: Optional[Dict[str, Any]] = None
):
    """Displays simulation results in a modal dialog with Charts, Recs, Risk, and Financials."""

    if not timeseries_data and not recommendations and not risk_summary and not financial_summary:
        st.warning("No simulation analysis data provided to display.")
        return

    tab_titles = []
    if timeseries_data: tab_titles.append("📊 Charts")
    if financial_summary: tab_titles.append("💰 Financials")
    if recommendations: tab_titles.append("💡 Recommendations")
    if risk_summary: tab_titles.append("🚨 Risk Info")

    if not tab_titles:
         st.error("Internal Error: No displayable content tabs.")
         return

    tabs = st.tabs(tab_titles)
    tab_index = 0

    # --- Charts Tab ---
    if timeseries_data:
        with tabs[tab_index]:
            st.subheader("Simulation Operational Charts")
            try:
                df = pd.DataFrame(timeseries_data)
                if not df.empty and 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    df.dropna(subset=['date'], inplace=True) # Drop rows where date conversion failed

                    if not df.empty:
                        fig = make_subplots(specs=[[{"secondary_y": True}]])
                        plot_made = False

                        if 'inventory_level_units' in df.columns and df['inventory_level_units'].notna().any():
                            fig.add_trace(go.Scatter(x=df['date'], y=df['inventory_level_units'], name="Inventory", line=dict(color='#1f77b4')), secondary_y=False)
                            plot_made = True
                        
                        if 'shortage_units' in df.columns and df['shortage_units'].notna().any():
                            shortage_series = pd.to_numeric(df['shortage_units'], errors='coerce').fillna(0)
                            if shortage_series.sum() > 0:
                                fig.add_trace(go.Bar(x=df['date'], y=shortage_series, name="Shortage", marker_color='#d62728', opacity=0.7), secondary_y=True)
                                plot_made = True
                            else:
                                st.caption("No shortage occurred.")
                        
                        # --- FIXED SECTION FOR VLINES ---
                        if risk_summary and risk_summary.get('applied_risk_events'):
                             applied_events_list = risk_summary.get('applied_risk_events', [])
                             severities_dict = risk_summary.get('risk_severity', {})
                             num_points = len(df)
                             num_risks_applied = len(applied_events_list)

                             if num_risks_applied > 0 and num_points > 0:
                                 # Distribute markers roughly across the timeline
                                 # Ensure at least num_risks_applied points if num_points is small
                                 effective_points_for_linspace = max(num_points, num_risks_applied)
                                 indices = np.linspace(0, effective_points_for_linspace - 1, num_risks_applied, dtype=int)
                                 # Clip indices to be within the actual DataFrame bounds
                                 indices = np.clip(indices, 0, num_points - 1)
                                 # Ensure unique indices if multiple risks map to the same point due to few data points
                                 indices = sorted(list(set(indices)))
                                 
                                 plotted_risk_count = 0
                                 for i, risk_event_name in enumerate(applied_events_list):
                                     if i < len(indices): # Ensure we don't go out of bounds for indices
                                         idx = indices[i]
                                         severity = severities_dict.get(risk_event_name, 0.5)
                                         
                                         # Use numeric dates for vlines instead of timestamp objects
                                         # This avoids the addition/subtraction issue with Timestamp objects
                                         date_val = df['date'].iloc[idx].timestamp() * 1000  # Convert to milliseconds for plotly
                                         
                                         fig.add_shape(
                                             type="line",
                                             x0=date_val,
                                             y0=0,
                                             x1=date_val,
                                             y1=1,
                                             yref="paper",
                                             line=dict(
                                                 color="grey",
                                                 width=1,
                                                 dash="dash",
                                             )
                                         )
                                         
                                         # Add annotation separately
                                         fig.add_annotation(
                                             x=date_val,
                                             y=1,
                                             yref="paper",
                                             text=f"{risk_event_name[:10]} ({severity:.1f})",
                                             showarrow=False,
                                             font=dict(size=9),
                                             xanchor="left",
                                             yanchor="bottom"
                                         )
                                         
                                         plotted_risk_count += 1
                                     else:
                                        logging.warning(f"Not enough unique indices to plot all risk events. Risk '{risk_event_name}' skipped for vline.")
                                 if plotted_risk_count < num_risks_applied:
                                     st.caption(f"Note: Some risk event vlines might overlap or be closely spaced due to data granularity.")


                        if plot_made:
                            fig.update_layout(
                                title="Inventory & Shortage Levels Over Time",
                                height=350,
                                margin=dict(l=20, r=20, t=50, b=20), # Adjusted top margin for title
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                            )
                            fig.update_yaxes(title_text="Inventory (Units)", secondary_y=False)
                            fig.update_yaxes(title_text="Shortage (Units)", secondary_y=True, showgrid=False)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No plottable inventory or shortage data found in the simulation results.")

                        try:
                            csv_data = df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Download Chart Data",
                                data=csv_data,
                                file_name="sim_chart_data.csv",
                                mime="text/csv"
                            )
                        except Exception as e:
                            logging.error(f"CSV download preparation failed: {e}")
                    else:
                        st.warning("No valid data available for charting after date processing.")
                else:
                    st.warning("Simulation timeseries data is empty or missing the 'date' column.")
            except Exception as e_chart_tab:
                st.error(f"An error occurred while generating charts: {e_chart_tab}")
                logging.error(f"Error in Charts Tab of Modal: {e_chart_tab}", exc_info=True)
        tab_index += 1

    # --- Financials Tab ---
    if financial_summary:
        with tabs[tab_index]:
            st.subheader("Simulation Financial Summary")
            if financial_summary.get("calculation_error"):
                 st.error(f"Financial calculation encountered an error: {financial_summary['calculation_error']}")
            else:
                 col_h, col_s, col_l, col_p, col_t = st.columns(5)
                 col_h.metric("Holding Cost", f"${financial_summary.get('total_holding_cost', 0.0):,.2f}")
                 col_s.metric("Stockout Cost", f"${financial_summary.get('total_stockout_cost', 0.0):,.2f}")
                 col_l.metric("Logistics Cost (Est)", f"${financial_summary.get('estimated_total_logistics_cost', 0.0):,.2f}")
                 col_p.metric("Production Cost (Est)", f"${financial_summary.get('estimated_total_production_cost', 0.0):,.2f}")
                 col_t.metric("Total SC Cost (Est)", f"${financial_summary.get('estimated_total_supply_chain_cost', 0.0):,.2f}")
                 st.markdown("---")
                 st.caption("Costs are estimated based on total units and provided cost rates. Does not include fixed costs or dynamic pricing.")
        tab_index += 1

    # --- Recommendations Tab ---
    if recommendations:
         with tabs[tab_index]:
             st.subheader("AI-Powered Recommendations")
             if recommendations: # Check if list is not empty
                  categories = sorted(list(set(rec.get('category', 'Unknown') for rec in recommendations)))
                  selected_cat = st.selectbox("Filter Category:", ["All"] + categories, key="modal_rec_cat_filter")
                  
                  filtered_recs = [
                      r for r in recommendations 
                      if selected_cat == "All" or r.get('category') == selected_cat
                  ]
                  filtered_recs.sort(key=lambda x: x.get('priority', 0), reverse=True) # Sort by priority

                  if not filtered_recs:
                      st.info("No recommendations match the current filter.")
                  else:
                      for rec in filtered_recs:
                           prio = rec.get("priority", 0)
                           cat = rec.get("category", "Info")
                           txt = rec.get("text", "N/A")
                           p_map = {5:"critical", 4:"high", 3:"medium", 2:"low", 1:"informational", 0:"unknown"} # Added 0 mapping
                           css_class = f"recommendation-{p_map.get(prio, 'unknown')}"
                           st.markdown(f'<div class="{css_class}"><strong>{cat} (Priority: {prio}):</strong> {txt}</div>', unsafe_allow_html=True)
             else:
                 st.info("No recommendations available for this simulation.")
         tab_index += 1

    # --- Risk Info Tab ---
    if risk_summary:
        with tabs[tab_index]:
            st.subheader("Applied Risk Scenario Information")
            applied_events = risk_summary.get('applied_risk_events')
            severities = risk_summary.get('risk_severity')
            
            if applied_events:
                st.markdown("**Risk Events Applied in Simulation:**")
                # Create a list of dictionaries for better DataFrame display
                risk_data_list = []
                for r_event in applied_events:
                    risk_data_list.append({
                        "Risk Event": r_event,
                        "Severity": f"{severities.get(r_event, 'N/A'):.1f}" if isinstance(severities.get(r_event), (int, float)) else str(severities.get(r_event, 'N/A'))
                    })
                if risk_data_list:
                    st.dataframe(pd.DataFrame(risk_data_list), use_container_width=True, hide_index=True)
                else:
                    st.caption("No detailed risk event data to display.")

            effective_factors = risk_summary.get('effective_risk_factors_applied')
            if effective_factors:
                 st.markdown("**Effective Simulation Risk Parameters (Overrides due to Risks):**")
                 factor_display_list = []
                 for risk_type, params_dict in effective_factors.items():
                      for param_name, param_value in params_dict.items():
                           factor_display_list.append({
                               "Risk Factor Category": risk_type,
                               "Parameter Modified": param_name,
                               "New Value": str(param_value) # Ensure value is string for display
                           })
                 if factor_display_list:
                     st.dataframe(pd.DataFrame(factor_display_list), use_container_width=True, hide_index=True)
                 else:
                     st.info("No specific risk parameters were programmatically overridden by the applied risks.")
            
            market_factor_applied_msg = risk_summary.get('market_factor_applied')
            if market_factor_applied_msg:
                 st.markdown(f"**Market Data Impact Considered:** {market_factor_applied_msg}")

            if not applied_events and not effective_factors and not market_factor_applied_msg:
                 st.info("No specific risk events, parameter overrides, or market factors were applied to this simulation run.")
        tab_index += 1
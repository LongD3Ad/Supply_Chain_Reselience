# src/tabs/sim_results_modal.py
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Dict, Any, Optional
import logging
import numpy as np
import json

def display_simulation_results_modal(
    timeseries_data: Optional[List[Dict[str, Any]]] = None,
    recommendations: Optional[List[Dict[str, Any]]] = None,
    risk_summary: Optional[Dict[str, Any]] = None,
    financial_summary: Optional[Dict[str, Any]] = None # <--- NEW: Accept financial summary
):
    """Displays simulation results in a modal dialog with Charts, Recs, Risk, and Financials."""

    # Check for any data before proceeding
    if not timeseries_data and not recommendations and not risk_summary and not financial_summary:
        st.warning("No simulation analysis data provided to display.")
        return

    # --- Define Tabs Based on Available Data ---
    tab_titles = []
    if timeseries_data: tab_titles.append("📊 Charts")
    if financial_summary: tab_titles.append("💰 Financials") # <--- NEW Tab
    if recommendations: tab_titles.append("💡 Recommendations")
    if risk_summary: tab_titles.append("🚨 Risk Info")

    if not tab_titles: # Should not happen if data is present, but safety
         st.error("Internal Error: No displayable content tabs.")
         return

    tabs = st.tabs(tab_titles)
    tab_index = 0 # Counter for iterating through tabs

    # --- Charts Tab ---
    if timeseries_data:
        with tabs[tab_index]:
            st.subheader("Simulation Operational Charts") # More specific title
            df = pd.DataFrame(timeseries_data)
            if not df.empty and 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df.dropna(subset=['date'], inplace=True)

                if not df.empty:
                    # Create figure with subplots
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    plot_made = False

                    # Inventory Plot
                    if 'inventory_level_units' in df.columns and df['inventory_level_units'].notna().any():
                        fig.add_trace(go.Scatter(x=df['date'], y=df['inventory_level_units'], name="Inventory", line=dict(color='#1f77b4')), secondary_y=False); plot_made = True
                    # Shortage Plot
                    if 'shortage_units' in df.columns and df['shortage_units'].notna().any():
                        shortage_series = pd.to_numeric(df['shortage_units'], errors='coerce').fillna(0)
                        if shortage_series.sum() > 0:
                            fig.add_trace(go.Bar(x=df['date'], y=shortage_series, name="Shortage", marker_color='#d62728', opacity=0.7), secondary_y=True); plot_made = True
                        else: st.caption("No shortage occurred.")

                    # Add Risk Event VLines (Keep existing - check if risk_summary has data)
                    if risk_summary and risk_summary.get('applied_risk_events'):
                         applied_events_list = risk_summary.get('applied_risk_events', [])
                         severities_dict = risk_summary.get('risk_severity', {})
                         num_points = len(df)
                         num_risks_applied = len(applied_events_list)
                         if num_risks_applied > 0 and num_points > num_risks_applied:
                             # Distribute markers roughly across the timeline
                             indices = np.linspace(0, num_points - 1, num_risks_applied, dtype=int)
                             # Ensure indices are within bounds
                             indices = np.clip(indices, 0, num_points - 1)

                             for i, risk in enumerate(applied_events_list):
                                 idx = indices[i]
                                 severity = severities_dict.get(risk, 0.5)
                                 if idx < len(df): # Safety check
                                    fig.add_vline(x=df['date'].iloc[idx], line_width=1, line_dash="dash", line_color="grey",
                                                  annotation_text=f"{risk[:10]} ({severity:.1f})", annotation_position="top left", annotation_font_size=9)

                    if plot_made:
                        fig.update_layout(title="Inventory & Shortage", height=350, margin=dict(l=20, r=20, t=40, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                        fig.update_yaxes(title_text="Inventory", secondary_y=False)
                        fig.update_yaxes(title_text="Shortage", secondary_y=True, showgrid=False)
                        st.plotly_chart(fig, use_container_width=True)
                    else: st.info("No plottable inventory or shortage data.")

                    # Download Button
                    try: csv_data = df.to_csv(index=False).encode('utf-8'); st.download_button(label="Download Chart Data", data=csv_data, file_name="sim_chart_data.csv", mime="text/csv")
                    except Exception as e: logging.error(f"CSV download prep failed: {e}")
                else: st.warning("No valid data after date processing.")
            else: st.warning("Timeseries data is empty or missing 'date' column.")
        tab_index += 1

    # --- NEW: Financials Tab ---
    if financial_summary:
        with tabs[tab_index]:
            st.subheader("Simulation Financial Summary")
            if financial_summary.get("calculation_error"):
                 st.error(f"Financial calculation encountered an error: {financial_summary['calculation_error']}")
            else:
                 # Display key financial metrics using st.metric
                 col_h, col_s, col_l, col_p, col_t = st.columns(5)
                 # Use .get() for safety
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
             st.subheader("AI-Powered Recommendations") # More specific title
             if recommendations:
                  categories = sorted(list(set(rec.get('category', 'Unknown') for rec in recommendations)))
                  selected_cat = st.selectbox("Filter Category:", ["All"] + categories, key="modal_rec_cat_filter")
                  filtered_recs = [r for r in recommendations if selected_cat == "All" or r.get('category') == selected_cat]
                  filtered_recs.sort(key=lambda x: x.get('priority', 0), reverse=True)
                  if not filtered_recs: st.info("No recommendations match filter.")
                  for rec in filtered_recs:
                       prio=rec.get("priority",0); cat=rec.get("category","Info"); txt=rec.get("text","N/A")
                       p_map={5:"critical", 4:"high", 3:"medium", 2:"low", 1:"informational"}; css=f"recommendation-{p_map.get(prio, 'unknown')}"
                       st.markdown(f'<div class="{css}"><strong>{cat} (P{prio}):</strong> {txt}</div>', unsafe_allow_html=True)
             else: st.info("No recommendations available.")
         tab_index += 1

    # --- Risk Info Tab ---
    if risk_summary:
        with tabs[tab_index]:
            st.subheader("Applied Risk Scenario Info") # More specific title
            applied_events = risk_summary.get('applied_risk_events')
            severities = risk_summary.get('risk_severity')
            if applied_events:
                st.markdown("**Risk Events Applied:**")
                risk_data_list = [{"Risk Event": r, "Severity": f"{severities.get(r, 0.5):.1f}"} for r in applied_events]
                st.dataframe(pd.DataFrame(risk_data_list), use_container_width=True, hide_index=True)

            # Display effective risk factors applied in the simulation
            effective_factors = risk_summary.get('effective_risk_factors_applied')
            if effective_factors:
                 st.markdown("**Effective Simulation Risk Parameters (Overrides):**")
                 # Format as a table or list
                 factor_list = []
                 for risk_type, params in effective_factors.items():
                      for param, value in params.items():
                           factor_list.append({"Risk Type": risk_type, "Parameter": param, "Applied Value": str(value)}) # Convert value to str for display
                 if factor_list:
                     st.dataframe(pd.DataFrame(factor_list), use_container_width=True, hide_index=True)
                 else:
                     st.info("No specific risk parameters were overridden by applied risks.")

            market_factor = risk_summary.get('market_factor_applied')
            if market_factor:
                 st.markdown(f"**Market Data Impact:** {market_factor}")

            if not applied_events and not effective_factors and not market_factor:
                 st.info("No specific risk events or market factors were applied to this simulation run.")

        tab_index += 1
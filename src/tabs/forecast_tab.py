# src/tabs/forecast_tab.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import logging

# Import FORECAST_PERIODS from config
from config import FORECAST_PERIODS

def render_tab(df_filtered, selected_run_file, start_date_filter, end_date_filter):
    """Renders the Predictive Forecast tab."""
    st.header("Predictive Forecast (Experimental)")

    if df_filtered.empty:
        if selected_run_file and start_date_filter <= end_date_filter: st.warning("No data available for the selected filters in this run to base forecast on.")
        elif not selected_run_file: st.info("Please select a simulation run file from the sidebar.")
        return # Stop rendering

    st.markdown("Basic forecast using Exponential Smoothing. Does not account for future disruptions.")
    factories_in_filtered_data_fc = sorted(df_filtered['factory_name'].unique().tolist())

    if not factories_in_filtered_data_fc:
         st.warning("No factories match the current filter selection for forecasting.")
         return # Stop rendering

    # --- Forecast Controls ---
    # (Copy forecast factory/metric selection from original Tab 4 here)
    current_fc_selection = st.session_state.get("forecast_factory_select", None)
    if current_fc_selection not in factories_in_filtered_data_fc:
        st.session_state.forecast_factory_select = factories_in_filtered_data_fc[0]
    forecast_factory = st.selectbox(
        "Select Factory to Forecast:",
        options=factories_in_filtered_data_fc,
        key="forecast_factory_select"
    )
    forecast_metric = st.selectbox(
        "Select Metric to Forecast:",
        options=['adjusted_production_units', 'inventory_level_units', 'lead_time_days', 'shortage_units'],
        index=0, # Default to production
        key="forecast_metric_select"
    )

    # --- Forecast Logic & Plot ---
    if forecast_factory:
        df_forecast_base = df_filtered[df_filtered['factory_name'] == forecast_factory].set_index('date').sort_index()

        # Check for sufficient data
        # (Copy forecast logic, model fitting, and plotting from original Tab 4 here)
        if df_forecast_base.empty or forecast_metric not in df_forecast_base.columns or df_forecast_base[forecast_metric].isnull().all():
            st.warning(f"No valid data for '{forecast_metric}' at {forecast_factory} within the selected range.")
        elif len(df_forecast_base.dropna(subset=[forecast_metric])) < 24: # Check after dropping NaNs
            st.warning(f"Not enough historical data points ({len(df_forecast_base.dropna(subset=[forecast_metric]))}) for forecasting '{forecast_metric}'. Need at least 24.")
        else:
            try:
                ts_data = df_forecast_base[forecast_metric].copy()
                # Ensure weekly frequency and fill gaps
                ts_data = ts_data.asfreq('W-SUN', method='ffill') # Use Pandas standard weekly frequency
                ts_data = ts_data.fillna(method='ffill').fillna(method='bfill') # Fill remaining gaps

                if len(ts_data) < 24:
                     st.warning(f"Still not enough data points ({len(ts_data)}) after frequency adjustment and filling for forecasting '{forecast_metric}'.")
                else:
                     # Fit ETS model (Holt-Winters)
                     model = ExponentialSmoothing(
                          ts_data,
                          trend='add',
                          seasonal='add',
                          seasonal_periods=52, # Assuming weekly data with yearly seasonality
                          initialization_method='estimated'
                      )
                     fit = model.fit()

                     # Make forecast
                     forecast_values = fit.forecast(FORECAST_PERIODS)
                     forecast_index = pd.date_range(
                          start=ts_data.index[-1] + pd.Timedelta(weeks=1),
                          periods=FORECAST_PERIODS,
                          freq='W-SUN' # Match frequency
                      )
                     forecast_series = pd.Series(forecast_values, index=forecast_index)

                     # Plot
                     fig_forecast = go.Figure()
                     fig_forecast.add_trace(go.Scatter(x=ts_data.index, y=ts_data, mode='lines', name='Historical Data'))
                     fig_forecast.add_trace(go.Scatter(x=forecast_series.index, y=forecast_series, mode='lines', name='Forecast', line=dict(color='red')))
                     fig_forecast.update_layout(
                         title=f"Forecast for {forecast_metric.replace('_', ' ').title()} at {forecast_factory}",
                         xaxis_title="Date",
                         yaxis_title=forecast_metric.replace('_', ' ').title(),
                         hovermode="x unified"
                      )
                     st.plotly_chart(fig_forecast, use_container_width=True)
            except Exception as e:
                st.error(f"Could not generate forecast for '{forecast_metric}': {e}")
                logging.error(f"Forecast failed for {forecast_factory} / {forecast_metric}: {e}", exc_info=True)
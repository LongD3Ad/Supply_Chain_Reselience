import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import os
import glob
import logging

# --- Configuration & Setup ---
DATA_DIR = "src\simulation_data"  # Relative path from dashboard.py to the data folder
DEFAULT_RUN_FILE = None
FORECAST_PERIODS = 12 # Number of weeks/periods to forecast ahead

# --- Logging Setup ---
# Basic logging for Streamlit app isn't as crucial as for the backend script,
# but can be helpful for debugging complex callbacks later.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Page Configuration (Set Wide Mode) ---
st.set_page_config(layout="wide", page_title="Semiconductor Supply Chain Digital Twin")

st.title("Semiconductor Supply Chain - Digital Twin Simulation Dashboard")
st.markdown("""
Welcome to the Semiconductor Supply Chain Digital Twin dashboard.
This tool visualizes data generated from simulations modeling various risk factors
(pandemic, geopolitical tensions, cyber attacks, etc.) impacting production,
lead times, and inventory levels across different factories.

**Instructions:**
1.  Select a simulation run from the dropdown in the sidebar.
2.  Use the filters to narrow down the date range and select specific factories.
3.  Explore the different tabs to see overall performance, factory details, risk analysis, and basic forecasts.
""")

# --- Helper Functions ---

@st.cache_data(ttl=3600) # Cache data loading for 1 hour
def load_data(file_path):
    """Loads and preprocesses data from a selected simulation run CSV."""
    try:
        df = pd.read_csv(file_path)
        # Convert date column to datetime objects
        df['date'] = pd.to_datetime(df['date'])
        # Ensure numeric types where expected
        numeric_cols = ['base_production_capacity', 'adjusted_production_units',
                        'production_impact_factor', 'lead_time_days',
                        'simulated_demand_units', 'shipped_units',
                        'inventory_level_units', 'shortage_units', 'safety_stock_level']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce') # Coerce errors to NaN

        # Sort by date for time series analysis
        df = df.sort_values(by=['factory_name', 'date'])
        logging.info(f"Successfully loaded and preprocessed data from {file_path}")
        return df
    except FileNotFoundError:
        st.error(f"Error: File not found at {file_path}. Make sure the simulation data exists.")
        logging.error(f"File not found: {file_path}")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading {os.path.basename(file_path)}: {e}")
        logging.error(f"Error loading {file_path}: {e}")
        return None

def get_available_runs(data_dir):
    """Finds all simulation CSV files in the specified directory."""
    search_path = os.path.join(data_dir, "semiconductor_supplychain_simulated_run_*.csv")
    files = glob.glob(search_path)
    if not files:
        st.warning(f"No simulation files found matching pattern in '{data_dir}'. "
                   f"Please run the `supply_chain_data.py` script first.")
        return []
    # Return just the filenames for the dropdown
    return sorted([os.path.basename(f) for f in files], reverse=True) # Show newest first potentially


@st.cache_data
def get_filtered_data(df, selected_factories, start_date, end_date):
    """Filters the dataframe based on user selections."""
    if df is None:
        return pd.DataFrame() # Return empty DataFrame if loading failed

    filtered = df[
        (df['factory_name'].isin(selected_factories)) &
        (df['date'] >= pd.to_datetime(start_date)) &
        (df['date'] <= pd.to_datetime(end_date))
    ].copy()
    return filtered

# --- Sidebar Controls ---
st.sidebar.header("Simulation Controls")

available_files = get_available_runs(DATA_DIR)

if not available_files:
    st.sidebar.error("No simulation data found.")
    st.stop() # Stop execution if no data is available

selected_run_file = st.sidebar.selectbox(
    "Select Simulation Run:",
    available_files,
    index=0 # Default to the first file in the sorted list
)

# Load the main dataframe for the selected run
file_path = os.path.join(DATA_DIR, selected_run_file)
df_main = load_data(file_path)

if df_main is not None and not df_main.empty:
    # Get available factories and date range from the loaded data
    available_factories = df_main['factory_name'].unique().tolist()
    min_date = df_main['date'].min().date()
    max_date = df_main['date'].max().date()

    st.sidebar.header("Filters")
    selected_factories = st.sidebar.multiselect(
        "Select Factories:",
        options=available_factories,
        default=available_factories # Default to all factories
    )

    # Ensure defaults are valid dates
    if 'start_date_filter' not in st.session_state:
        st.session_state.start_date_filter = min_date
    if 'end_date_filter' not in st.session_state:
        st.session_state.end_date_filter = max_date

    # Use session state to prevent widget state loss on rerun
    start_date_filter = st.sidebar.date_input(
        "Start Date:",
        value=st.session_state.start_date_filter,
        min_value=min_date,
        max_value=max_date,
        key='start_date_filter'
    )

    end_date_filter = st.sidebar.date_input(
        "End Date:",
        value=st.session_state.end_date_filter,
        min_value=min_date, # start_date_filter, # Min should be overall min, max should be overall max
        max_value=max_date,
        key='end_date_filter'
    )

    # Basic validation
    if start_date_filter > end_date_filter:
        st.sidebar.error("Error: Start date must be before end date.")
        st.stop() # Stop execution if dates are invalid

    # Filter data based on selections
    df_filtered = get_filtered_data(df_main, selected_factories, start_date_filter, end_date_filter)

else:
    st.error("Failed to load data for the selected run. Please check the file and try again.")
    st.stop() # Stop execution if data loading fails completely

# --- Main Dashboard Area ---

if df_filtered.empty:
    st.warning("No data available for the selected factories and date range.")
else:
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Overall Summary",
        "🏭 Factory Deep Dive",
        "🚨 Risk Analysis",
        "🔮 Predictive Forecast"
    ])

    with tab1:
        st.header("Overall Supply Chain Performance")
        st.markdown(f"Displaying aggregated data for **{len(selected_factories)} selected factories** from **{start_date_filter}** to **{end_date_filter}**.")

        # Calculate KPIs
        total_production = df_filtered['adjusted_production_units'].sum()
        avg_lead_time = df_filtered['lead_time_days'].mean()
        total_shortage = df_filtered['shortage_units'].sum()
        avg_inventory = df_filtered['inventory_level_units'].mean() # Avg inventory level across time/factories

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Adjusted Production", f"{total_production:,.0f} units")
        col2.metric("Average Lead Time", f"{avg_lead_time:.1f} days")
        col3.metric("Total Shortage", f"{total_shortage:,.0f} units")
        col4.metric("Average Inventory Level", f"{avg_inventory:,.0f} units")

        st.subheader("Performance Over Time (Aggregated)")

        # Aggregate data by date for plotting
        df_agg = df_filtered.groupby('date').agg(
            total_adjusted_production=('adjusted_production_units', 'sum'),
            average_lead_time=('lead_time_days', 'mean'),
            total_inventory=('inventory_level_units', 'sum'),
            total_shortage=('shortage_units', 'sum')
        ).reset_index()

        # Plot aggregated metrics
        fig_prod = px.line(df_agg, x='date', y='total_adjusted_production', title='Total Production Over Time', labels={'total_adjusted_production':'Total Units Produced'})
        fig_prod.update_layout(hovermode="x unified")
        st.plotly_chart(fig_prod, use_container_width=True)

        fig_inv_short = go.Figure()
        fig_inv_short.add_trace(go.Scatter(x=df_agg['date'], y=df_agg['total_inventory'], mode='lines', name='Total Inventory'))
        fig_inv_short.add_trace(go.Scatter(x=df_agg['date'], y=df_agg['total_shortage'], mode='lines', name='Total Shortage', line=dict(color='red')))
        fig_inv_short.update_layout(title='Total Inventory vs. Shortage Over Time', hovermode="x unified", yaxis_title="Units")
        st.plotly_chart(fig_inv_short, use_container_width=True)

        fig_lead = px.line(df_agg, x='date', y='average_lead_time', title='Average Lead Time Over Time', labels={'average_lead_time':'Average Days'})
        fig_lead.update_layout(hovermode="x unified")
        st.plotly_chart(fig_lead, use_container_width=True)


    with tab2:
        st.header("Factory Deep Dive")
        st.markdown("Select a specific factory from the list below to see its detailed performance metrics.")

        # Allow selecting only ONE factory for the deep dive view
        factory_to_inspect = st.selectbox(
            "Select Factory for Details:",
            options=selected_factories, # Only show factories already selected in sidebar
            index=0 if selected_factories else -1 # Default to first selected, or none if none selected
        )

        if factory_to_inspect:
            df_factory = df_filtered[df_filtered['factory_name'] == factory_to_inspect].copy()
            st.subheader(f"Details for: {factory_to_inspect}")

            if df_factory.empty:
                st.warning("No data available for this factory within the selected date range.")
            else:
                # KPIs for the selected factory
                f_total_prod = df_factory['adjusted_production_units'].sum()
                f_avg_lead = df_factory['lead_time_days'].mean()
                f_total_shortage = df_factory['shortage_units'].sum()
                f_avg_inv = df_factory['inventory_level_units'].mean()

                fcol1, fcol2, fcol3, fcol4 = st.columns(4)
                fcol1.metric("Total Production", f"{f_total_prod:,.0f} units")
                fcol2.metric("Avg Lead Time", f"{f_avg_lead:.1f} days")
                fcol3.metric("Total Shortage", f"{f_total_shortage:,.0f} units")
                fcol4.metric("Avg Inventory", f"{f_avg_inv:,.0f} units")

                # Detailed plots
                fig_f_prod = px.line(df_factory, x='date', y='adjusted_production_units', title='Adjusted Production', labels={'adjusted_production_units':'Units Produced'})
                fig_f_prod.update_layout(hovermode="x unified")

                fig_f_inv = px.line(df_factory, x='date', y='inventory_level_units', title='Inventory Level', labels={'inventory_level_units':'Units in Stock'})
                fig_f_inv.add_trace(go.Scatter(x=df_factory['date'], y=df_factory['safety_stock_level'], mode='lines', name='Safety Stock', line=dict(dash='dash', color='grey')))
                fig_f_inv.update_layout(hovermode="x unified")


                fig_f_lead = px.line(df_factory, x='date', y='lead_time_days', title='Lead Time', labels={'lead_time_days':'Days'})
                fig_f_lead.update_layout(hovermode="x unified")


                fig_f_short = px.bar(df_factory, x='date', y='shortage_units', title='Shortage Events', labels={'shortage_units':'Unmet Demand (Units)'})
                fig_f_short.update_layout(hovermode="x unified")

                # Add risk highlights to production chart
                risk_events = df_factory[df_factory['active_risks'] != 'none']
                for _, event in risk_events.iterrows():
                    # Find consecutive periods with the same risk for shading
                    # This is a simplified approach; robust grouping would be needed for complex overlaps
                    fig_f_prod.add_vrect(
                        x0=event['date'] - pd.Timedelta(days=3.5), # Center shade on week start approx
                        x1=event['date'] + pd.Timedelta(days=3.5),
                        fillcolor="rgba(255, 0, 0, 0.1)", # Light red shade for risk
                        layer="below", line_width=0,
                        annotation_text=event['active_risks'], annotation_position="top left"
                    )


                st.plotly_chart(fig_f_prod, use_container_width=True)
                st.plotly_chart(fig_f_inv, use_container_width=True)
                st.plotly_chart(fig_f_lead, use_container_width=True)
                st.plotly_chart(fig_f_short, use_container_width=True)

                st.subheader("Risk Events Log")
                st.dataframe(risk_events[['date', 'active_risks', 'production_impact_factor', 'adjusted_production_units', 'lead_time_days']], use_container_width=True)
        else:
            st.info("Select a factory above to see detailed charts.")


    with tab3:
        st.header("Risk Analysis")
        st.markdown("Analyzing the frequency and impact of simulated risk events across selected factories and time range.")

        # Prepare data for risk analysis (explode comma-separated risks)
        df_risk = df_filtered.copy()
        df_risk['active_risks'] = df_risk['active_risks'].apply(lambda x: [r.strip() for r in x.split(',')] if x != 'none' else ['none'])
        df_risk_exploded = df_risk.explode('active_risks')
        df_risk_events = df_risk_exploded[df_risk_exploded['active_risks'] != 'none'].copy()

        if df_risk_events.empty:
            st.info("No risk events occurred within the selected filters.")
        else:
            # Risk Frequency
            st.subheader("Risk Event Frequency")
            risk_counts = df_risk_events['active_risks'].value_counts().reset_index()
            risk_counts.columns = ['Risk Type', 'Frequency (Weeks Active)']
            fig_risk_freq = px.bar(risk_counts, x='Risk Type', y='Frequency (Weeks Active)', title="Frequency of Each Risk Type")
            st.plotly_chart(fig_risk_freq, use_container_width=True)

            # Risk Impact Analysis
            st.subheader("Production Impact by Risk Type")
            # Filter out recovery phases for clearer impact view, if identifiable (e.g., based on name)
            # Or just show all impacts including recovery:
            df_impact = df_risk_events[df_risk_events['production_impact_factor'] < 1.0] # Only where impact occurred
            fig_impact_box = px.box(df_impact, x='active_risks', y='production_impact_factor',
                                    title="Distribution of Production Impact Factor by Risk Type (Lower is worse)",
                                    labels={'active_risks': 'Risk Type', 'production_impact_factor': 'Production Impact Factor (1 = No Impact)'},
                                    points="all") # Show individual points
            fig_impact_box.update_yaxes(range=[0, 1.1]) # Set y-axis from 0 to 1.1
            st.plotly_chart(fig_impact_box, use_container_width=True)

            st.subheader("Lead Time Impact by Risk Type")
            fig_lead_box = px.box(df_risk_events, x='active_risks', y='lead_time_days',
                                  title="Distribution of Lead Time by Risk Type",
                                  labels={'active_risks': 'Risk Type', 'lead_time_days': 'Lead Time (Days)'},
                                  points="all")
            st.plotly_chart(fig_lead_box, use_container_width=True)

            st.subheader("Risk Events per Factory")
            risk_per_factory = df_risk_events.groupby('factory_name')['active_risks'].value_counts().unstack(fill_value=0)
            st.dataframe(risk_per_factory)


    with tab4:
        st.header("Predictive Forecast (Experimental)")
        st.markdown("""
        This section demonstrates a simple forecast using Exponential Smoothing (Holt-Winters)
        based on the historical simulated data for a selected factory and metric.
        *Note: This is a basic model and does not account for future simulated disruptions.*
        """)

        forecast_factory = st.selectbox(
            "Select Factory to Forecast:",
            options=selected_factories,
             index=0 if selected_factories else -1,
            key="forecast_factory_select" # Unique key needed
        )

        forecast_metric = st.selectbox(
            "Select Metric to Forecast:",
            options=['adjusted_production_units', 'inventory_level_units', 'lead_time_days', 'shortage_units'],
            index=0
        )

        if forecast_factory:
            df_forecast_base = df_filtered[
                (df_filtered['factory_name'] == forecast_factory)
            ].set_index('date').sort_index()

            if df_forecast_base.empty or df_forecast_base[forecast_metric].isnull().all():
                 st.warning(f"Not enough valid data for {forecast_metric} at {forecast_factory} in the selected range.")
            elif len(df_forecast_base) < 24: # Need sufficient data points for model
                 st.warning(f"Not enough historical data points ({len(df_forecast_base)}) for forecasting. Need at least 24 weeks.")
            else:
                try:
                    # Prepare series for modeling
                    ts_data = df_forecast_base[forecast_metric].copy()
                    ts_data = ts_data.asfreq('W-SUN', method='ffill').fillna(method='ffill') # Ensure weekly freq, forward fill NaNs

                    # Fit Exponential Smoothing model
                    # Trying additive trend and seasonality, letting model choose damping etc.
                    # Seasonal periods=52 assumes weekly data with yearly seasonality
                    model = ExponentialSmoothing(
                        ts_data,
                        trend='add',
                        seasonal='add',
                        seasonal_periods=52, # Assume yearly seasonality for weekly data
                        initialization_method='estimated'
                    )
                    fit = model.fit()

                    # Generate forecast
                    forecast_values = fit.forecast(FORECAST_PERIODS)
                    forecast_index = pd.date_range(ts_data.index[-1] + pd.Timedelta(weeks=1), periods=FORECAST_PERIODS, freq='W-SUN')
                    forecast_series = pd.Series(forecast_values, index=forecast_index)

                    # Create plot
                    fig_forecast = go.Figure()
                    # Historical data
                    fig_forecast.add_trace(go.Scatter(x=ts_data.index, y=ts_data, mode='lines', name='Historical Data'))
                    # Fitted values (optional)
                    # fig_forecast.add_trace(go.Scatter(x=ts_data.index, y=fit.fittedvalues, mode='lines', name='Model Fit', line=dict(dash='dot')))
                    # Forecast
                    fig_forecast.add_trace(go.Scatter(x=forecast_series.index, y=forecast_series, mode='lines', name='Forecast', line=dict(color='red')))

                    # Add confidence intervals (if model provides them - ETS usually doesn't directly in statsmodels fit.forecast)
                    # Simple placeholder for illustrating the concept:
                    # conf_int = fit.get_prediction(start=forecast_series.index[0], end=forecast_series.index[-1]).conf_int(alpha=0.10) # 90% CI
                    # fig_forecast.add_trace(go.Scatter(x=conf_int.index, y=conf_int.iloc[:, 0], fill=None, mode='lines', line_color='rgba(255,0,0,0.2)', name='Lower CI 90%'))
                    # fig_forecast.add_trace(go.Scatter(x=conf_int.index, y=conf_int.iloc[:, 1], fill='tonexty', mode='lines', line_color='rgba(255,0,0,0.2)', name='Upper CI 90%'))


                    fig_forecast.update_layout(
                        title=f"Forecast for {forecast_metric} at {forecast_factory}",
                        xaxis_title="Date",
                        yaxis_title=forecast_metric,
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig_forecast, use_container_width=True)

                except Exception as e:
                    st.error(f"Could not generate forecast: {e}")
                    logging.error(f"Forecast failed for {forecast_factory} / {forecast_metric}: {e}")


        else:
             st.info("Select a factory to generate a forecast.")


# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.info("Dashboard developed based on simulated supply chain data.")
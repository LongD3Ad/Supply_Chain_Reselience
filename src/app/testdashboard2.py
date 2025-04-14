import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import os
import glob
import logging
import json
import random
from datetime import datetime, timedelta
import simpy

# ---------------- Configuration & Setup ----------------

CONFIG_PATH = "../../../DEFAULT_CONFIG.json" # Adjust if needed
DEFAULT_SIM_CONFIG = {
    # --- Default Config remains the same ---
    "factories": [
        {
            "name": "Foundry_Taiwan_A", "location": "Taiwan", "type": "Foundry",
            "base_production_capacity_weekly": [10000, 15000],
            "production_noise_std_dev_percent": 0.05,
            "base_lead_time_days": [25, 35],
            "initial_inventory_units": [50000, 80000],
            "base_weekly_demand": [6000, 11000]
        },
        {
            "name": "IDM_USA_B", "location": "USA", "type": "IDM",
            "base_production_capacity_weekly": [8000, 12000],
            "production_noise_std_dev_percent": 0.04,
            "base_lead_time_days": [20, 30],
            "initial_inventory_units": [40000, 60000],
            "base_weekly_demand": [5000, 9000]
        },
        {
            "name": "OSAT_Malaysia_C", "location": "Malaysia", "type": "OSAT",
            "base_production_capacity_weekly": [15000, 20000],
            "production_noise_std_dev_percent": 0.07,
            "base_lead_time_days": [15, 25],
            "initial_inventory_units": [60000, 90000],
            "base_weekly_demand": [10000, 16000]
        },
        {
            "name": "Foundry_Germany_D", "location": "Germany", "type": "Foundry",
            "base_production_capacity_weekly": [7000, 11000],
            "production_noise_std_dev_percent": 0.035,
            "base_lead_time_days": [28, 38],
            "initial_inventory_units": [35000, 55000],
            "base_weekly_demand": [4000, 8000]
        }
    ],
    "risk_factors": {
        "pandemic_closure": {"prob_per_day": 0.002, "impact_factor_range": [0.2, 0.6], "duration_range_days": [28, 84], "lead_time_multiplier_range": [1.5, 3.0], "mean_days_between_attempts": 180},
        "geopolitical_tension": {"prob_per_day": 0.003, "impact_factor_range": [0.7, 0.9], "duration_range_days": [14, 56], "lead_time_multiplier_range": [1.2, 1.8], "mean_days_between_attempts": 120},
        "trade_restrictions": {"prob_per_day": 0.004, "impact_factor_range": [0.6, 0.85], "duration_range_days": [21, 70], "lead_time_multiplier_range": [1.3, 2.0], "mean_days_between_attempts": 90},
        "cyber_attack": {"prob_per_day": 0.001, "impact_factor_range": [0.1, 0.5], "duration_range_days": [7, 21], "lead_time_multiplier_range": [1.1, 1.5], "mean_days_between_attempts": 250},
        "natural_disaster": {"prob_per_day": 0.0015, "impact_factor_range": [0.0, 0.7], "duration_range_days": [7, 42], "lead_time_multiplier_range": [1.4, 4.0], "mean_days_between_attempts": 200},
        "logistics_bottleneck": {"prob_per_day": 0.005, "impact_factor_range": [1.0, 1.0], "duration_range_days": [7, 28], "lead_time_multiplier_range": [1.5, 2.5], "mean_days_between_attempts": 60}
    },
    "simulation_defaults": {
        "sim_duration_weeks": 52,
        "time_step_days": 1
    }
}

DATA_DIR = "src/simulation_data"  # Adjust file path if needed relative to script location
FORECAST_PERIODS = 12

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Page Config and Styling ---
st.set_page_config(layout="wide", page_title="Supply Chain Simulation") # Generalized Page Title
st.markdown("""
    <style>
        .css-18e3th9 {padding-top: 2rem; padding-bottom: 2rem;} /* Adjust top/bottom padding */
        .css-1d391kg {font-size: 1.1rem;} /* Adjust sidebar font size if needed */
        .stSelectbox label {font-weight: bold;} /* Make selectbox labels bold */
    </style>
    """, unsafe_allow_html=True)

# ---------------- Load Configuration ----------------
# load_simulation_config function remains the same
@st.cache_data
def load_simulation_config(config_path=None):
    # ... (function content unchanged) ...
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                logging.info(f"Loading simulation config from {config_path}")
                return json.load(f)
        except Exception as e:
            logging.error(f"Failed to load config file: {e}. Using default config.")
            st.warning("Could not load external config. Using default simulation parameters.")
            return DEFAULT_SIM_CONFIG
    else:
        logging.info("Using default simulation configuration.")
        return DEFAULT_SIM_CONFIG

sim_config = load_simulation_config()

# ---------------- Helper Functions (Data Loading, Filtering) ----------------
# load_data, get_available_runs, get_filtered_data functions remain the same
@st.cache_data(ttl=3600)
def load_data(file_path):
    # ... (function content unchanged) ...
    try:
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        numeric_cols = ['base_production_capacity', 'adjusted_production_units',
                        'production_impact_factor', 'lead_time_days',
                        'simulated_demand_units', 'shipped_units',
                        'inventory_level_units', 'shortage_units', 'safety_stock_level']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.sort_values(by=['factory_name', 'date'])
        logging.info(f"Loaded data from {file_path}")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def get_available_runs(data_dir):
    # ... (function content unchanged) ...
    search_path = os.path.join(data_dir, "semiconductor_supplychain_simulated_run_*.csv")
    files = glob.glob(search_path)
    # Filter out non-CSV files or directories if any weird files exist
    csv_files = [f for f in files if os.path.isfile(f) and f.lower().endswith('.csv')]
    return sorted([os.path.basename(f) for f in csv_files], reverse=True) if csv_files else []


@st.cache_data
def get_filtered_data(df, selected_factories, start_date, end_date):
    # ... (function content unchanged) ...
    if df is None:
        return pd.DataFrame()
    # Ensure dates are datetime objects before comparison
    start_date_dt = pd.to_datetime(start_date)
    end_date_dt = pd.to_datetime(end_date)
    filtered = df[
        (df['factory_name'].isin(selected_factories)) &
        (df['date'] >= start_date_dt) &
        (df['date'] <= end_date_dt)
    ].copy()
    return filtered

# ---------------- SimPy Simulation Engine Functions ----------------
# run_factory_simulation, production_process, risk_event_generator functions remain the same
def run_factory_simulation(factory_config, risk_factors_config, sim_duration_weeks, sim_start_date):
    # ... (function content unchanged) ...
    env = simpy.Environment()
    TIME_STEP_DAYS = 1
    SIM_DURATION_DAYS = sim_duration_weeks * 7

    base_weekly_capacity = random.randint(*factory_config['base_production_capacity_weekly'])
    base_daily_capacity = base_weekly_capacity / 7.0
    daily_noise_std_dev = base_daily_capacity * factory_config['production_noise_std_dev_percent']

    factory_state = {
        "name": factory_config['name'],
        "base_daily_capacity": base_daily_capacity,
        "current_daily_capacity": base_daily_capacity,
        "daily_noise_std_dev": daily_noise_std_dev,
        "base_lead_time": random.randint(*factory_config['base_lead_time_days']),
        "current_lead_time_multiplier": 1.0,
        "inventory": random.randint(*factory_config['initial_inventory_units']),
        "weekly_demand_range": factory_config.get('base_weekly_demand', [5000, 10000]),
        "risk_log": [],
        "active_risks": {}
    }

    output_log = []

    # --- Nested SimPy Processes ---
    def production_process(env, factory_state, output_log):
        # ... (process logic unchanged) ...
        weekly_production = 0
        weekly_demand = 0
        weekly_shipped = 0
        weekly_shortage = 0
        current_week = 0

        while True:
            daily_prod = max(0, np.random.normal(factory_state["current_daily_capacity"], factory_state["daily_noise_std_dev"]))
            weekly_production += daily_prod
            factory_state["inventory"] += daily_prod

            daily_demand_mean = random.uniform(*factory_state["weekly_demand_range"]) / 7.0
            daily_demand = max(0, np.random.normal(daily_demand_mean, daily_demand_mean * 0.3))
            weekly_demand += daily_demand

            shipped = min(factory_state["inventory"], daily_demand)
            factory_state["inventory"] -= shipped
            weekly_shipped += shipped
            weekly_shortage += max(0, daily_demand - shipped)

            yield env.timeout(TIME_STEP_DAYS)

            if env.now > 0 and env.now % 7 == 0:
                current_week += 1
                current_date = sim_start_date + timedelta(weeks=current_week - 1)
                effective_lead_time = int(factory_state["base_lead_time"] * factory_state["current_lead_time_multiplier"])

                output_log.append({
                    "week": current_week,
                    "date": current_date,
                    "factory_name": factory_state["name"],
                    "adjusted_production_units": int(weekly_production),
                    "simulated_demand_units": int(weekly_demand),
                    "shipped_units": int(weekly_shipped),
                    "inventory_level_units": int(factory_state["inventory"]),
                    "shortage_units": int(weekly_shortage),
                    "lead_time_days": effective_lead_time,
                    "active_risks_count": len(factory_state["active_risks"])
                })
                weekly_production = 0
                weekly_demand = 0
                weekly_shipped = 0
                weekly_shortage = 0

    def risk_event_generator(env, risk_name, risk_params, factory_state):
        # ... (process logic unchanged) ...
        while True:
            # Use configured mean time between attempts if available, else default
            mean_wait = risk_params.get("mean_days_between_attempts", 90)
            time_to_next_attempt = np.random.exponential(mean_wait) if mean_wait > 0 else float('inf') # Avoid division by zero or negative wait
            if time_to_next_attempt == float('inf'): break # Stop if no attempts should be made
            yield env.timeout(time_to_next_attempt)

            # Calculate probability based on daily rate and time passed (approximate check)
            prob_trigger = risk_params.get("prob_per_day", 0.01) * time_to_next_attempt
            if random.random() < prob_trigger:
                # Check if this risk is not already active (simplification)
                if risk_name not in factory_state["active_risks"] or env.now >= factory_state["active_risks"][risk_name]:
                    impact_factor = random.uniform(*risk_params["impact_factor_range"])
                    duration_days = random.randint(*risk_params["duration_range_days"])
                    lead_time_multiplier = random.uniform(*risk_params["lead_time_multiplier_range"])
                    start_time = env.now
                    end_time = start_time + duration_days

                    logging.info(f"Event: {risk_name} at day {start_time:.1f} for {duration_days} days (Impact: {impact_factor:.2f})")

                    factory_state["risk_log"].append({
                        "risk_type": risk_name,
                        "start_day": start_time,
                        "end_day": end_time,
                        "duration_days": duration_days,
                        "impact_factor": impact_factor,
                        "lead_time_multiplier": lead_time_multiplier
                    })

                    # --- Apply Impact ---
                    # Store original values before this specific risk hits
                    # Note: This needs careful thought if multiple impacts modify base differently.
                    # Current approach multiplies the *current* capacity, assuming sequential impacts.
                    factory_state["current_daily_capacity"] *= impact_factor
                    factory_state["current_lead_time_multiplier"] = max(factory_state["current_lead_time_multiplier"], lead_time_multiplier)
                    factory_state["active_risks"][risk_name] = end_time # Mark as active until end_time

                    # Wait for the duration
                    yield env.timeout(duration_days)

                    # --- Recovery Logic ---
                    # Recalculate state based on *other* risks still active *now*
                    current_impact_factor = 1.0
                    current_max_lt_multiplier = 1.0
                    risks_still_active = {}
                    for r_name, r_end_time in factory_state["active_risks"].items():
                        # Check if a different risk is still ongoing
                        if r_name != risk_name and env.now < r_end_time:
                            # Find its parameters from the log (last occurrence before now)
                            r_details = next((item for item in reversed(factory_state["risk_log"]) if item["risk_type"] == r_name and item["start_day"] <= env.now), None)
                            if r_details:
                                current_impact_factor *= r_details["impact_factor"] # Re-apply its impact
                                current_max_lt_multiplier = max(current_max_lt_multiplier, r_details["lead_time_multiplier"])
                            risks_still_active[r_name] = r_end_time # Keep it in the active list

                    # Reset capacity based on base * remaining active risks
                    factory_state["current_daily_capacity"] = factory_state["base_daily_capacity"] * current_impact_factor
                    # Reset lead time based on remaining active risks (or 1.0 if none)
                    factory_state["current_lead_time_multiplier"] = max(1.0, current_max_lt_multiplier)
                    # Update the dictionary of active risks
                    factory_state["active_risks"] = risks_still_active

                    logging.info(f"Recovered: {risk_name} ended at day {env.now:.1f}. Active risks: {list(risks_still_active.keys())}")


    # --- Start Processes & Run ---
    env.process(production_process(env, factory_state, output_log))
    for risk_name, risk_params in risk_factors_config.items():
        env.process(risk_event_generator(env, risk_name, risk_params, factory_state))

    env.run(until=SIM_DURATION_DAYS)
    # --- Process Results ---
    df_sim_results = pd.DataFrame(output_log)
    df_risk_log = pd.DataFrame(factory_state["risk_log"])
    if not df_risk_log.empty:
        df_risk_log['start_date'] = df_risk_log['start_day'].apply(lambda d: sim_start_date + timedelta(days=int(d)))
        df_risk_log['end_date'] = df_risk_log['end_day'].apply(lambda d: sim_start_date + timedelta(days=int(d)))
    return df_sim_results, df_risk_log

# ---------------- Streamlit App Layout ----------------

# --- Main Title ---
st.title("Supply Chain Risk Simulation Dashboard") # <-- Generalized Title
st.markdown("""
Explore simulation runs capturing supply chain performance affected by various risks.
Use the sidebar to select a run and filter data. The tabs below cover overall performance,
factory details, risk analysis, predictive forecasting, and a live simulation.
""")

# --- Sidebar ---
st.sidebar.header("Scenario Selection & Controls")

# --- NEW: List of Real-World Events ---
st.sidebar.subheader("Disruption Scenarios")
event_names = [
    "Semiconductor Shortage (Baseline)", # Keep this as the conceptual baseline for generated data
    "Suez Canal Blockage (Ever Given)",
    "COVID-19 Vaccine Distribution Challenges",
    "Russia-Ukraine War",
    "Port Congestion at Los Angeles/Long Beach",
    "Texas Winter Storm Uri",
    "Renesas Factory Fire in Japan",
    "China COVID Zero Policy Shutdowns",
    "Colonial Pipeline Ransomware Attack",
    "Panama Canal Drought",
    "Global Semiconductor Shortage (Specific Event)" # Adding this explicitly if needed
]
# Display names - These are for display only, not interactive data loading
for event in event_names:
    st.sidebar.caption(f"- {event}") # Use caption for less emphasis than markdown
st.sidebar.markdown("---") # Separator

# --- Controls for Historical Data (Generated CSVs) ---
st.sidebar.subheader("Load Generated Simulation Run") # Added subheader for clarity
available_files = get_available_runs(DATA_DIR)
if not available_files:
    st.sidebar.error(f"No simulation CSV files found in '{DATA_DIR}'. Please generate data first.")
    st.stop() # Stop if no data to analyze

selected_run_file = st.sidebar.selectbox(
    "Select Run File:", # Simplified label
    available_files,
    index=0,
    key="historical_run_select",
    help="Select a pre-generated CSV file representing a simulation run." # Added tooltip
)
file_path = os.path.join(DATA_DIR, selected_run_file)
# Load the main dataframe after selecting the file
df_main = load_data(file_path)

# --- Filtering Controls (Remain the same) ---
if df_main is not None and not df_main.empty:
    available_factories = sorted(df_main['factory_name'].unique().tolist())
    min_date = df_main['date'].min().date()
    max_date = df_main['date'].max().date()

    st.sidebar.header("Filters for Selected Run")
    selected_factories = st.sidebar.multiselect(
        "Select Factories:",
        options=available_factories,
        default=available_factories,
        key="hist_factory_filter" # Added key for stability
        )

    # Date range selection using session state for persistence
    if 'start_date_filter' not in st.session_state or st.session_state.start_date_filter < min_date or st.session_state.start_date_filter > max_date :
        st.session_state.start_date_filter = min_date
    if 'end_date_filter' not in st.session_state or st.session_state.end_date_filter < min_date or st.session_state.end_date_filter > max_date:
        st.session_state.end_date_filter = max_date

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
        min_value=min_date, # Allow selection up to start date
        max_value=max_date,
        key='end_date_filter'
        )

    # Ensure start_date is not after end_date
    if start_date_filter > end_date_filter:
        st.sidebar.error("Error: Start date must be on or before end date.")
        # Prevent further processing with invalid date range
        # Option 1: Show error and stop (st.stop())
        # Option 2: Show error and default to full range or disable tabs (more user-friendly)
        df_filtered = pd.DataFrame() # Use empty dataframe to avoid errors in tabs
        st.warning("Invalid date range selected. Please correct the dates in the sidebar.")

    else:
        # Apply filters
        df_filtered = get_filtered_data(df_main, selected_factories, start_date_filter, end_date_filter)

else:
    # Handle case where df_main failed to load
    st.error(f"Failed to load or process data from {selected_run_file}. Cannot proceed.")
    st.stop()


# ---------------- Tabs for Display (Tabs remain the same) ----------------
tab_titles = ["📈 Overall Summary", "🏭 Factory Deep Dive", "🚨 Risk Analysis", "🔮 Predictive Forecast", "🔬 Live Simulation"]
tabs = st.tabs(tab_titles)

# ---------- Tab 1: Overall Summary ----------
with tabs[0]:
    st.header("Overall Supply Chain Performance")
    if df_filtered.empty and start_date_filter <= end_date_filter: # Check if empty due to filtering, not date error
         st.warning("No data available for the selected filters in this run.")
    elif not df_filtered.empty:
        st.markdown(f"Aggregated data for **{len(selected_factories)}** selected factories from **{start_date_filter}** to **{end_date_filter}** (Run: `{selected_run_file}`).")
        # --- KPIs ---
        total_production = df_filtered['adjusted_production_units'].sum()
        avg_lead_time = df_filtered['lead_time_days'].mean()
        total_shortage = df_filtered['shortage_units'].sum()
        avg_inventory = df_filtered['inventory_level_units'].mean()
        # Display KPIs...
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Production", f"{total_production:,.0f} units")
        col2.metric("Average Lead Time", f"{avg_lead_time:.1f} days" if not pd.isna(avg_lead_time) else "N/A")
        col3.metric("Total Shortage", f"{total_shortage:,.0f} units")
        col4.metric("Average Inventory", f"{avg_inventory:,.0f} units" if not pd.isna(avg_inventory) else "N/A")

        # --- Charts ---
        st.subheader("Performance Over Time")
        df_agg = df_filtered.groupby('date').agg(
            total_adjusted_production=('adjusted_production_units', 'sum'),
            average_lead_time=('lead_time_days', 'mean'),
            total_inventory=('inventory_level_units', 'sum'),
            total_shortage=('shortage_units', 'sum')
        ).reset_index()
        # Plotting code... (unchanged)
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
    else:
        # This case is now handled by the date range check above or if df_filtered is empty
         st.info("Adjust filters or date range to view data.")

# ---------- Tab 2: Factory Deep Dive ----------
with tabs[1]:
    st.header("Factory Deep Dive")
    if df_filtered.empty and start_date_filter <= end_date_filter:
        st.warning("No data available for the selected filters in this run.")
    elif not df_filtered.empty:
        st.markdown("Select a specific factory below to see its detailed performance metrics.")
        # Ensure factory list is based on the *filtered* data if filters are active
        factories_in_filtered_data = sorted(df_filtered['factory_name'].unique().tolist())
        if not factories_in_filtered_data:
             st.warning("No factories match the current filter selection.")
        else:
            factory_to_inspect = st.selectbox(
                "Select Factory for Details:",
                options=factories_in_filtered_data,
                index=0, # Default to first available
                key="hist_factory_inspect"
            )
            if factory_to_inspect: # Should always be true if list is not empty
                df_factory = df_filtered[df_filtered['factory_name'] == factory_to_inspect].copy()
                st.subheader(f"Details for: {factory_to_inspect}")
                # Check again if df_factory is empty (can happen with very narrow date ranges)
                if df_factory.empty:
                    st.warning("No data available for this specific factory within the selected date range.")
                else:
                    # KPIs for the factory... (unchanged)
                    f_total_prod = df_factory['adjusted_production_units'].sum()
                    f_avg_lead = df_factory['lead_time_days'].mean()
                    f_total_shortage = df_factory['shortage_units'].sum()
                    f_avg_inv = df_factory['inventory_level_units'].mean()
                    fcol1, fcol2, fcol3, fcol4 = st.columns(4)
                    fcol1.metric("Total Production", f"{f_total_prod:,.0f} units")
                    fcol2.metric("Avg Lead Time", f"{f_avg_lead:.1f} days" if not pd.isna(f_avg_lead) else "N/A")
                    fcol3.metric("Total Shortage", f"{f_total_shortage:,.0f} units")
                    fcol4.metric("Avg Inventory", f"{f_avg_inv:,.0f} units" if not pd.isna(f_avg_inv) else "N/A")

                    # Plotting code... (unchanged, includes risk shading)
                    fig_f_prod = px.line(df_factory, x='date', y='adjusted_production_units', title='Adjusted Production', labels={'adjusted_production_units':'Units Produced'})
                    fig_f_prod.update_layout(hovermode="x unified")
                    risk_events = df_factory[df_factory['active_risks'] != 'none']
                    for _, event in risk_events.iterrows():
                         fig_f_prod.add_vrect(
                              x0=event['date'] - pd.Timedelta(days=3.5), x1=event['date'] + pd.Timedelta(days=3.5),
                              fillcolor="rgba(255, 0, 0, 0.1)", layer="below", line_width=0,
                              annotation_text=event['active_risks'], annotation_position="top left"
                         )
                    st.plotly_chart(fig_f_prod, use_container_width=True)
                    # ... other factory charts (inventory, lead time, shortage) ...
                    fig_f_inv = px.line(df_factory, x='date', y='inventory_level_units', title='Inventory Level', labels={'inventory_level_units':'Units'})
                    if 'safety_stock_level' in df_factory.columns:
                         fig_f_inv.add_trace(go.Scatter(x=df_factory['date'], y=df_factory['safety_stock_level'], mode='lines', name='Safety Stock', line=dict(dash='dash', color='grey')))
                    fig_f_inv.update_layout(hovermode="x unified")
                    st.plotly_chart(fig_f_inv, use_container_width=True)

                    fig_f_lead = px.line(df_factory, x='date', y='lead_time_days', title='Lead Time', labels={'lead_time_days':'Days'})
                    fig_f_lead.update_layout(hovermode="x unified")
                    st.plotly_chart(fig_f_lead, use_container_width=True)

                    fig_f_short = px.bar(df_factory, x='date', y='shortage_units', title='Shortage Events', labels={'shortage_units':'Unmet Demand (Units)'})
                    fig_f_short.update_layout(hovermode="x unified")
                    st.plotly_chart(fig_f_short, use_container_width=True)


                    st.subheader("Risk Events Log")
                    st.dataframe(risk_events[['date', 'active_risks', 'production_impact_factor', 'adjusted_production_units', 'lead_time_days']], use_container_width=True)
    else:
         st.info("Adjust filters or date range to view data.")


# ---------- Tab 3: Risk Analysis ----------
with tabs[2]:
    st.header("Risk Analysis")
    if df_filtered.empty and start_date_filter <= end_date_filter:
        st.warning("No data available for the selected filters in this run.")
    elif not df_filtered.empty:
        st.markdown("Analyzing the frequency and impact of simulated risk events across selected factories.")
        # Risk analysis code... (unchanged)
        df_risk = df_filtered.copy()
        # Handle potential NaN or non-string values in 'active_risks' before splitting
        df_risk['active_risks_list'] = df_risk['active_risks'].fillna('none').astype(str).apply(lambda x: [r.strip() for r in x.split(',')] if x != 'none' else ['none'])
        df_risk_exploded = df_risk.explode('active_risks_list')
        df_risk_events = df_risk_exploded[df_risk_exploded['active_risks_list'] != 'none'].copy()

        if df_risk_events.empty:
            st.info("No risk events occurred within the selected filters.")
        else:
            st.subheader("Risk Event Frequency")
            risk_counts = df_risk_events['active_risks_list'].value_counts().reset_index()
            risk_counts.columns = ['Risk Type', 'Frequency (Weeks Active)']
            fig_risk_freq = px.bar(risk_counts, x='Risk Type', y='Frequency (Weeks Active)', title="Frequency of Each Risk Type")
            st.plotly_chart(fig_risk_freq, use_container_width=True)

            st.subheader("Production Impact by Risk Type")
            # Ensure impact factor is numeric before filtering
            df_impact = df_risk_events[pd.to_numeric(df_risk_events['production_impact_factor'], errors='coerce') < 1.0].copy()
            if not df_impact.empty:
                 fig_impact_box = px.box(df_impact, x='active_risks_list', y='production_impact_factor',
                                        title="Distribution of Production Impact Factor by Risk Type (Lower is worse)",
                                        labels={'active_risks_list': 'Risk Type', 'production_impact_factor': 'Impact Factor'},
                                        points="all")
                 fig_impact_box.update_yaxes(range=[0, 1.1])
                 st.plotly_chart(fig_impact_box, use_container_width=True)
            else:
                 st.info("No events with production impact factor < 1.0 found in selection.")


            st.subheader("Lead Time Impact by Risk Type")
            # Ensure lead time is numeric
            df_risk_events['lead_time_days_numeric'] = pd.to_numeric(df_risk_events['lead_time_days'], errors='coerce')
            df_lead_impact = df_risk_events.dropna(subset=['lead_time_days_numeric'])
            if not df_lead_impact.empty:
                 fig_lead_box = px.box(df_lead_impact, x='active_risks_list', y='lead_time_days_numeric',
                                     title="Distribution of Lead Time by Risk Type",
                                     labels={'active_risks_list': 'Risk Type', 'lead_time_days_numeric': 'Lead Time (Days)'},
                                     points="all")
                 st.plotly_chart(fig_lead_box, use_container_width=True)
            else:
                 st.info("No valid lead time data found for risk events in selection.")


            st.subheader("Risk Events per Factory")
            # Use the exploded list column for grouping
            risk_per_factory = df_risk_events.groupby('factory_name')['active_risks_list'].value_counts().unstack(fill_value=0)
            st.dataframe(risk_per_factory)

    else:
         st.info("Adjust filters or date range to view data.")


# ---------- Tab 4: Predictive Forecast ----------
with tabs[3]:
    st.header("Predictive Forecast (Experimental)")
    if df_filtered.empty and start_date_filter <= end_date_filter:
         st.warning("No data available for the selected filters in this run to base forecast on.")
    elif not df_filtered.empty:
        st.markdown("""
        This section demonstrates a forecast using Exponential Smoothing based on historical data.
        Note: The model is basic and does not account for future disruptions.
        """)
        factories_in_filtered_data_fc = sorted(df_filtered['factory_name'].unique().tolist())
        if not factories_in_filtered_data_fc:
             st.warning("No factories match the current filter selection for forecasting.")
        else:
            forecast_factory = st.selectbox(
                "Select Factory to Forecast:",
                options=factories_in_filtered_data_fc,
                index=0,
                key="forecast_factory_select"
            )
            forecast_metric = st.selectbox(
                "Select Metric to Forecast:",
                options=['adjusted_production_units', 'inventory_level_units', 'lead_time_days', 'shortage_units'],
                index=0,
                key="forecast_metric_select" # Added key
            )
            if forecast_factory: # Should always be true here
                # Forecast logic... (unchanged)
                df_forecast_base = df_filtered[df_filtered['factory_name'] == forecast_factory].set_index('date').sort_index()
                # Check for sufficient non-null data points for the chosen metric
                if df_forecast_base.empty or df_forecast_base[forecast_metric].isnull().sum() >= len(df_forecast_base) - 1:
                    st.warning(f"Not enough valid data points for '{forecast_metric}' at {forecast_factory} within the selected range.")
                elif len(df_forecast_base.dropna(subset=[forecast_metric])) < 24: # Check after dropping NaNs for the specific metric
                    st.warning(f"Not enough historical data points ({len(df_forecast_base.dropna(subset=[forecast_metric]))}) for forecasting '{forecast_metric}'. Need at least 24.")
                else:
                    try:
                        ts_data = df_forecast_base[forecast_metric].copy()
                        # Ensure weekly frequency and fill gaps potentially needed by model
                        # Using 'W-SUN' assuming week ends on Sunday, adjust if needed
                        ts_data = ts_data.asfreq('W-SUN', method='ffill').fillna(method='ffill')

                        # Check again after filling if enough points
                        if len(ts_data) < 24:
                             st.warning(f"Still not enough data points ({len(ts_data)}) after frequency adjustment for forecasting '{forecast_metric}'.")
                        else:
                             # Fit ETS model (Holt-Winters)
                             model = ExponentialSmoothing(
                                  ts_data,
                                  trend='add',      # Assumes additive trend
                                  seasonal='add',   # Assumes additive seasonality
                                  seasonal_periods=52, # Assumes weekly data with yearly seasonality
                                  initialization_method='estimated' # Let statsmodels estimate initial levels/trend/seasonality
                              )
                             fit = model.fit()

                             # Make forecast
                             forecast_values = fit.forecast(FORECAST_PERIODS)
                             # Create forecast index starting after the last historical date
                             forecast_index = pd.date_range(
                                  start=ts_data.index[-1] + pd.Timedelta(weeks=1),
                                  periods=FORECAST_PERIODS,
                                  freq='W-SUN' # Match frequency
                              )
                             forecast_series = pd.Series(forecast_values, index=forecast_index)

                             # Plot historical data and forecast
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
                        logging.error(f"Forecast failed for {forecast_factory} / {forecast_metric}: {e}", exc_info=True) # Log traceback
            # else: # Case where forecast_factory is somehow None (shouldn't happen with current logic)
            #     st.info("Select a factory to generate a forecast.")
    else:
         st.info("Adjust filters or date range to view data.")


# ---------- Tab 5: Live Simulation ----------
with tabs[4]:
    st.header("🔬 Live Factory Simulation")
    st.markdown("""
    Configure and run a live simulation using the SimPy engine.
    Adjust risk parameters interactively and view updated KPIs and event timelines.
    """)
    sim_col1, sim_col2 = st.columns([1, 2]) # Setup and Parameters columns

    # --- Simulation Setup Column ---
    with sim_col1:
        st.subheader("Simulation Setup")
        # Select Factory Profile from Config
        factory_options = [f['name'] for f in sim_config.get('factories', [])]
        if not factory_options:
            st.error("Error: No factories defined in the simulation configuration.")
            st.stop()

        selected_factory_sim = st.selectbox(
            "Select Factory Profile:",
            options=factory_options,
            key="sim_factory_select" # Unique key for widget
        )
        # Get the configuration dictionary for the selected factory
        factory_sim_config = next((f for f in sim_config['factories'] if f['name'] == selected_factory_sim), None)

        # Simulation Duration Input
        sim_duration_weeks = st.number_input(
            "Simulation Duration (Weeks):",
            min_value=4, max_value=156, # 1 month to 3 years
            value=sim_config.get('simulation_defaults', {}).get('sim_duration_weeks', 52), # Use config default or 52
            step=4,
            key="sim_duration"
        )

        # Simulation Start Date Input
        default_start = datetime.now().date() - timedelta(weeks=sim_duration_weeks)
        sim_start_date = st.date_input(
            "Simulation Start Date:",
            value=default_start,
            key="sim_start_date"
        )

        # Run Button
        run_button = st.button("🚀 Run Live Simulation", key="run_sim_button")

    # --- Risk Parameter Adjustment Column ---
    with sim_col2:
        st.subheader("Adjust Risk Parameters")
        interactive_risk_params = {} # Dictionary to hold parameters modified by sliders
        # Dynamically create sliders for each risk factor found in the config
        for risk, params in sim_config.get('risk_factors', {}).items():
            # Use an expander for each risk type to keep the UI cleaner
            with st.expander(f"Configure: {risk.replace('_', ' ').title()}", expanded=False):
                # Probability Slider
                prob_key = f"{risk}_prob"
                prob = st.slider(f"Probability per day", min_value=0.0, max_value=0.01, value=params.get("prob_per_day", 0.001), step=0.0001, format="%.4f", key=prob_key, help="Average chance per day this risk event might occur.")
                # Impact Range Sliders
                impact_key_min = f"{risk}_impact_min"
                impact_key_max = f"{risk}_impact_max"
                impact_min = st.slider(f"Min Impact Factor", 0.0, 1.0, params.get("impact_factor_range", [0.5, 0.8])[0], 0.01, key=impact_key_min, help="Minimum production capacity multiplier (0=full stop, 1=no effect).")
                impact_max = st.slider(f"Max Impact Factor", 0.0, 1.0, params.get("impact_factor_range", [0.5, 0.8])[1], 0.01, key=impact_key_max, help="Maximum production capacity multiplier.")
                # Duration Range Sliders
                dur_key_min = f"{risk}_dur_min"
                dur_key_max = f"{risk}_dur_max"
                duration_min = st.slider(f"Min Duration (Days)", 1, 100, params.get("duration_range_days", [7, 28])[0], key=dur_key_min)
                duration_max = st.slider(f"Max Duration (Days)", 1, 100, params.get("duration_range_days", [7, 28])[1], key=dur_key_max)
                # Lead Time Multiplier Range Sliders
                lt_key_min = f"{risk}_lt_min"
                lt_key_max = f"{risk}_lt_max"
                lead_time_min = st.slider(f"Min Lead Time Multiplier", 1.0, 5.0, params.get("lead_time_multiplier_range", [1.0, 1.5])[0], 0.1, key=lt_key_min, help="Minimum factor applied to base lead time during disruption.")
                lead_time_max = st.slider(f"Max Lead Time Multiplier", 1.0, 5.0, params.get("lead_time_multiplier_range", [1.0, 1.5])[1], 0.1, key=lt_key_max, help="Maximum factor applied to base lead time.")

                # Store the interactive values
                interactive_risk_params[risk] = {
                    "prob_per_day": prob,
                    "impact_factor_range": [min(impact_min, impact_max), max(impact_min, impact_max)], # Ensure min <= max
                    "duration_range_days": [min(duration_min, duration_max), max(duration_min, duration_max)], # Ensure min <= max
                    "lead_time_multiplier_range": [min(lead_time_min, lead_time_max), max(lead_time_min, lead_time_max)], # Ensure min <= max
                    "mean_days_between_attempts": params.get("mean_days_between_attempts", 90) # Keep this from config for now
                }
        st.caption("Adjust sliders to override default risk parameters for this run.")

    # --- Simulation Execution and Results Display ---
    st.markdown("---") # Visual separator
    # Check if the button was pressed AND we have a valid factory config
    if run_button and factory_sim_config:
        with st.spinner(f"Running {sim_duration_weeks}-week simulation for {selected_factory_sim}... Please wait."):
            try:
                # Execute the SimPy simulation function with selected factory and *interactive* risk params
                df_sim_results, df_risk_log = run_factory_simulation(
                    factory_config=factory_sim_config,
                    risk_factors_config=interactive_risk_params, # Pass the adjusted parameters
                    sim_duration_weeks=sim_duration_weeks,
                    sim_start_date=pd.to_datetime(sim_start_date) # Convert date input to datetime
                )
                # Store results in session state to persist them across reruns until the next simulation
                st.session_state['sim_results'] = df_sim_results
                st.session_state['sim_risk_log'] = df_risk_log
                st.session_state['sim_factory_name'] = selected_factory_sim # Store which factory was simulated
                st.success(f"Live simulation for '{selected_factory_sim}' complete!")
            except Exception as e:
                st.error(f"Simulation run failed: {e}")
                logging.error(f"SimPy simulation error: {e}", exc_info=True) # Log full error
                # Clear previous results if run failed
                if 'sim_results' in st.session_state: del st.session_state['sim_results']
                if 'sim_risk_log' in st.session_state: del st.session_state['sim_risk_log']

    # Display results if they exist in session state
    if 'sim_results' in st.session_state and not st.session_state['sim_results'].empty:
        st.subheader(f"Live Simulation Results for: {st.session_state['sim_factory_name']}")
        df_display_sim = st.session_state['sim_results']
        df_display_risk = st.session_state['sim_risk_log']

        # KPIs from the live run... (unchanged)
        scol1, scol2, scol3, scol4 = st.columns(4)
        sim_total_prod = df_display_sim['adjusted_production_units'].sum()
        sim_avg_lead = df_display_sim['lead_time_days'].mean()
        sim_total_shortage = df_display_sim['shortage_units'].sum()
        sim_avg_inv = df_display_sim['inventory_level_units'].mean()
        scol1.metric("Total Production", f"{sim_total_prod:,.0f} units")
        scol2.metric("Avg Lead Time", f"{sim_avg_lead:.1f} days" if not pd.isna(sim_avg_lead) else "N/A")
        scol3.metric("Total Shortage", f"{sim_total_shortage:,.0f} units")
        scol4.metric("Avg Inventory", f"{sim_avg_inv:,.0f} units" if not pd.isna(sim_avg_inv) else "N/A")

        # Plotting results... (unchanged, includes timeline)
        st.plotly_chart(px.line(df_display_sim, x='date', y='adjusted_production_units', title='Simulated Production'), use_container_width=True)
        st.plotly_chart(px.line(df_display_sim, x='date', y='inventory_level_units', title='Simulated Inventory Level'), use_container_width=True)
        st.plotly_chart(px.line(df_display_sim, x='date', y='lead_time_days', title='Simulated Lead Time'), use_container_width=True)
        st.plotly_chart(px.bar(df_display_sim, x='date', y='shortage_units', title='Simulated Shortage'), use_container_width=True)

        # Risk Timeline Chart
        if not df_display_risk.empty:
             # Prepare data for timeline chart
             df_display_risk['Task'] = df_display_risk['risk_type'].str.replace('_', ' ').str.title() # Use formatted risk type for y-axis
             df_display_risk['Start'] = df_display_risk['start_date']
             df_display_risk['Finish'] = df_display_risk['end_date']
             # Create timeline
             fig_timeline = px.timeline(
                  df_display_risk,
                  x_start="Start",
                  x_end="Finish",
                  y="Task", # Display different risk types on y-axis
                  title="Simulated Risk Events Timeline",
                  color="Task", # Color bars by risk type
                  hover_data=["duration_days", "impact_factor", "lead_time_multiplier"]
              )
             fig_timeline.update_yaxes(autorange="reversed") # Often makes timelines easier to read
             st.plotly_chart(fig_timeline, use_container_width=True)
        else:
             st.info("No major risk events were triggered during this simulation run.")

        # Risk Event Log Table
        st.subheader("Live Simulation Risk Event Log")
        if not df_display_risk.empty:
            # Display relevant columns from the risk log
            st.dataframe(df_display_risk[['risk_type', 'start_date', 'end_date', 'duration_days', 'impact_factor', 'lead_time_multiplier']], use_container_width=True)
        else:
            st.info("No risk events recorded.")

    elif run_button: # If button was pressed but results are not in session state (implies failure)
         st.warning("Simulation did not produce results. Please check logs or parameters.")


# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.info("Supply Chain Digital Twin Dashboard V2.1")
# src/config.py
import os
import json
import logging
import streamlit as st # Needed for st.cache_data

# --- Constants ---
CONFIG_PATH = "../../../DEFAULT_CONFIG.json" # Adjust if needed relative to your project root if different
DATA_DIR = os.path.join(os.path.dirname(__file__), 'simulation_data') # Define relative to src/
FORECAST_PERIODS = 12
BACKEND_URL = "http://localhost:8000" # URL for the FastAPI backend (Adjust port if needed)

# --- Default Simulation Configuration ---
# (Your DEFAULT_SIM_CONFIG dictionary remains here - unchanged)
DEFAULT_SIM_CONFIG = {
    "factories": [
        {
            "name": "Foundry_Taiwan_A", "location": "Taiwan", "type": "Foundry",
            "base_production_capacity_weekly": [10000, 14000],
            "production_noise_std_dev_percent": 0.05,
            "base_lead_time_days": [25, 35],
            "initial_inventory_units": [20000, 40000],
            "base_weekly_demand": [7000, 12000]
        },
        {
            "name": "IDM_USA_B", "location": "USA", "type": "IDM",
            "base_production_capacity_weekly": [8000, 11000],
            "production_noise_std_dev_percent": 0.04,
            "base_lead_time_days": [20, 30],
            "initial_inventory_units": [15000, 30000],
            "base_weekly_demand": [7000, 10000]
        },
        {
            "name": "OSAT_Malaysia_C", "location": "Malaysia", "type": "OSAT",
            "base_production_capacity_weekly": [15000, 19000],
            "production_noise_std_dev_percent": 0.07,
            "base_lead_time_days": [15, 25],
            "initial_inventory_units": [30000, 50000],
            "base_weekly_demand": [12000, 17000]
        },
        {
            "name": "Foundry_Germany_D", "location": "Germany", "type": "Foundry",
            "base_production_capacity_weekly": [7000, 10000],
            "production_noise_std_dev_percent": 0.035,
            "base_lead_time_days": [28, 38],
            "initial_inventory_units": [10000, 25000],
            "base_weekly_demand": [7500, 11000]
        }
    ],
    "risk_factors": {
        "pandemic_closure": {"prob_per_day": 0.01, "impact_factor_range": [0.2, 0.6], "duration_range_days": [28, 84], "lead_time_multiplier_range": [1.5, 3.0], "mean_days_between_attempts": 60},
        "geopolitical_tension": {"prob_per_day": 0.015, "impact_factor_range": [0.7, 0.9], "duration_range_days": [14, 56], "lead_time_multiplier_range": [1.2, 1.8], "mean_days_between_attempts": 45},
        "trade_restrictions": {"prob_per_day": 0.02, "impact_factor_range": [0.6, 0.85], "duration_range_days": [21, 70], "lead_time_multiplier_range": [1.3, 2.0], "mean_days_between_attempts": 40},
        "cyber_attack": {"prob_per_day": 0.005, "impact_factor_range": [0.1, 0.5], "duration_range_days": [7, 21], "lead_time_multiplier_range": [1.1, 1.5], "mean_days_between_attempts": 90},
        "natural_disaster": {"prob_per_day": 0.008, "impact_factor_range": [0.0, 0.7], "duration_range_days": [7, 42], "lead_time_multiplier_range": [1.4, 4.0], "mean_days_between_attempts": 75},
        "logistics_bottleneck": {"prob_per_day": 0.025, "impact_factor_range": [1.0, 1.0], "duration_range_days": [7, 28], "lead_time_multiplier_range": [1.5, 2.5], "mean_days_between_attempts": 30}
    },
    "simulation_defaults": {
        "sim_duration_weeks": 52,
        "time_step_days": 1
    }
}


# --- Config Loading Function ---
@st.cache_data
def load_simulation_config(config_path=None):
    """Loads simulation config from file or returns default."""
    effective_path = config_path if config_path else CONFIG_PATH
    if not os.path.isabs(effective_path):
         # Go up two levels from config.py (src/) to reach project root, then potentially down
         project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
         config_in_project = os.path.join(project_root, effective_path.lstrip('../')) # Handle relative path like ../../..
         # Check if config is directly in src or project root for flexibility
         config_in_src = os.path.join(os.path.dirname(__file__), os.path.basename(effective_path))

         if os.path.exists(config_in_project):
              effective_path = config_in_project
         elif os.path.exists(config_in_src):
              effective_path = config_in_src
         else: # Try original relative path logic as last resort
              effective_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', effective_path))


    if effective_path and os.path.exists(effective_path):
        try:
            with open(effective_path, 'r') as f:
                logging.info(f"Loading simulation config from {effective_path}")
                return json.load(f)
        except Exception as e:
            logging.error(f"Failed to load config file '{effective_path}': {e}. Using default config.")
            st.warning(f"Could not load external config '{os.path.basename(effective_path)}'. Using default parameters.")
            return DEFAULT_SIM_CONFIG
    else:
        logging.info(f"Config file '{effective_path}' not found. Using default simulation configuration.")
        if config_path:
             st.warning(f"Specified config file '{os.path.basename(config_path)}' not found. Using default parameters.")
        return DEFAULT_SIM_CONFIG

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

# --- Function to get the loaded simulation config ---
@st.cache_data
def get_config():
    return load_simulation_config()
import pandas as pd
import numpy as np
import random
from datetime import timedelta, datetime
import json
import os
import argparse
import logging

# --- Configuration ---
# Moved parameters to a dictionary for better organization and potential loading from JSON/YAML
# This makes the script more configurable without touching the core logic.
DEFAULT_CONFIG = {
    "simulation_timeline": {
        "start_date": "2020-01-01", # Extended start date to capture more baseline
        "end_date": "2023-12-31",   # Extended end date
        "frequency": "W"           # Weekly granularity ('D' for daily, 'W' for weekly)
    },
    "factories": [
        {
            "name": "Foundry_Taiwan_A", "location": "Taiwan", "type": "Foundry",
            "base_production_capacity_weekly": [10000, 15000], # Range [min, max]
            "production_noise_std_dev": 500, # Weekly fluctuation
            "base_lead_time_days": [25, 35], # Range [min, max]
            "initial_inventory_units": [50000, 80000] # Range [min, max]
        },
        {
            "name": "IDM_USA_B", "location": "USA", "type": "IDM",
            "base_production_capacity_weekly": [8000, 12000],
            "production_noise_std_dev": 400,
            "base_lead_time_days": [20, 30],
            "initial_inventory_units": [40000, 60000]
        },
        {
            "name": "OSAT_Malaysia_C", "location": "Malaysia", "type": "OSAT",
            "base_production_capacity_weekly": [15000, 20000], # Assembly/Test often higher volume
            "production_noise_std_dev": 700,
            "base_lead_time_days": [15, 25],
            "initial_inventory_units": [60000, 90000]
        },
        {
            "name": "Foundry_Germany_D", "location": "Germany", "type": "Foundry",
            "base_production_capacity_weekly": [7000, 11000],
            "production_noise_std_dev": 350,
            "base_lead_time_days": [28, 38],
            "initial_inventory_units": [35000, 55000]
        }
        # Add more factories based on the provided list (TSMC, Samsung, Intel etc.)
        # Can specify location to potentially link to region-specific risks later
    ],
    "risk_factors": {
        # Prob: Chance per factory per time step (week/day)
        # Impact Factor Range: Multiplier applied to production (1 = no impact, 0 = full stop)
        # Duration Range: How many time steps the disruption lasts [min, max]
        # Lead Time Multiplier Range: Factor to increase base lead time [min, max]
        # Recovery Weeks: Weeks needed after disruption ends for production to normalize
        "pandemic_closure": {"prob": 0.015, "impact_factor_range": [0.2, 0.6], "duration_range_weeks": [4, 12], "lead_time_multiplier_range": [1.5, 3.0], "recovery_weeks": 4},
        "geopolitical_tension": {"prob": 0.02, "impact_factor_range": [0.7, 0.9], "duration_range_weeks": [2, 8], "lead_time_multiplier_range": [1.2, 1.8], "recovery_weeks": 2},
        "trade_restrictions": {"prob": 0.025, "impact_factor_range": [0.6, 0.85], "duration_range_weeks": [3, 10], "lead_time_multiplier_range": [1.3, 2.0], "recovery_weeks": 3},
        "cyber_attack": {"prob": 0.01, "impact_factor_range": [0.1, 0.5], "duration_range_weeks": [1, 3], "lead_time_multiplier_range": [1.1, 1.5], "recovery_weeks": 2},
        "natural_disaster": {"prob": 0.01, "impact_factor_range": [0.0, 0.7], "duration_range_weeks": [1, 6], "lead_time_multiplier_range": [1.4, 4.0], "recovery_weeks": 5},
        "logistics_bottleneck": {"prob": 0.03, "impact_factor_range": [1.0, 1.0], "duration_range_weeks": [1, 4], "lead_time_multiplier_range": [1.5, 2.5], "recovery_weeks": 1} # No production impact, only lead time
    },
    "inventory_simulation": {
        "base_weekly_demand_per_factory": [5000, 10000], # Range for simulated consumption/shipment
        "demand_std_dev_percent": 0.15, # Percentage of base demand
        "safety_stock_factor": 0.1 # As a fraction of initial inventory (simple approach)
    },
    "output": {
        "filename_template": "semiconductor_supplychain_simulated_{run_id}.csv"
    },
    "logging": {
        "level": "INFO" # DEBUG, INFO, WARNING, ERROR, CRITICAL
    }
}

# --- Logging Setup ---
logging.basicConfig(level=DEFAULT_CONFIG['logging']['level'], format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---

def load_config(config_path=None):
    """Loads configuration from a JSON file if provided, otherwise uses default."""
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                logging.info(f"Loading configuration from {config_path}")
                return json.load(f)
        except Exception as e:
            logging.error(f"Failed to load config file {config_path}: {e}. Using default config.")
            return DEFAULT_CONFIG
    else:
        logging.info("Using default configuration.")
        return DEFAULT_CONFIG

def generate_date_range(config):
    """Generates the date range based on simulation timeline config."""
    try:
        start_date = datetime.strptime(config['simulation_timeline']['start_date'], '%Y-%m-%d')
        end_date = datetime.strptime(config['simulation_timeline']['end_date'], '%Y-%m-%d')
        freq = config['simulation_timeline']['frequency']
        return pd.date_range(start=start_date, end=end_date, freq=freq)
    except Exception as e:
        logging.error(f"Error creating date range: {e}")
        raise

# --- Core Simulation Logic ---

def initialize_factory_states(config):
    """Initializes dynamic state for each factory (inventory, disruptions)."""
    factory_states = {}
    for factory in config['factories']:
        name = factory['name']
        initial_inventory = random.randint(*factory['initial_inventory_units'])
        factory_states[name] = {
            "current_inventory": initial_inventory,
            "active_disruptions": [], # List of active disruptions {risk_type, end_date, impact_factor, lead_time_multiplier, recovery_end_date}
            "base_production_capacity": random.randint(*factory['base_production_capacity_weekly']),
            "base_lead_time": random.randint(*factory['base_lead_time_days']),
            "safety_stock": int(initial_inventory * config['inventory_simulation']['safety_stock_factor'])
        }
    return factory_states

def check_for_new_disruptions(config, factory_name, current_date):
    """Checks if new disruptions start for a factory on the current date."""
    new_disruptions = []
    factory_config = next(f for f in config['factories'] if f['name'] == factory_name) # Find factory specific config if needed
    
    for risk_type, details in config['risk_factors'].items():
        if random.random() < details['prob']:
            duration_weeks = random.randint(*details['duration_range_weeks'])
            impact_factor = round(random.uniform(*details['impact_factor_range']), 3)
            lead_time_multiplier = round(random.uniform(*details['lead_time_multiplier_range']), 2)
            recovery_weeks = details['recovery_weeks']

            # Calculate end dates based on simulation frequency
            if config['simulation_timeline']['frequency'] == 'W':
                 end_date = current_date + timedelta(weeks=duration_weeks -1) # -1 because current week counts
                 recovery_end_date = end_date + timedelta(weeks=recovery_weeks)
            elif config['simulation_timeline']['frequency'] == 'D':
                 end_date = current_date + timedelta(days=duration_weeks*7 - 1) # Convert weeks to days
                 recovery_end_date = end_date + timedelta(days=recovery_weeks*7)
            else: # Default to weeks if frequency unknown
                 end_date = current_date + timedelta(weeks=duration_weeks -1)
                 recovery_end_date = end_date + timedelta(weeks=recovery_weeks)


            new_disruptions.append({
                "risk_type": risk_type,
                "start_date": current_date,
                "end_date": end_date,
                "impact_factor": impact_factor,
                "lead_time_multiplier": lead_time_multiplier,
                "recovery_end_date": recovery_end_date
            })
            logging.debug(f"New disruption: {factory_name} - {risk_type} starting {current_date}, ending {end_date}, recovery until {recovery_end_date}")

    return new_disruptions

def update_factory_state(factory_state, current_date):
    """Removes expired disruptions and handles recovery period."""
    active_disruptions = factory_state["active_disruptions"]
    
    # Filter out disruptions whose recovery period has ended
    factory_state["active_disruptions"] = [
        d for d in active_disruptions if current_date <= d["recovery_end_date"]
    ]
    return factory_state

def calculate_current_production(config, factory_config, factory_state, current_date):
    """Calculates adjusted production considering noise, disruptions, and recovery."""
    base_capacity = factory_state['base_production_capacity']
    noise_std_dev = factory_config['production_noise_std_dev']

    # Apply base noise/fluctuation
    base_production = max(0, np.random.normal(base_capacity, noise_std_dev))

    # Apply impacts from active disruptions
    final_impact_factor = 1.0 # Start with full production
    active_risk_types = []

    for disruption in factory_state["active_disruptions"]:
        if current_date <= disruption["end_date"]: # Main disruption phase
            final_impact_factor *= disruption["impact_factor"]
            active_risk_types.append(f"{disruption['risk_type']} (Active)")
        elif current_date <= disruption["recovery_end_date"]: # Recovery phase
            # Simple linear recovery model
            recovery_duration = (disruption["recovery_end_date"] - disruption["end_date"]).days
            days_into_recovery = (current_date - disruption["end_date"]).days
            
            if recovery_duration > 0:
                 # Calculate recovery progress (0 to 1)
                 recovery_progress = min(1.0, days_into_recovery / recovery_duration)
                 # Adjust impact factor back towards 1.0 based on recovery progress
                 current_impact = disruption["impact_factor"] + (1.0 - disruption["impact_factor"]) * recovery_progress
                 final_impact_factor *= current_impact
                 active_risk_types.append(f"{disruption['risk_type']} (Recovery)")
            else: # No recovery period or immediate recovery
                 # No impact during this phase if recovery duration is 0
                 pass
        # else: Disruption and recovery are fully over, no impact


    adjusted_production = int(base_production * final_impact_factor)
    
    # Ensure production is not negative
    adjusted_production = max(0, adjusted_production) 

    return adjusted_production, active_risk_types, final_impact_factor


def calculate_current_lead_time(config, factory_state, current_date):
    """Calculates adjusted lead time based on active disruptions."""
    base_lead_time = factory_state['base_lead_time']
    max_lead_time_multiplier = 1.0

    # Find the highest lead time multiplier among active (non-recovery) disruptions
    for disruption in factory_state["active_disruptions"]:
         if current_date <= disruption["end_date"]: # Only consider active phase for lead time impact
              max_lead_time_multiplier = max(max_lead_time_multiplier, disruption["lead_time_multiplier"])

    # Add some random noise to lead time as well
    noise = np.random.normal(0, 2) # Small daily/weekly fluctuation noise
    adjusted_lead_time = int(base_lead_time * max_lead_time_multiplier + noise)

    # Clamp lead time to a reasonable min/max (e.g., min 10 days, max 90 days)
    adjusted_lead_time = np.clip(adjusted_lead_time, 10, 90)

    return adjusted_lead_time

def simulate_inventory(config, factory_state, adjusted_production):
    """Simulates inventory changes based on production and stochastic demand."""
    prev_inventory = factory_state['current_inventory']

    # Simulate demand for this period
    demand_range = config['inventory_simulation']['base_weekly_demand_per_factory']
    demand_std_dev_percent = config['inventory_simulation']['demand_std_dev_percent']
    base_demand = random.uniform(*demand_range)
    demand_std_dev = base_demand * demand_std_dev_percent
    simulated_demand = max(0, int(np.random.normal(base_demand, demand_std_dev)))

    # Calculate units available and shipped
    available_units = prev_inventory + adjusted_production
    shipped_units = min(available_units, simulated_demand) # Can't ship more than available

    # Update inventory, ensuring it doesn't drop below zero (or safety stock if implemented differently)
    current_inventory = max(0, available_units - shipped_units) # Ensure non-negative inventory

    # Update factory state
    factory_state['current_inventory'] = current_inventory

    # Calculate shortage (demand that couldn't be met)
    shortage = max(0, simulated_demand - shipped_units)

    return current_inventory, simulated_demand, shipped_units, shortage

# --- Main Execution ---

def run_simulation(config, run_id="run1"):
    """Executes the entire supply chain simulation."""
    logging.info(f"Starting simulation run: {run_id}")

    date_range = generate_date_range(config)
    factory_states = initialize_factory_states(config)
    all_data = []

    logging.info(f"Simulating {len(config['factories'])} factories over {len(date_range)} time steps...")

    progress_interval = max(1, len(date_range) // 10) # Log progress every 10%

    for i, current_date in enumerate(date_range):
        if (i + 1) % progress_interval == 0:
             logging.info(f"Progress: Simulated up to {current_date.strftime('%Y-%m-%d')} ({((i+1)/len(date_range)*100):.0f}%)")

        for factory_config in config['factories']:
            factory_name = factory_config['name']
            factory_state = factory_states[factory_name]

            # 1. Update factory state (remove old disruptions)
            factory_state = update_factory_state(factory_state, current_date)

            # 2. Check for and add new disruptions
            new_disruptions = check_for_new_disruptions(config, factory_name, current_date)
            factory_state["active_disruptions"].extend(new_disruptions)

            # 3. Calculate current production
            adjusted_production, active_risks_list, impact_factor = calculate_current_production(config, factory_config, factory_state, current_date)

            # 4. Calculate current lead time
            adjusted_lead_time = calculate_current_lead_time(config, factory_state, current_date)

            # 5. Simulate inventory change
            current_inventory, simulated_demand, shipped_units, shortage = simulate_inventory(config, factory_state, adjusted_production)

            # 6. Record data for this time step
            all_data.append({
                "run_id": run_id,
                "factory_name": factory_name,
                "factory_location": factory_config['location'],
                "factory_type": factory_config['type'],
                "date": current_date,
                "base_production_capacity": factory_state['base_production_capacity'],
                "adjusted_production_units": adjusted_production,
                "production_impact_factor": round(impact_factor, 3), # Track the overall impact
                "active_risks": ", ".join(sorted(list(set(active_risks_list)))) if active_risks_list else "none",
                "lead_time_days": adjusted_lead_time,
                "simulated_demand_units": simulated_demand,
                "shipped_units": shipped_units,
                "inventory_level_units": current_inventory,
                "shortage_units": shortage, # How much demand was unmet
                "safety_stock_level": factory_state['safety_stock']
            })

            # Update the state in the main dictionary
            factory_states[factory_name] = factory_state

    logging.info("Simulation complete.")

    # Create DataFrame
    df = pd.DataFrame(all_data)

    # Save DataFrame
    output_filename = config['output']['filename_template'].format(run_id=run_id)
    try:
        df.to_csv(output_filename, index=False)
        logging.info(f"Dataset saved to {output_filename}")
    except Exception as e:
        logging.error(f"Failed to save dataset to {output_filename}: {e}")

    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Semiconductor Supply Chain Simulation.")
    parser.add_argument("--config", type=str, help="Path to JSON configuration file (optional).")
    parser.add_argument("--runs", type=int, default=1, help="Number of simulation runs to perform.")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save output CSV files.")

    args = parser.parse_args()

    config = load_config(args.config)

    # Ensure output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        logging.info(f"Created output directory: {args.output_dir}")
        
    # Update output path in config
    config['output']['filename_template'] = os.path.join(args.output_dir, os.path.basename(config['output']['filename_template']))


    for i in range(args.runs):
        run_id = f"run_{i+1:03d}" # e.g., run_001, run_002
        run_simulation(config, run_id=run_id)

    logging.info(f"Successfully completed {args.runs} simulation run(s).")
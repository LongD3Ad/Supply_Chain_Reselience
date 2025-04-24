# src/simulation/engine.py
import simpy
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

# Note: Configuration (like risk params, factory details) will be passed IN to run_factory_simulation

def run_factory_simulation(factory_config, risk_factors_config, sim_duration_weeks, sim_start_date):
    """Runs the SimPy simulation for a single factory."""
    env = simpy.Environment()
    TIME_STEP_DAYS = 1
    SIM_DURATION_DAYS = sim_duration_weeks * 7

    # --- Factory State Initialization ---
    try:
        base_weekly_capacity = random.randint(*factory_config['base_production_capacity_weekly'])
        base_daily_capacity = base_weekly_capacity / 7.0
        daily_noise_std_dev = base_daily_capacity * factory_config['production_noise_std_dev_percent']

        factory_state = {
            "name": factory_config['name'],
            "base_daily_capacity": base_daily_capacity,
            "daily_noise_std_dev": daily_noise_std_dev,
            "base_lead_time": random.randint(*factory_config['base_lead_time_days']),
            "inventory": random.randint(*factory_config['initial_inventory_units']),
            "weekly_demand_range": factory_config.get('base_weekly_demand', [5000, 10000]), # Use .get for safety
            "risk_log": [],
            "active_risks": {}, # risk_name -> end_time
            "current_risk_details": {} # risk_name -> {impact_factor, lead_time_multiplier}
        }
        logging.info(f"Starting simulation for {factory_state['name']}. Base daily capacity: {base_daily_capacity:.0f}, Base lead time: {factory_state['base_lead_time']}, Initial Inv: {factory_state['inventory']}")
    except KeyError as e:
        logging.error(f"Missing key in factory_config for {factory_config.get('name', 'Unknown')}: {e}")
        raise ValueError(f"Configuration error for factory {factory_config.get('name', 'Unknown')}: Missing key {e}") from e
    except Exception as e:
         logging.error(f"Error initializing factory state for {factory_config.get('name', 'Unknown')}: {e}", exc_info=True)
         raise

    output_log = []

    # --- Nested SimPy Processes ---
    # (Copy the production_process function definition from your original script here)
    def production_process(env, factory_state, output_log):
        # (Content of production_process function - NO CHANGES NEEDED YET)
        weekly_production = 0
        weekly_demand = 0
        weekly_shipped = 0
        weekly_shortage = 0
        current_week = 0

        while True:
            current_production_factor = 1.0
            current_max_lt_multiplier = 1.0
            active_risk_names = list(factory_state["active_risks"].keys())

            for r_name in active_risk_names:
                details = factory_state["current_risk_details"].get(r_name, {})
                current_production_factor *= details.get("impact_factor", 1.0)
                current_max_lt_multiplier = max(current_max_lt_multiplier, details.get("lead_time_multiplier", 1.0))

            effective_daily_capacity = factory_state["base_daily_capacity"] * current_production_factor
            daily_prod = max(0, np.random.normal(effective_daily_capacity, factory_state["daily_noise_std_dev"]))

            weekly_production += daily_prod
            factory_state["inventory"] += daily_prod

            daily_demand_mean = random.uniform(*factory_state["weekly_demand_range"]) / 7.0
            daily_demand = max(0, np.random.normal(daily_demand_mean, daily_demand_mean * 0.3))
            weekly_demand += daily_demand

            shipped = min(factory_state["inventory"], daily_demand)
            factory_state["inventory"] -= shipped
            weekly_shipped += shipped
            shortage_today = max(0, daily_demand - shipped)
            weekly_shortage += shortage_today

            yield env.timeout(TIME_STEP_DAYS)

            if env.now > 0 and env.now % 7 == 0:
                current_week += 1
                current_date = sim_start_date + timedelta(days=env.now - 1)
                effective_lead_time = int(factory_state["base_lead_time"] * current_max_lt_multiplier)
                active_risks_str = ', '.join(active_risk_names) if active_risk_names else 'none'

                output_log.append({
                    "week": current_week, "date": current_date, "factory_name": factory_state["name"],
                    "base_production_capacity": int(factory_state["base_daily_capacity"] * 7),
                    "adjusted_production_units": int(weekly_production),
                    "production_impact_factor": round(current_production_factor, 3),
                    "simulated_demand_units": int(weekly_demand), "shipped_units": int(weekly_shipped),
                    "inventory_level_units": int(factory_state["inventory"]),
                    "shortage_units": int(weekly_shortage), "lead_time_days": effective_lead_time,
                    "lead_time_multiplier": round(current_max_lt_multiplier, 2),
                    "active_risks": active_risks_str, "active_risks_count": len(active_risk_names),
                })
                weekly_production, weekly_demand, weekly_shipped, weekly_shortage = 0, 0, 0, 0


    # (Copy the risk_event_generator function definition from your original script here)
    def risk_event_generator(env, risk_name, risk_params, factory_state):
        # (Content of risk_event_generator function - NO CHANGES NEEDED YET)
        risk_counter = 0
        while True:
            mean_wait = risk_params.get("mean_days_between_attempts", 90)
            prob_per_day = risk_params.get("prob_per_day", 0.01)
            time_to_next_attempt = np.random.exponential(mean_wait) if mean_wait > 0 else float('inf')
            if time_to_next_attempt == float('inf'): break
            yield env.timeout(time_to_next_attempt)
            logging.debug(f"[{factory_state['name']}@{env.now:.1f}] Attempting check for risk '{risk_name}' (Prob: {prob_per_day}, Mean Wait: {mean_wait})")

            if random.random() < prob_per_day:
                 if risk_name not in factory_state["active_risks"]:
                    risk_counter += 1
                    impact_factor = random.uniform(*risk_params["impact_factor_range"])
                    duration_days = random.randint(*risk_params["duration_range_days"])
                    lead_time_multiplier = random.uniform(*risk_params["lead_time_multiplier_range"])
                    start_time = env.now
                    end_time = start_time + duration_days
                    logging.info(f"RISK TRIGGER {risk_counter}: '{risk_name}' at Day {start_time:.1f} for {duration_days} days. Impact: {impact_factor:.2f}, LT Mult: {lead_time_multiplier:.2f}. Factory: {factory_state['name']}")

                    factory_state["risk_log"].append({
                        "risk_type": risk_name, "start_day": start_time, "end_day": end_time,
                        "duration_days": duration_days, "impact_factor": impact_factor,
                        "lead_time_multiplier": lead_time_multiplier
                    })
                    factory_state["active_risks"][risk_name] = end_time
                    factory_state["current_risk_details"][risk_name] = {"impact_factor": impact_factor, "lead_time_multiplier": lead_time_multiplier}

                    yield env.timeout(duration_days)

                    logging.info(f"RISK END: '{risk_name}' ended at Day {env.now:.1f}. Factory: {factory_state['name']}")
                    if risk_name in factory_state["active_risks"]: del factory_state["active_risks"][risk_name]
                    if risk_name in factory_state["current_risk_details"]: del factory_state["current_risk_details"][risk_name]


    # --- Start Processes & Run Simulation ---
    env.process(production_process(env, factory_state, output_log))
    if risk_factors_config: # Only add risk generators if config is provided
        for risk_name, risk_params in risk_factors_config.items():
            if risk_params: # Check if params are valid
                 env.process(risk_event_generator(env, risk_name, risk_params, factory_state))
            else:
                 logging.warning(f"Missing or invalid parameters for risk '{risk_name}' in factory {factory_state['name']}. Skipping.")

    env.run(until=SIM_DURATION_DAYS)

    # --- Process Results ---
    # (Copy the results processing part from the end of your original run_factory_simulation here)
    df_sim_results = pd.DataFrame(output_log)
    df_risk_log = pd.DataFrame(factory_state["risk_log"])
    # ... (rest of the results processing, date conversion, column checks, logging) ...
    if not df_risk_log.empty:
        df_risk_log['start_day_int'] = df_risk_log['start_day'].fillna(0).astype(int)
        df_risk_log['end_day_int'] = df_risk_log['end_day'].fillna(0).astype(int)
        sim_start_date_naive = pd.to_datetime(sim_start_date).tz_localize(None) # Ensure naive
        df_risk_log['start_date'] = df_risk_log['start_day_int'].apply(lambda d: sim_start_date_naive + timedelta(days=d))
        df_risk_log['end_date'] = df_risk_log['end_day_int'].apply(lambda d: sim_start_date_naive + timedelta(days=d))
        df_risk_log = df_risk_log.drop(columns=['start_day_int', 'end_day_int'])
    else:
        df_risk_log = pd.DataFrame(columns=['risk_type', 'start_day', 'end_day', 'duration_days', 'impact_factor', 'lead_time_multiplier', 'start_date', 'end_date'])

    expected_sim_cols = ["week", "date", "factory_name", "base_production_capacity", "adjusted_production_units",
                        "production_impact_factor", "simulated_demand_units", "shipped_units",
                        "inventory_level_units", "shortage_units", "lead_time_days", "lead_time_multiplier",
                        "active_risks", "active_risks_count"]
    for col in expected_sim_cols:
        if col not in df_sim_results.columns: df_sim_results[col] = pd.NA

    numeric_sim_cols = ['adjusted_production_units', 'production_impact_factor', 'lead_time_days',
                        'lead_time_multiplier', 'simulated_demand_units', 'shipped_units',
                        'inventory_level_units', 'shortage_units', 'active_risks_count']
    for col in numeric_sim_cols:
         if col in df_sim_results.columns: df_sim_results[col] = pd.to_numeric(df_sim_results[col], errors='coerce')

    df_sim_results['date'] = pd.to_datetime(df_sim_results['date'], errors='coerce')

    final_inv = factory_state['inventory']
    total_shortage = df_sim_results['shortage_units'].sum() if 'shortage_units' in df_sim_results.columns else 0
    logging.info(f"Simulation finished for {factory_state['name']}. Final Inventory: {final_inv:.0f}. Total Shortage: {total_shortage:.0f}. Risks Logged: {len(df_risk_log)}")

    return df_sim_results, df_risk_log
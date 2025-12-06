import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import shutil
import glob

# --- 1. Configuration des chemins et des données ---

# Base path for the original, non-cleaned files (This path must contain all the CM61-LR-scen files)
ORIGINAL_DATA_BASE_PATH = "C:\\Users\\clari\\Documents\\MA 2\\atm processes\\project\\"

# Total analysis period
PERIOD_START = 2025
PERIOD_END = 2094

# Reference period for anomaly calculation (first decade)
REF_START = 2025 
REF_END = 2034

# Scenario and region definitions
REGIONS = ['arctic', 'europe']
SCENARIOS = ['ssp245', 'ssp585']
SCENARIO_COLORS = {'ssp245': 'blue', 'ssp585': 'red'}
TIME_COORD_NAME = 'time_counter' 

# --- Functions for Data Loading (Unified RAW loading) ---

def load_raw_variable(scenario, region, variable_name, base_path):
    """
    Loads a specific variable from the ORIGINAL (non-cleaned) file path.
    Returns the time series of the spatial mean of that variable across lat/lon.
    If the variable is T2m or SICF, it will be 2D (time).
    If the variable is OVAP or THETA, it will be 3D (time, vertical_level).
    """
    print(f"Loading RAW variable '{variable_name}' for {scenario.upper()} - {region.upper()}")
    
    # Pattern: CM61-LR-scen-sspXXXX_YYYYMMDD_YYYYMMDD_1M_histmth_REGION.nc
    file_pattern = os.path.join(base_path, f'CM61-LR-scen-{scenario}_*_{region}.nc')
    file_list = glob.glob(file_pattern)

    if not file_list:
        print(f"Error: No RAW files found for {scenario}-{region} at {base_path}")
        return None
        
    datasets = []
    try:
        for file_path in file_list:
            # Load only the specified variable to save memory
            ds = xr.open_dataset(file_path, engine='netcdf4')
            
            if variable_name not in ds.data_vars:
                 print(f"Warning: Variable '{variable_name}' not found in {os.path.basename(file_path)}. Skipping.")
                 ds.close()
                 continue
                 
            # Extract the variable, convert to Dataset (to keep metadata) and load into memory
            ds_var = ds[variable_name].to_dataset().load()
            datasets.append(ds_var)
            
        if not datasets:
            return None
            
        # Concat and calculate the regional mean of the variable across lat/lon
        ds_full = xr.concat(datasets, dim=TIME_COORD_NAME)
        
        # Return the DataArray. If it has vertical levels, the mean over lat/lon will preserve them.
        return ds_full[variable_name].mean(dim=('lat', 'lon'))
        
    except Exception as e:
        print(f"Error loading RAW variable '{variable_name}' for {scenario}-{region}: {e}")
        return None


def calculate_regional_anomalies(ds_t2m_mean):
    """Calculates the regional T2m anomaly based on the reference period."""
    # ds_t2m_mean is already the regional mean (2D: time)
    regional_ref = ds_t2m_mean.sel({TIME_COORD_NAME: slice(f'{REF_START}', f'{REF_END}')}).mean()
    regional_anomaly = ds_t2m_mean - regional_ref
    return regional_anomaly

# Define the exponential function for fitting
def exponential_func(t, a, b):
    # a: scaling factor, b: growth rate
    return a * np.exp(b * t)

def plot_t2m_anomalies(analysis_results):
    """
    Plots the T2m temperature anomaly (6-Month Moving Average + Exponential Fit excluding boundaries).
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # NEW CONFIGURATION
    MA_WINDOW_MONTHS = 12 # 6-month moving average
    EXCLUDE_PERIOD_MONTHS = 6 # Exclude first/last 6 months for fitting
    
    print("\n--- Exponential Fit Results (Annual Growth Rate) ---")

    for region, results in analysis_results.items():
        for scenario, data in results.items():
            
            # --- 1. SMOOTHED DATA (6-MONTH MOVING AVERAGE) ---
            anom_raw = data['regional_anom']
            # Apply 6-month rolling mean
            anom_smoothed_6mo = anom_raw.rolling(
                {TIME_COORD_NAME: MA_WINDOW_MONTHS}, 
                center=True, 
                min_periods=1
            ).mean() 
            
            base_color = data['color']
            
            # Plot 6-month smoothed data (thin line, high transparency)
            ax.plot(
                anom_raw[TIME_COORD_NAME], anom_smoothed_6mo.values,
                label=f'{region.capitalize()} ({scenario.upper()}) - 6-Month MA', 
                color=base_color, 
                linewidth=1.0,
                alpha=0.6,
                linestyle='-'
            )
            
            # --- 2. EXPONENTIAL REGRESSION (TREND) ---
            
            # 2a. Define the safe data subset for fitting
            N_total = len(anom_raw)
            # Indices: Exclude first EXCLUDE_PERIOD_MONTHS and last EXCLUDE_PERIOD_MONTHS
            start_index = EXCLUDE_PERIOD_MONTHS
            end_index = N_total - EXCLUDE_PERIOD_MONTHS
            
            # Data used for fitting
            anom_fit_data = anom_smoothed_6mo.values[start_index:end_index]
            
            # Time array used for fitting (starts from 0, scaled to years)
            time_numeric = np.arange(N_total) / 12.0
            time_fit_data = time_numeric[start_index:end_index]
            
            try:
                # Calculate the fit
                popt, pcov = curve_fit(exponential_func, time_fit_data, anom_fit_data, p0=[0.1, 0.01], maxfev=5000)
                b = popt[1] # Extract growth rate 'b'
                
                # Trend line values calculated over the entire period
                trend_line = exponential_func(time_numeric, *popt)
                
                # Plot the exponential trend line (Thick line)
                ax.plot(
                    anom_raw[TIME_COORD_NAME], trend_line,
                    label=f'{region.capitalize()} ({scenario.upper()}) - Exp. Trend (Growth Rate: {b*100:.2f}%/yr)', 
                    color=base_color, 
                    linewidth=3.0,
                    linestyle='-' if region == 'Arctic' else '--'
                )
                
                # Print regression results
                print(f"  {region.capitalize()} ({scenario.upper()}) - Exp. Growth Rate (b): {b:.6f} (Annual)")
                
            except RuntimeError:
                print(f"  {region.capitalize()} ({scenario.upper()}): WARNING - Exponential fit failed to converge.")
            
    ax.set_title(f"T2m Temperature Anomaly (2025-2094) - Smoothed Data and Exponential Trend")
    ax.set_ylabel("T2m Anomaly (K)")
    ax.set_xlabel("Date")
    ax.axhline(0.0, color='black', linestyle=':', alpha=0.7)
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


def plot_arctic_vs_europe_ratio(analysis_results):
    """
    Plots the Arctic vs. Europe Ratio: Delta T_Europe / Delta T_Arctic.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    arctic_results = analysis_results['Arctic']
    europe_results = analysis_results['Europe']
    
    for scenario in SCENARIOS:
        # Raw regional anomalies
        anom_arctic = arctic_results[scenario.upper()]['regional_anom']
        anom_europe = europe_results[scenario.upper()]['regional_anom']
        
        # Calculate the Ratio: Europe / Arctic
        ratio_ap = np.where(anom_arctic != 0, anom_europe / anom_arctic, np.nan)
        ratio_ap = xr.DataArray(ratio_ap, coords=anom_arctic.coords)
        
        # Smoothing (10-year rolling mean = 120 months)
        ratio_smoothed = ratio_ap.rolling(
            {TIME_COORD_NAME: 120}, 
            center=True, 
            min_periods=1
        ).mean() 
        
        # Plot
        ratio_smoothed.plot(
            ax=ax, 
            label=f'Europe / Arctic Ratio ({scenario.upper()}) - 10-Year Rolling Mean', 
            color=SCENARIO_COLORS[scenario], 
            linewidth=2
        )

    ax.set_title(f"Europe Warming Relative to Arctic Warming ({PERIOD_START}-{PERIOD_END})")
    ax.set_ylabel(r"Relative Warming Ratio ($\Delta T_{Europe}/\Delta T_{Arctic}$)")
    ax.set_xlabel("Date")
    ax.axhline(1.0, color='black', linestyle=':', alpha=0.7, label='Ratio = 1 (Warming is Equal)')
    
    # We expect the ratio to be < 1.0, showing that Europe warms slower than the Arctic
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Ratio = 0.5 (Europe warms half as fast)')
    ax.set_ylim(0.0, 1.2) # Focus the y-axis to confirm the ratio < 1.0
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


# --- 4. Additional Variable Plots (Water Vapor and Potential Temperature) ---

def plot_additional_anomalies(analysis_results_dict, variable_name, title, ylabel):
    """
    Plots the anomaly of any additional variable (Water Vapor or Theta).
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    for region, results in analysis_results_dict.items():
        for scenario, data in results.items():
            
            # Apply 12-month rolling mean for smoothing
            anom_smoothed_1yr = data['regional_anom'].rolling(
                {TIME_COORD_NAME: 12}, 
                center=True, 
                min_periods=1
            ).mean() 
            
            # Plot
            anom_smoothed_1yr.plot(
                ax=ax, 
                label=f'{region.capitalize()} ({scenario.upper()}) - 1-Year MA', 
                color=data['color'], 
                linewidth=2,
                linestyle='-' if region == 'Arctic' else '--'
            )
            
    ax.set_title(f"{title} Anomaly ({PERIOD_START}-{PERIOD_END})")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Date")
    ax.axhline(0.0, color='black', linestyle=':', alpha=0.7)
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

# --- 5. Main Execution Block ---

if __name__ == '__main__':
    
    # Initialization of results dictionaries
    t2m_analysis_results = {'Arctic': {}, 'Europe': {}}
    sicf_analysis_results = {'Arctic': {}, 'Europe': {}}
    vapor_analysis_results = {'Arctic': {}, 'Europe': {}}
    theta_analysis_results = {'Arctic': {}, 'Europe': {}}
    
    # --- Step 1: T2M Analysis (Primary Quantification) ---
    print("\n--- T2M TEMPERATURE ANALYSIS (Primary Quantification) ---")

    for scenario in SCENARIOS:
        for region in REGIONS:
            
            # Load mean T2m time series
            ds_t2m_mean = load_raw_variable(scenario, region, 't2m', ORIGINAL_DATA_BASE_PATH)

            if ds_t2m_mean is None:
                continue

            # Calculate regional T2m anomalies
            regional_anom = calculate_regional_anomalies(ds_t2m_mean)

            # Store T2m results
            t2m_analysis_results[region.capitalize()][scenario.upper()] = {
                'regional_anom': regional_anom,
                'color': SCENARIO_COLORS[scenario]
            }
            
            # --- DEBUGGING T2M (End anomalies) ---
            end_index = -1
            regional_end = regional_anom.isel({TIME_COORD_NAME: end_index}).item()
            print(f"  --- DEBUG ({scenario.upper()}-{region.upper()}) ---")
            print(f"  Regional Anomaly (Start): {regional_anom[0].item():.4f} K")
            print(f"  Regional Anomaly (End): {regional_end:.4f} K")
            print("  --------------------------------------")

    # Final preparation for AP Ratio: copy Arctic anomaly to Global Anom field
    for scenario in SCENARIOS:
        if 'Arctic' in t2m_analysis_results and scenario.upper() in t2m_analysis_results['Arctic']:
            arctic_anom = t2m_analysis_results['Arctic'][scenario.upper()]['regional_anom']
            
            # Assign Arctic anomaly as the denominator (global_anom) for all regions
            t2m_analysis_results['Arctic'][scenario.upper()]['global_anom'] = arctic_anom
            if 'Europe' in t2m_analysis_results and scenario.upper() in t2m_analysis_results['Europe']:
                 t2m_analysis_results['Europe'][scenario.upper()]['global_anom'] = arctic_anom
    
    # --- Step 2: Plot T2m Anomaly + Exponential Trend (Focus on Magnitude) ---
    plot_t2m_anomalies(t2m_analysis_results)
    
    # --- Step 3: Plot Arctic vs Europe Ratio (Focus on Relative Amplification) ---
    plot_arctic_vs_europe_ratio(t2m_analysis_results)


    # --- Step 4: Water Vapor ('ovap') and Potential Temperature ('theta') Analysis ---
    print("\n--- Atmospheric Feedback Analysis (RAW files) ---")
    
    VAPOR_VARIABLE = 'ovap' 
    THETA_VARIABLE = 'theta'
    
    for scenario in SCENARIOS:
        for region in REGIONS:
            
            # --- A. VAPOR ANALYSIS ---
            ovap_mean = load_raw_variable(scenario, region, VAPOR_VARIABLE, ORIGINAL_DATA_BASE_PATH)
            
            if ovap_mean is not None:
                # Calculate anomaly
                ovap_mean_time_series = ovap_mean.mean(dim=[dim for dim in ovap_mean.dims if dim != TIME_COORD_NAME])
                ovap_ref = ovap_mean_time_series.sel({TIME_COORD_NAME: slice(f'{REF_START}', f'{REF_END}')}).mean()
                ovap_anomaly = ovap_mean_time_series - ovap_ref
                
                vapor_analysis_results[region.capitalize()][scenario.upper()] = {
                    'regional_anom': ovap_anomaly,
                    'color': SCENARIO_COLORS[scenario]
                }
            
            # --- B. THETA ANALYSIS ---
            theta_mean = load_raw_variable(scenario, region, THETA_VARIABLE, ORIGINAL_DATA_BASE_PATH)
            
            if theta_mean is not None:
                # Calculate anomaly
                theta_mean_time_series = theta_mean.mean(dim=[dim for dim in theta_mean.dims if dim != TIME_COORD_NAME])
                theta_ref = theta_mean_time_series.sel({TIME_COORD_NAME: slice(f'{REF_START}', f'{REF_END}')}).mean()
                theta_anomaly = theta_mean_time_series - theta_ref
                
                theta_analysis_results[region.capitalize()][scenario.upper()] = {
                    'regional_anom': theta_anomaly,
                    'color': SCENARIO_COLORS[scenario]
                }

    # Plot Water Vapor Analysis (only if data was successfully loaded)
    if any(vapor_analysis_results['Arctic']) or any(vapor_analysis_results['Europe']):
        plot_additional_anomalies(vapor_analysis_results, VAPOR_VARIABLE, "Specific Humidity (Water Vapor) Anomaly", "Specific Humidity Anomaly (kg/kg)")
    else:
        print("Skipping Water Vapor analysis: Variable 'ovap' not found in original files.")

    # Plot Potential Temperature Analysis (only if data was successfully loaded)
    if any(theta_analysis_results['Arctic']) or any(theta_analysis_results['Europe']):
        plot_additional_anomalies(theta_analysis_results, THETA_VARIABLE, "Potential Air Temperature Anomaly", "Potential Temperature Anomaly (K)")
    else:
        print("Skipping Potential Temperature analysis: Variable 'theta' not found in original files.")


    # --- Step 5: Sea Ice Evolution (sicf) ---

    print("\n--- Sea Ice Fraction (sicf) Analysis ---")
    
    for scenario in SCENARIOS:
        # Load mean SICF time series
        ds_sicf_mean = load_raw_variable(scenario, 'arctic', 'sicf', ORIGINAL_DATA_BASE_PATH)
        
        if ds_sicf_mean is None:
            continue
        
        # Calculate annual mean SICF
        sicf_annual = ds_sicf_mean.resample({TIME_COORD_NAME: 'YE'}).mean()
        
        sicf_analysis_results['Arctic'][scenario.upper()] = {
            'regional_anom': sicf_annual, # SICF is plotted directly, not as an anomaly relative to the start
            'color': SCENARIO_COLORS[scenario]
        }


    if any(sicf_analysis_results['Arctic']):
        plt.figure(figsize=(10, 5))
        
        for scenario, data in sicf_analysis_results['Arctic'].items():
            data['regional_anom'].plot(label=f'Mean SICF ({scenario})', color=data['color'], linewidth=2)
        
        plt.title(f"Evolution of Mean Sea Ice Fraction (SICF) in the Arctic ({PERIOD_START}-{PERIOD_END})")
        plt.ylabel("Mean Sea Ice Fraction")
        plt.xlabel("Year")
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend()
        plt.show()
    else:
        print("Skipping SICF analysis due to loading errors.")

    print("\nFinal analysis script executed with Arctic vs Europe ratio and added vapor/theta analysis.")
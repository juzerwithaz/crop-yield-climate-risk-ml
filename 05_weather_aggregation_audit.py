import os
import glob
import json
import calendar
import pandas as pd
import numpy as np

def run_weather_aggregation_audit():
    print("=" * 80)
    print(" 05_WEATHER_AGGREGATION_AUDIT: NASA POWER METADATA & SCALING")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. INSPECT METADATA FROM AT LEAST 3 DISTRICT FILES
    # -------------------------------------------------------------------------
    print("\n--- 1. METADATA INSPECTION (Sample of 3 Districts) ---")
    weather_files = glob.glob("weather_raw_*.json")
    
    if len(weather_files) < 3:
        print(f"WARNING: Found only {len(weather_files)} weather JSON files. Proceeding with available files.")
        if len(weather_files) == 0:
            print("ERROR: No weather_raw_*.json files found. Terminating audit.")
            return
            
    sample_files = weather_files[:3]
    metadata_records = []
    
    for file in sample_files:
        district_name = os.path.basename(file).replace("weather_raw_", "").replace(".json", "").replace("_", " ").strip().upper()
        with open(file, 'r') as f:
            data = json.load(f)
            
        header = data.get('header', {})
        parameters = data.get('parameters', {})
        
        # NASA POWER sometimes puts metadata in 'parameters'
        precip_meta = parameters.get('PRECTOTCORR', {})
        temp_meta = parameters.get('T2M', {})
        humid_meta = parameters.get('RH2M', {})
        
        metadata_records.append({
            'District': district_name,
            'Temporal_Resolution': header.get('temporal_resolution', 'Unknown'),
            'PRECTOTCORR_Units': precip_meta.get('units', 'Unknown'),
            'T2M_Units': temp_meta.get('units', 'Unknown'),
            'RH2M_Units': humid_meta.get('units', 'Unknown')
        })
        
        print(f"District: {district_name}")
        print(f"  - Temporal Resolution: {header.get('temporal_resolution', 'Unknown')}")
        print(f"  - PRECTOTCORR Units:   {precip_meta.get('units', 'Unknown')}")
        print(f"  - T2M Units:           {temp_meta.get('units', 'Unknown')}")
        print(f"  - RH2M Units:          {humid_meta.get('units', 'Unknown')}\n")

    # Verify if metadata is identical across the sample
    is_uniform = all(
        (r['Temporal_Resolution'] == metadata_records[0]['Temporal_Resolution']) and
        (r['PRECTOTCORR_Units'] == metadata_records[0]['PRECTOTCORR_Units']) and
        (r['T2M_Units'] == metadata_records[0]['T2M_Units']) and
        (r['RH2M_Units'] == metadata_records[0]['RH2M_Units'])
        for r in metadata_records
    )
    if is_uniform:
        print("Metadata is UNIFORM across the inspected sample files.")
    else:
        print("WARNING: Metadata is INCONSISTENT across the inspected sample files.")

    # -------------------------------------------------------------------------
    # 2 & 3. PRECTOTCORR EXTRACTION AND ILLUSTRATIVE AGGREGATION
    # -------------------------------------------------------------------------
    print("\n--- 2 & 3. PRECTOTCORR MONTHLY EXTRACTION & CALCULATION ---")
    
    target_file = sample_files[0]
    target_district = metadata_records[0]['District']
    target_year = 2008  # Using a leap year to test calendar logic
    
    with open(target_file, 'r') as f:
        data = json.load(f)
        
    param_meta = data.get('parameters', {})
    param_data = data.get('properties', {}).get('parameter', {})
    
    precip_units = param_meta.get('PRECTOTCORR', {}).get('units', 'Unknown').lower()
    precip_longname = param_meta.get('PRECTOTCORR', {}).get('longname', 'Unknown')
    
    print(f"Target District: {target_district} | Target Year: {target_year} (Leap Year)")
    print(f"Parameter: PRECTOTCORR ({precip_longname}) | Units: {precip_units}")
    print("Monthly Values:")
    
    raw_monthly_precip = []
    calculated_monthly_totals = []
    
    for month in range(1, 13):
        ym_key = f"{target_year}{str(month).zfill(2)}"
        val = param_data.get('PRECTOTCORR', {}).get(ym_key, -999)
        raw_monthly_precip.append(val)
        
        days_in_month = calendar.monthrange(target_year, month)[1]
        
        if val != -999:
            if 'day' in precip_units:
                # Value is mm/day. Multiply by days in month to get total mm/month.
                monthly_total = val * days_in_month
            else:
                # Value is already accumulated mm/month.
                monthly_total = val
            calculated_monthly_totals.append(monthly_total)
        else:
            calculated_monthly_totals.append(np.nan)
            
        print(f"  - {ym_key} (Days: {days_in_month}): {val} {precip_units}")

    if 'day' in precip_units:
        illustrative_annual_precip = np.nansum(calculated_monthly_totals)
        naive_sum = sum([v for v in raw_monthly_precip if v != -999])
        print(f"\n[CALCULATION SUPPORTED BY METADATA]: 'mm/day' detected.")
        print(f"Correct Annual Total (Days * Rate): {illustrative_annual_precip:.2f} mm")
        print(f"Naive Sum (Flawed - simply adding daily rates): {naive_sum:.2f} mm")
    elif 'month' in precip_units or precip_units == 'mm':
        illustrative_annual_precip = np.nansum(calculated_monthly_totals)
        print(f"\n[CALCULATION SUPPORTED BY METADATA]: Accumulated 'mm' detected.")
        print(f"Annual Total (Direct Summation): {illustrative_annual_precip:.2f} mm")
    else:
        print(f"\n[WARNING] Unrecognized unit '{precip_units}'. Cannot confidently perform annual aggregation.")
        illustrative_annual_precip = None

    # -------------------------------------------------------------------------
    # 4. T2M & RH2M ANNUAL MEANS
    # -------------------------------------------------------------------------
    print("\n--- 4. T2M & RH2M ANNUAL MEAN CALCULATION ---")
    
    t2m_units = param_meta.get('T2M', {}).get('units', 'Unknown')
    rh2m_units = param_meta.get('RH2M', {}).get('units', 'Unknown')
    
    print(f"T2M Units: {t2m_units} | RH2M Units: {rh2m_units}")
    
    t2m_vals = []
    rh2m_vals = []
    
    for month in range(1, 13):
        ym_key = f"{target_year}{str(month).zfill(2)}"
        
        t_val = param_data.get('T2M', {}).get(ym_key, -999)
        h_val = param_data.get('RH2M', {}).get(ym_key, -999)
        
        if t_val != -999: t2m_vals.append(t_val)
        if h_val != -999: rh2m_vals.append(h_val)
        
    illustrative_annual_temp = np.mean(t2m_vals) if t2m_vals else np.nan
    illustrative_annual_humid = np.mean(rh2m_vals) if rh2m_vals else np.nan
    
    print(f"Illustrative Annual Mean Temp: {illustrative_annual_temp:.2f} {t2m_units}")
    print(f"Illustrative Annual Mean Humidity: {illustrative_annual_humid:.2f} {rh2m_units}")

    # -------------------------------------------------------------------------
    # 5. MASTER DATASET COMPARISON
    # -------------------------------------------------------------------------
    print("\n--- 5. MASTER DATASET COMPARISON ---")
    master_file = "tn_master_dataset_clean.csv"
    
    if os.path.exists(master_file):
        df_master = pd.read_csv(master_file)
        
        # Find the record for the target district and year
        # Note: A district might have multiple crops, we just need one row to check weather features
        mask = (df_master['District'].str.upper() == target_district) & (df_master['Year'] == target_year)
        record = df_master[mask]
        
        if not record.empty:
            csv_rain = record.iloc[0].get('Rainfall', 'Not Found')
            csv_temp = record.iloc[0].get('Temp', 'Not Found')
            csv_humid = record.iloc[0].get('Humidity', 'Not Found')
            
            print(f"Existing Data in {master_file} for {target_district} ({target_year}):")
            print(f"  - CSV Rainfall: {csv_rain}")
            print(f"  - CSV Temp:     {csv_temp}")
            print(f"  - CSV Humidity: {csv_humid}")
            
            print("\nDIAGNOSIS & DIFFERENCES:")
            
            # Rainfall Diagnosis
            if isinstance(csv_rain, (int, float)) and illustrative_annual_precip is not None:
                if np.isclose(csv_rain, illustrative_annual_precip, rtol=1e-2, atol=1):
                    print("  [Rainfall]: MATCH. The CSV correctly scales the daily rates by the days in the month.")
                elif np.isclose(csv_rain, naive_sum, rtol=1e-2, atol=1):
                    print("  [Rainfall]: SEVERE FLAW. The CSV simply summed the 'mm/day' values without multiplying by days in the month.")
                    print("              This means your model is using ~40-80mm as annual rainfall instead of the actual ~600-1500mm.")
                else:
                    print("  [Rainfall]: MISMATCH. The CSV rainfall does not match the correct calculation nor the naive sum. Different aggregation logic or source data may have been used.")
            
            # Temp Diagnosis
            if isinstance(csv_temp, (int, float)):
                if np.isclose(csv_temp, illustrative_annual_temp, rtol=1e-2, atol=0.1):
                    print("  [Temp]:     MATCH. The CSV correctly calculated the annual mean temperature.")
                else:
                    print("  [Temp]:     MISMATCH. The CSV mean temperature differs from the calculated mean.")
                    
            # Humid Diagnosis
            if isinstance(csv_humid, (int, float)):
                if np.isclose(csv_humid, illustrative_annual_humid, rtol=1e-2, atol=0.1):
                    print("  [Humidity]: MATCH. The CSV correctly calculated the annual mean humidity.")
                else:
                    print("  [Humidity]: MISMATCH. The CSV mean humidity differs from the calculated mean.")
                    
        else:
            print(f"Record for {target_district} in {target_year} not found in {master_file}.")
    else:
        print(f"{master_file} not found. Skipping comparison.")

    print("\n" + "=" * 80)
    print(" END OF WEATHER AGGREGATION AUDIT")
    print("=" * 80)

if __name__ == "__main__":
    run_weather_aggregation_audit()
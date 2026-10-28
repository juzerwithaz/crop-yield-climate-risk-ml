import pandas as pd
import urllib.request
import json
import re
import time
import os

# Ensure the CSV is in the same directory
csv_filename = "tn_district_coords.csv"
df = pd.read_csv(csv_filename)

def parse_coord(coord_str):
    match = re.search(r"([\d\.]+)", str(coord_str))
    if match:
        val = float(match.group(1))
        if 'S' in str(coord_str) or 'W' in str(coord_str):
            val = -val
        return val
    return None

success_count = 0

for index, row in df.iterrows():
    district = row['District_Name']
    
    # Format the expected filename to check if it exists
    safe_name = str(district).lower().replace(" ", "_")
    filename = f"weather_raw_{safe_name}.json"
    
    # Check if we already downloaded this file
    if os.path.exists(filename):
        print(f"Skipping {district} - File already exists.")
        continue
        
    lat = parse_coord(row['Latitude'])
    lon = parse_coord(row['Longitude'])
    
    if lat is None or lon is None:
        print(f"Skipping {district} due to invalid coordinates.")
        continue
        
    url = f"https://power.larc.nasa.gov/api/temporal/monthly/point?parameters=PRECTOTCORR,T2M,RH2M&community=AG&longitude={lon}&latitude={lat}&start=1997&end=2013&format=JSON"
    
    # Retry logic: Try up to 3 times per missing district
    max_retries = 3
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=15) as response: # added a timeout
                data = json.loads(response.read().decode())
                
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=2)
                
                print(f"Successfully saved {filename}")
                success_count += 1
                break # Break out of the retry loop on success
                
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for {district}: {e}")
            if attempt < max_retries - 1:
                sleep_time = 3 * (attempt + 1)
                print(f"Waiting {sleep_time} seconds before retrying...")
                time.sleep(sleep_time)
            else:
                print(f"Could not fetch {district} after {max_retries} attempts.")
        
    # Wait 2 seconds between successful requests to be safe
    time.sleep(2)

print(f"\nDone! Successfully fetched {success_count} missing files.")
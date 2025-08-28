import json
import pandas as pd
import os

def json_to_csv(json_file):
    # Get the base filename without extension
    base_name = os.path.splitext(json_file)[0]
    csv_file = base_name + '.csv'
    
    # Read JSON file
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(data['results'])
    df.to_csv(csv_file, index=False)
    print(f"Successfully converted {json_file} to {csv_file}")

# Convert both files
json_to_csv('original_packages.json')      # Creates original_packages.csv
json_to_csv('high_impact_packages.json')   # Creates high_impact_packages.csv
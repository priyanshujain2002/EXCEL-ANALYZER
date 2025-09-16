import json
import pandas as pd
import os

def json_to_excel(json_file):
    # Get the base filename without extension
    base_name = os.path.splitext(json_file)[0]
    excel_file = base_name + '.xlsx'
    
    # Read JSON file
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    # Convert to DataFrame and save to Excel
    df = pd.DataFrame(data['results'])
    df.to_excel(excel_file, index=False)
    print(f"Successfully converted {json_file} to {excel_file}")

# Convert both files
json_to_excel('original_packages.json')      # Creates original_packages.xlsx
json_to_excel('high_impact_packages.json')   # Creates high_impact_packages.xlsx
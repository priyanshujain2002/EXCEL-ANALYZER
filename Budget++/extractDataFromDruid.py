import requests
import json
import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
import os

url = "https://druid.use1-rprod.k8s.adgear.com/druid/v2/sql"

payload = json.dumps({
    "query": """
WITH first_split_response AS (
    SELECT 
        CASE 
            WHEN UPPER(TRIM(genre_individual)) = 'AP' THEN 'Audience Participation'
            WHEN UPPER(TRIM(genre_individual)) = 'AC' THEN 'Award Ceremonies & Pageants'
            WHEN UPPER(TRIM(genre_individual)) = 'CP' THEN 'Children''s Programming'
            WHEN UPPER(TRIM(genre_individual)) = 'CV' THEN 'Comedy Variety'
            WHEN UPPER(TRIM(genre_individual)) = 'CM' THEN 'Concert Music'
            WHEN UPPER(TRIM(genre_individual)) = 'CC' THEN 'Conversation, Colloquies'
            WHEN UPPER(TRIM(genre_individual)) = 'DD' THEN 'Daytime Drama'
            WHEN UPPER(TRIM(genre_individual)) = 'D' THEN 'Devotional'
            WHEN UPPER(TRIM(genre_individual)) = 'DO' THEN 'Documentary, General'
            WHEN UPPER(TRIM(genre_individual)) = 'DN' THEN 'Documentary, News'
            WHEN UPPER(TRIM(genre_individual)) = 'EA' THEN 'Evening Animation'
            WHEN UPPER(TRIM(genre_individual)) = 'FF' THEN 'Feature Film'
            WHEN UPPER(TRIM(genre_individual)) = 'GD' THEN 'General Drama'
            WHEN UPPER(TRIM(genre_individual)) = 'GV' THEN 'General Variety'
            WHEN UPPER(TRIM(genre_individual)) = 'IA' THEN 'Instructions, Advice'
            WHEN UPPER(TRIM(genre_individual)) = 'MD' THEN 'Musical Drama'
            WHEN UPPER(TRIM(genre_individual)) = 'N' THEN 'News'
            WHEN UPPER(TRIM(genre_individual)) = 'OP' THEN 'Official Police'
            WHEN UPPER(TRIM(genre_individual)) = 'P' THEN 'Paid Political'
            WHEN UPPER(TRIM(genre_individual)) = 'PV' THEN 'Participation Variety'
            WHEN UPPER(TRIM(genre_individual)) = 'PC' THEN 'Popular Music'
            WHEN UPPER(TRIM(genre_individual)) = 'PD' THEN 'Private Detective'
            WHEN UPPER(TRIM(genre_individual)) = 'QG' THEN 'Quiz -Give Away'
            WHEN UPPER(TRIM(genre_individual)) = 'QP' THEN 'Quiz -Panel'
            WHEN UPPER(TRIM(genre_individual)) = 'SF' THEN 'Science Fiction'
            WHEN UPPER(TRIM(genre_individual)) = 'CS' THEN 'Situation Comedy'
            WHEN UPPER(TRIM(genre_individual)) = 'SA' THEN 'Sports Anthology'
            WHEN UPPER(TRIM(genre_individual)) = 'SC' THEN 'Sports Commentary'
            WHEN UPPER(TRIM(genre_individual)) = 'SE' THEN 'Sports Event'
            WHEN UPPER(TRIM(genre_individual)) = 'SN' THEN 'Sports News'
            WHEN UPPER(TRIM(genre_individual)) = 'SM' THEN 'Suspense/Mystery'
            WHEN UPPER(TRIM(genre_individual)) = 'EW' THEN 'Western Drama'
            ELSE TRIM(genre_individual)
        END as expanded_genre,
        creativetype,
        "row_count"
    FROM (
        SELECT 
            CASE WHEN genre IS NULL THEN 'Unknown' ELSE genre END as genre,
            creativetype,
            "row_count"
        FROM "ctv_untargeted_bid_response"
        WHERE 
            country = 'US'
            AND exchange_id = 55
            AND __time >= '2025-09-03T00:00:00.000Z'
            AND __time <= '2025-09-09T00:00:00.000Z'
    ) CROSS JOIN UNNEST(STRING_TO_ARRAY(genre, ',')) AS t(genre_individual)
    WHERE UPPER(TRIM(genre_individual)) != 'A'
),
first_split_request AS (
    SELECT 
        CASE 
            WHEN UPPER(TRIM(genre_individual)) = 'AP' THEN 'Audience Participation'
            WHEN UPPER(TRIM(genre_individual)) = 'AC' THEN 'Award Ceremonies & Pageants'
            WHEN UPPER(TRIM(genre_individual)) = 'CP' THEN 'Children''s Programming'
            WHEN UPPER(TRIM(genre_individual)) = 'CV' THEN 'Comedy Variety'
            WHEN UPPER(TRIM(genre_individual)) = 'CM' THEN 'Concert Music'
            WHEN UPPER(TRIM(genre_individual)) = 'CC' THEN 'Conversation, Colloquies'
            WHEN UPPER(TRIM(genre_individual)) = 'DD' THEN 'Daytime Drama'
            WHEN UPPER(TRIM(genre_individual)) = 'D' THEN 'Devotional'
            WHEN UPPER(TRIM(genre_individual)) = 'DO' THEN 'Documentary, General'
            WHEN UPPER(TRIM(genre_individual)) = 'DN' THEN 'Documentary, News'
            WHEN UPPER(TRIM(genre_individual)) = 'EA' THEN 'Evening Animation'
            WHEN UPPER(TRIM(genre_individual)) = 'FF' THEN 'Feature Film'
            WHEN UPPER(TRIM(genre_individual)) = 'GD' THEN 'General Drama'
            WHEN UPPER(TRIM(genre_individual)) = 'GV' THEN 'General Variety'
            WHEN UPPER(TRIM(genre_individual)) = 'IA' THEN 'Instructions, Advice'
            WHEN UPPER(TRIM(genre_individual)) = 'MD' THEN 'Musical Drama'
            WHEN UPPER(TRIM(genre_individual)) = 'N' THEN 'News'
            WHEN UPPER(TRIM(genre_individual)) = 'OP' THEN 'Official Police'
            WHEN UPPER(TRIM(genre_individual)) = 'P' THEN 'Paid Political'
            WHEN UPPER(TRIM(genre_individual)) = 'PV' THEN 'Participation Variety'
            WHEN UPPER(TRIM(genre_individual)) = 'PC' THEN 'Popular Music'
            WHEN UPPER(TRIM(genre_individual)) = 'PD' THEN 'Private Detective'
            WHEN UPPER(TRIM(genre_individual)) = 'QG' THEN 'Quiz -Give Away'
            WHEN UPPER(TRIM(genre_individual)) = 'QP' THEN 'Quiz -Panel'
            WHEN UPPER(TRIM(genre_individual)) = 'SF' THEN 'Science Fiction'
            WHEN UPPER(TRIM(genre_individual)) = 'CS' THEN 'Situation Comedy'
            WHEN UPPER(TRIM(genre_individual)) = 'SA' THEN 'Sports Anthology'
            WHEN UPPER(TRIM(genre_individual)) = 'SC' THEN 'Sports Commentary'
            WHEN UPPER(TRIM(genre_individual)) = 'SE' THEN 'Sports Event'
            WHEN UPPER(TRIM(genre_individual)) = 'SN' THEN 'Sports News'
            WHEN UPPER(TRIM(genre_individual)) = 'SM' THEN 'Suspense/Mystery'
            WHEN UPPER(TRIM(genre_individual)) = 'EW' THEN 'Western Drama'
            ELSE TRIM(genre_individual)
        END as expanded_genre,
        creativetype,
        "row_count"
    FROM (
        SELECT 
            CASE WHEN genre IS NULL THEN 'Unknown' ELSE genre END as genre,
            creativetype,
            "row_count"
        FROM "ctv_untargeted_bid_request"
        WHERE 
            exchange_id = 55
            AND country = 'US'
            AND __time >= '2025-09-03T00:00:00.000Z'
            AND __time <= '2025-09-09T00:00:00.000Z'
    ) CROSS JOIN UNNEST(STRING_TO_ARRAY(genre, ',')) AS t(genre_individual)
    WHERE UPPER(TRIM(genre_individual)) != 'A'
),
split_response AS (
    SELECT 
        LOWER(TRIM(final_genre)) as genre,
        creativetype,
        "row_count"
    FROM first_split_response
    CROSS JOIN UNNEST(STRING_TO_ARRAY(expanded_genre, ',')) AS t(final_genre)
),
split_request AS (
    SELECT 
        LOWER(TRIM(final_genre)) as genre,
        creativetype,
        "row_count"
    FROM first_split_request
    CROSS JOIN UNNEST(STRING_TO_ARRAY(expanded_genre, ',')) AS t(final_genre)
),
cte1 AS (
    SELECT 
        genre,
        creativetype,
        SUM("row_count") AS sum_response
    FROM split_response
    GROUP BY 1, 2
),
cte2 AS (
    SELECT
        genre,
        creativetype,
        SUM("row_count") AS sum_request
    FROM split_request
    GROUP BY 1, 2
)
SELECT 
    cte1.genre,
    cte1.creativetype,
    cte2.sum_request,
    cte1.sum_response
FROM cte1
INNER JOIN cte2 
ON cte1.genre = cte2.genre 
AND cte1.creativetype = cte2.creativetype
"""
})
headers = {
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload, verify=False)

# Parse JSON response and create dataframe
data = json.loads(response.text)
df = pd.DataFrame(data)

# Add ID column at the beginning (starting from 1)
df.insert(0, 'id', range(1, len(df) + 1))

# Display the dataframe
print("Dataframe created successfully:")
print(df)

# Create Excel file with formulas and sorting
excel_filename = "knowledge/druid_query_results.xlsx"

# Calculate X values for sorting (temporary calculation)
df['unsold_supply_calc'] = df['sum_request'] - df['sum_response']
total_unsold = df['unsold_supply_calc'].sum()
df['x_calc'] = (df['unsold_supply_calc'] / total_unsold * 100) if total_unsold != 0 else 0

# Sort by X values (descending)
df_sorted = df.sort_values('x_calc', ascending=False).reset_index(drop=True)

# Create workbook with sorted data and Excel formulas
wb = Workbook()
ws = wb.active

# Add headers
headers = ['id', 'genre', 'creativetype', 'sum_request', 'sum_response', 'unsold_supply', 'x', 'y']
ws.append(headers)

# Add sorted data rows
for i, (_, row) in enumerate(df_sorted.iterrows(), 2):
    ws.append([i-1, row['genre'], row['creativetype'], row['sum_request'], row['sum_response'], '', '', ''])

# Add Excel formulas to new columns
total_rows = len(df_sorted) + 1
for row in range(2, total_rows + 1):
    # Update ID to sequential
    ws[f'A{row}'] = row - 1
    
    # Column F: Unsold Supply = sum_request - sum_response
    ws[f'F{row}'] = f'=D{row}-E{row}'
    
    # Column G: X = (Unsold supply of that type/Sum of all unsold supply)*100
    ws[f'G{row}'] = f'=IF(SUM(F$2:F${total_rows})=0,0,(F{row}/SUM(F$2:F${total_rows}))*100)'
    
    # Column H: Y = (Unsold supply of that type / sum_request of that type)*100
    ws[f'H{row}'] = f'=IF(D{row}=0,0,(F{row}/D{row})*100)'

# Ensure the knowledge directory exists
os.makedirs("Budget++/knowledge", exist_ok=True)

# Save Excel file
wb.save(excel_filename)
print(f"Data saved to Excel file: {excel_filename}")
print("File is sorted by X values (highest to lowest)")
print("Columns added: Unsold Supply, X, Y with Excel formulas (full precision)")

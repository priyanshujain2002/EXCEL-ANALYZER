import requests
import json
import pandas as pd

url = "https://druid.use1-rprod.k8s.adgear.com/druid/v2/sql"

payload = json.dumps({
    "query": """
WITH split_response AS (
    SELECT 
        TRIM(genre_individual) as genre,
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
),
split_request AS (
    SELECT 
        TRIM(genre_individual) as genre,
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

# Display the dataframe
print("Dataframe created successfully:")
print(df)

# Save dataframe to files for later access

excel_filename = "druid_query_results.xlsx"



df.to_excel(excel_filename, index=False)
print(f"Data saved to Excel file: {excel_filename}")

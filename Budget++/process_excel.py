import pandas as pd
import os

def process_excel_file(input_file, output_file):
    """
    Process the Excel file to merge genres and handle multi-genre entries.
    
    Args:
        input_file (str): Path to input Excel file
        output_file (str): Path to output Excel file
    """
    print(f"Reading input file: {input_file}")
    
    # Read the Excel file
    df = pd.read_excel(input_file)
    
    print(f"Original data shape: {df.shape}")
    print(f"Original columns: {df.columns.tolist()}")
    
    # Check if required columns exist
    required_columns = ['Genre', 'Sum_Request', 'Sum_Response']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the Excel file")
    
    # Initialize list to store processed rows
    processed_rows = []
    
    print("Processing rows...")
    
    # Process each row
    for index, row in df.iterrows():
        genre = str(row['Genre'])
        sum_request = row['Sum_Request']
        sum_response = row['Sum_Response']
        
        # Split genre by various separators except '&' and 'and'
        import re
        
        # Replace common separators with commas, but preserve '&' and 'and'
        # Split on: commas, periods, slashes (/), plus signs (+)
        processed_genre = re.sub(r'[./+]', ',', genre)
        
        # Split by commas and strip whitespace
        genres = [g.strip() for g in processed_genre.split(',')]
        
        # Additional processing: handle entries that might have been split but contain '&' or 'and'
        # These should be kept as single genre entries
        final_genres = []
        for g in genres:
            if '&' in g or ' and ' in g.lower():
                final_genres.append(g)
            else:
                # If no '&' or 'and', check if it needs further splitting
                final_genres.append(g)
        
        # Create separate row for each genre
        for g in final_genres:
            if g:  # Skip empty genres
                # Convert to lowercase
                normalized_genre = g.lower()
                processed_rows.append({
                    'Genre': normalized_genre,
                    'Sum_Request': sum_request,
                    'Sum_Response': sum_response
                })
    
    # Create new DataFrame with processed rows
    processed_df = pd.DataFrame(processed_rows)
    
    print(f"Processed data shape after splitting: {processed_df.shape}")
    
    # Aggregate data by genre
    aggregated_df = processed_df.groupby('Genre').agg({
        'Sum_Request': 'sum',
        'Sum_Response': 'sum'
    }).reset_index()
    
    # Sort by Sum_Request in descending order
    aggregated_df = aggregated_df.sort_values('Sum_Request', ascending=False)
    
    print(f"Final data shape after aggregation: {aggregated_df.shape}")
    print(f"Number of unique genres: {len(aggregated_df)}")
    
    # Save to output file
    aggregated_df.to_excel(output_file, index=False)
    
    print(f"Processed data saved to: {output_file}")
    
    # Display some statistics
    print("\n=== Processing Statistics ===")
    print(f"Original rows: {len(df)}")
    print(f"Rows after splitting multi-genre entries: {len(processed_df)}")
    print(f"Final unique genres: {len(aggregated_df)}")
    print(f"Top 10 genres by Sum_Request:")
    print(aggregated_df.head(10)[['Genre', 'Sum_Request']].to_string(index=False))
    
    return aggregated_df

if __name__ == "__main__":
    # Define file paths
    input_file = "Copy of Request Response Database.xlsx"
    output_file = "Processed_Request_Response_Database.xlsx"
    
    try:
        # Process the file
        result_df = process_excel_file(input_file, output_file)
        print("\nProcessing completed successfully!")
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        raise

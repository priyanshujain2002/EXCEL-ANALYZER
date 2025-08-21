#!/usr/bin/env python3
"""
Combined Excel Processor - Merges functionality from process_excel.py and multi_sheet_excel_processor.py
Processes multi-sheet Excel files by splitting them and cleaning each sheet individually.
"""

import sys
import os
import pandas as pd
from pathlib import Path
import shutil
import glob
import openpyxl
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters
from typing import List, Tuple

def split_excel_by_sheets(input_file_path: str, temp_dir: str = "temp_sheets") -> List[Tuple[str, str, str]]:
    """
    Split an Excel file with multiple sheets into separate Excel files.
    
    Args:
        input_file_path (str): Path to the input Excel file
        temp_dir (str): Directory to save the split files
    
    Returns:
        List[Tuple[str, str, str]]: List of tuples containing (sheet_name, split_file_path, output_file_name)
    """
    # Create temp directory if it doesn't exist
    os.makedirs(temp_dir, exist_ok=True)
    
    # Get the base filename without extension
    base_filename = Path(input_file_path).stem
    
    split_info = []
    
    try:
        # Load the Excel file with openpyxl to check sheet visibility
        wb = openpyxl.load_workbook(input_file_path, read_only=True)
        excel_file = pd.ExcelFile(input_file_path)
        
        print(f"Found {len(excel_file.sheet_names)} sheets in {input_file_path}")
        
        for sheet_name in excel_file.sheet_names:
            # Skip all types of hidden sheets (hidden and veryHidden)
            if wb[sheet_name].sheet_state != 'visible':
                print(f"Skipping hidden sheet ({wb[sheet_name].sheet_state}): {sheet_name}")
                continue
            # Clean sheet name for filename (remove invalid characters)
            clean_sheet_name = "".join(c for c in sheet_name if c.isalnum() or c in (' ', '-', '_', '+')).rstrip()
            clean_sheet_name = clean_sheet_name.replace(' ', '_')
            
            # Create split file path
            split_filename = f"{base_filename}_{clean_sheet_name}.xlsx"
            split_file_path = os.path.join(temp_dir, split_filename)
            
            # Create output file name (what the agent will produce)
            output_filename = f"cleaned_{base_filename}_{clean_sheet_name}.xlsx"
            
            # Read the specific sheet and save as new Excel file
            df = pd.read_excel(input_file_path, sheet_name=sheet_name, header=None)
            
            with pd.ExcelWriter(split_file_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
            
            split_info.append((sheet_name, os.path.abspath(split_file_path), output_filename))
            print(f"Created: {split_filename} -> Output will be: {output_filename}")
    
    except Exception as e:
        print(f"Error splitting Excel file: {str(e)}")
        raise
    
    return split_info

def process_single_sheet_file(input_file: str, output_file: str, sheet_name: str) -> bool:
    """
    Process a single Excel file (containing one sheet) using the agent workflow.
    
    Args:
        input_file (str): Path to input Excel file
        output_file (str): Path to output Excel file  
        sheet_name (str): Name of the original sheet (for context)
        
    Returns:
        bool: True if successful, False otherwise
    """
    
    # Convert to absolute paths
    input_file = os.path.abspath(input_file)
    output_file = os.path.abspath(output_file)
    
    server_params = StdioServerParameters(
        command="python3",
        args=["-m", "excel_mcp", "stdio"],
        env={"UV_PYTHON": "3.12", **os.environ},
    )
    
    llm = LLM(
        model="bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        aws_region_name="us-east-1",
        temperature=0.2,
    )
    
    try:
        print(f"\n{'='*60}")
        print(f"Processing Sheet: {sheet_name}")
        print(f"Input File: {os.path.basename(input_file)}")
        print(f"Output File: {os.path.basename(output_file)}")
        print(f"{'='*60}")
        
        with MCPServerAdapter(server_params) as mcp_tools:
            print(f"Available tools: {[tool.name for tool in mcp_tools]}")

            # Agent 1: Excel Reader
            excel_reader = Agent(
                role='Excel File Reader',
                goal='Read Excel files and extract targeted data from specific ranges',
                backstory=f"""You are an expert Excel file reader processing the '{sheet_name}' sheet. 
                You specialize in reading Excel files efficiently by targeting specific data ranges. 
                You avoid reading massive ranges that could cause performance issues. You focus on 
                getting workbook metadata first, then reading smaller, targeted ranges to understand 
                the data structure.""",
                tools=mcp_tools,
                verbose=True,
                allow_delegation=False,
                llm=llm
            )

            # Agent 2: Data Processor
            data_processor = Agent(
                role='Data Analysis Specialist',
                goal='Analyze Excel data and identify clean tabular data ranges efficiently',
                backstory=f"""You are a data analysis expert processing the '{sheet_name}' sheet. 
                You specialize in identifying meaningful tabular data within Excel files by examining 
                small sample ranges first. You can distinguish between headers, formatting rows, empty 
                cells, and actual data. You work efficiently by examining targeted ranges rather than 
                processing massive amounts of data at once.""",
                tools=mcp_tools,
                verbose=True,
                allow_delegation=False,
                llm=llm
            )

            # Agent 3: Excel Writer
            excel_writer = Agent(
                role='Excel File Writer',
                goal='Create new Excel files with ALL extracted data preserved completely',
                backstory=f"""You are an Excel file creation specialist processing the '{sheet_name}' sheet.
                You take cleaned data and create new, properly formatted Excel files. You are CRITICAL about
                data preservation - you MUST ensure that EVERY SINGLE CELL of data that was extracted from
                the input range is written to the output file. You NEVER leave columns blank or skip any data.
                Your primary responsibility is complete data integrity and preservation.""",
                tools=mcp_tools,
                verbose=True,
                allow_delegation=False,
                llm=llm
            )

            # Task 1: Read Excel File Metadata and Sample Data
            read_task = Task(
                description=f"""
                Read the Excel file located at '{input_file}' which contains data from the original '{sheet_name}' sheet.
                
                Your task is to:
                1. Use get_workbook_metadata tool to understand the file structure and sheet names
                2. Read the full width sample range (e.g., A1:AC20) to understand the complete data structure
                3. Focus on identifying where the actual data starts and what ALL the column headers are
                4. Make sure to capture the full horizontal extent of the data (columns A through AC or beyond)
                5. Provide a summary of the sheet's structure and content type including ALL columns
                
                IMPORTANT: This file contains data from the '{sheet_name}' sheet. Read the full width of data but limit rows to avoid performance issues.
                """,
                expected_output=f"Metadata about the Excel file from '{sheet_name}' sheet, including sample data showing the structure and headers.",
                agent=excel_reader
            )

            # Task 2: Process and Identify Clean Data Ranges
            process_task = Task(
                description=f"""
                Based on the sample data from the previous task, identify the clean tabular data ranges from the '{sheet_name}' sheet.
                
                CRITICAL REQUIREMENTS FOR COMPLETE DATA EXTRACTION:
                1. Analyze the sample data to identify where the main tabular data is located
                2. Determine the exact row where data starts (after headers/titles)
                3. Identify the COMPLETE column range that contains ALL meaningful data
                4. Estimate a data range (e.g., A49:AC58) that captures the ENTIRE data table with ALL columns
                5. INCLUDE all columns that have any data, even if some cells are empty
                6. Avoid including only:
                   - Title rows at the top
                   - Completely empty rows and columns at the edges
                   - Summary/total rows at the bottom
                7. Provide specific range recommendations for COMPLETE data extraction
                
                MANDATORY: Your range MUST include ALL columns that contain any data. Do not exclude columns
                just because they have some empty cells. The goal is COMPLETE data preservation, not selective extraction.
                
                Focus on extracting ALL tabular data and provide a range that captures the FULL width of the data table.
                """,
                expected_output=f"Specific recommendations for COMPLETE data ranges to extract from '{sheet_name}' sheet, including starting row, ending row, and the FULL column range that captures ALL data columns.",
                agent=data_processor
            )

            # Task 3: Write Cleaned Data to New Excel File
            write_task = Task(
                description=f"""
                Create a new Excel file with ALL the cleaned tabular data from the '{sheet_name}' sheet based on the identified ranges.
                
                CRITICAL DATA PRESERVATION REQUIREMENTS:
                1. Use the specific data range identified in the previous task
                2. Read ONLY that specific range using read_data_from_excel with the exact range
                3. Create a new workbook using create_workbook
                4. Write ALL the extracted data to the new file using write_data_to_excel
                5. Save the complete data to '{output_file}'
                
                MANDATORY DATA INTEGRITY CHECKS:
                - VERIFY that every column from the extracted range is written to the output file
                - ENSURE no columns are left blank or missing in the output
                - CONFIRM that the number of columns in output matches the extracted range
                - VALIDATE that all data values are preserved exactly as extracted
                
                FAILURE TO PRESERVE ALL DATA IS UNACCEPTABLE. You must write every single cell of data
                that was extracted from the input range. If any column appears blank in the output,
                you have failed your primary responsibility.
                
                The output file name '{os.path.basename(output_file)}' is specifically designed to map back to the original '{sheet_name}' sheet.
                """,
                expected_output=f"A new Excel file '{output_file}' containing ALL the clean tabular data from the '{sheet_name}' sheet with complete data preservation - no missing columns or blank data.",
                agent=excel_writer
            )

            # Create crew with all agents and tasks
            crew = Crew(
                agents=[excel_reader, data_processor, excel_writer],
                tasks=[read_task, process_task, write_task],
                verbose=True
            )

            # Execute the crew
            result = crew.kickoff()
            print(f"✅ Successfully processed sheet '{sheet_name}'")
            return True
            
    except Exception as e:
        print(f"❌ Error processing sheet '{sheet_name}': {str(e)}")
        return False

def process_multi_sheet_excel(input_excel_path: str, output_directory: str = "processed_sheets"):
    """
    Main function to process a multi-sheet Excel file by splitting it and processing each sheet individually.
    
    Args:
        input_excel_path (str): Path to the input Excel file with multiple sheets
        output_directory (str): Directory to save processed files
    """
    print(f"🚀 Starting multi-sheet Excel processing...")
    print(f"Input file: {input_excel_path}")
    
    # Check if input file exists
    if not os.path.exists(input_excel_path):
        print(f"❌ Input file not found: {input_excel_path}")
        return
    
    # Create output directory
    os.makedirs(output_directory, exist_ok=True)
    
    # Create temporary directory for split sheets
    temp_dir = "temp_split_sheets"
    
    successful_files = []
    failed_files = []
    
    try:
        # Step 1: Split sheets into individual files
        print(f"\n✂️ Splitting sheets into individual files...")
        split_info = split_excel_by_sheets(input_excel_path, temp_dir)
        
        # Step 2: Process each file individually
        print(f"\n🔄 Processing {len(split_info)} sheets sequentially...")
        
        for i, (sheet_name, split_file_path, output_filename) in enumerate(split_info, 1):
            # Create full output path
            output_file_path = os.path.join(output_directory, output_filename)
            
            print(f"\n[{i}/{len(split_info)}] Processing '{sheet_name}' sheet...")
            
            success = process_single_sheet_file(split_file_path, output_file_path, sheet_name)
            
            if success:
                successful_files.append((sheet_name, output_filename))
                print(f"✅ Sheet '{sheet_name}' completed successfully -> {output_filename}")
            else:
                failed_files.append((sheet_name, output_filename))
                print(f"❌ Sheet '{sheet_name}' failed")
    
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"\n🧹 Cleaned up temporary files")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📈 PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successfully processed: {len(successful_files)} sheets")
    print(f"❌ Failed: {len(failed_files)} sheets")
    
    if successful_files:
        print(f"\n✅ Successful sheets:")
        for sheet_name, output_file in successful_files:
            print(f"  - '{sheet_name}' -> {output_file}")
    
    if failed_files:
        print(f"\n❌ Failed sheets:")
        for sheet_name, output_file in failed_files:
            print(f"  - '{sheet_name}' -> {output_file}")
    
    print(f"\n📁 Output directory: {os.path.abspath(output_directory)}")
    print(f"🎉 Multi-sheet processing completed!")

# Configuration section
CONFIG = {
    'input_dir': 'input_excels',  # Directory containing Excel files
    'output_base': 'processed_files'  # Base output directory
}

def find_excel_files(path):
    """Find all Excel files in a directory or return single file path."""
    if os.path.isfile(path):
        return [path] if path.lower().endswith(('.xlsx', '.xls')) else []
    return glob.glob(os.path.join(path, '*.xlsx')) + glob.glob(os.path.join(path, '*.xls'))

def main():
    """Main function to process all Excel files in input directory."""
    
    print("📊 Batch Excel Processor")
    print("=" * 35)
    
    # Find all Excel files
    excel_files = find_excel_files(CONFIG['input_dir'])
    
    if not excel_files:
        print(f"❌ No Excel files found in: {CONFIG['input_dir']}")
        return
    
    print(f"\nFound {len(excel_files)} Excel files to process:")
    for file in excel_files:
        print(f"  - {os.path.basename(file)}")
    
    # Create base output directory
    os.makedirs(CONFIG['output_base'], exist_ok=True)
    
    # Process each file
    for input_file in excel_files:
        try:
            base_name = Path(input_file).stem
            output_dir = os.path.join(CONFIG['output_base'], base_name)
            
            print(f"\n{'='*60}")
            print(f"Processing: {os.path.basename(input_file)}")
            print(f"Output to: {output_dir}/")
            print(f"{'='*60}")
            
            process_multi_sheet_excel(input_file, output_dir)
        except Exception as e:
            print(f"❌ Error processing {os.path.basename(input_file)}: {str(e)}")
            continue

if __name__ == "__main__":
    main()

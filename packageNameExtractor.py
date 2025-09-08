"""
In this script we input the cleaned media plans and this script analyses it and gives two json file, one for original package and
the other for high impact package. 
"""

from crewai import Agent, Task, Crew, LLM
from crewai.knowledge.source.excel_knowledge_source import ExcelKnowledgeSource
import os
import glob
import boto3
import pandas as pd
import re

# Setup LLM
llm = LLM(
    model="bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    aws_region_name="us-east-1",
    temperature=0.3,
)

def extract_placement_names_from_sheet(excel_file: str, sheet_name: str) -> list:
    """Extract placement names from a specific sheet with enhanced placement column detection"""
    placement_names = []
    
    # Define package-specific column patterns - ONLY Package Name
    placement_column_patterns = [
        ['package', 'name'],             # "Package Name" - ONLY this pattern
    ]
    
    try:
        # Read the sheet - try different header row positions
        df = None
        placement_column = None
        
        for header_row in [0, 1, 2]:  # Try different header positions
            try:
                temp_df = pd.read_excel(excel_file, sheet_name=sheet_name, header=header_row)
                
                # Find placement column using enhanced patterns
                found_column = find_placement_column(temp_df.columns, placement_column_patterns)
                if found_column:
                    df = temp_df
                    placement_column = found_column
                    break
            except:
                continue
        
        # If no header found, read without header and look for placement data
        if df is None:
            df = pd.read_excel(excel_file, sheet_name=sheet_name, header=None)
            
            # Look for the header row using placement patterns
            header_row_idx = None
            for idx, row in df.iterrows():
                row_values = [str(val).lower() for val in row.values if pd.notna(val)]
                if find_placement_column_in_values(row_values, placement_column_patterns):
                    header_row_idx = idx
                    break
            
            if header_row_idx is not None:
                # Re-read with correct header
                df = pd.read_excel(excel_file, sheet_name=sheet_name, header=header_row_idx)
                placement_column = find_placement_column(df.columns, placement_column_patterns)
        
        if df is not None and placement_column:
            # Extract placement names
            placements = df[placement_column].dropna()
            placements = placements[placements.astype(str).str.strip() != '']
            placements = placements[placements.astype(str).str.strip().str.lower() != 'nan']
            
            # Convert to list and clean up
            for placement in placements:
                placement_str = str(placement).strip()
                # Skip if it looks like a header, total row, or contains "added value"
                if (placement_str and
                    not is_header_like(placement_str) and
                    not has_added_value(placement_str) and
                    len(placement_str) >= 3):
                    placement_names.append(placement_str)
            
            # Remove duplicates while preserving order
            placement_names = list(dict.fromkeys(placement_names))
                    
    except Exception as e:
        print(f"Error extracting placement names from {sheet_name} in {excel_file}: {e}")
    
    return placement_names

def find_placement_column(columns, patterns):
    """Find the best matching placement column using pattern priority"""
    for pattern in patterns:
        for col in columns:
            col_str = str(col).lower().strip()
            # Now we WANT package columns specifically
            # Check if all keywords in pattern are present in column name
            if all(keyword in col_str for keyword in pattern):
                return col
    return None

def find_placement_column_in_values(values, patterns):
    """Find placement column patterns in row values"""
    for pattern in patterns:
        for val in values:
            val_str = str(val).lower().strip()
            # Now we WANT package values specifically
            # Check if all keywords in pattern are present in value
            if all(keyword in val_str for keyword in pattern):
                return True
    return False

def has_added_value(text):
    """Check if text contains 'added value' and should be filtered out"""
    text_lower = text.lower().strip()
    return 'added value' in text_lower

def is_header_like(text):
    """Check if text looks like a header or system row"""
    text_lower = text.lower().strip()
    
    # Exact header matches (these are definitely headers)
    exact_headers = [
        'placement name', 'package name', 'placement', 'name', 'total', 'sum',
        'header', 'title', 'description', 'unnamed:', 'column'
    ]
    
    # If text exactly matches header indicators, it's a header
    if text_lower in exact_headers:
        return True
    
    # If text is very short (but allow valid 3-letter packages like CTV) or contains only special characters
    stripped_text = text.strip()
    if len(stripped_text) <= 2 or stripped_text in ['', '-', '_', '=', '+']:
        return True
    
    # Special case: if it's exactly 3 characters and all uppercase, it might be a valid package (like CTV)
    if len(stripped_text) == 3 and stripped_text.isupper() and stripped_text.isalpha():
        return False  # Don't filter out 3-letter uppercase packages
    
    # If text starts with "placement name" or "package name" (common headers)
    if text_lower.startswith('placement name') or text_lower.startswith('package name'):
        return True
        
    return False

def analyze_media_plan_structure(excel_files: list):
    """Analyze the structure of media plan files to identify sheet relationships and extract placement data"""
    media_plan_structure = {}
    
    for excel_file in excel_files:
        try:
            xl_file = pd.ExcelFile(excel_file)
            filename = os.path.basename(excel_file)
            
            # Categorize sheets based on naming patterns
            original_sheets = []
            high_impact_sheets = []
            combined_sheets = []
            
            for sheet_name in xl_file.sheet_names:
                sheet_lower = sheet_name.lower()
                
                # High Impact sheets (contain "high impact" in name)
                if "high impact" in sheet_lower:
                    high_impact_sheets.append(sheet_name)
                # High Impact sheets (have "+" indicating higher budget - but only if it's clearly higher)
                elif "+" in sheet_name and re.search(r'\$?\d+k\+', sheet_lower):
                    high_impact_sheets.append(sheet_name)
                # Combined sheets (contain "combined" or both budget amounts)
                elif "combined" in sheet_lower or ("200k" in sheet_lower and "100k" in sheet_lower):
                    combined_sheets.append(sheet_name)
                # Original plan sheets (contain budget amounts but not "+" or "high impact")
                elif re.search(r'\$?\d+k', sheet_lower) and "+" not in sheet_name and "high impact" not in sheet_lower:
                    original_sheets.append(sheet_name)
                # Other sheets that might be original plans
                else:
                    # If it's not clearly high impact or combined, assume it's original
                    if "summary" not in sheet_lower and "overview" not in sheet_lower:
                        original_sheets.append(sheet_name)
                        
            
            # Extract actual placement names from each sheet type
            original_placements = []
            for sheet in original_sheets:
                placements = extract_placement_names_from_sheet(excel_file, sheet)
                original_placements.extend(placements)
            original_placements = list(dict.fromkeys(original_placements))  # Remove duplicates
            
            # Extract placements from high impact sheets
            high_impact_all_placements = []
            for sheet in high_impact_sheets:
                placements = extract_placement_names_from_sheet(excel_file, sheet)
                high_impact_all_placements.extend(placements)
            high_impact_all_placements = list(dict.fromkeys(high_impact_all_placements))  # Remove duplicates
            
            combined_placements = []
            for sheet in combined_sheets:
                placements = extract_placement_names_from_sheet(excel_file, sheet)
                combined_placements.extend(placements)
            combined_placements = list(dict.fromkeys(combined_placements))  # Remove duplicates
            
            # Determine high impact placements based on available sheets
            high_impact_placements = []
            original_set = set(original_placements)
            
            if high_impact_all_placements:
                # If we have direct high impact sheets, find placements that are NOT in original
                high_impact_placements = [p for p in high_impact_all_placements if p not in original_set]
                
            elif combined_placements:
                # If no direct high impact sheets but combined sheets exist,
                # find placements that are in combined but not in original
                high_impact_placements = [p for p in combined_placements if p not in original_set]
            
            media_plan_structure[filename] = {
                'original_sheets': original_sheets,
                'high_impact_sheets': high_impact_sheets,
                'combined_sheets': combined_sheets,
                'all_sheets': xl_file.sheet_names,
                'original_placements': original_placements,
                'high_impact_placements': high_impact_placements,
                'combined_placements': combined_placements
            }
            
        except Exception as e:
            print(f"Warning: Could not analyze structure of {excel_file}: {e}")
            media_plan_structure[filename] = {
                'original_sheets': [],
                'high_impact_sheets': [],
                'combined_sheets': [],
                'all_sheets': [],
                'original_placements': [],
                'high_impact_placements': [],
                'combined_placements': []
            }
    
    return media_plan_structure

def create_separate_json_files(result_data):
    """Create two separate JSON files from the combined result data"""
    import json
    
    try:
        # Handle CrewOutput object - extract the raw result
        if hasattr(result_data, 'raw'):
            result_str = result_data.raw
        elif hasattr(result_data, '__str__'):
            result_str = str(result_data)
        else:
            result_str = result_data
        
        # Clean the result string - remove markdown code blocks if present
        if isinstance(result_str, str):
            result_str = result_str.strip()
            if result_str.startswith('```json'):
                result_str = result_str[7:]  # Remove ```json
            if result_str.endswith('```'):
                result_str = result_str[:-3]  # Remove ```
            result_str = result_str.strip()
        
        # Parse the JSON
        data = json.loads(result_str)
        
        # Create original packages file structure
        original_file_data = {
            "results": []
        }
        
        # Create high impact packages file structure
        high_impact_file_data = {
            "results": []
        }
        
        # Process each result entry
        for entry in data.get("results", []):
            # Original packages entry
            original_entry = {
                "id": entry.get("id"),
                "original_package_names": entry.get("original_package_names", [])
            }
            original_file_data["results"].append(original_entry)
            
            # High impact packages entry
            high_impact_entry = {
                "id": entry.get("id"),  # Same ID as original
                "high_impact_package_names": entry.get("high_impact_package_names", []),
                "reasoning": entry.get("reasoning", "")
            }
            high_impact_file_data["results"].append(high_impact_entry)
        
        # Save to separate files
        with open("original_packages.json", "w", encoding="utf-8") as f:
            json.dump(original_file_data, f, indent=2, ensure_ascii=False)
        
        with open("high_impact_packages.json", "w", encoding="utf-8") as f:
            json.dump(high_impact_file_data, f, indent=2, ensure_ascii=False)
        
        return True
        
    except Exception as e:
        return False

def extract_package_names_with_reasoning(excel_files: list):
    """Extract package names from original and high impact plans with AI-powered reasoning"""
    
    # Analyze media plan structure first
    media_plan_structure = analyze_media_plan_structure(excel_files)
    
    session = boto3.Session(region_name="us-east-1")
    embedder_config = {
        "provider": "bedrock",
        "config": {
            "model": "amazon.titan-embed-text-v1",
            "session": session
        }
    }
    
    # Create Excel knowledge sources
    excel_sources = []
    for excel_file in excel_files:
        filename = os.path.basename(excel_file)
        excel_source = ExcelKnowledgeSource(
            file_paths=[filename],
            embedder=embedder_config
        )
        excel_sources.append(excel_source)
    
    # Create single reasoning agent
    reasoning_agent = Agent(
        role='Package Analysis and Reasoning Specialist',
        goal='Extract package names and provide intelligent reasoning for high impact classifications',
        backstory=f"""You are a media planning expert who analyzes placement data and provides
        strategic reasoning for why certain packages are classified as high impact upselling opportunities.
        
        **PRE-EXTRACTED PLACEMENT DATA:**
        {media_plan_structure}
        
        **YOUR TASK:**
        For each media plan file, you must:
        1. List exact placement names from original plans
        2. List exact placement names from high impact plans
        3. Provide intelligent business reasoning for why the high impact placements are premium upselling opportunities
        
        **REASONING EXPERTISE:**
        - Analyze placement characteristics (targeting, format, positioning)
        - Identify premium features that justify higher costs
        - Understand media planning strategy and upselling logic
        - Provide concise, actionable business insights
        """,
        knowledge_sources=excel_sources,
        embedder=embedder_config,
        llm=llm,
        verbose=True,
        allow_delegation=False
    )
    
    # Create single task for extraction and reasoning
    extraction_task = Task(
        description=f"""
        Extract package names and provide reasoning for each media plan file in JSON format.
        
        **REQUIRED JSON OUTPUT FORMAT:**
        
        {{
            "results": [
                {{
                    "id": 1,
                    "filename": "Excel filename",
                    "original_package_names": ["list", "of", "original", "placement", "names"],
                    "high_impact_package_names": ["list", "of", "high", "impact", "placement", "names"],
                    "reasoning": "One sentence explaining why the high impact packages are premium upselling opportunities"
                }},
                {{
                    "id": 2,
                    "filename": "Next Excel filename",
                    "original_package_names": ["original", "placements", "from", "this", "file"],
                    "high_impact_package_names": ["high", "impact", "placements", "from", "this", "file"],
                    "reasoning": "One sentence reasoning for this file's high impact classification"
                }}
            ]
        }}
        
        **MEDIA PLAN STRUCTURE DATA:**
        {media_plan_structure}
        
        **EXTRACTION REQUIREMENTS:**
        - Create one entry per Excel file with sequential ID numbers starting from 1
        - Use exact placement names from 'original_placements' and 'high_impact_placements' arrays
        - Include complete lists of all package names for each file
        - Provide EXACTLY ONE SENTENCE reasoning (maximum 25 words) per file
        - Focus on key differentiators (enhanced targeting, premium formats, better positioning, etc.)
        - Output ONLY valid JSON format - no additional text or explanations
        """,
        expected_output="Valid JSON object with results array containing id, filename, original_package_names array, high_impact_package_names array, and reasoning for each Excel file",
        agent=reasoning_agent
    )
    
    # Create and run simplified Crew
    crew = Crew(
        agents=[reasoning_agent],
        tasks=[extraction_task],
        verbose=True
    )
    
    result = crew.kickoff()
    
    # Create separate JSON files
    create_separate_json_files(result)
    
    return result

# Direct extraction mode
if __name__ == "__main__":
    # Check setup
    if not os.path.exists("knowledge/"):
        exit()
    
    # Get the full relative paths from glob
    excel_files = glob.glob("knowledge/*.xlsx") + glob.glob("knowledge/*.xls")
    if not excel_files:
        exit()
    
    # Extract package names and generate JSON files silently
    try:
        extract_package_names_with_reasoning(excel_files)
    except Exception as e:
        pass

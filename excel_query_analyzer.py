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
                # Skip if it looks like a header or total row
                if (placement_str and
                    not is_header_like(placement_str) and
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

def ask_question(question: str, excel_files: list):
    """Answer user's question about media plans with sheet-level awareness"""
    
    # Analyze media plan structure first
    media_plan_structure = analyze_media_plan_structure(excel_files)
    
    session = boto3.Session(region_name="us-east-1")
    embedder_config = {
        "provider": "bedrock",
        "config": {
            "model": "amazon.titan-embed-text-v1",  # or your preferred embedding model
            "session": session  # Uses default AWS session
        }
    }
    
    # Create Excel knowledge sources for all files in knowledge directory
    # Since ExcelKnowledgeSource searches under knowledge directory automatically,
    # we only need the filenames without the "knowledge/" prefix
    excel_sources = []
    for excel_file in excel_files:
        # Extract just the filename (remove "knowledge/" prefix)
        filename = os.path.basename(excel_file)
        excel_source = ExcelKnowledgeSource(
            file_paths=[filename],
            embedder=embedder_config
        )
        excel_sources.append(excel_source)
    
    # Create specialized agents with enhanced sheet-level awareness
    query_interpreter = Agent(
        role='Media Plan Query Interpreter',
        goal='Understand user queries and identify relevant data sources and sheet-level analysis requirements',
        backstory=f"""You are a query interpretation specialist who understands natural language
        questions about media plans and advertising packages. You excel at breaking down complex
        questions into specific data requirements and identifying which Excel files, sheets, and columns
        are most relevant for answering user questions. 
        
        MEDIA PLAN STRUCTURE KNOWLEDGE:
        {media_plan_structure}
        
        You understand that:
        - Each Excel file represents a unique media plan with multiple sheets
        - Original Plan sheets contain packages within the client's budget
        - High Impact Plan sheets contain upselling packages beyond the client's budget
        - Combined sheets show both original and high impact packages together
        - Somethimes the combined sheet will contain all the same placement names as in original sheet but they will increase the net cost of placements in it.
        - **PLACEMENT NAME IS THE MOST IMPORTANT COLUMN** - this is the primary key for all comparisons
        - CRITICAL LOGIC: Understand high impact sheet structure and mapping:
          * If original + high impact sheets exist → high impact sheet contains ONLY upselling placements, map these to show "for this original plan, this is the corresponding high impact upselling plan"
          * If only original + combined sheets exist → high impact placements = placement names in combined sheet that are NOT in original sheet
        - You need to map original plans with their corresponding high impact plans within the same file using placement name comparison
        
        You understand media planning terminology, package structures, and can distinguish between 
        different types of analysis needed (comparisons, summaries, calculations, etc.).""",
        knowledge_sources=excel_sources,
        embedder=embedder_config,
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    package_analyzer = Agent(
        role='Sheet-Level Package Mapping Specialist',
        goal='Analyze and map packages between original and high impact sheets within each media plan file',
        backstory=f"""You are a package analysis expert specializing in mapping and comparing packages
        between different sheets within the same media plan file. You have access to pre-extracted placement data:
        
        **PRE-EXTRACTED PLACEMENT DATA AVAILABLE:**
        {media_plan_structure}
        
        **PLACEMENT NAME IS THE MOST IMPORTANT ELEMENT** - Focus entirely on placement names for all analysis.
        
        PLACEMENT NAME COMPARISON EXPERTISE (MOST CRITICAL):
        - **PLACEMENT NAME IS THE MOST IMPORTANT COLUMN** - this is your primary focus for all analysis
        - CORE COMPARISON LOGIC: Compare placement names between sheets to identify high impact placements:
          * If original + high impact sheets exist → Map the original placement names to high impact placement names that for this original plan, this will be the high impact plan.
          * If only original + combined sheets exist → find placement names in combined sheet that are NOT in original sheet (these are high impact)
        - Sometimes the combined sheet will contain all the same placement names as in original sheet but they will increase the net cost of placements in it.
        - Understanding that high impact packages are upselling opportunities beyond client budget
        - Mapping relationships using placement name as the definitive identifier
        - Identifying which placement names are exclusive to high impact plans through name comparison
        - Analyzing package variations for matching placement names across sheet types
        
        You understand that each Excel file is a complete media plan containing:
        - Original Plan sheets (client's budget packages), always present.
        - High Impact Plan sheets (upselling packages)
        - Sometimes combined sheets showing both together
        
        **USE THE PRE-EXTRACTED PLACEMENT DATA** from the media plan structure to provide accurate placement name lists and analysis.""",
        knowledge_sources=excel_sources,
        embedder=embedder_config,
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    data_analyst = Agent(
        role='Multi-Sheet Media Plan Data Analyst',
        goal='Perform detailed analysis across multiple sheets within media plan files and extract comparative insights',
        backstory=f"""You are an expert data analyst specializing in multi-sheet media plan Excel data analysis.
        
        **MANDATORY DATA EXTRACTION REQUIREMENTS:**
        - **YOU MUST EXTRACT ACTUAL PLACEMENT NAMES FROM EXCEL FILES** - never say "not available"
        - **SEARCH ALL KNOWLEDGE SOURCES THOROUGHLY** for placement data
        - **ACCESS EVERY SHEET** in each Excel file to find placement names
        - **EXTRACT EXACT PLACEMENT NAMES** as they appear in the Excel cells
        - Look in columns: "Placement Name", "Package Name", "Placement", "Product", "Placement/Product"
        
        SHEET-LEVEL ANALYSIS CAPABILITIES:
        - Package specifications and targeting details across original and high impact sheets
        - Budget allocations and cost analysis comparing original vs high impact plans
        - Media plan metrics and performance indicators by sheet type
        - Display and video specifications variations between plan types
        - Availability and targeting summaries across different sheets
        - Placement name mapping and package relationship analysis
        
        MEDIA PLAN STRUCTURE AWARENESS:
        {media_plan_structure}
        
        You understand that:
        - Each file contains multiple related sheets (original, high impact, combined)
        - Original plans represent client's budget constraints
        - **High impact sheets contain ONLY upselling placements** (no duplicates of original)
        - **PLACEMENT NAMES ARE THE MOST CRITICAL COLUMN** for mapping between sheets
        - CORE MAPPING LOGIC: Map original placements to their corresponding high impact upselling placements to show "for this original placement, this is the high impact upselling option"
        - Analysis should focus on placement name relationships within each file
        
        You excel at placement name mapping analysis within individual media plan files and can identify how original placements relate to their high impact upselling counterparts.""",
        knowledge_sources=excel_sources,
        embedder=embedder_config,
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    report_generator = Agent(
        role='Multi-Plan Report Generator',
        goal='Generate comprehensive reports showing relationships between original and high impact plans across all media plan files',
        backstory=f"""You are a report generation specialist who creates clear, well-structured
        reports from multi-sheet media plan data analysis results. You excel at presenting complex 
        sheet-level relationships and package mappings in an understandable format.
        
        REPORTING SPECIALIZATIONS:
        - Synthesize findings from multiple sheets within each media plan file
        - Create executive summaries showing original vs high impact plan differences
        - Provide strategic recommendations based on placement name mapping analysis
        - Present comparative analysis results clearly across sheet types
        - Highlight upselling opportunities from high impact plans
        - Show package relationships and mapping within each media plan file
        
        MEDIA PLAN STRUCTURE CONTEXT:
        {media_plan_structure}
        
        You understand the business context:
        - Original plans align with client budgets and requirements
        - High impact plans are upselling opportunities beyond client budget
        - Each file represents a complete media plan with related sheets
        - Placement names are the primary key for mapping relationships
        
        Your reports help stakeholders understand upselling opportunities and make informed 
        decisions about media planning strategies across original and high impact offerings.""",
        knowledge_sources=excel_sources,
        embedder=embedder_config,
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    # Create specialized tasks with sheet-level awareness
    interpret_task = Task(
        description=f"""
        Analyze the user query: "{question}"
        
        Your task is to:
        1. Understand what the user is asking for in the context of multi-sheet media plans
        2. Identify which Excel files and specific sheets are most relevant
        3. Determine if the query requires sheet-level mapping within files
        4. Identify if placement/package name mapping is needed between original and high impact sheets
        5. Provide clear guidance on the sheet-level analysis approach needed
        6. Specify which columns and data points should be examined across different sheet types
        
        MULTI-SHEET MEDIA PLAN CONTEXT:
        - Each Excel file is a unique media plan with multiple related sheets
        - Original Plan sheets: Packages within client's budget
        - High Impact Plan sheets: Upselling packages beyond client's budget  
        - Combined sheets: Both original and high impact packages together
        - Somethimes the combined sheet will contain all the same placement names as in original sheet but they will increase the net cost of placements in it.
        - Placement Name is the primary mapping key between sheets
        
        AVAILABLE MEDIA PLAN STRUCTURE:
        {media_plan_structure}
        
        Use the available knowledge sources to understand the data structure and content across all sheets.
        """,
        expected_output="Clear interpretation of the user query with specific sheet-level analysis recommendations and relevant data sources identified",
        agent=query_interpreter
    )
    
    analyze_task = Task(
        description=f"""
        Based on the query interpretation, perform detailed sheet-level analysis to answer: "{question}"
        
        **ABSOLUTE REQUIREMENT: EXTRACT ACTUAL PLACEMENT NAMES FROM EXCEL DATA**
        
        **MANDATORY STEPS - NO EXCEPTIONS:**
        1. **QUERY EVERY KNOWLEDGE SOURCE** - Access all Excel files in the knowledge base
        2. **EXAMINE EVERY SHEET** - Look through all sheets in each Excel file
        3. **FIND PLACEMENT COLUMNS** - Search for "Placement Name", "Package Name", "Placement", "Product" columns
        4. **EXTRACT EXACT NAMES** - Copy placement names exactly as written in Excel cells
        5. **NEVER SAY "NOT AVAILABLE"** - If you can't find data, search harder in different sheets/columns
        
        **FOR EACH EXCEL FILE YOU MUST:**
        - List ALL placement names from original plan sheets (exact names from Excel)
        - List ALL placement names from high impact sheets (exact names from Excel)
        - List ALL placement names from combined sheets (exact names from Excel)
        - Provide the actual placement names, not descriptions or summaries
        
        
        
        SHEET MAPPING STRATEGY:
        1. Always prioritize "Placement Name" columns for mapping
        2. If original plan and high impact sheets are available then directly map the original sheet's placement with te high impact
        placement, that for this plan this will be the high impact plan.
        3. If original and combined sheets are available then compare the placement names of both sheets and the placement names present
        only in the combined sheets are the high imapct packages, map them to the original plan.
        4. Somethimes the combined sheet will contain all the same placement names as in original sheet but they will increase the net 
        cost of placements in it.
        
        MEDIA PLAN STRUCTURE TO ANALYZE:
        {media_plan_structure}
        
        Use the knowledge sources to access the actual data for comprehensive sheet-level analysis.
        """,
        expected_output="Detailed sheet-level data analysis results with specific findings, package mappings between original and high impact sheets, and insights",
        agent=package_analyzer if "package" in question.lower() or "compare" in question.lower() or "map" in question.lower() else data_analyst
    )
    
    report_task = Task(
        description=f"""
        Create a focused response to the user query: "{question}"
        
        **USE THE PRE-EXTRACTED PLACEMENT DATA PROVIDED IN THE MEDIA PLAN STRUCTURE**
        
        **For EACH media plan file separately, provide ONLY the following three things:**
        
        For each Excel file in the knowledge base:
        1. **List of all placement names in original plan sheets** - Use the 'original_placements' data from media plan structure
        2. **List of all placement names in high impact plan sheets** - Use the 'high_impact_placements' data from media plan structure
        3. **Reasoning of why this particular plan is high impact plan** - Provide business reasoning based on the placement differences and targeting strategies
        
        **STRICT OUTPUT FORMAT FOR EACH FILE:**
        
        # [FILENAME]
        
        ## 1. Original Plan Placement Names:
        [List all placements from 'original_placements' in the media plan structure]
        
        ## 2. High Impact Plan Placement Names:
        [List all placements from 'high_impact_placements' in the media plan structure]
        
        ## 3. Why These Are High Impact Plans:
        [Provide ONE brief sentence explaining the key differentiator that makes these high impact placements premium upselling opportunities]
        
        ---
        
        **MEDIA PLAN STRUCTURE WITH PRE-EXTRACTED DATA:**
        {media_plan_structure}
        
        **REASONING GUIDELINES:**
        - Provide ONE concise sentence explaining why these are high impact placements
        - Focus on the key differentiator (roadblock, enhanced targeting, premium format, etc.)
        - Keep reasoning under 20 words maximum
        """,
        expected_output="For each media plan file separately: 1) Exact placement names from original sheets only, 2) Exact placement names from high impact sheets only, 3) Reasoning for high impact classification - with no cross-contamination between lists",
        agent=report_generator
    )

    # Create and run Crew with multiple agents and tasks
    crew = Crew(
        agents=[query_interpreter, package_analyzer, data_analyst, report_generator],
        tasks=[interpret_task, analyze_task, report_task],
        verbose=True
    )
    
    result = crew.kickoff()
    return result

# Interactive mode
if __name__ == "__main__":
    print("🎯 Multi-Sheet Media Plan Agent - Ask questions about your Excel files!")
    
    # Check setup
    if not os.path.exists("knowledge/"):
        print("❌ 'knowledge' directory not found!")
        exit()
    
    # Get the full relative paths from glob
    excel_files = glob.glob("knowledge/*.xlsx") + glob.glob("knowledge/*.xls")
    if not excel_files:
        print("❌ No Excel files found in knowledge/ directory")
        exit()
    
    print(f"📁 Available Excel files:")
    for file in excel_files:
        print(f"  • {os.path.basename(file)}")
    

    
    print("\n💬 Ask your questions (type 'quit' to exit):")
    print("\n🧪 SHEET-LEVEL MAPPING TEST QUESTIONS:")
   
    print("Map the relationship between original and high impact packages using placement names")
   
    
    while True:
        question = input("\n> ").strip()
        
        if question.lower() in ['quit', 'exit']:
            print("👋 Goodbye!")
            break
            
        if question:
            print("\n🔍 Analyzing across multiple sheets...")
            try:
                result = ask_question(question, excel_files)
                print(f"\n📊 Answer:\n{result}")
            except Exception as e:
                print(f"❌ Error: {str(e)}")

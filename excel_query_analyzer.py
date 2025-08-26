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

def analyze_media_plan_structure(excel_files: list):
    """Analyze the structure of media plan files to identify sheet relationships"""
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
                # Combined sheets (contain "+" or "combined" or both budget amounts)
                elif "+" in sheet_name or "combined" in sheet_lower or ("200k" in sheet_lower and "100k" in sheet_lower):
                    combined_sheets.append(sheet_name)
                # Original plan sheets (contain budget amounts like "200k", "100k" but not high impact)
                elif re.search(r'\d+k', sheet_lower) and "high impact" not in sheet_lower:
                    original_sheets.append(sheet_name)
                # Other sheets that might be original plans
                else:
                    # If it's not clearly high impact or combined, assume it's original
                    if "summary" not in sheet_lower and "overview" not in sheet_lower:
                        original_sheets.append(sheet_name)
            
            media_plan_structure[filename] = {
                'original_sheets': original_sheets,
                'high_impact_sheets': high_impact_sheets,
                'combined_sheets': combined_sheets,
                'all_sheets': xl_file.sheet_names
            }
            
        except Exception as e:
            print(f"Warning: Could not analyze structure of {excel_file}: {e}")
            media_plan_structure[filename] = {
                'original_sheets': [],
                'high_impact_sheets': [],
                'combined_sheets': [],
                'all_sheets': []
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
        between different sheets within the same media plan file. You excel at:
        
        PLACEMENT NAME COMPARISON EXPERTISE (MOST CRITICAL):
        - **PLACEMENT NAME IS THE MOST IMPORTANT COLUMN** - this is your primary focus for all analysis
        - CORE COMPARISON LOGIC: Compare placement names between sheets to identify high impact placements:
          * If original + high impact sheets exist → Map the original placement names to high impact placement names that for this original plan, this will be the high impact plan.
          * If only original + combined sheets exist → find placement names in combined sheet that are NOT in original sheet (these are high impact)
        - Somethimes the combined sheet will contain all the same placement names as in original sheet but they will increase the net cost of placements in it.
        - Understanding that high impact packages are upselling opportunities beyond client budget
        - Mapping relationships using placement name as the definitive identifier
        - Identifying which placement names are exclusive to high impact plans through name comparison
        - Analyzing package variations for matching placement names across sheet types
        
        MEDIA PLAN STRUCTURE:
        {media_plan_structure}
        
        You understand that each Excel file is a complete media plan containing:
        - Original Plan sheets (client's budget packages), always present.
        - High Impact Plan sheets (upselling packages)
        - Sometimes combined sheets showing both together
        
        **YOUR PRIMARY FOCUS IS PLACEMENT NAME COMPARISON** to identify which placement names are extra in combined plans compared to original plans, as these represent the high impact upselling opportunities.""",
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
        You can analyze data patterns, trends, and provide detailed insights about:
        
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
        
        Your task is to:
        1. Examine the relevant datasets from the identified knowledge sources across multiple sheets
        2. If package mapping is needed, map packages between original and high impact sheets within each file
        3. Use placement names as the primary mapping key between sheet types
        4. Identify relationships between original plan and high impact plan offerings:
           - **High impact sheet contains ONLY upselling placements** (no duplicates of original)
           - Map original placements to their corresponding high impact upselling options
           - Show "for this original placement, this is the corresponding high impact upselling placement"
           - Compare specifications and details between original placements and their high impact counterparts
        5. Perform appropriate calculations, aggregations, or comparisons across sheet types
        
        
        
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
        Create a comprehensive response to the user query: "{question}"
        
        Based on the query interpretation and sheet-level data analysis, generate a clear,
        well-structured response that:
        1. Directly answers the user's question with sheet-level context
        2. Provides supporting data and evidence from multiple sheets within each media plan file
        3. Includes relevant context about media plan structure and sheet relationships
        4. Clearly distinguishes between original plan and high impact packages within each file
        5. Shows placement name mappings between original and high impact sheets
        6. Offers actionable insights about upselling opportunities from high impact plans
        7. Mentions specific data sources, files, and sheets used from the knowledge base
        8. Presents findings in a logical, easy-to-understand format with sheet-level organization
        9. Highlights key business implications of original vs high impact plan differences
        
        SHEET-LEVEL REPORTING GUIDELINES:
        - Organize findings by media plan file, then by sheet type within each file
        - Use clear headings showing file names and sheet relationships
        - Provide specific examples of package mappings between sheets
        - Explain the significance of high impact plan differences
        - Include comparative analysis between original and high impact offerings
        - Highlight upselling opportunities and business value
        
        MEDIA PLAN CONTEXT FOR REPORTING:
        {media_plan_structure}
        """,
        expected_output="A clear, comprehensive response to the user query with supporting sheet-level analysis, package mappings, and actionable insights about original vs high impact plan relationships",
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

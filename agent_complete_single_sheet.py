from crewai import Agent
from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters
import os
import json
import pytz
from crewai import Agent, Task, Crew, Process, LLM

# Define input and output file paths (absolute paths required for excel_mcp in stdio mode)
import os
input_file = os.path.abspath("NBCE_The Voice S28_Fall_25_RFP Template (Samsung Ads) 5.13.xlsx")
output_file = os.path.abspath("complete_cleaned_data.xlsx")

server_params = StdioServerParameters(
    command="python3",
    args=["-m", "excel_mcp", "stdio"],
    env={"UV_PYTHON": "3.12", **os.environ},
)

llm = LLM(
    model="bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    aws_region_name="us-east-1",
    temperature=0.2,        # Higher for more creative outputs
    max_tokens=4096,        # Reduce max tokens to avoid context limit
)

with MCPServerAdapter(server_params) as mcp_tools:
    print(f"Available tools: {[tool.name for tool in mcp_tools]}")

    # Agent 1: Single Sheet Excel Reader - Reads one sheet and identifies all tabular data
    excel_reader = Agent(
        role='Single Sheet Excel File Reader and Data Mapper',
        goal='Read Excel file and identify ALL tabular data from the primary sheet, mapping complete dimensions including all rows and columns',
        backstory="""You are an expert Excel file reader who specializes in processing single sheet workbooks
        and identifying the complete dimensions of ALL tabular data within one worksheet. You systematically explore
        both horizontal (columns) and vertical (rows) extents of data tables. You understand that
        data tables can extend far to the right with many columns (A through V, W, X or beyond)
        and you ensure no columns or rows are missed during analysis. You consolidate all data ranges
        from one sheet into a comprehensive view.""",
        tools=mcp_tools,
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    # Agent 2: Single Sheet Data Processor - Analyzes and consolidates all data from one sheet
    data_processor = Agent(
        role='Single Sheet Data Consolidator',
        goal='Analyze Excel data from one sheet and create a consolidated view of ALL tabular data without splitting into multiple ranges',
        backstory="""You are a data analysis expert who specializes in processing single sheet Excel workbooks
        and consolidating ALL tabular data into one comprehensive dataset. You understand that a single sheet
        may contain multiple data sections (headers, specifications, media plans, etc.) that should be
        combined into one unified output. You identify the complete boundaries of all data and create
        a consolidation plan that preserves all information in a single, organized structure.""",
        tools=mcp_tools,
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    # Agent 3: Single Sheet Excel Writer - Creates one consolidated output sheet
    excel_writer = Agent(
        role='Single Sheet Excel Writer and Data Consolidator',
        goal='Create ONE Excel sheet with ALL tabular data consolidated from the source sheet without splitting into multiple worksheets',
        backstory="""You are an Excel file creation specialist who ensures that ALL data from a single source
        sheet is consolidated into ONE output worksheet. You do NOT create multiple sheets or split data.
        Instead, you organize all the tabular data from the source into one comprehensive, well-structured
        output sheet that contains ALL rows and ALL columns from the original data, properly organized
        and cleaned but kept together as one unified dataset.""",
        tools=mcp_tools,
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    # Task 1: Identify All Data in Primary Sheet
    read_task = Task(
        description=f"""
        Read the Excel file at '{input_file}' and identify ALL tabular data in the primary sheet.
        
        Steps:
        1. Use get_workbook_metadata to get sheet names
        2. Focus on the primary sheet (usually first sheet or main data sheet)
        3. Identify ALL data ranges that contain tabular information:
           - Check A49:Y58 (main media plan area)
           - Check A1:AC5 (header/specs area)
           - Check for any other data ranges with content
        4. Map the COMPLETE extent of all data (find the furthest row and column with data)
        
        Goal: Find ALL data ranges but plan to consolidate them into ONE output sheet.
        """,
        expected_output="Complete mapping of all data ranges in the primary sheet with plan for consolidation",
        agent=excel_reader
    )

    # Task 2: Create Consolidation Plan
    process_task = Task(
        description=f"""
        Create a consolidation plan to combine ALL data from the primary sheet into ONE unified output.
        
        For the data ranges identified:
        1. Determine how to best organize all data into one consolidated structure
        2. Plan the layout to include ALL information without losing any data
        3. Ensure headers, specifications, and main data are all included in one sheet
        
        IMPORTANT: Do NOT plan multiple output sheets. Plan ONE consolidated output sheet.
        """,
        expected_output="Consolidation plan for creating ONE output sheet with all data combined",
        agent=data_processor
    )

    # Task 3: Extract and Write Consolidated Data
    write_task = Task(
        description=f"""
        Extract ALL data from the primary sheet and create ONE consolidated output file.
        
        Steps:
        1. Create new workbook using create_workbook
        2. Read ALL data ranges identified in the consolidation plan
        3. Create ONE worksheet named "Consolidated_Data"
        4. Write ALL data to this single worksheet in an organized manner:
           - Include headers/specifications at the top
           - Include main data table below
           - Ensure ALL columns and rows are preserved
        5. Save to '{output_file}'
        
        CRITICAL: Create only ONE worksheet with ALL data consolidated together.
        """,
        expected_output=f"Excel file '{output_file}' with ONE sheet containing all consolidated data",
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
    print("All tasks completed!")
    print(f"Result: {result}")
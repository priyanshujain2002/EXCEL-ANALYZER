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
output_file = os.path.abspath("cleaned_data.xlsx")

server_params = StdioServerParameters(
    command="python3",
    args=["-m", "excel_mcp", "stdio"],
    env={"UV_PYTHON": "3.12", **os.environ},
)

llm = LLM(
    model="bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    aws_region_name="us-east-1",
    temperature=0.2,        # Higher for more creative outputs
)

with MCPServerAdapter(server_params) as mcp_tools:
    print(f"Available tools: {[tool.name for tool in mcp_tools]}")

    # Agent 1: Excel Reader - Reads the Excel file and extracts targeted data
    excel_reader = Agent(
        role='Excel File Reader',
        goal='Read ALL the sheets present in the given excel file and extract targeted data from specific ranges',
        backstory="""You are an expert Excel file reader who specializes in reading Excel files
        efficiently by targeting specific data ranges. You avoid reading massive ranges that could
        cause performance issues. You focus on getting workbook metadata first for ALL the sheets, then reading
        smaller, targeted ranges to understand the data structure.""",
        tools=mcp_tools,
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    # Agent 2: Data Processor - Analyzes data and identifies clean tabular sections
    data_processor = Agent(
        role='Data Analysis Specialist',
        goal='Analyze Excel data of ALL the sheets and identify clean tabular data ranges efficiently from ALL the sheets',
        backstory="""You are a data analysis expert who specializes in identifying meaningful
        tabular data from ALL the sheets within Excel files by examining small sample ranges first. You can distinguish 
        between headers, formatting rows, empty cells, and actual data. You work efficiently by 
        examining targeted ranges of ALL the sheets rather than processing massive amounts of data at once.""",
        tools=mcp_tools,
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    # Agent 3: Excel Writer - Creates new Excel file with cleaned data
    excel_writer = Agent(
        role='Excel Data Writer and Worksheet Populator',
        goal='Execute MCP tools to create Excel files and POPULATE each worksheet with actual tabular data from ALL source sheets',
        backstory="""You are a data transfer specialist who EXECUTES Excel MCP tools to create files and populate them with data.
        Your primary responsibility is to:
        1. Execute create_workbook to make the output file
        2. Execute read_data_from_excel to extract data from each source sheet
        3. Execute create_worksheet for each sheet (when multiple sheets exist)
        4. Execute write_data_to_excel to POPULATE each worksheet with the actual extracted data
        
        You do NOT provide summaries or descriptions. You EXECUTE the tools and ensure every worksheet contains
        the actual tabular data from the corresponding source sheet. You verify data transfer completion.""",
        tools=mcp_tools,
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    # Task 1: Read Excel File Metadata and Find Data Boundaries
    read_task = Task(
        description=f"""
        Read the Excel file located at '{input_file}' and find the exact data boundaries for ALL sheets.
        
        MANDATORY MULTI-SHEET PROCESSING:
        1. Use get_workbook_metadata tool to get ALL sheet names - there should be MULTIPLE sheets
        2. You MUST process EVERY SINGLE sheet found - do not skip any sheets
        3. For EACH AND EVERY sheet found:
           a. Use read_data_from_excel with sheet_name parameter for each individual sheet
           b. Read a large range (A1:AZ100) to find actual data boundaries for that specific sheet
           c. Identify the rightmost column containing data for that sheet
           d. Identify the bottommost row containing data for that sheet
           e. Record the EXACT data range for that specific sheet
        
        CRITICAL: Process ALL sheets individually. If there are 5 sheets, you must process all 5 sheets.
        Report results for EVERY sheet: Sheet1: range, Sheet2: range, Sheet3: range, etc.
        """,
        expected_output="Complete list with exact sheet names and precise data ranges for EVERY sheet in the workbook (minimum 2+ sheets expected)",
        agent=excel_reader
    )

    # Task 2: Validate and Confirm Exact Data Ranges
    process_task = Task(
        description=f"""
        Validate the exact data ranges found for ALL sheets in the previous task.
        
        MULTI-SHEET VALIDATION:
        1. Take the exact data ranges identified for EVERY sheet in the previous task
        2. You must have data ranges for MULTIPLE sheets (not just one)
        3. Confirm these ranges contain the complete tabular data for EACH individual sheet
        4. Do NOT modify or estimate - use the EXACT ranges found for ALL sheets
        5. Prepare the extraction plan with exact sheet names and exact ranges for ALL sheets
        
        CRITICAL: Your output must include ALL sheets found in the previous task.
        Output format for ALL sheets: "Sheet1": A1:Y58, "Sheet2": A49:AC65, "Sheet3": B10:Z30, etc.
        """,
        expected_output="Confirmed exact data ranges for ALL sheets with precise coordinates (multiple sheets required)",
        agent=data_processor,
        context = [read_task]
    )

    # Task 3: Replicate Excel Structure with Exact Data
    write_task = Task(
        description=f"""
        MANDATORY MULTI-SHEET TOOL EXECUTION:
        
        1. FIRST: Execute create_workbook(file_path='{output_file}')
        
        2. FOR EVERY SINGLE SHEET from previous task (you must process ALL sheets):
           REPEAT this sequence for EACH sheet:
           a. Execute read_data_from_excel(file_path='{input_file}', sheet_name='EXACT_SHEET_NAME', range='EXACT_RANGE')
           b. Execute create_worksheet(file_path='{output_file}', sheet_name='SAME_EXACT_SHEET_NAME')
           c. Execute write_data_to_excel(file_path='{output_file}', sheet_name='SAME_EXACT_SHEET_NAME', data=extracted_data, start_cell='A1')
        
        CRITICAL MULTI-SHEET RULES:
        - Process EVERY sheet identified in previous task (not just the first one)
        - If previous task found 5 sheets, you must create 5 output sheets
        - Use the EXACT sheet names from the source file for ALL sheets
        - Use the EXACT data ranges identified for ALL sheets
        - Show tool execution results for EVERY sheet processed
        
        You must execute tools for ALL sheets, not just one sheet.
        """,
        expected_output=f"Tool execution results showing '{output_file}' created with ALL sheets from source file, each populated with data",
        agent=excel_writer,
        context = [process_task]
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
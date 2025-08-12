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
)

with MCPServerAdapter(server_params) as mcp_tools:
    print(f"Available tools: {[tool.name for tool in mcp_tools]}")

    # Agent 1: Excel Reader - Reads the Excel file and extracts targeted data
    excel_reader = Agent(
        role='Excel File Reader and Data Mapper',
        goal='Read Excel files and map the complete dimensions of tabular data including all rows and columns',
        backstory="""You are an expert Excel file reader who specializes in identifying the complete
        dimensions of data tables. You systematically explore both horizontal (columns) and vertical (rows)
        extents of data tables. You understand that media plan tables can extend far to the right with
        many columns (A through V, W, X or beyond) and you ensure no columns or rows are missed.""",
        tools=mcp_tools,
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    # Agent 2: Data Processor - Analyzes data and identifies complete tabular sections
    data_processor = Agent(
        role='Complete Data Range Analyst',
        goal='Analyze Excel data and identify the full dimensional boundaries of tabular data (all rows AND all columns)',
        backstory="""You are a data analysis expert who specializes in identifying the complete
        boundaries of tabular data. You understand that data tables can be very wide (extending to
        column V, W, X or beyond) and you systematically determine both the row range AND the complete
        column range. You ensure that ALL columns of the media plan are captured, including financial
        data, dates, specifications, and metadata columns.""",
        tools=mcp_tools,
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    # Agent 3: Excel Writer - Creates new Excel file with complete cleaned data
    excel_writer = Agent(
        role='Complete Data Excel Writer',
        goal='Create new Excel files with ALL rows and ALL columns of cleaned tabular data',
        backstory="""You are an Excel file creation specialist who ensures that NO data is lost
        during the extraction process. You create output files that contain the complete tabular
        data with ALL rows and ALL columns from the original media plan table. You verify that
        the output matches the full dimensions of the source data.""",
        tools=mcp_tools,
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    # Task 1: Map Complete Data Table Dimensions
    read_task = Task(
        description=f"""
        Read the Excel file located at '{input_file}' and map the complete dimensions of the tabular data.
        
        Your task is to:
        1. Use get_workbook_metadata tool to understand the file structure and sheet names
        2. For the main sheet (likely '$400K Media Plan'), systematically explore the data dimensions:
           - Read A1:AA20 to see the wide header structure (extending beyond column V)
           - Read A45:AA65 to capture the main data table area with full width
           - Read A49:AA56 to focus on the core media plan table with complete columns
        3. Identify ALL columns in the media plan table, which should include:
           - Basic info: Site Name, Package Name, Placement Name, Audience Demo, Audience Targeting
           - Technical: Device Type, Ad Unit/Size, Site Served Only, Buy Model, Media Type
           - Dates: Start Date, End Date
           - Capabilities: Can be purchased on Viewability, Can Utilize MOAT, Can Purchase Programmatically
           - Specifications: AutoPlay or Click to Play, Is the Unit Skippable, Creative Restrictions
           - Financial: Rate, Impressions, Net Cost, Cancellation Date, % SOV, Minimum amounts
        4. Determine the exact row range (likely rows 49-55 or similar)
        5. Determine the complete column range (likely A through V, W, X, Y, Z, AA or beyond)
        
        CRITICAL: Explore wide ranges to ensure you capture ALL columns of the media plan table.
        """,
        expected_output="Complete dimensional mapping of the media plan table including exact row boundaries and the full column extent (A through the last populated column).",
        agent=excel_reader
    )

    # Task 2: Define Complete Data Extraction Range
    process_task = Task(
        description=f"""
        Based on the dimensional mapping from the previous task, define the complete extraction range.
        
        Your task is to:
        1. Analyze the dimensional data to identify the exact boundaries of the complete media plan table
        2. Define the precise starting row where the column headers begin
        3. Define the precise ending row that captures ALL media plan entries (all 6+ data rows)
        4. Define the complete column range from A to the last populated column (likely V, W, X, Y, Z, AA or beyond)
        5. Ensure the range captures ALL columns including:
           - All descriptive columns (Site Name through Media Type)
           - All date columns (Start Date, End Date)
           - All capability/specification columns
           - All financial and business columns (Rate, Impressions, Net Cost, etc.)
        6. Provide a single comprehensive range that captures the ENTIRE media plan table
        
        CRITICAL: The range must include ALL rows AND ALL columns - no truncation allowed.
        """,
        expected_output="A single, comprehensive range specification (e.g., A49:AA55) that captures the complete media plan table with ALL rows and ALL columns.",
        agent=data_processor,
        context = 
    )

    # Task 3: Extract and Write Complete Data
    write_task = Task(
        description=f"""
        Extract the complete tabular data and create a comprehensive output file.
        
        Your task is to:
        1. Use the complete range identified in the previous task
        2. Read the ENTIRE range using read_data_from_excel with the comprehensive range
        3. Verify that you have captured:
           - ALL column headers (should be 20+ columns from Site Name to Minimum amounts)
           - ALL media plan data rows (should be 6+ complete entries)
           - ALL data points for each entry (no missing columns or truncated data)
        4. Create a new workbook using create_workbook
        5. Write ALL the data to the new file using write_data_to_excel
        6. Save the complete data to '{output_file}'
        
        VERIFICATION REQUIREMENTS:
        - The output must contain ALL columns from the original table
        - The output must contain ALL rows of media plan data
        - No data should be truncated or omitted
        - Each row should have complete information across all columns
        """,
        expected_output=f"A comprehensive Excel file '{output_file}' containing the complete media plan table with ALL rows and ALL columns - no data omitted.",
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
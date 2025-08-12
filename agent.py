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

    # Agent 1: Excel Reader - Reads the Excel file and extracts raw data
    excel_reader = Agent(
        role='Excel File Reader',
        goal='Read Excel files and extract raw data from all sheets',
        backstory="""You are an expert Excel file reader who specializes in reading Excel files
        and extracting all available data. You can read different sheets, understand the structure
        of Excel files, and provide comprehensive data extraction.""",
        tools=mcp_tools,
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    # Agent 2: Data Processor - Analyzes data and identifies clean tabular sections
    data_processor = Agent(
        role='Data Analysis Specialist',
        goal='Analyze Excel data and identify clean tabular data ranges',
        backstory="""You are a data analysis expert who specializes in identifying meaningful
        tabular data within messy Excel files. You can distinguish between headers, formatting rows,
        empty cells, and actual data. Your expertise lies in determining the exact row and column
        ranges that contain clean, structured data.""",
        tools=mcp_tools,
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    # Agent 3: Excel Writer - Creates new Excel file with cleaned data
    excel_writer = Agent(
        role='Excel File Writer',
        goal='Create new Excel files with cleaned and structured data',
        backstory="""You are an Excel file creation specialist who takes cleaned data and creates
        new, properly formatted Excel files. You ensure that the output files contain only the
        essential tabular data without any formatting artifacts or unnecessary elements.""",
        tools=mcp_tools,
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    # Task 1: Read Excel File
    read_task = Task(
        description=f"""
        Read the Excel file located at '{input_file}' and extract all available data.
        
        Your task is to:
        1. Use read_data_from_excel tool to read the Excel file
        2. Get workbook metadata to understand the structure
        3. Extract data from all sheets in the workbook
        4. Provide a comprehensive overview of the file contents
        
        Use the appropriate MCP tools like read_data_from_excel and get_workbook_metadata.
        """,
        expected_output="Complete data extraction from the Excel file with detailed information about all sheets and their contents.",
        agent=excel_reader
    )

    # Task 2: Process and Identify Clean Data Ranges
    process_task = Task(
        description=f"""
        Analyze the extracted Excel data and identify the clean tabular data ranges.
        
        Your task is to:
        1. Analyze the raw data from the previous task
        2. Identify rows and columns that contain meaningful tabular data
        3. Filter out garbage data such as:
           - Extra headers and titles
           - Formatting rows
           - Empty rows and columns
           - Merged cells content that's not part of the data table
           - Summary rows or totals that are not part of the main data
        4. Determine the exact start and end row/column ranges for clean data
        5. Specify which sheet contains the main tabular data
        
        Provide specific row and column ranges (e.g., "Rows 5-25, Columns A-F") for the clean data.
        """,
        expected_output="Detailed analysis specifying the exact row and column ranges containing clean tabular data, including sheet name and data boundaries.",
        agent=data_processor
    )

    # Task 3: Write Cleaned Data to New Excel File
    write_task = Task(
        description=f"""
        Create a new Excel file with only the cleaned tabular data based on the identified ranges.
        
        Your task is to:
        1. Use the identified clean data ranges from the previous task
        2. Extract only the specified clean data using read_data_from_excel with proper range
        3. Create a new workbook using create_workbook
        4. Write the cleaned data to the new file using write_data_to_excel
        5. Save the cleaned data to '{output_file}'
        
        Use MCP tools: create_workbook, write_data_to_excel, and read_data_from_excel.
        """,
        expected_output=f"A new Excel file '{output_file}' containing only the clean tabular data without any formatting or garbage content.",
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
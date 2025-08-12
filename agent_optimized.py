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
        goal='Read Excel files and extract targeted data from specific ranges',
        backstory="""You are an expert Excel file reader who specializes in reading Excel files
        efficiently by targeting specific data ranges. You avoid reading massive ranges that could
        cause performance issues. You focus on getting workbook metadata first, then reading
        smaller, targeted ranges to understand the data structure.""",
        tools=mcp_tools,
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    # Agent 2: Data Processor - Analyzes data and identifies clean tabular sections
    data_processor = Agent(
        role='Data Analysis Specialist',
        goal='Analyze Excel data and identify clean tabular data ranges efficiently',
        backstory="""You are a data analysis expert who specializes in identifying meaningful
        tabular data within Excel files by examining small sample ranges first. You can distinguish 
        between headers, formatting rows, empty cells, and actual data. You work efficiently by 
        examining targeted ranges rather than processing massive amounts of data at once.""",
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

    # Task 1: Read Excel File Metadata and Sample Data
    read_task = Task(
        description=f"""
        Read the Excel file located at '{input_file}' and extract metadata and sample data.
        
        Your task is to:
        1. Use get_workbook_metadata tool to understand the file structure and sheet names
        2. For each sheet, read the full width sample range (e.g., A1:AC20) to understand the complete data structure
        3. Focus on identifying where the actual data starts and what ALL the column headers are
        4. Make sure to capture the full horizontal extent of the data (columns A through AC or beyond)
        5. Provide a summary of each sheet's structure and content type including ALL columns
        
        IMPORTANT: Read the full width of data but limit rows to avoid performance issues. Use ranges like A1:AC20 to capture all columns.
        """,
        expected_output="Metadata about the Excel file including sheet names, and sample data from small ranges showing the structure and headers of each sheet.",
        agent=excel_reader
    )

    # Task 2: Process and Identify Clean Data Ranges
    process_task = Task(
        description=f"""
        Based on the sample data from the previous task, identify the clean tabular data ranges.
        
        Your task is to:
        1. Analyze the sample data to identify which sheet contains the main tabular data
        2. Determine the approximate row where data starts (after headers/titles)
        3. Identify the column range that contains meaningful data
        4. Estimate a reasonable data range (e.g., A49:AC58) that captures the main data table with ALL columns
        5. Avoid including:
           - Title rows at the top
           - Empty rows and columns
           - Summary/total rows at the bottom
        6. Provide specific range recommendations for data extraction
        
        Focus on the main data sheet and provide a targeted range that includes ALL columns (A through AC or beyond) for complete data extraction.
        """,
        expected_output="Specific recommendations for data ranges to extract, including sheet name, starting row, ending row, and column range for the clean tabular data.",
        agent=data_processor
    )

    # Task 3: Write Cleaned Data to New Excel File
    write_task = Task(
        description=f"""
        Create a new Excel file with only the cleaned tabular data based on the identified ranges.
        
        Your task is to:
        1. Use the specific data range identified in the previous task
        2. Read ONLY that specific range using read_data_from_excel with the exact range
        3. Create a new workbook using create_workbook
        4. Write the cleaned data to the new file using write_data_to_excel
        5. Save the cleaned data to '{output_file}'
        
        Make sure to use the exact range specified by the data processor to avoid reading unnecessary data.
        """,
        expected_output=f"A new Excel file '{output_file}' containing only the clean tabular data from the specified range.",
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
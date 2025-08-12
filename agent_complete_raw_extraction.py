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
    temperature=0.1,        # Lower temperature for more precise extraction
    max_tokens=4096,
)

with MCPServerAdapter(server_params) as mcp_tools:
    print(f"Available tools: {[tool.name for tool in mcp_tools]}")

    # Agent 1: Raw Data Extractor - Copies data exactly as-is
    excel_reader = Agent(
        role='Raw Excel Data Extractor',
        goal='Extract ALL tabular data from Excel sheet exactly as-is without any processing, summarization, or modification',
        backstory="""You are a precise data extraction specialist who copies Excel data exactly as it appears
        in the source file. You do NOT summarize, process, analyze, or modify any data. Your job is to
        identify where tabular data exists and extract it in its original form - every cell value, every
        formula result, every number, text, and date exactly as it appears. You preserve the original
        structure, formatting, and content without any interpretation or changes.""",
        tools=mcp_tools,
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    # Agent 2: Data Range Mapper - Maps exact ranges without processing
    data_processor = Agent(
        role='Exact Data Range Mapper',
        goal='Map the exact cell ranges containing data without analyzing or processing the content',
        backstory="""You are a technical specialist who identifies the precise cell ranges that contain
        data in Excel sheets. You do NOT analyze what the data means or how it should be organized.
        You simply identify the exact ranges (like A1:Z100) that contain non-empty cells and need to
        be copied. You work with raw cell coordinates and ranges, not data interpretation.""",
        tools=mcp_tools,
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    # Agent 3: Raw Data Writer - Copies data exactly to output
    excel_writer = Agent(
        role='Raw Data Copy Writer',
        goal='Copy extracted data to output file exactly as-is without any modifications, processing, or summarization',
        backstory="""You are a data copying specialist who transfers Excel data from source to destination
        without making ANY changes. You do NOT summarize, reorganize, clean, or process the data in any way.
        You simply copy the raw data values from the identified ranges and write them to the output file
        in their original form. Every cell value must be preserved exactly as it was in the source.""",
        tools=mcp_tools,
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    # Task 1: Find Data Ranges (No Processing)
    read_task = Task(
        description=f"""
        Read the Excel file at '{input_file}' and identify the exact cell ranges that contain data.
        
        Steps:
        1. Use get_workbook_metadata to get sheet names
        2. Focus on the first/primary sheet
        3. Identify the exact ranges with data:
           - Check A49:Y58 area for data
           - Check A1:AC5 area for data
           - Find the actual boundaries of data (last row and column with content)
        4. Report the EXACT cell ranges (e.g., A49:Y58, A1:AC5)
        
        DO NOT analyze or interpret the data content. Just find where data exists.
        """,
        expected_output="Exact cell ranges containing data (e.g., A49:Y58, A1:AC5)",
        agent=excel_reader
    )

    # Task 2: Plan Raw Extraction
    process_task = Task(
        description=f"""
        Create a simple extraction plan for copying the identified data ranges exactly as-is.
        
        For each data range identified:
        1. Note the exact range coordinates
        2. Plan to copy this range directly to output
        3. NO processing, NO summarization, NO analysis
        
        Keep it simple: just list the ranges to copy.
        """,
        expected_output="Simple list of exact ranges to copy (e.g., Copy A49:Y58, Copy A1:AC5)",
        agent=data_processor
    )

    # Task 3: Raw Data Copy
    write_task = Task(
        description=f"""
        Copy the data from identified ranges to output file exactly as-is.
        
        Steps:
        1. Create new workbook using create_workbook
        2. For each range in the extraction plan:
           - Use read_data_from_excel to get the raw data
           - Use write_data_to_excel to copy it exactly to output
        3. Create ONE worksheet with all data copied as-is
        4. Save to '{output_file}'
        
        CRITICAL: Do NOT modify, summarize, or process any data. Copy exactly as-is.
        """,
        expected_output=f"Excel file '{output_file}' with raw data copied exactly from source",
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
from crewai import Agent
from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters
import os
from crewai import Agent, Task, Crew, LLM

# Define input and output file paths
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
    temperature=0.1,
    max_tokens=2048,
)

with MCPServerAdapter(server_params) as mcp_tools:
    print(f"Available tools: {[tool.name for tool in mcp_tools]}")

    # Single Agent: Table Extractor
    table_extractor = Agent(
        role='Simple Table Data Extractor',
        goal='Extract only the main tabular data from the Excel sheet and copy it to output file',
        backstory="""You are a straightforward data extractor. Your only job is to:
        1. Find the main data table in the Excel sheet
        2. Extract that table exactly as-is
        3. Write it to a new Excel file
        You do not analyze, summarize, or process the data - just copy the table.""",
        tools=mcp_tools,
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    # Single Task: Extract Table
    extract_task = Task(
        description=f"""
        Extract the main tabular data from '{input_file}' and save to '{output_file}'.
        
        Steps:
        1. Open the Excel file and get the first sheet
        2. Find the main data table (likely around A49:Y58 based on the file)
        3. Read the complete table data using read_data_from_excel
        4. Create a new workbook and write the table data exactly as-is
        5. Save the output file
        
        Focus only on the main data table - ignore headers, specs, or other content.
        """,
        expected_output=f"Excel file '{output_file}' containing only the main tabular data",
        agent=table_extractor
    )

    # Create simple crew
    crew = Crew(
        agents=[table_extractor],
        tasks=[extract_task],
        verbose=True
    )

    # Execute
    result = crew.kickoff()
    print("Table extraction completed!")
    print(f"Result: {result}")
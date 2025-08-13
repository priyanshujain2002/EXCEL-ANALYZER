from crewai import Agent
from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters
import os
from crewai import Task, Crew, Process, LLM

# Define input and output file paths
input_file = os.path.abspath("NBCE_The Voice S28_Fall_25_RFP Template (Samsung Ads) 5.13.xlsx")
output_file = os.path.abspath("cleaned_data.xlsx")

server_params = StdioServerParameters(
    command="python3",
    args=["-m", "excel_mcp", "stdio"],
    env={"UV_PYTHON": "3.12.3", **os.environ},
)

llm = LLM(
    model="bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    aws_region_name="us-east-1",
    temperature=0.2,
)

with MCPServerAdapter(server_params) as mcp_tools:
    print("Starting Excel processing...")
    print("\nStarting sequential sheet processing...\n")

    # Agent 1: Excel Reader - Finds data ranges
    excel_reader = Agent(
        role='Excel Range Finder',
        goal='Identify exact data ranges in Excel sheets',
        backstory="""You specialize in finding data boundaries in Excel sheets.
        You quickly scan sheets to identify the main tabular data ranges.""",
        tools=mcp_tools,
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    # Agent 2: Excel Writer - Copies data
    excel_writer = Agent(
        role='Excel Data Copier',
        goal='Copy identified data ranges to new Excel file',
        backstory="""You efficiently copy data between Excel files using exact ranges.
        You ensure all data is transferred accurately without modification.""",
        tools=mcp_tools,
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    # Task 1: Get all sheet names
    get_sheets_task = Task(
        description=f"Get all sheet names from '{input_file}'",
        expected_output="List of all sheet names in the workbook",
        agent=excel_reader,
        async_execution=False,
        output_file=output_file,
        context=[],
        tools=[tool for tool in mcp_tools if tool.name == 'get_workbook_metadata']
    )

    # Create crew for initial sheet listing
    sheet_crew = Crew(
        agents=[excel_reader],
        tasks=[get_sheets_task],
        verbose=True,
        process=Process.sequential
    )

    # Execute to get sheet names
    sheet_listing = sheet_crew.kickoff()
    sheet_names = eval(sheet_listing)['sheet_names']
    print(f"\nFound {len(sheet_names)} sheets: {', '.join(sheet_names)}\n")

    # Create output workbook
    mcp_tools.create_workbook(file_path=output_file)
    print(f"Created output workbook at {output_file}\n")

    # Process each sheet sequentially
    for sheet_name in sheet_names:
        print(f"Processing sheet: {sheet_name}")

        # Task 2: Find data range for current sheet
        find_range_task = Task(
            description=f"""Find data range for sheet '{sheet_name}' in '{input_file}':
            1. Read initial range (A1:AZ100)
            2. Identify exact data boundaries
            3. Return range in format: A1:Y58""",
            expected_output=f"Exact data range for sheet '{sheet_name}'",
            agent=excel_reader,
            async_execution=False,
            context=[]
        )

        # Task 3: Copy data for current sheet
        copy_data_task = Task(
            description=f"""Copy data from sheet '{sheet_name}' to output file:
            1. Create worksheet in output
            2. Read data using exact range
            3. Write data to output sheet""",
            expected_output=f"Data copied for sheet '{sheet_name}'",
            agent=excel_writer,
            async_execution=False,
            context=[find_range_task]
        )

        # Create crew for current sheet processing
        sheet_crew = Crew(
            agents=[excel_reader, excel_writer],
            tasks=[find_range_task, copy_data_task],
            verbose=True,
            process=Process.sequential
        )

        # Execute processing for current sheet
        sheet_crew.kickoff()
        print(f"Completed processing for sheet: {sheet_name}\n")

    print("All sheets processed successfully!")
    print(f"Clean data saved to: {output_file}")

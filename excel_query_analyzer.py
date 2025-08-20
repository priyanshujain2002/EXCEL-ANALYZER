from crewai import Agent, Task, Crew, LLM
from crewai.knowledge.source.excel_knowledge_source import ExcelKnowledgeSource
import os
import glob
import boto3

# Setup LLM
llm = LLM(
    model="bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    aws_region_name="us-east-1",
    temperature=0.3,
)

def ask_question(question: str, excel_files: list):
    """Answer user's question about media plans"""
    
    session = boto3.Session(region_name="us-east-1")
    embedder_config = {
        "provider": "bedrock",
        "config": {
            "model": "amazon.titan-embed-text-v1",  # or your preferred embedding model
            "session": session  # Uses default AWS session
        }
    }
    
    # Automatically create Excel knowledge sources for all files in knowledge directory
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
    
    # Create specialized agents
    query_interpreter = Agent(
        role='Media Plan Query Interpreter',
        goal='Understand user queries and identify relevant data sources and analysis requirements',
        backstory="""You are a query interpretation specialist who understands natural language
        questions about media plans and advertising packages. You excel at breaking down complex
        questions into specific data requirements and identifying which Excel files and columns
        are most relevant for answering user questions. You understand media planning terminology,
        package structures, and can distinguish between different types of analysis needed
        (comparisons, summaries, calculations, etc.).""",
        knowledge_sources=excel_sources,
        embedder=embedder_config,
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    package_analyzer = Agent(
        role='Package Comparison Specialist',
        goal='Analyze and compare package information across different Excel files to identify patterns and differences',
        backstory="""You are a package analysis expert specializing in comparing package names,
        specifications, and characteristics across multiple Excel files. You excel at identifying:
        - Original plan packages vs High impact packages
        - Package differences between files
        - Cross-file package relationships and dependencies
        - Package categorization and classification
        You have deep expertise in media planning package structures and can perform sophisticated
        comparisons to identify upselling opportunities and package variations.""",
        knowledge_sources=excel_sources,
        embedder=embedder_config,
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    data_analyst = Agent(
        role='Media Plan Data Analyst',
        goal='Perform detailed analysis of Excel data and extract insights based on specific requirements',
        backstory="""You are an expert data analyst specializing in media plan Excel data analysis.
        You can analyze data patterns, trends, and provide detailed insights about:
        - Package specifications and targeting details
        - Budget allocations and cost analysis
        - Media plan metrics and performance indicators
        - Display and video specifications
        - Availability and targeting summaries
        You excel at understanding business data, financial information, and operational metrics
        within the context of media planning and advertising campaigns.""",
        knowledge_sources=excel_sources,
        embedder=embedder_config,
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    report_generator = Agent(
        role='Media Plan Report Generator',
        goal='Generate comprehensive reports and summaries based on data analysis results',
        backstory="""You are a report generation specialist who creates clear, well-structured
        reports from media plan data analysis results. You excel at presenting complex data insights
        in an understandable format with proper context and actionable recommendations. You can:
        - Synthesize findings from multiple data sources
        - Create executive summaries and detailed reports
        - Provide strategic recommendations based on package analysis
        - Present comparative analysis results clearly
        - Highlight key insights and business implications
        Your reports help stakeholders make informed decisions about media planning strategies.""",
        knowledge_sources=excel_sources,
        embedder=embedder_config,
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    # Create specialized tasks
    interpret_task = Task(
        description=f"""
        Analyze the user query: "{question}"
        
        Your task is to:
        1. Understand what the user is asking for in the context of media plans
        2. Identify which Excel files and data sources are most relevant
        3. Determine what specific analysis should be performed (comparison, summary, calculation, etc.)
        4. Identify if the query requires package comparison across files
        5. Provide clear guidance on the analysis approach needed
        6. Specify which columns and data points should be examined
        
        MEDIA PLAN CONTEXT:
        - Original Plan Packages: Packages aligned with customer requirements and budget
        - High Impact Packages: Upselling packages that are additions to the budget
        - Files may contain different types of specifications (Display, Video, Targeting, etc.)
        
        Use the available knowledge sources to understand the data structure and content.
        """,
        expected_output="Clear interpretation of the user query with specific analysis recommendations and relevant data sources identified",
        agent=query_interpreter
    )
    
    analyze_task = Task(
        description=f"""
        Based on the query interpretation, perform detailed analysis to answer: "{question}"
        
        Your task is to:
        1. Examine the relevant datasets from the identified knowledge sources
        2. If package comparison is needed, compare package names across different files
        3. Identify original plan packages vs high impact packages using comparison methodology:
           - Find files with different package counts
           - Compare package names between files to identify differences
           - Classify packages based on their presence/absence in different files
        4. Perform appropriate calculations, aggregations, or comparisons
        5. Extract specific data points and metrics
        6. Identify key patterns, trends, and insights
        7. Note any limitations or assumptions in your analysis
        
        PACKAGE IDENTIFICATION STRATEGY:
        1. Always start by finding the "Package Name" column in each Excel file
        2. Compare package columns between files to identify high impact packages
        3. High impact packages will be present in combined files but absent in original-only files
        
        Use the knowledge sources to access the actual data for comprehensive analysis.
        """,
        expected_output="Detailed data analysis results with specific findings, package comparisons, and insights",
        agent=package_analyzer if "package" in question.lower() or "compare" in question.lower() else data_analyst
    )
    
    report_task = Task(
        description=f"""
        Create a comprehensive response to the user query: "{question}"
        
        Based on the query interpretation and data analysis, generate a clear,
        well-structured response that:
        1. Directly answers the user's question
        2. Provides supporting data and evidence from the knowledge sources
        3. Includes relevant context about media plan structure
        4. Clearly distinguishes between original plan and high impact packages when relevant
        5. Offers actionable insights and recommendations if applicable
        6. Mentions specific data sources and files used from the knowledge base
        7. Presents findings in a logical, easy-to-understand format
        8. Highlights key business implications
        
        REPORTING GUIDELINES:
        - Use clear headings and structure
        - Provide specific examples and data points
        - Explain the significance of findings
        - Include comparative analysis when relevant
        """,
        expected_output="A clear, comprehensive response to the user query with supporting analysis and actionable insights",
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
    print("🎯 Media Plan Agent - Ask questions about your Excel files!")
    
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
    print("Examples:")
    print("- What package names are available?")
    print("- Compare package names across different files to find high impact packages")
    print("- Summarize all package information and specifications")
    print("- What are the differences between original and high impact packages?")
    print("- Analyze targeting specifications across all packages")
    print("- What display and video specs are available for each package?")
    
    while True:
        question = input("\n> ").strip()
        
        if question.lower() in ['quit', 'exit']:
            print("👋 Goodbye!")
            break
            
        if question:
            print("\n🔍 Analyzing...")
            try:
                result = ask_question(question, excel_files)
                print(f"\n📊 Answer:\n{result}")
            except Exception as e:
                print(f"❌ Error: {str(e)}")

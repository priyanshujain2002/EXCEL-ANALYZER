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
    
    # Create separate Excel knowledge sources for each of the 7 files (without explicit engine)
    excel_source_1 = ExcelKnowledgeSource(
        file_paths=["cleaned_Copy of USA_ The Rainmaker_RFP Template 5.5_350K1_Media_Plan.xlsx"],
        embedder=embedder_config
    )
    
    excel_source_2 = ExcelKnowledgeSource(
        file_paths=["cleaned_Copy of USA_ The Rainmaker_RFP Template 5.5_350K_Media_Plan.xlsx"],
        embedder=embedder_config
    )
    
    excel_source_3 = ExcelKnowledgeSource(
        file_paths=["cleaned_Copy of USA_ The Rainmaker_RFP Template 5.5_Display_Specs.xlsx"],
        embedder=embedder_config
    )
    
    excel_source_4 = ExcelKnowledgeSource(
        file_paths=["cleaned_Copy of USA_ The Rainmaker_RFP Template 5.5_Drop_Downs.xlsx"],
        embedder=embedder_config
    )
    
    excel_source_5 = ExcelKnowledgeSource(
        file_paths=["cleaned_Copy of USA_ The Rainmaker_RFP Template 5.5_Max_Avails.xlsx"],
        embedder=embedder_config
    )
    
    excel_source_6 = ExcelKnowledgeSource(
        file_paths=["cleaned_Copy of USA_ The Rainmaker_RFP Template 5.5_Targeting_Summary.xlsx"],
        embedder=embedder_config
    )
    
    excel_source_7 = ExcelKnowledgeSource(
        file_paths=["cleaned_Copy of USA_ The Rainmaker_RFP Template 5.5_Video_Specs.xlsx"],
        embedder=embedder_config
    )
    
    # Create Agent with knowledge source
    agent = Agent(
        role='Media Plan Expert',
        goal='Answer questions about media plans by analyzing Excel files with package information',
        backstory="""You are a media planning expert who analyzes comprehensive media plans (advertising plans) consisting of different packages according to customer requirements and budget.
        
        UNDERSTANDING MEDIA PLAN STRUCTURE:
        - Media plans contain different types of packages based on customer requirements and budget
        - Original Plan Packages: Packages perfectly aligned with customer requirements and budget
        - High Impact Packages: Upselling packages that suit client requirements but are additions to their budget
        - Combined Files: Some files may contain both original and high impact packages together
        
        PACKAGE IDENTIFICATION STRATEGY:
        1. Always start by finding the "Package Name" column in each Excel file
        2. If there's a dedicated high impact file, identify high impact packages from there
        3. If no separate high impact file exists, compare package columns between files:
           - Look for 2 files: one with original plan, one with original + high impact combined
           - Compare package names between these files to identify which are high impact packages
           - High impact packages will be present in the combined file but absent in the original-only file
        
        ANALYSIS APPROACH:
        - Analyze all provided Excel files comprehensively
        - Cross-reference package information across all files for comprehensive answers
        - Clearly distinguish between original plan packages and high impact (upselling) packages
        
        You must ONLY use information from the provided Excel knowledge sources.
        If information is not available, clearly state this limitation.""",
        knowledge_sources=[excel_source_1, excel_source_2, excel_source_3, excel_source_4, excel_source_5, excel_source_6, excel_source_7],
        embedder=embedder_config,
        llm=llm,
        verbose=True,
        allow_delegation=False
    )
    
    # Create Task
    task = Task(
        description=f"""
        Answer this question about media plans: "{question}"
        
        COMPREHENSIVE ANALYSIS PROCESS:
        1. Identify all available Excel files and analyze their content
        2. Locate "Package Name" columns in all relevant files
        3. Distinguish between package types:
           - Original plan packages (aligned with customer budget/requirements)
           - High impact packages (upselling opportunities)
        4. If separate high impact file exists, use it to identify high impact packages
        5. If no separate high impact file, compare package columns between files to identify high impact packages
        6. Cross-reference package information across all files for complete analysis
        7. Provide comprehensive answer distinguishing between original and high impact packages when relevant
        
        You must ONLY use information from the provided Excel knowledge sources.
        """,
        agent=agent,
        expected_output="Answer based on package information from Excel files"
    )
    
    # Create and run Crew
    crew = Crew(
        agents=[agent],
        tasks=[task],
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
    print("- Summarize all package information")
    print("- What high-impact packages do we have?")
    
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

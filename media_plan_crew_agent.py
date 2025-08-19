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
    
    # Create separate Excel knowledge sources for each file (without explicit engine)
    excel_source_1 = ExcelKnowledgeSource(
        file_paths=["cleaned_NBCE_The Voice S28_Fall_25_RFP Template (Samsung Ads) 5.13_400K_Media_Plan.xlsx"],
        embedder=embedder_config
    )
    
    excel_source_2 = ExcelKnowledgeSource(
        file_paths=["cleaned_NBCE_The Voice S28_Fall_25_RFP Template (Samsung Ads) 5.13_High_Impact.xlsx"],
        embedder=embedder_config
    )
    
    # Create Agent with knowledge source
    agent = Agent(
        role='Media Plan Expert',
        goal='Answer questions about media plans by analyzing Excel files with package information',
        backstory="""You are a media planning expert who analyzes all provided Excel files containing advertising packages.
        
        IMPORTANT: Always start by finding the "Package Name" column in each Excel file first, 
        then use that as your reference to answer user queries about packages across all files.
        
        You must ONLY use information from the provided Excel knowledge sources.
        If information is not available, clearly state this limitation.""",
        knowledge_sources=[excel_source_1, excel_source_2],
        embedder=embedder_config,
        llm=llm,
        verbose=True,
        allow_delegation=False
    )
    
    # Create Task
    task = Task(
        description=f"""
        Answer this question about media plans: "{question}"
        
        Process:
        1. First, find the "Package Name" column in all available Excel files
        2. Then answer the user's query using the package information from all files
        
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

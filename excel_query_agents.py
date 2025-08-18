#!/usr/bin/env python3
"""
Excel Query Agents - Simple implementation for querying Excel data using AI agents
Uses the output from excel_processor.py as knowledge source for question answering
"""

import os
import glob
from crewai import Agent, Task, Crew, Process, LLM
from crewai.knowledge.source.excel_knowledge_source import ExcelKnowledgeSource

# Setup LLM
llm = LLM(
    model="bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    aws_region_name="us-east-1",
    temperature=0.3,
)

# Create agents
data_analyst = Agent(
    role='Excel Data Analyst',
    goal='Analyze Excel data and provide insights based on user queries',
    backstory="""You are an expert data analyst specializing in Excel data analysis. 
    You can analyze data patterns, trends, and provide detailed insights. You excel at 
    understanding business data, financial information, and operational metrics.""",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

query_interpreter = Agent(
    role='Query Interpreter',
    goal='Understand user queries and identify relevant data sources',
    backstory="""You are a query interpretation specialist who understands natural 
    language questions. You can identify which datasets and columns are most relevant 
    for answering user questions. You excel at breaking down complex questions into 
    specific data requirements.""",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

report_generator = Agent(
    role='Report Generator',
    goal='Generate comprehensive reports and summaries based on data analysis',
    backstory="""You are a report generation specialist who creates clear, well-structured 
    reports from data analysis results. You excel at presenting complex data insights in 
    an understandable format with proper context and actionable recommendations.""",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# Create knowledge sources from processed Excel files
processed_files_dir = "processed_files"
knowledge_sources = []

if os.path.exists(processed_files_dir):
    excel_files = []
    for root, dirs, files in os.walk(processed_files_dir):
        for file in files:
            if file.endswith('.xlsx') and file.startswith('cleaned_'):
                excel_files.append(os.path.join(root, file))
    
    print(f"📊 Creating knowledge sources for {len(excel_files)} Excel files")
    
    for file_path in excel_files:
        try:
            relative_path = os.path.relpath(file_path, processed_files_dir)
            excel_source = ExcelKnowledgeSource(file_paths=[relative_path])
            knowledge_sources.append(excel_source)
            print(f"✅ Created knowledge source for: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"❌ Error creating knowledge source for {file_path}: {str(e)}")

def query_data(user_query):
    """Process a user query and return insights"""
    print(f"\n🔍 Processing query: '{user_query}'")
    
    if not knowledge_sources:
        return "❌ No processed Excel files found. Please run excel_processor.py first."
    
    # Create tasks
    interpret_task = Task(
        description=f"""
        Analyze the user query: "{user_query}"
        
        Your task is to:
        1. Understand what the user is asking for
        2. Identify which datasets from the knowledge sources are most relevant
        3. Determine what specific analysis should be performed
        4. Provide clear guidance on how to answer the query
        
        Use the available knowledge sources to understand the data structure and content.
        """,
        expected_output="Clear interpretation of the user query with specific analysis recommendations",
        agent=query_interpreter
    )
    
    analyze_task = Task(
        description=f"""
        Based on the query interpretation, analyze the relevant data to answer: "{user_query}"
        
        Your task is to:
        1. Examine the relevant datasets from the knowledge sources
        2. Perform appropriate analysis (calculations, aggregations, comparisons)
        3. Identify key insights and patterns
        4. Extract specific answers to the user's question
        5. Note any limitations or assumptions in your analysis
        
        Use the knowledge sources to access the actual data for analysis.
        """,
        expected_output="Detailed data analysis results with specific findings and insights",
        agent=data_analyst
    )
    
    report_task = Task(
        description=f"""
        Create a comprehensive response to the user query: "{user_query}"
        
        Based on the query interpretation and data analysis, generate a clear,
        well-structured response that:
        1. Directly answers the user's question
        2. Provides supporting data and evidence from the knowledge sources
        3. Includes relevant context and insights
        4. Offers actionable recommendations if applicable
        5. Mentions specific data sources used from the knowledge base
        """,
        expected_output="A clear, comprehensive response to the user query with supporting analysis",
        agent=report_generator
    )
    
    # Create and execute crew
    crew = Crew(
        agents=[query_interpreter, data_analyst, report_generator],
        tasks=[interpret_task, analyze_task, report_task],
        verbose=True,
        process=Process.sequential,
        knowledge_sources=knowledge_sources
    )
    
    try:
        result = crew.kickoff()
        return str(result)
    except Exception as e:
        return f"❌ Error processing query: {str(e)}"

def list_available_data():
    """List all available datasets"""
    if not os.path.exists(processed_files_dir):
        return "❌ Processed files directory not found."
    
    excel_files = []
    for root, dirs, files in os.walk(processed_files_dir):
        for file in files:
            if file.endswith('.xlsx') and file.startswith('cleaned_'):
                excel_files.append(os.path.join(root, file))
    
    if not excel_files:
        return "❌ No processed Excel files found."
    
    result = [f"📊 Found {len(excel_files)} processed Excel files:\n"]
    for file_path in excel_files:
        relative_path = os.path.relpath(file_path, processed_files_dir)
        result.append(f"🔹 {relative_path}")
    
    return "\n".join(result)

# Main execution
if __name__ == "__main__":
    print("🤖 Excel Query Agents System")
    print("=" * 40)
    
    if not os.path.exists(processed_files_dir):
        print("❌ No processed Excel files found. Please run excel_processor.py first.")
        exit(1)
    
    print("\n📋 Available commands:")
    print("  • 'list' - Show all available datasets")
    print("  • 'query <your question>' - Ask a question about the data")
    print("  • 'quit' - Exit the system")
    
    while True:
        try:
            user_input = input("\n🔍 Enter your command or query: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            elif user_input.lower() == 'list':
                print(list_available_data())
            
            elif user_input.lower().startswith('query '):
                query = user_input[6:].strip()
                if query:
                    result = query_data(query)
                    print(f"\n📊 Query Result:\n{result}")
                else:
                    print("❌ Please provide a query after 'query '")
            
            else:
                # Treat as direct query
                result = query_data(user_input)
                print(f"\n📊 Query Result:\n{result}")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except EOFError:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")
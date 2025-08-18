#!/usr/bin/env python3
"""
Excel Query Agents - Creates intelligent agents that can answer queries based on processed Excel data
Uses the output from excel_processor.py as knowledge source for question answering
"""

import os
import sys
import pandas as pd
import glob
from pathlib import Path
from typing import Dict, List, Any, Optional
from crewai import Agent, Task, Crew, Process, LLM
from crewai.knowledge.source.excel_knowledge_source import ExcelKnowledgeSource
from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters
import json

class ExcelQuerySystem:
    """Main system for querying Excel data using AI agents"""
    
    def __init__(self, processed_files_dir: str = "processed_files"):
        self.processed_files_dir = processed_files_dir
        self.llm = self._setup_llm()
    
    def _setup_llm(self):
        """Setup the LLM for agents"""
        return LLM(
            model="bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            aws_region_name="us-east-1",
            temperature=0.3,
        )
    
    def _create_agents(self):
        """Create agents (knowledge will be provided via crew knowledge_sources)"""
        
        # Data Analyst Agent
        data_analyst = Agent(
            role='Excel Data Analyst',
            goal='Analyze Excel data and provide insights based on user queries',
            backstory="""You are an expert data analyst specializing in Excel data analysis. 
            You can analyze data patterns, trends, and provide detailed insights. You excel at 
            understanding business data, financial information, and operational metrics.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        # Query Interpreter Agent
        query_interpreter = Agent(
            role='Query Interpreter',
            goal='Understand user queries and identify relevant data sources',
            backstory="""You are a query interpretation specialist who understands natural 
            language questions. You can identify which datasets and columns are most relevant 
            for answering user questions. You excel at breaking down complex questions into 
            specific data requirements.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        # Report Generator Agent
        report_generator = Agent(
            role='Report Generator',
            goal='Generate comprehensive reports and summaries based on data analysis',
            backstory="""You are a report generation specialist who creates clear, well-structured 
            reports from data analysis results. You excel at presenting complex data insights in 
            an understandable format with proper context and actionable recommendations.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        return query_interpreter, data_analyst, report_generator
    
    def _create_knowledge_sources(self):
        """Create knowledge sources from all processed Excel files in directory"""
        if not os.path.exists(self.processed_files_dir):
            print(f"❌ Processed files directory not found: {self.processed_files_dir}")
            return []
        
        # Find all processed Excel files
        excel_files = []
        for root, dirs, files in os.walk(self.processed_files_dir):
            for file in files:
                if file.endswith('.xlsx') and file.startswith('cleaned_'):
                    excel_files.append(os.path.join(root, file))
        
        if not excel_files:
            print(f"❌ No processed Excel files found in: {self.processed_files_dir}")
            return []
        
        print(f"📊 Creating knowledge sources for {len(excel_files)} Excel files")
        
        # Create ExcelKnowledgeSource for each file
        knowledge_sources = []
        for file_path in excel_files:
            try:
                # Convert to relative path from knowledge directory
                # ExcelKnowledgeSource expects files to be in knowledge/ directory
                relative_path = os.path.relpath(file_path, self.processed_files_dir)
                excel_source = ExcelKnowledgeSource(
                    file_paths=[relative_path]
                )
                knowledge_sources.append(excel_source)
                print(f"✅ Created knowledge source for: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"❌ Error creating knowledge source for {file_path}: {str(e)}")
                continue
        
        return knowledge_sources
    
    def query_data(self, user_query: str, project_filter: str = None) -> str:
        """Process a user query and return insights"""
        
        print(f"\n🔍 Processing query: '{user_query}'")
        if project_filter:
            print(f"📁 Project filter: {project_filter}")
        
        # Create knowledge sources for all Excel files
        knowledge_sources = self._create_knowledge_sources()
        if not knowledge_sources:
            return f"❌ No processed Excel files found. Please run excel_processor.py first."
        
        # Create agents
        query_interpreter, data_analyst, report_generator = self._create_agents()
        
        # Task 1: Interpret the query
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
        
        # Task 2: Analyze the data
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
        
        # Task 3: Generate report
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
        
        # Create and execute crew with all knowledge sources
        crew = Crew(
            agents=[query_interpreter, data_analyst, report_generator],
            tasks=[interpret_task, analyze_task, report_task],
            verbose=True,
            process=Process.sequential,
            knowledge_sources=knowledge_sources  # Pass all knowledge sources to crew
        )
        
        try:
            result = crew.kickoff()
            return str(result)
        except Exception as e:
            return f"❌ Error processing query: {str(e)}"
    
    def list_available_data(self) -> str:
        """List all available datasets"""
        if not os.path.exists(self.processed_files_dir):
            return "❌ Processed files directory not found."
        
        excel_files = []
        for root, dirs, files in os.walk(self.processed_files_dir):
            for file in files:
                if file.endswith('.xlsx') and file.startswith('cleaned_'):
                    excel_files.append(os.path.join(root, file))
        
        if not excel_files:
            return "❌ No processed Excel files found."
        
        summary_parts = [f"📊 Found {len(excel_files)} processed Excel files:\n"]
        
        for file_path in excel_files:
            relative_path = os.path.relpath(file_path, self.processed_files_dir)
            summary_parts.append(f"🔹 {relative_path}")
        
        return "\n".join(summary_parts)
    
    def get_data_details(self, project_name: str = None, sheet_name: str = None) -> str:
        """Get detailed information about specific datasets"""
        if not os.path.exists(self.processed_files_dir):
            return "❌ Processed files directory not found."
        
        excel_files = []
        for root, dirs, files in os.walk(self.processed_files_dir):
            for file in files:
                if file.endswith('.xlsx') and file.startswith('cleaned_'):
                    file_path = os.path.join(root, file)
                    if project_name and project_name.lower() not in file_path.lower():
                        continue
                    if sheet_name and sheet_name.lower() not in file_path.lower():
                        continue
                    excel_files.append(file_path)
        
        if not excel_files:
            return "No matching datasets found."
        
        results = []
        for file_path in excel_files:
            try:
                df = pd.read_excel(file_path)
                relative_path = os.path.relpath(file_path, self.processed_files_dir)
                
                details = [
                    f"📊 {relative_path}",
                    f"   • Dimensions: {len(df)} rows × {len(df.columns)} columns",
                    f"   • Columns: {', '.join(df.columns.tolist())}",
                ]
                
                if not df.empty:
                    details.append("   • Sample data:")
                    for i, record in enumerate(df.head(2).to_dict('records')):
                        details.append(f"     Row {i+1}: {record}")
                
                results.append("\n".join(details))
                
            except Exception as e:
                results.append(f"❌ Error reading {relative_path}: {str(e)}")
        
        return "\n\n".join(results)

def main():
    """Main interactive function"""
    print("🤖 Excel Query Agents System")
    print("=" * 40)
    
    # Initialize the query system
    query_system = ExcelQuerySystem()
    
    # Check if data is available
    if not os.path.exists(query_system.processed_files_dir):
        print("❌ No processed Excel files found. Please run excel_processor.py first.")
        return
    
    print("\n📋 Available commands:")
    print("  • 'list' - Show all available datasets")
    print("  • 'details [project] [sheet]' - Show details for specific data")
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
                print(query_system.list_available_data())
            
            elif user_input.lower().startswith('details'):
                parts = user_input.split()
                project = parts[1] if len(parts) > 1 else None
                sheet = parts[2] if len(parts) > 2 else None
                print(query_system.get_data_details(project, sheet))
            
            elif user_input.lower().startswith('query '):
                query = user_input[6:].strip()  # Remove 'query ' prefix
                if query:
                    result = query_system.query_data(query)
                    print(f"\n📊 Query Result:\n{result}")
                else:
                    print("❌ Please provide a query after 'query '")
            
            else:
                # Treat as direct query
                result = query_system.query_data(user_input)
                print(f"\n📊 Query Result:\n{result}")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except EOFError:
            # Handle EOF (end of input) gracefully - happens with piped input
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
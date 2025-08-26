import json
import pandas as pd
import os
import glob
from crewai import Agent, Task, Crew, LLM
from crewai.knowledge.source.excel_knowledge_source import ExcelKnowledgeSource
import boto3

# Setup LLM
llm = LLM(
    model="bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    aws_region_name="us-east-1",
    temperature=0.3,
)

class KnowledgeTestAgent:
    def __init__(self):
        self.session = boto3.Session(region_name="us-east-1")
        self.embedder_config = {
            "provider": "bedrock",
            "config": {
                "model": "amazon.titan-embed-text-v1",
                "session": self.session
            }
        }
        self.excel_sources = []
        self._load_knowledge_base()
    
    def _load_knowledge_base(self):
        """Load all Excel files from knowledge directory as knowledge sources"""
        if not os.path.exists("knowledge/"):
            raise FileNotFoundError("'knowledge' directory not found!")
        
        excel_files = glob.glob("knowledge/*.xlsx") + glob.glob("knowledge/*.xls")
        if not excel_files:
            raise FileNotFoundError("No Excel files found in knowledge/ directory")
        
        print(f"📁 Loading knowledge base from {len(excel_files)} Excel files:")
        for excel_file in excel_files:
            filename = os.path.basename(excel_file)
            print(f"  • {filename}")
            
            excel_source = ExcelKnowledgeSource(
                file_paths=[filename],
                embedder=self.embedder_config
            )
            self.excel_sources.append(excel_source)
    
    def ask_question(self, question: str) -> str:
        """
        Ask a specific question to test the agent's understanding of original vs high impact plan relationships
        
        Args:
            question: The question to ask about the knowledge base
            
        Returns:
            Answer from the agent
        """
        
        # Create knowledge analyzer agent
        knowledge_analyzer = Agent(
            role='Media Plan Knowledge Expert',
            goal='Answer specific questions about media plan structures and relationships between original and high impact plans',
            backstory="""You are a media planning expert who has deep knowledge of Excel-based media plans.
            You understand the structure of different sheet types and can identify relationships between:
            - Original Plan sheets (like "200k Media Plan")
            - High Impact Plan sheets (containing "high impact")
            - Combined Plan sheets (like "200k+ Media Plan")
            
            You can analyze package names, placement names, and understand how original plans relate to their
            corresponding high impact alternatives. You provide accurate, specific answers based on the actual
            data in the Excel files.""",
            knowledge_sources=self.excel_sources,
            embedder=self.embedder_config,
            llm=llm,
            verbose=True,
            allow_delegation=False
        )
        
        # Create task to answer the question
        answer_task = Task(
            description=f"""
            Answer the following question based on your knowledge of the Excel media plans:
            
            Question: {question}
            
            Your task is to:
            1. Analyze the relevant Excel files in the knowledge base
            2. Identify the specific sheets and data that relate to the question
            3. Provide a clear, accurate answer based on the actual data
            4. Include specific examples from the Excel files when relevant
            5. Explain the relationships between original and high impact plans if applicable
            
            Focus on providing factual information from the knowledge base rather than general assumptions.
            """,
            expected_output="Clear, specific answer based on actual data from the Excel knowledge base",
            agent=knowledge_analyzer
        )
        
        # Create and run crew
        crew = Crew(
            agents=[knowledge_analyzer],
            tasks=[answer_task],
            verbose=True
        )
        
        result = crew.kickoff()
        return result

def main():
    """Interactive mode for testing knowledge base understanding"""
    print("🧪 Knowledge Base Test Agent")
    print("=" * 50)
    print("This agent helps test whether the system can properly link original plans with high impact plans")
    
    try:
        test_agent = KnowledgeTestAgent()
        print("✅ Knowledge base loaded successfully!")
        
        # Suggested test questions
        print("\n💡 Suggested test questions to verify original-to-high-impact plan linking:")
        print("1. List all the Excel files in the knowledge base and their sheet names")
        print("2. For each Excel file, identify which sheets are original plans and which are high impact plans")
        print("3. Show me the package names from an original plan and its corresponding high impact plan")
        print("4. What placement names are available in the high impact sheets?")
        print("5. Can you identify the relationship between '200k Media Plan' and 'High Impact' sheets in the same file?")
        print("6. Show me examples of how original packages differ from high impact packages")
        
        print("\n💬 Ask your test questions (type 'quit' to exit):")
        
        while True:
            question = input("\n❓ Your question: ").strip()
            
            if question.lower() in ['quit', 'exit']:
                print("👋 Goodbye!")
                break
            
            if question:
                print("\n🔍 Analyzing knowledge base...")
                try:
                    answer = test_agent.ask_question(question)
                    print(f"\n📋 Answer:\n")
                    print("=" * 60)
                    print(answer)
                    print("=" * 60)
                except Exception as e:
                    print(f"❌ Error: {str(e)}")
    
    except Exception as e:
        print(f"❌ Error initializing system: {str(e)}")

if __name__ == "__main__":
    main()
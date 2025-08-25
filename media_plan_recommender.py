import json
import pandas as pd
import os
import glob
from crewai import Agent, Task, Crew, LLM
from crewai.knowledge.source.excel_knowledge_source import ExcelKnowledgeSource
import boto3
from typing import Dict, List, Any

# Setup LLM
llm = LLM(
    model="bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    aws_region_name="us-east-1",
    temperature=0.3,
)

# Hardcoded JSON media plan data
SAMPLE_JSON_DATA = {
    "media_plan": {
        "plan": [
            {
            "placement_name": "First Screen Masthead - US",
            "audience_segment": "WBHE - The Flash - Superhero/Action PVOD/EST Viewers OR Superhero/DC/Marvel Fans (USID: 245802)",
            "start_datetime": "2025-06-15",
            "end_datetime": "2025-06-27",
            "rate": 24,
            "budget_allocated": 25350,
            "impressions": 1034701,
            "max_impression": 1034701,
            "additional_notes": "High impact primary placement targeting superhero/action fans for full campaign duration"
            },
            {
            "placement_name": "Universal Guide Masthead - US",
            "audience_segment": "WBHE - The Flash - Superhero/Action PVOD/EST Viewers OR Superhero/DC/Marvel Fans (USID: 245802)",
            "start_datetime": "2025-06-15",
            "end_datetime": "2025-06-27",
            "rate": 22,
            "budget_allocated": 58381,
            "impressions": 2653698,
            "max_impression": 2653698,
            "additional_notes": "Secondary high impact placement providing additional reach throughout campaign period"
            },
            {
            "placement_name": "Universal Guide Masthead - US (Weekend Heavy-Up)",
            "audience_segment": "WBHE - The Flash - Superhero/Action PVOD/EST Viewers OR Superhero/DC/Marvel Fans (USID: 245802)",
            "start_datetime": "2025-06-20",
            "end_datetime": "2025-06-21",
            "rate": 22,
            "budget_allocated": 4947,
            "impressions": 224855,
            "max_impression": 224855,
            "additional_notes": "Weekend heavy-up as specified in RFP for increased visibility during peak conversion period"
            },
            {
            "placement_name": "Apps Store Masthead - US",
            "audience_segment": "WBHE - The Flash - Superhero/Action PVOD/EST Viewers OR Superhero/DC/Marvel Fans (USID: 245802)",
            "start_datetime": "2025-06-15",
            "end_datetime": "2025-06-27",
            "rate": 24,
            "budget_allocated": 3973,
            "impressions": 162152,
            "max_impression": 162152,
            "additional_notes": "Complementary high-impact placement to extend reach across different Samsung TV touchpoints"
            }
        ]
    }
}

class MediaPlanRecommender:
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
    
    def _create_agents(self):
        """Create specialized agents for media plan analysis"""
        
        # Knowledge Analyzer Agent
        knowledge_analyzer = Agent(
            role='Media Plan Knowledge Analyzer',
            goal='Analyze Excel knowledge base to understand package-placement relationships and high impact alternatives',
            backstory="""You are a media planning expert who specializes in analyzing historical media plans
            to understand the relationships between packages and placements. You excel at:
            - Identifying package names (also called group names) and their associated placement names
            - Understanding the structure of original plans vs high impact plans
            - Recognizing patterns in successful high impact placements
            - Extracting placement characteristics and specifications from Excel data
            
            You focus specifically on the two key columns: Package Name/Group Name and Placement Name.
            You understand that high impact packages are upselling opportunities that provide additional value.""",
            knowledge_sources=self.excel_sources,
            embedder=self.embedder_config,
            llm=llm,
            verbose=True,
            allow_delegation=False
        )
        
        # JSON Analyzer Agent
        json_analyzer = Agent(
            role='JSON Media Plan Analyzer',
            goal='Analyze input JSON media plan to extract package names and placement requirements',
            backstory="""You are a JSON data analyst specializing in media plan structure analysis.
            You excel at parsing JSON media plans to identify:
            - Package names and their characteristics
            - Current placement names and specifications
            - Media plan structure and requirements
            - Budget and targeting information
            
            You focus on extracting the key information needed for high impact placement recommendations,
            specifically package names and placement names from the JSON structure.""",
            llm=llm,
            verbose=True,
            allow_delegation=False
        )
        
        # Media Plan Matcher Agent
        media_plan_matcher = Agent(
            role='Media Plan Matching Specialist',
            goal='Find the most similar media plan in knowledge base that matches the user input JSON media plan',
            backstory="""You are a media plan matching expert who specializes in comparing user media plans
            with historical media plans in the knowledge base. You excel at:
            - Analyzing package names and placement names from user JSON input
            - Comparing them against all available media plans in Excel knowledge base
            - Identifying the most similar or matching media plan based on package/placement similarity
            - Understanding media plan structures and finding the best match
            - Recognizing similar targeting, budget ranges, and campaign objectives
            
            Your primary task is to find which media plan in the knowledge base most closely matches
            the user's input media plan based on package names and placement names.""",
            knowledge_sources=self.excel_sources,
            embedder=self.embedder_config,
            llm=llm,
            verbose=True,
            allow_delegation=False
        )
        
        # High Impact Recommender Agent
        high_impact_recommender = Agent(
            role='High Impact Plan Recommender',
            goal='Extract and recommend high impact packages from the matched media plan with detailed reasoning',
            backstory="""You are a high impact package specialist who focuses on identifying and recommending
            high impact alternatives from matched media plans. You excel at:
            - Analyzing the matched media plan to identify high impact sheets/packages
            - Extracting high impact placement names from the matched plan
            - Understanding the relationship between original and high impact packages
            - Providing detailed reasoning for why specific high impact placements are recommended
            - Explaining the additional value and benefits of high impact alternatives
            
            You work with the matched media plan to find its corresponding high impact packages
            and recommend the best high impact placements with proper reasoning.""",
            knowledge_sources=self.excel_sources,
            embedder=self.embedder_config,
            llm=llm,
            verbose=True,
            allow_delegation=False
        )
        
        return knowledge_analyzer, json_analyzer, media_plan_matcher, high_impact_recommender
    
    def recommend_high_impact_placements(self) -> str:
        """
        Analyze hardcoded JSON media plan and recommend high impact placements
        
        Returns:
            Recommendations with placement names and reasoning
        """
        
        # Use hardcoded JSON data
        json_data = SAMPLE_JSON_DATA
        print("📄 Using hardcoded JSON media plan data:")
        print(json.dumps(json_data, indent=2))
        
        # Create agents
        knowledge_analyzer, json_analyzer, media_plan_matcher, high_impact_recommender = self._create_agents()
        
        # Create tasks
        knowledge_analysis_task = Task(
            description=f"""
            Analyze the Excel knowledge base to understand the structure and relationships of media plans.
            
            Your task is to:
            1. Examine all Excel files in the knowledge base
            2. Identify different sheet types (Original Plans, High Impact Plans, Combined Plans)
            3. Extract package names (also called group names) and their associated placement names
            4. Understand the relationships between original packages and high impact alternatives
            5. Catalog high impact placements and their characteristics
            6. Identify patterns in successful high impact package implementations
            
            Focus specifically on:
            - Package Name / Group Name columns
            - Placement Name columns
            - High impact alternatives and their specifications
            
            Sheet naming patterns to recognize:
            - High Impact sheets: contain "high impact"
            - Original Plan sheets: like "200k Media Plan" (budget + "Media Plan")
            - Combined Plan sheets: like "200k+ Media Plan" (budget + "+" + "Media Plan")
            """,
            expected_output="Comprehensive analysis of knowledge base with package-placement relationships and high impact alternatives cataloged",
            agent=knowledge_analyzer
        )
        
        json_analysis_task = Task(
            description=f"""
            Analyze the provided JSON media plan to extract package and placement information.
            
            JSON Data: {json.dumps(json_data, indent=2)}
            
            Your task is to:
            1. Parse the JSON structure to identify media plan components
            2. Extract all package names (may also be called group names)
            3. Extract current placement names associated with each package
            4. Identify package characteristics, targeting, and specifications
            5. Understand the current media plan structure and requirements
            6. Prepare package information for high impact placement matching
            
            Focus on extracting:
            - Package names and their current placements
            - Package specifications and requirements
            - Any targeting or budget information
            - Current placement characteristics
            """,
            expected_output="Detailed extraction of packages and placements from JSON with their characteristics",
            agent=json_analyzer
        )
        
        matching_task = Task(
            description=f"""
            Find the most similar media plan in the knowledge base that matches the user's JSON media plan.
            
            JSON Data: {json.dumps(json_data, indent=2)}
            
            Your task is to:
            1. Analyze the package names and placement names from the JSON input
            2. Compare them against all media plans available in the Excel knowledge base
            3. Find the media plan that has the most similar packages and placements
            4. Identify which Excel file and sheets contain the best matching media plan
            5. Determine the similarity score and matching criteria
            
            Focus on matching:
            - Package names (exact or similar matches)
            - Placement names (exact or similar matches)
            - Campaign types and objectives
            - Budget ranges and targeting demographics
            
            Output should identify:
            - The specific Excel file that contains the best matching media plan
            - Which sheets in that file are most relevant (original plan sheets)
            - Similarity analysis and matching reasoning
            """,
            expected_output="Identification of the best matching media plan from knowledge base with similarity analysis",
            agent=media_plan_matcher
        )
        
        recommendation_task = Task(
            description=f"""
            Based on the matched media plan, provide exactly TOP 3 high impact placement recommendations.
            
            Your task is to:
            1. Use the matched media plan identified in the previous task
            2. Find the high impact sheets/packages within that same Excel file
            3. Extract the TOP 3 best high impact placement names from the matched media plan
            4. Provide ONLY the top 3 recommendations with single-line reasoning
            
            STRICT OUTPUT FORMAT - provide exactly this format with no additional text:
            
            TOP 3 HIGH IMPACT RECOMMENDATIONS:
            
            1. [High Impact Placement Name] - [Single line reasoning]
            2. [High Impact Placement Name] - [Single line reasoning]
            3. [High Impact Placement Name] - [Single line reasoning]
            
            Requirements:
            - ONLY provide the top 3 recommendations
            - Each reasoning must be exactly ONE line
            - No additional explanations, headers, or text
            - Focus on the most relevant high impact placements from the matched media plan
            """,
            expected_output="Exactly 3 high impact placement recommendations with single-line reasoning each",
            agent=high_impact_recommender
        )
        
        # Create and run crew
        crew = Crew(
            agents=[knowledge_analyzer, json_analyzer, media_plan_matcher, high_impact_recommender],
            tasks=[knowledge_analysis_task, json_analysis_task, matching_task, recommendation_task],
            verbose=True
        )
        
        result = crew.kickoff()
        return result

def main():
    """Generate recommendations for hardcoded JSON media plan"""
    print("🎯 Media Plan High Impact Placement Recommender")
    print("=" * 50)
    
    try:
        recommender = MediaPlanRecommender()
        print("✅ Knowledge base loaded successfully!")
        
        print("\n🔍 Analyzing hardcoded media plan and generating recommendations...")
        try:
            recommendations = recommender.recommend_high_impact_placements()
            print(f"\n📊 High Impact Placement Recommendations:\n")
            print("=" * 60)
            print(recommendations)
            print("=" * 60)
        except Exception as e:
            print(f"❌ Error generating recommendations: {str(e)}")
    
    except Exception as e:
        print(f"❌ Error initializing system: {str(e)}")

if __name__ == "__main__":
    main()
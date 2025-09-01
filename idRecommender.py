from crewai import Agent, Task, Crew, LLM
from crewai.knowledge.source.excel_knowledge_source import ExcelKnowledgeSource
import json
import pandas as pd
import os
import glob
import boto3

llm = LLM(
    model="bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    aws_region_name="us-east-1",
    temperature=0.3,
)

placement_names = ['First Screen Masthead - US', 'Universal Guide Masthead - US', 'Universal Guide Masthead - US (Weekend Heavy-Up)',
                    'Apps Store Masthead - US']

session = boto3.Session(region_name="us-east-1")
embedder_config = {
        "provider": "bedrock",
        "config": {
            "model": "amazon.titan-embed-text-v1",
            "session": session
        }
    }

excel_source = ExcelKnowledgeSource(
        file_paths=["original_packages.xlsx"],
        embedder=embedder_config
    )

knowledge_extractor = Agent(
    role = 'Excel File Reader',
    goal = 'Read the complete excel file successfully and extract all package information',
    backstory = 'You are an expert excel reader, who has mastered the art of reading and extracting knowledge from the excel files.',
    llm = llm,
    knowledge_sources = [excel_source],
    embedder=embedder_config,
    verbose = False,
    allow_delegation = False
)

placement_reader = Agent(
    role = 'Input list reader',
    goal = 'Read all the placement names present in the list and understand their characteristics',
    backstory = 'You are an expert in reading and analyzing placement name data from lists',
    llm = llm,
    verbose = False,
    embedder=embedder_config,
    allow_delegation = False
)

name_matcher = Agent(
    role = 'Name matcher and ID recommender',
    goal = 'Match placement names with package names from the knowledge source and recommend the top 3 most relevant package IDs',
    backstory = 'You are an expert in matching placement names with package names and providing accurate ID recommendations based on similarity and relevance.',
    knowledge_sources = [excel_source],
    llm = llm,
    verbose = False,
    embedder=embedder_config,
    allow_delegation = False
)

# Define tasks for each agent
extract_knowledge_task = Task(
    description="""
    Read and extract all the package information from the Excel file.
    Understand the structure of the data including:
    - Package IDs
    - Package names and their variations
    - Any patterns in the naming conventions
    
    Provide a summary of what packages are available in the knowledge base.
    """,
    agent=knowledge_extractor,
    expected_output="Summary of all packages available in the Excel knowledge base"
)

read_placements_task = Task(
    description=f"""
    Analyze the following placement names list completely:
    {placement_names}
    
    Understand the characteristics and patterns of these placement names.
    Extract key information like:
    - Platform/location indicators (e.g., 'US', 'Apps Store', 'Universal Guide')
    - Placement types (e.g., 'Masthead', 'Screen')
    - Special attributes (e.g., 'Weekend Heavy-Up', 'First Screen')
    """,
    agent=placement_reader,
    expected_output="Analysis of the placement names and their characteristics",
    context=[extract_knowledge_task]
)

match_and_recommend_task = Task(
    description=f"""
    Using the knowledge from the Excel file and the placement analysis, match each of the following placement names:
    {placement_names}
    
    Find packages that contain terms like:
    - 'First Screen' (matches with 'First Screen Masthead - US')
    - 'Universal' or 'Guide' (matches with 'Universal Guide Masthead - US')
    - 'Apps' or 'Store' (matches with 'Apps Store Masthead - US')
    - 'CTV' or connected TV related packages
    
    Return ONLY the top 3 package IDs that best match the placement names.
    Format your final answer as: [ID1, ID2, ID3]
    """,
    agent=name_matcher,
    expected_output="Top 3 package IDs in format [ID1, ID2, ID3]",
    context=[extract_knowledge_task, read_placements_task]
)

# Create and run the crew
crew = Crew(
    agents=[knowledge_extractor, placement_reader, name_matcher],
    tasks=[extract_knowledge_task, read_placements_task, match_and_recommend_task],
    verbose=False
)

# Execute the crew
if __name__ == "__main__":
    result = crew.kickoff()
    print(result)

"""
This script contains two crews, one for recommending id which takes original_packages.xlsx as its knowledge source
and the other for recommending top 3 high impact packages which takes high_impact_packages.xlsx as its knowledge source.
"""

from crewai import Agent, Task, Crew, LLM
from crewai.knowledge.source.excel_knowledge_source import ExcelKnowledgeSource
import json
import pandas as pd
import os
import glob
import boto3

llm_crew1 = LLM(
    model="bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    aws_region_name="us-east-1",
    temperature=0.0,  # Lower temperature for consistency in crew1
)

llm_crew2 = LLM(
    model="bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    aws_region_name="us-east-1",
    temperature=0.3,  # Higher temperature for creativity in crew2's reasoning
)

# placement_names will be provided as a parameter to the recommend_packages function

session = boto3.Session(region_name="us-east-1")
embedder_config = {
        "provider": "bedrock",
        "config": {
            "model": "amazon.titan-embed-text-v1",
            "session": session
        }
    }

# --- New code to list files in the subfolder ---
knowledge_subfolder_name = "Original Packages" # Name of the folder inside 'knowledge/'
base_knowledge_dir = "High_Impact/knowledge" # Updated to point to the correct knowledge directory

# Construct the full path to the subfolder to list its contents
# This path is used for os.listdir, so it needs to be relative to the script location
full_path_to_subfolder = os.path.join(base_knowledge_dir, knowledge_subfolder_name)

excel_file_paths = []
if os.path.exists(full_path_to_subfolder) and os.path.isdir(full_path_to_subfolder):
    for filename in os.listdir(full_path_to_subfolder):
        # We assume all files in this specific folder are relevant Excel files
        # ExcelKnowledgeSource expects paths relative to knowledge directory
        # Since it prepends 'knowledge/', we need to provide path from knowledge directory
        relative_path = os.path.join(knowledge_subfolder_name, filename)
        excel_file_paths.append(relative_path)
else:
    # Handle error: folder not found or is not a directory
    print(f"Error: Knowledge subfolder not found at {full_path_to_subfolder}")
    # Depending on desired behavior, you might want to exit or raise an exception here.

if not excel_file_paths:
    # This will likely cause ExcelKnowledgeSource to raise an error, which is appropriate.
    print(f"Warning: No Excel files found in {full_path_to_subfolder}. The knowledge source will be empty.")

excel_source = ExcelKnowledgeSource(
        file_paths=excel_file_paths,
        embedder=embedder_config
    )

# --- Setup for High Impact Packages Knowledge Source ---
high_impact_subfolder_name = "High Impact Packages" # Name of the folder inside 'knowledge/'
full_path_to_high_impact_subfolder = os.path.join(base_knowledge_dir, high_impact_subfolder_name)

high_impact_excel_file_paths = []
if os.path.exists(full_path_to_high_impact_subfolder) and os.path.isdir(full_path_to_high_impact_subfolder):
    for filename in os.listdir(full_path_to_high_impact_subfolder):
        # ExcelKnowledgeSource expects paths relative to knowledge directory
        # Since it prepends 'knowledge/', we need to provide path from knowledge directory
        relative_path = os.path.join(high_impact_subfolder_name, filename)
        high_impact_excel_file_paths.append(relative_path)
else:
    print(f"Error: High Impact knowledge subfolder not found at {full_path_to_high_impact_subfolder}")

if not high_impact_excel_file_paths:
    print(f"Warning: No Excel files found in {full_path_to_high_impact_subfolder}. The high impact knowledge source will be empty.")

high_impact_excel_source = ExcelKnowledgeSource(
        file_paths=high_impact_excel_file_paths,
        embedder=embedder_config
    )

knowledge_extractor = Agent(
    role = 'Excel File Reader',
    goal = 'Read the complete excel file successfully and extract all package information',
    backstory = 'You are an expert excel reader, who has mastered the art of reading and extracting knowledge from the excel files.',
    llm = llm_crew1,
    knowledge_sources = [excel_source],
    embedder=embedder_config,
    verbose = False,
    allow_delegation = False
)

placement_reader = Agent(
    role = 'Input list reader',
    goal = 'Read all the placement names present in the list and understand their characteristics',
    backstory = 'You are an expert in reading and analyzing placement name data from lists',
    llm = llm_crew1,
    verbose = False,
    embedder=embedder_config,
    allow_delegation = False
)

name_matcher = Agent(
    role = 'Name matcher and ID recommender',
    goal = 'Match placement names with package names from the knowledge source and recommend the top 3 most relevant package IDs',
    backstory = 'You are an expert in matching placement names with package names and providing accurate ID recommendations based on similarity and relevance.',
    knowledge_sources = [excel_source],
    llm = llm_crew1,
    verbose = False,
    embedder=embedder_config,
    allow_delegation = False
)

# --- Agent and Task for High Impact Analysis ---
high_impact_analyzer = Agent(
    role='High Impact Package Specialist',
    goal="To identify and rank the top 3 high-impact packages based on a given list of package IDs, using the 'High Impact Packages' data.",
    backstory="You are an expert analyst specializing in evaluating package impact. You can cross-reference provided IDs with a detailed database of high-impact packages, considering their frequency and associated reasoning to determine the most significant ones.",
    llm=llm_crew2,
    knowledge_sources=[high_impact_excel_source],
    embedder=embedder_config,
    verbose=False,
    allow_delegation=False
)

analyze_high_impact_task = Task(
    description="", # This will be dynamically set before execution
    agent=high_impact_analyzer,
    expected_output="A list of the top 3 high-impact package names with their reasoning, formatted as:\n1. [Package Name 1]: [Reasoning 1]\n2. [Package Name 2]: [Reasoning 2]\n3. [Package Name 3]: [Reasoning 3]"
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
    description="",
    agent=placement_reader,
    expected_output="Analysis of the placement names and their characteristics",
    context=[extract_knowledge_task]
)

match_and_recommend_task = Task(
    description="",
    agent=name_matcher,
    expected_output="Top 3 package IDs in format [ID1, ID2, ID3]",
    context=[extract_knowledge_task, read_placements_task]
)

# Create and run the first crew
crew1 = Crew(
    agents=[knowledge_extractor, placement_reader, name_matcher],
    tasks=[extract_knowledge_task, read_placements_task, match_and_recommend_task],
    verbose=False
)

# Create the second crew for high impact analysis
crew2 = Crew(
    agents=[high_impact_analyzer],
    tasks=[analyze_high_impact_task],
    verbose=False # Set to True for more detailed output from crew2
)

# Execute the crews sequentially
def recommend_packages(placement_names):
    """
    Recommend packages based on placement names.
    
    Args:
        placement_names (list): List of placement names to analyze
        
    Returns:
        tuple: (id_list_result, high_impact_packages_result)
    """
    # Update the read_placements_task description with the provided placement names
    read_placements_task.description = f"""
    Analyze the following placement names list completely:
    {placement_names}
    
    Understand the characteristics and patterns of these placement names.
    Extract key information like:
    - Platform/location indicators (e.g., 'US', 'Apps Store', 'Universal Guide')
    - Placement types (e.g., 'Masthead', 'Screen')
    - Special attributes (e.g., 'Weekend Heavy-Up', 'First Screen')
    """
    
    # Update the match_and_recommend_task description with the provided placement names
    match_and_recommend_task.description = f"""
    Using the knowledge from the Excel file and the placement analysis, match each of the following placement names:
    {placement_names}
    
    Find packages that contain terms like:
    - 'First Screen' (matches with 'First Screen Masthead - US')
    - 'Universal' or 'Guide' (matches with 'Universal Guide Masthead - US')
    - 'Apps' or 'Store' (matches with 'Apps Store Masthead - US')
    - 'CTV' or connected TV related packages
    
    Return ONLY the top 3 package IDs that best match the placement names.
    Format your final answer as: [ID1, ID2, ID3]
    """
    
    print("Starting Crew 1: ID Recommendation...")
    id_list_result = crew1.kickoff()
    print(f"Crew 1 finished. Recommended IDs: {id_list_result}")

    # Dynamically set the description for the second crew's task
    analyze_high_impact_task.description = f"""
    The previous crew recommended the following package IDs: {id_list_result}.
    Your task is to:
    1.  Parse these IDs.
    2.  Consult your knowledge source (the 'High Impact Packages' Excel file(s)) to find all rows associated with these IDs. The file has columns: 'ID', 'High Impact Package Names', 'Reasoning'.
    3.  Analyze the 'High Impact Package Names' and 'Reasoning' for these rows.
    4.  Recommend the top 3 high-impact packages. A key criterion should be the frequency of each high-impact package name appearing across the provided IDs. Also, consider the strength and clarity of the 'Reasoning'.
    5.  Output ONLY the top 3 high-impact package names with their reasoning. Do NOT include any introductory text, explanations, or references to the provided IDs. Start directly with the numbered list in this exact format:
        1. [Package Name 1]: [Reasoning 1]
        2. [Package Name 2]: [Reasoning 2]
        3. [Package Name 3]: [Reasoning 3]
    """

    print("\nStarting Crew 2: High Impact Package Analysis...")
    high_impact_packages_result = crew2.kickoff()
    print(f"Crew 2 finished. Top High Impact Packages:\n{high_impact_packages_result}")
    
    return high_impact_packages_result

if __name__ == "__main__":
    # Example usage for testing
    placement_names = []
    recommend_packages(placement_names)

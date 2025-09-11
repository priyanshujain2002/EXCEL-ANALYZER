"""
Genre Recommendation Agent

This script contains a CrewAI agent that reads genre data from an Excel knowledge source
and recommends related genres based on semantic similarity found within the dataset.
"""

from crewai import Agent, Task, Crew, LLM
from crewai.knowledge.source.excel_knowledge_source import ExcelKnowledgeSource
import pandas as pd
import os
import boto3

# Initialize Bedrock session and LLM configuration
session = boto3.Session(region_name="us-east-1")

llm = LLM(
    model="bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    aws_region_name="us-east-1",
    temperature=0.1,  # Lower temperature for consistent genre analysis
)

embedder_config = {
    "provider": "bedrock",
    "config": {
        "model": "amazon.titan-embed-text-v1",
        "session": session
    }
}

# Setup Excel Knowledge Source for genre data
genre_excel_path = "Testing Sheet.xlsx"

# Verify the file exists (ExcelKnowledgeSource looks in knowledge directory by default)
full_path = os.path.join("knowledge", genre_excel_path)
if not os.path.exists(full_path):
    raise FileNotFoundError(f"Genre knowledge source not found at: {full_path}")

excel_source = ExcelKnowledgeSource(
    file_paths=[genre_excel_path],
    embedder=embedder_config
)

# Create the genre recommendation agent
genre_analyst = Agent(
    role='Advanced Genre Relationship Specialist',
    goal='Identify and recommend the most relevant related genres based on multi-dimensional analysis including audience overlap, thematic similarity, emotional resonance, and cultural context',
    backstory='''You are a world-renowned genre relationship specialist with over 20 years of experience in entertainment
    industry analysis, audience psychology, and cultural anthropology. Your expertise spans across multiple domains:
    
    CORE EXPERTISE:
    - Audience Demographics & Psychographics: You understand how different genres appeal to specific age groups,
      personality types, cultural backgrounds, and lifestyle preferences
    - Thematic Analysis: You identify deep thematic connections between genres beyond surface-level similarities
    - Emotional Mapping: You analyze the emotional journey and psychological satisfaction each genre provides
    - Cross-Cultural Genre Evolution: You understand how genres influence and blend with each other over time
    - Market Dynamics: You know which genres have overlapping fan bases and consumption patterns
    
    RECOMMENDATION PHILOSOPHY:
    - You NEVER recommend the exact input genre - only genuinely related but distinct genres
    - You prioritize quality over quantity - better to recommend 3 perfect matches than 10 mediocre ones
    - You consider multiple relationship vectors: audience overlap, thematic resonance, emotional satisfaction,
      cultural context, and consumption patterns
    - You provide evidence-based reasoning for each recommendation, explaining the specific connection mechanisms
    - You rank recommendations by strength of relationship, considering both breadth and depth of connections
    
    ANALYTICAL FRAMEWORK:
    1. Audience Overlap Analysis: Identify genres with similar target demographics and psychographics
    2. Thematic Resonance: Find genres sharing core themes, values, or narrative structures
    3. Emotional Satisfaction: Match genres providing similar emotional experiences or catharsis
    4. Cultural Context: Consider genres from similar cultural movements or historical periods
    5. Consumption Patterns: Identify genres frequently enjoyed by the same audiences
    6. Gateway Relationships: Find genres that serve as natural progression paths for fans
    
    You are meticulous, insightful, and always provide actionable recommendations with clear, compelling reasoning.''',
    llm=llm,
    knowledge_sources=[excel_source],
    embedder=embedder_config,
    verbose=True,
    allow_delegation=False
)

# Create the genre analysis task
analyze_genres_task = Task(
    description="""
    Conduct a comprehensive multi-dimensional analysis of the genre dataset from the Excel knowledge source:
    
    PRIMARY ANALYSIS OBJECTIVES:
    1. GENRE INVENTORY: Catalog all unique genres present in the dataset with frequency analysis
    2. AUDIENCE SEGMENTATION: Identify potential audience clusters and demographic patterns
    3. THEMATIC MAPPING: Group genres by underlying themes, moods, and narrative structures
    4. CULTURAL CATEGORIZATION: Classify genres by cultural origin, time period, and social context
    5. EMOTIONAL PROFILING: Map the emotional experiences and psychological satisfaction each genre provides
    6. RELATIONSHIP NETWORKS: Identify natural genre progression paths and cross-pollination patterns
    
    ANALYTICAL DEPTH REQUIREMENTS:
    - Examine not just what genres exist, but WHY they might appeal to similar audiences
    - Consider the psychological and sociological factors that drive genre preferences
    - Identify potential bridge genres that connect different audience segments
    - Analyze genre evolution patterns and influence relationships
    - Map emotional and thematic DNA of each genre category
    
    Create a sophisticated genre relationship matrix that will enable precise, multi-criteria recommendations.
    """,
    agent=genre_analyst,
    expected_output="Comprehensive genre landscape analysis with relationship matrix, audience insights, and thematic categorization"
)

# Create the genre recommendation task
recommend_genres_task = Task(
    description="", # This will be dynamically set
    agent=genre_analyst,
    expected_output="List of related genres with reasoning",
    context=[analyze_genres_task]
)

# Create the crew
genre_crew = Crew(
    agents=[genre_analyst],
    tasks=[analyze_genres_task, recommend_genres_task],
    verbose=True
)

def find_related_genres(input_genre):
    """
    Find genres that are related to the input genre using advanced multi-criteria analysis.
    
    Args:
        input_genre (str): The genre to find related genres for
        
    Returns:
        str: Formatted results with IDs, related genres, and comprehensive reasoning
    """
    # Update the recommend_genres_task description with the input genre
    recommend_genres_task.description = f"""
    Using your comprehensive genre analysis, identify genres most closely related to '{input_genre}'
    through advanced multi-dimensional evaluation. Apply your full analytical framework:
    
    MANDATORY REQUIREMENTS:
    1. EXCLUSION RULE: NEVER include '{input_genre}' itself in recommendations - only distinct related genres
    2. SOURCE VALIDATION: STRICTLY recommend ONLY genres that exist in the Excel knowledge source - do not create or suggest any genres not in the dataset
    3. KNOWLEDGE BASE CONSTRAINT: Every recommended genre MUST be found in the provided Excel file - verify each recommendation against the knowledge source
    4. QUALITY THRESHOLD: Be extremely selective - recommend only genres with strong, multi-faceted relationships
    5. RANKING SYSTEM: Order by relationship strength (strongest connections first)
    6. OPTIMAL COUNT: Limit to 5-8 top recommendations (prioritize quality over quantity)
    
    MULTI-CRITERIA EVALUATION FRAMEWORK:
    
    A. AUDIENCE OVERLAP ANALYSIS (Weight: 25%)
    - Identify genres with similar target demographics (age, gender, education, income)
    - Consider psychographic similarities (personality traits, values, lifestyle preferences)
    - Analyze consumption behavior patterns and cross-genre preferences
    
    B. THEMATIC RESONANCE ANALYSIS (Weight: 25%)
    - Examine shared core themes, values, and philosophical underpinnings
    - Identify similar narrative structures and storytelling approaches
    - Consider archetypal characters and plot patterns
    
    C. EMOTIONAL SATISFACTION MAPPING (Weight: 20%)
    - Match genres providing similar emotional experiences and catharsis
    - Consider mood, atmosphere, and psychological impact
    - Analyze stress relief, escapism, or intellectual stimulation factors
    
    D. CULTURAL CONTEXT EVALUATION (Weight: 15%)
    - Identify genres from similar cultural movements or historical periods
    - Consider geographical origins and cultural values
    - Analyze social commentary and cultural significance
    
    E. CONSUMPTION PATTERN ANALYSIS (Weight: 10%)
    - Identify genres frequently consumed together by same audiences
    - Consider seasonal preferences and consumption contexts
    - Analyze binge-watching or marathon consumption patterns
    
    F. GATEWAY RELATIONSHIP ASSESSMENT (Weight: 5%)
    - Find genres serving as natural progression paths for '{input_genre}' fans
    - Identify stepping-stone genres for audience development
    - Consider complexity and accessibility factors
    
    OUTPUT FORMAT REQUIREMENTS:
    For each recommendation, provide ONLY these three elements in this exact format:
    
    ID: [Excel_Row_ID]
    Genre: [Exact_Genre_Name_From_Knowledge_Base]
    Reasoning: [Concise explanation of why this genre is related to '{input_genre}', focusing on the strongest connection - audience overlap, thematic similarity, emotional resonance, or cultural context]
    
    ---
    
    ANALYTICAL RIGOR REQUIREMENTS:
    - STRICTLY use only genres from the Excel knowledge source - no external genres allowed
    - Provide evidence-based reasoning for each connection
    - Explain WHY the relationship exists, not just THAT it exists
    - Consider both obvious and subtle connections
    - Balance breadth of appeal with depth of connection
    - Prioritize actionable insights for genre discovery
    - Double-check that every recommended genre exists in the provided dataset
    """
    
    print(f"\n🎯 Initiating advanced multi-criteria analysis for genre: '{input_genre}'")
    
    
    # Run the crew
    result = genre_crew.kickoff()
    
    return result

def load_and_preview_data():
    """
    Load and preview the genre data to verify it's working correctly.
    """
    try:
        df = pd.read_excel(full_path)
        print("📊 Genre Knowledge Base Preview:")
        
        # Additional analysis for genre column
        if 'Genre' in df.columns:
            unique_genres = df['Genre'].dropna().nunique()
            print(f"   • Unique genres: {unique_genres}")
            
        return df
    except Exception as e:
        print(f"❌ Error loading genre data: {e}")
        return None

def validate_genre_exists(input_genre, df):
    """
    Check if the input genre exists in the knowledge base.
    
    Args:
        input_genre (str): The genre to validate
        df (pd.DataFrame): The genre dataset
        
    Returns:
        tuple: (bool, list) - (exists, similar_genres)
    """
    if df is None or 'Genre' not in df.columns:
        return False, []
    
    # Exact match (case insensitive)
    exact_match = df[df['Genre'].str.lower() == input_genre.lower()]
    if not exact_match.empty:
        return True, []
    
    # Find similar genres (partial matches)
    similar_genres = df[df['Genre'].str.contains(input_genre, case=False, na=False)]['Genre'].unique()
    
    return False, list(similar_genres)

def get_genre_suggestions(df, limit=15):
    """
    Get a list of genre suggestions from the knowledge base.
    
    Args:
        df (pd.DataFrame): The genre dataset
        limit (int): Maximum number of suggestions to return
        
    Returns:
        list: List of genre suggestions
    """
    if df is None or 'Genre' not in df.columns:
        return []
    
    return df['Genre'].dropna().unique()[:limit].tolist()

if __name__ == "__main__":
    print("Genre Recommendation System")
    
    # Load and preview data
    df = load_and_preview_data()
    
    if df is not None:
        print(f"\n✅ Successfully loaded genre knowledge base with {len(df)} entries")
        
        while True:
            # Get user input
            user_input = input("\n🎯 Enter a genre to discover related genres (or 'quit' to exit): ").strip()
            
            # Check if user wants to exit
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n🎬 Thank you for using the Genre Recommendation System!")
                break
            
            # Check if input is empty
            if not user_input:
                print("⚠️  Please enter a valid genre name.")
                continue
            
            # Validate genre exists in knowledge base
            genre_exists, similar_genres = validate_genre_exists(user_input, df)
            
            if not genre_exists and similar_genres:
                print(f"\n⚠️  '{user_input}' not found exactly, but found similar genres:")
                for i, genre in enumerate(similar_genres[:5], 1):
                    print(f"   {i}. {genre}")
                
                choice = input(f"\n💡 Would you like to use one of these instead? Enter number (1-{min(len(similar_genres), 5)}) or 'n' for no: ").strip()
                
                if choice.isdigit() and 1 <= int(choice) <= min(len(similar_genres), 5):
                    user_input = similar_genres[int(choice) - 1]
                    print(f"✅ Using '{user_input}' for analysis")
                elif choice.lower() == 'n':
                    print("🔄 Please try a different genre.")
                    continue
                else:
                    print("❌ Invalid choice. Please try again.")
                    continue
            elif not genre_exists and not similar_genres:
                print(f"\n❌ '{user_input}' not found in the knowledge base.")
                suggestions = get_genre_suggestions(df, 8)
                if suggestions:
                    print("\n💡 Here are some available genres you can try:")
                    for i, genre in enumerate(suggestions, 1):
                        print(f"   {i}. {genre}")
                continue
            
            # Find related genres
            print(f"\n🔍 Analyzing relationships for: '{user_input}'")
            
            
            try:
                results = find_related_genres(user_input)
                print("GENRE RECOMMENDATION RESULTS:")
                print(results)
                
            except Exception as e:
                print(f"\n❌ An error occurred during analysis: {e}")
                


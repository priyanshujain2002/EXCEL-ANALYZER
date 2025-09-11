"""
Enhanced Genre Recommendation System

This script contains a dual-agent CrewAI system that:
1. Finds related genres based on semantic similarity (Genre Analyst)
2. Ranks recommendations by supply priority using X and Y values (Supply Priority Analyst)

The system provides genre recommendations ranked by supply opportunity metrics where:
- X = % of unsold supply belonging to that supply type
- Y = % of that supply type that is unsold
- Priority given to X when ranking (higher X = higher priority)
"""

import os
# Disable telemetry before importing CrewAI to prevent connection issues
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TELEMETRY_ENABLED"] = "false"
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

from crewai import Agent, Task, Crew, LLM
from crewai.knowledge.source.excel_knowledge_source import ExcelKnowledgeSource
import pandas as pd
import os
import boto3
import time
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.WARNING)
logging.getLogger("crewai").setLevel(logging.WARNING)

# Disable CrewAI telemetry to prevent connection issues
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TELEMETRY_ENABLED"] = "false"

# Initialize Bedrock session and LLM configuration with retry settings
session = boto3.Session(region_name="us-east-1")

llm = LLM(
    model="bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    aws_region_name="us-east-1",
    temperature=0.1,  # Lower temperature for consistent genre analysis
    timeout=120,  # Increase timeout to 2 minutes
    max_retries=3,  # Add retry mechanism
)

embedder_config = {
    "provider": "bedrock",
    "config": {
        "model": "amazon.titan-embed-text-v1",
        "session": session
    }
}

# Setup Excel Knowledge Source for genre data
genre_excel_path = "Request Response Database.xlsx"

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

# Create the supply priority analyst agent
supply_priority_analyst = Agent(
    role='Supply Chain Priority Strategist',
    goal='Analyze and rank recommended genres based on supply opportunity metrics (X and Y values) to identify high-priority business opportunities',
    backstory='''You are a senior supply chain strategist and business intelligence expert with 15+ years of experience
    in media inventory optimization and revenue maximization. Your specialized expertise includes:
    
    CORE COMPETENCIES:
    - Supply-Demand Analytics: Expert in analyzing unsold inventory patterns and supply chain inefficiencies
    - Revenue Optimization: Skilled at identifying high-value opportunities in underutilized content categories
    - Market Gap Analysis: Proficient in spotting supply-demand mismatches that represent business opportunities
    - Priority Scoring: Advanced in creating weighted scoring systems for business decision-making
    - Risk-Reward Assessment: Experienced in balancing opportunity size with market penetration difficulty
    
    ANALYTICAL FRAMEWORK:
    - X Value Analysis: % of unsold supply belonging to that supply type (market share of unsold inventory)
    - Y Value Analysis: % of that supply type that is unsold (inefficiency rate within category)
    - Combined Priority Scoring: Sophisticated weighting of X and Y to identify optimal opportunities
    - Business Impact Assessment: Evaluation of potential ROI and strategic value
    
    PRIORITY PHILOSOPHY:
    - High X + High Y = Maximum Priority (large unsold market share + high inefficiency rate)
    - High X + Low Y = Strategic Priority (large market presence but efficient - competitive advantage opportunity)
    - Low X + High Y = Niche Priority (small but highly inefficient - quick wins possible)
    - Low X + Low Y = Low Priority (small market share and efficient - limited opportunity)
    
    You provide data-driven, actionable business intelligence with clear priority rankings and strategic reasoning.''',
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

# Create the supply priority ranking task
rank_by_supply_priority_task = Task(
    description="", # This will be dynamically set
    agent=supply_priority_analyst,
    expected_output="Ranked list of genres with same ID/Genre/Reasoning format, ordered by supply priority",
    context=[recommend_genres_task]
)

# Create the crew with both agents
genre_crew = Crew(
    agents=[genre_analyst, supply_priority_analyst],
    tasks=[analyze_genres_task, recommend_genres_task, rank_by_supply_priority_task],
    verbose=True
)

def find_related_genres(input_genre, max_retries: int = 3, retry_delay: int = 5):
    """
    Find genres that are related to the input genre using advanced multi-criteria analysis.
    
    Args:
        input_genre (str): The genre to find related genres for
        max_retries (int): Maximum number of retry attempts
        retry_delay (int): Delay between retries in seconds
        
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
    
    # Update the supply priority ranking task description
    rank_by_supply_priority_task.description = f"""
    Take the genre recommendations from the Genre Analyst and re-rank them based on supply opportunity metrics (X and Y values).
    
    RANKING METHODOLOGY:
    1. For each recommended genre, extract X and Y values from the Excel knowledge source:
       - X Value: % of unsold supply belonging to that supply type
       - Y Value: % of that supply type that is unsold
    
    2. RANKING PRIORITY SYSTEM:
       - Primary Sort: X value (descending) - prioritize genres with higher % of unsold supply
       - Secondary Sort: Y value (descending) - when X values are similar, prioritize higher % unsold rate
       - When X values are equal or very close, give priority to higher X value
    
    3. MAINTAIN EXACT OUTPUT FORMAT:
       Keep the exact same format as the Genre Analyst provided:
       
       ID: [Excel_Row_ID]
       Genre: [Exact_Genre_Name_From_Knowledge_Base]
       Reasoning: [Original reasoning from Genre Analyst]
       
       ---
    
    4. REQUIREMENTS:
       - Maintain all original recommendations from Genre Analyst
       - Only change the ORDER based on X and Y values (X priority, then Y)
       - Keep exact same ID, Genre, and Reasoning text
       - Do not add any additional information or metrics to the output
       - Simply reorder the list with highest priority (high X, high Y) first
    
    CRITICAL: Output must look identical to Genre Analyst format, just reordered by supply priority.
    """
    
    print(f"\n🎯 Initiating advanced multi-criteria analysis for genre: '{input_genre}'")
    print("📊 Phase 1: Genre Relationship Analysis")
    print("📈 Phase 2: Supply Priority Ranking")
    
    # Implement retry logic for crew execution
    for attempt in range(max_retries):
        try:
            print(f"🚀 Attempt {attempt + 1}/{max_retries}")
            
            # Run the crew with timeout handling
            result = genre_crew.kickoff()
            
            # If we get here, the execution was successful
            return result
            
        except Exception as e:
            error_msg = str(e)
            print(f"⚠️  Attempt {attempt + 1} failed: {error_msg}")
            
            # Check if this is a connection-related error
            if any(keyword in error_msg.lower() for keyword in [
                'connection', 'timeout', 'disconnected', 'bedrock', 'api', 'network'
            ]):
                if attempt < max_retries - 1:
                    print(f"🔄 Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                else:
                    print("❌ All retry attempts exhausted. Providing fallback response.")
                    return generate_fallback_response(input_genre)
            else:
                # For non-connection errors, don't retry
                raise e
    
    # This shouldn't be reached, but just in case
    return generate_fallback_response(input_genre)

def generate_fallback_response(input_genre: str) -> str:
    """
    Generate a fallback response when the AI analysis fails.
    
    Args:
        input_genre (str): The input genre
        
    Returns:
        str: Fallback response with basic genre relationships
    """
    # Load the data to provide basic recommendations
    try:
        df = pd.read_excel(full_path)
        if 'Genre' in df.columns:
            # Find genres that contain similar keywords
            genre_lower = input_genre.lower()
            related_keywords = []
            
            # Define basic genre relationship mappings
            genre_relationships = {
                'crime': ['mystery', 'thriller', 'drama', 'suspense', 'law', 'detective'],
                'mystery': ['crime', 'thriller', 'suspense', 'detective', 'drama'],
                'comedy': ['sitcom', 'variety', 'entertainment', 'family'],
                'drama': ['romance', 'family', 'general drama', 'crime drama'],
                'action': ['adventure', 'thriller', 'crime', 'suspense'],
                'horror': ['thriller', 'suspense', 'mystery', 'paranormal'],
                'romance': ['drama', 'comedy', 'romantic comedy', 'family'],
                'documentary': ['education', 'history', 'science', 'nature'],
                'sports': ['competition', 'reality', 'entertainment'],
                'music': ['variety', 'entertainment', 'concert', 'dance']
            }
            
            # Find related genres based on keywords
            for key, related_list in genre_relationships.items():
                if key in genre_lower:
                    related_keywords.extend(related_list)
            
            # Search for matching genres in the dataset
            fallback_genres = []
            for keyword in related_keywords[:10]:  # Limit search
                matches = df[df['Genre'].str.contains(keyword, case=False, na=False)]
                for _, row in matches.head(2).iterrows():  # Max 2 per keyword
                    if row['Genre'].lower() != input_genre.lower():
                        fallback_genres.append({
                            'id': row.get('ID', 'N/A'),
                            'genre': row['Genre'],
                            'reason': f"Shares thematic elements with {input_genre}"
                        })
            
            # Remove duplicates and limit results
            seen_genres = set()
            unique_fallback = []
            for item in fallback_genres:
                if item['genre'] not in seen_genres:
                    seen_genres.add(item['genre'])
                    unique_fallback.append(item)
                if len(unique_fallback) >= 5:
                    break
            
            if unique_fallback:
                response = f"🔄 Fallback Analysis Results for '{input_genre}':\n\n"
                for item in unique_fallback:
                    response += f"ID: {item['id']}\n"
                    response += f"Genre: {item['genre']}\n"
                    response += f"Reasoning: {item['reason']}\n\n---\n\n"
                return response
    
    except Exception as e:
        print(f"⚠️  Fallback generation failed: {e}")
    
    return f"❌ Unable to analyze '{input_genre}' due to technical difficulties. Please try again later."

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
    print("🎬 Enhanced Genre Recommendation System")
    print("📊 Dual-Agent Analysis: Genre Relations + Supply Priority Ranking")
    print("="*60)
    
    # Load and preview data
    df = load_and_preview_data()
    
    if df is not None:
        print(f"\n✅ Successfully loaded genre knowledge base with {len(df)} entries")
        print("🤖 Dual-agent system ready: Genre Analyst + Supply Priority Analyst")
        
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
            
            # Find related genres with supply priority ranking
            print(f"\n🔍 Analyzing relationships for: '{user_input}'")
            print("🤖 Running dual-agent analysis...")
            
            try:
                results = find_related_genres(user_input)
                print("\n" + "="*60)
                print("🎬 GENRE RECOMMENDATION RESULTS:")
                print("📊 Ranked by Supply Priority (X & Y values)")
                print("="*60)
                print(results)
                print("="*60)
                
            except KeyboardInterrupt:
                print("\n\n🛑 Analysis interrupted by user.")
                break
            except Exception as e:
                print(f"\n❌ An error occurred during analysis: {e}")
                print("💡 This might be due to network connectivity issues. Please try again.")
                


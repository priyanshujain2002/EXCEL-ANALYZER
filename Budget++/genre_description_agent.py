"""
CrewAI Agent for Excel Genre Processing

This script creates a CrewAI agent that:
1. Reads the Excel file from the knowledge directory
2. Extracts genres from the genre column
3. Generates one-line keyword descriptions for each genre
4. Adds a new description column to the Excel file
5. Saves the updated Excel file with all existing columns and the new description column

Uses the excel_mcp tools for Excel operations.
"""

import os
# Disable telemetry before importing CrewAI to prevent connection issues
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TELEMETRY_ENABLED"] = "false"
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

import json
import pandas as pd
from crewai import Agent, Task, Crew, LLM
from crewai.knowledge.source.excel_knowledge_source import ExcelKnowledgeSource
import boto3
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable CrewAI telemetry to prevent connection issues
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TELEMETRY_ENABLED"] = "false"

# Initialize Bedrock session and LLM configuration
session = boto3.Session(region_name="us-east-1")

llm = LLM(
    model="bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    aws_region_name="us-east-1",
    temperature=0.3,  # Moderate temperature for creative but focused descriptions
    timeout=60,
    max_retries=2,
)

# Embedder configuration for knowledge source
embedder_config = {
    "provider": "bedrock",
    "config": {
        "model": "amazon.titan-embed-text-v1",
        "session": session
    }
}

# Setup Excel Knowledge Source for genre data
GENRE_EXCEL_FILENAME = "druid_query_results.xlsx"

# Verify the file exists (ExcelKnowledgeSource looks in knowledge directory by default)
full_path = os.path.join("knowledge", GENRE_EXCEL_FILENAME)
if not os.path.exists(full_path):
    raise FileNotFoundError(f"Genre knowledge source not found at: {full_path}")

excel_source = ExcelKnowledgeSource(
    file_paths=[GENRE_EXCEL_FILENAME],
    embedder=embedder_config
)

# Excel file paths for pandas operations
EXCEL_FILE_PATH = "knowledge/druid_query_results.xlsx"
OUTPUT_EXCEL_PATH = "knowledge/druid_query_results_with_descriptions.xlsx"

class ExcelProcessor:
    """Helper class to handle Excel operations using excel_mcp tools"""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.data = None
        self.genres = []
        
    def read_excel_data(self) -> bool:
        """Read Excel file and extract genre data"""
        try:
            # Use pandas to read the Excel file
            self.data = pd.read_excel(self.filepath)
            logger.info(f"Successfully read Excel file with {len(self.data)} rows")
            
            # Check if genre column exists
            if 'genre' not in self.data.columns:
                logger.error("Genre column not found in Excel file")
                return False
                
            # Extract unique genres
            self.genres = self.data['genre'].dropna().unique().tolist()
            logger.info(f"Found {len(self.genres)} unique genres")
            
            return True
            
        except Exception as e:
            logger.error(f"Error reading Excel file: {e}")
            return False
    
    def get_genres_with_ids(self) -> List[Dict[str, Any]]:
        """Get genres with their corresponding IDs"""
        if self.data is None:
            return []
            
        genre_info = []
        for _, row in self.data.iterrows():
            if pd.notna(row['genre']):
                genre_info.append({
                    'id': row['id'],
                    'genre': row['genre']
                })
        
        return genre_info
    
    def add_descriptions_to_data(self, descriptions: Dict[str, str]) -> None:
        """Add description column to the data"""
        if self.data is None:
            return
            
        # Add description column
        self.data['description'] = self.data['genre'].map(descriptions)
        
        # Fill any NaN values with empty string
        self.data['description'] = self.data['description'].fillna('')
        
    def save_updated_excel(self, output_path: str) -> bool:
        """Save the updated Excel file"""
        try:
            self.data.to_excel(output_path, index=False)
            logger.info(f"Successfully saved updated Excel file to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving Excel file: {e}")
            return False

# Create the genre description agent
genre_description_agent = Agent(
    role='Expert Genre Description Specialist',
    goal='Generate concise, impactful one-line keyword descriptions for entertainment genres that capture their essence and appeal',
    backstory='''You are a master entertainment genre analyst with decades of experience in content categorization,
    audience psychology, and media marketing. Your expertise includes:

    CORE COMPETENCIES:
    - Genre Analysis: Deep understanding of what makes each genre unique and appealing
    - Audience Psychology: Knowledge of what draws viewers to different types of content
    - Marketing Copy: Ability to create compelling, concise descriptions that resonate
    - Keyword Optimization: Skill in identifying the most impactful terms for each genre
    - Trend Awareness: Understanding of current genre trends and audience preferences

    DESCRIPTION PHILOSOPHY:
    - CONCISE: Each description must be exactly one line (under 15 words)
    - KEYWORD-FOCUSED: Use powerful, searchable terms that capture the genre's essence
    - EMOTIONAL: Tap into the core emotional appeal of each genre
    - DISTINCTIVE: Highlight what makes each genre unique from others
    - ACTION-ORIENTED: Use dynamic language that conveys the genre's energy

    DESCRIPTION FRAMEWORK:
    1. Core Theme: What is the central subject matter or premise?
    2. Emotional Tone: What feelings does the genre typically evoke?
    3. Style Elements: What are the characteristic production or narrative styles?
    4. Target Appeal: What type of viewer is most drawn to this genre?

    EXAMPLE DESCRIPTIONS:
    - "Action": High-energy thrills with explosive stunts and heroic adventures
    - "Drama": Emotional character studies exploring complex human relationships
    - "Comedy": Lighthearted humor designed to entertain and amuse audiences
    - "Horror": Suspenseful tales designed to frighten and thrill viewers
    - "Documentary": Factual explorations of real people, places, and events

    You excel at distilling complex genre characteristics into powerful, memorable one-liners.''',
    llm=llm,
    knowledge_sources=[excel_source],
    embedder=embedder_config,
    verbose=True,
    allow_delegation=False
)

# Create the genre processing task
process_genres_task = Task(
    description="""
    Analyze the entertainment genres in the Excel knowledge source and generate compelling one-line keyword descriptions for each unique genre.
    
    KNOWLEDGE SOURCE: You have access to an Excel file containing genre data with columns including id, genre, creativetype, sum_request, sum_response, unsold_supply, x, and y.
    
    REQUIREMENTS:
    1. ONE-LINE ONLY: Each description must be exactly one line (under 15 words)
    2. KEYWORD-RICH: Use powerful, searchable terms that capture the genre's essence
    3. EMOTIONAL APPEAL: Tap into the core emotional draw of each genre
    4. DISTINCTIVE: Highlight what makes each genre unique from others
    5. CONSISTENT: Maintain a similar style and tone across all descriptions
    
    PROCESS:
    1. Access the Excel knowledge source to extract all unique genres from the genre column
    2. For each unique genre found in the dataset:
       a. Analyze the genre's core characteristics and audience appeal
       b. Identify the most impactful keywords and emotional triggers
       c. Craft a concise, powerful one-line description
       d. Ensure the description is unique and distinctive
    3. Use the additional data columns (creativetype, x, y values) to inform your understanding of each genre's context and popularity
    
    OUTPUT FORMAT:
    Return a JSON object where keys are the genre names (exactly as found in the Excel file) and values are the one-line descriptions.
    Example:
    {
      "general variety": "Diverse entertainment mix with broad audience appeal",
      "drama": "Emotional character studies exploring complex human relationships",
      "news": "Current events and factual reporting with real-time relevance"
    }
    
    CRITICAL: 
    - Extract genres directly from the Excel knowledge source
    - Every unique genre in the dataset must have exactly one description line
    - Use genre names exactly as they appear in the Excel file
    - Leverage the knowledge source embeddings to understand genre relationships and characteristics
    """,
    agent=genre_description_agent,
    expected_output="JSON object with genre names as keys and one-line descriptions as values"
)

# Create the crew
genre_crew = Crew(
    agents=[genre_description_agent],
    tasks=[process_genres_task],
    verbose=True
)

def generate_genre_descriptions(genres_with_ids: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Generate one-line descriptions for the provided genres using Excel knowledge source
    
    Args:
        genres_with_ids: List of dictionaries containing 'id' and 'genre'
        
    Returns:
        Dictionary mapping genre names to descriptions
    """
    # Prepare the input for the agent
    genre_list = [item['genre'] for item in genres_with_ids]
    
    print(f"🎯 Generating descriptions for {len(genre_list)} genres using Excel knowledge source...")
    
    # Process genres in smaller batches to avoid timeout
    batch_size = 50
    all_descriptions = {}
    
    for i in range(0, len(genre_list), batch_size):
        batch_genres = genre_list[i:i + batch_size]
        print(f"🔄 Processing batch {i//batch_size + 1}/{(len(genre_list) + batch_size - 1)//batch_size} ({len(batch_genres)} genres)")
        
        # Create a new task for each batch
        batch_task = Task(
            description=f"""
            Analyze the entertainment genres in the Excel knowledge source and generate compelling one-line keyword descriptions for each unique genre.
            
            KNOWLEDGE SOURCE: You have access to an Excel file containing genre data with columns including id, genre, creativetype, sum_request, sum_response, unsold_supply, x, and y.
            
            REQUIREMENTS:
            1. ONE-LINE ONLY: Each description must be exactly one line (under 15 words)
            2. KEYWORD-RICH: Use powerful, searchable terms that capture the genre's essence
            3. EMOTIONAL APPEAL: Tap into the core emotional draw of each genre
            4. DISTINCTIVE: Highlight what makes each genre unique from others
            5. CONSISTENT: Maintain a similar style and tone across all descriptions
            
            PROCESS:
            1. Access the Excel knowledge source to extract the following genres from the genre column: {', '.join(batch_genres)}
            2. For each genre in this list:
               a. Analyze the genre's core characteristics and audience appeal
               b. Identify the most impactful keywords and emotional triggers
               c. Craft a concise, powerful one-line description
               d. Ensure the description is unique and distinctive
            3. Use the additional data columns (creativetype, x, y values) to inform your understanding of each genre's context and popularity
            
            OUTPUT FORMAT:
            Return a JSON object where keys are the genre names (exactly as found in the Excel file) and values are the one-line descriptions.
            Example:
            {{
              "general variety": "Diverse entertainment mix with broad audience appeal",
              "drama": "Emotional character studies exploring complex human relationships",
              "news": "Current events and factual reporting with real-time relevance"
            }}
            
            CRITICAL: 
            - Extract genres directly from the Excel knowledge source
            - Every genre in the provided list must have exactly one description line
            - Use genre names exactly as they appear in the Excel file
            - Leverage the knowledge source embeddings to understand genre relationships and characteristics
            """,
            agent=genre_description_agent,
            expected_output="JSON object with genre names as keys and one-line descriptions as values"
        )
        
        # Create a new crew for each batch
        batch_crew = Crew(
            agents=[genre_description_agent],
            tasks=[batch_task],
            verbose=True
        )
        
        try:
            # Run the crew for this batch
            result = batch_crew.kickoff()
            
            # Parse the JSON result
            try:
                # Handle CrewOutput object by converting to string first
                result_str = str(result)
                batch_descriptions = json.loads(result_str)
                all_descriptions.update(batch_descriptions)
                logger.info(f"Successfully generated {len(batch_descriptions)} descriptions in batch {i//batch_size + 1}")
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON result in batch {i//batch_size + 1}: {e}")
                logger.error(f"Raw result: {result}")
                continue
                
        except Exception as e:
            logger.error(f"Error generating genre descriptions in batch {i//batch_size + 1}: {e}")
            continue
    
    logger.info(f"Successfully generated {len(all_descriptions)} genre descriptions in total")
    return all_descriptions

def main():
    """Main function to process the Excel file and generate genre descriptions"""
    print("🎬 Genre Description Generator")
    print("📊 Processing Excel file to generate one-line keyword descriptions")
    print("="*60)
    
    # Initialize Excel processor
    excel_processor = ExcelProcessor(EXCEL_FILE_PATH)
    
    # Read Excel data
    if not excel_processor.read_excel_data():
        print("❌ Failed to read Excel file")
        return
    
    # Get genres with IDs
    genres_with_ids = excel_processor.get_genres_with_ids()
    print(f"📋 Found {len(genres_with_ids)} genres to process")
    
    # Generate descriptions using CrewAI agent with Excel knowledge source
    descriptions = generate_genre_descriptions(genres_with_ids)
    
    if not descriptions:
        print("❌ Failed to generate genre descriptions")
        return
    
    print(f"✅ Successfully generated {len(descriptions)} genre descriptions")
    
    # Add descriptions to Excel data
    excel_processor.add_descriptions_to_data(descriptions)
    
    # Save updated Excel file
    if excel_processor.save_updated_excel(OUTPUT_EXCEL_PATH):
        print(f"🎉 Successfully saved updated Excel file to: {OUTPUT_EXCEL_PATH}")
        print("\n📊 Summary:")
        print(f"   • Original file: {EXCEL_FILE_PATH}")
        print(f"   • Updated file: {OUTPUT_EXCEL_PATH}")
        print(f"   • Total genres processed: {len(descriptions)}")
        print(f"   • New column added: 'description'")
        
        # Show sample descriptions
        print("\n📝 Sample descriptions:")
        sample_count = min(5, len(descriptions))
        for i, (genre, desc) in enumerate(list(descriptions.items())[:sample_count]):
            print(f"   {i+1}. {genre}: {desc}")
        
    else:
        print("❌ Failed to save updated Excel file")

if __name__ == "__main__":
    main()

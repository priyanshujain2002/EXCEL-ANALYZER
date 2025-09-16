"""
Enhanced Genre Similarity Scorer using Pre-existing Descriptions and Amazon Titan Text Embeddings V2

This script reads genres and their descriptions from the druid query results file and uses:
1. Pre-existing descriptions from the Excel file for genres in the dataset
2. CrewAI with AWS Bedrock to generate meaningful one-line descriptions for new input genres only
3. Amazon Titan Text Embeddings V2 to calculate 
   relevance scores between an input genre and all genres in the dataset

Features:
- Uses pre-existing descriptions from Excel file for known genres
- Uses CrewAI agent to generate intelligent descriptions only for new input genres
- Uses Amazon Titan Text Embeddings V2 model
- Calculates cosine similarity between genre description embeddings
- Returns ranked list of similar genres with similarity scores
- Includes supply metrics (X and Y values) for business insights
"""

import os
# Disable telemetry before importing CrewAI to prevent connection issues
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TELEMETRY_ENABLED"] = "false"
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sys
from typing import List, Tuple, Dict, Optional
import warnings
import ssl
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json
import time
import logging

# CrewAI imports
from crewai import Agent, Task, Crew, LLM
from crewai.knowledge.source.excel_knowledge_source import ExcelKnowledgeSource
import boto3

# Disable SSL certificate verification (for development/testing only)
ssl._create_default_https_context = ssl._create_unverified_context

# Create a requests session with SSL verification disabled
def create_unverified_session():
    # Call the original Session class to avoid recursion
    session = original_session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.verify = False  # Disable SSL verification
    return session

# Store original Session class and monkey patch requests to use our unverified session
original_session = requests.Session
requests.Session = create_unverified_session

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.WARNING)
logging.getLogger("crewai").setLevel(logging.WARNING)

# Initialize Bedrock session and LLM configuration with retry settings
session = boto3.Session(region_name="us-east-1")

llm = LLM(
    model="bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    aws_region_name="us-east-1",
    temperature=0.1,  # Lower temperature for consistent genre descriptions
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
genre_excel_path = "druid_query_results_with_descriptions.xlsx"

# Verify the file exists (ExcelKnowledgeSource looks in knowledge directory by default)
full_path = os.path.join("knowledge", genre_excel_path)
if not os.path.exists(full_path):
    raise FileNotFoundError(f"Genre knowledge source not found at: {full_path}")

excel_source = ExcelKnowledgeSource(
    file_paths=[genre_excel_path],
    embedder=embedder_config
)

# Create the genre description specialist agent
genre_description_agent = Agent(
    role='Genre Description Specialist',
    goal='Generate concise, meaningful one-line descriptions that capture the essence and key characteristics of any given genre',
    backstory='''You are an expert entertainment genre analyst with deep knowledge across music, film, literature, and gaming genres. 
    Your specialized expertise includes:

    CORE COMPETENCIES:
    - Genre Essence Extraction: Identify the fundamental characteristics that define each genre
    - Cultural Context Understanding: Recognize the historical and cultural significance of genres
    - Audience Analysis: Understand what makes genres appealing to specific demographics
    - Cross-Media Genre Knowledge: Expertise in how genres manifest across different entertainment forms
    - Descriptive Precision: Ability to capture complex genre concepts in single, impactful sentences

    DESCRIPTION PHILOSOPHY:
    - CONCISENESS: Create exactly one sentence per description - no more, no less
    - ESSENCE CAPTURE: Focus on the core identity and unique characteristics of each genre
    - ACCESSIBILITY: Make descriptions understandable to both experts and casual audiences
    - DISTINCTIVENESS: Highlight what makes each genre unique and different from others
    - NEUTRALITY: Maintain objective, factual tone without subjective judgments

    ANALYTICAL FRAMEWORK:
    1. Core Identity: What fundamental elements define this genre?
    2. Key Characteristics: What are the most recognizable features or conventions?
    3. Cultural Context: What historical or cultural background is relevant?
    4. Audience Appeal: What emotional or intellectual needs does this genre fulfill?
    5. Medium-Specific Traits: How does this genre manifest in its primary medium?

    You are precise, insightful, and consistently deliver high-quality, informative descriptions that capture the true essence of any genre presented to you.''',
    llm=llm,
    knowledge_sources=[excel_source],
    embedder=embedder_config,
    verbose=True,
    allow_delegation=False
)

# Create the genre description task
generate_description_task = Task(
    description="",  # This will be dynamically set
    agent=genre_description_agent,
    expected_output="A single, concise sentence describing the genre"
)

# Create the crew for description generation
description_crew = Crew(
    agents=[genre_description_agent],
    tasks=[generate_description_task],
    verbose=True
)

class GenreSimilarityScorerWithDescriptions:
    """
    Enhanced genre similarity scorer using pre-existing descriptions and CrewAI for new input genres.
    """
    
    def __init__(self, excel_file_path: str = "knowledge/druid_query_results_with_descriptions.xlsx"):
        """
        Initialize the Enhanced Genre Similarity Scorer.
        
        Args:
            excel_file_path (str): Path to the Excel file containing genre data with descriptions
        """
        self.excel_file_path = excel_file_path
        self.model_id = "amazon.titan-embed-text-v2:0"  # Titan Text Embeddings V2 model ID
        self.bedrock_runtime = None
        self.genres_df = None
        self.genre_embeddings = None
        self.genres_list = None
        self.genre_descriptions = {}  # Cache for genre descriptions (loaded from Excel)
        
        # Load data and model
        self._load_data()
        self._initialize_bedrock_client()
        self._generate_embeddings()
    
    def _load_data(self):
        """Load genre data and descriptions from Excel file."""
        try:
            if not os.path.exists(self.excel_file_path):
                raise FileNotFoundError(f"Excel file not found: {self.excel_file_path}")
            
            # Read the Excel file
            self.genres_df = pd.read_excel(self.excel_file_path)
            
            # Verify required columns exist
            required_columns = ['genre', 'id', 'sum_request', 'sum_response']
            # Check if description column exists
            if 'description' in self.genres_df.columns:
                required_columns.append('description')
            else:
                print("⚠️  No 'description' column found in Excel file")
            
            missing_columns = [col for col in required_columns if col not in self.genres_df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Calculate x and y values based on Excel formulas
            # x = (unsold_supply / total_unsold_supply) * 100
            # y = (unsold_supply / sum_request) * 100
            # where unsold_supply = sum_request - sum_response
            
            self.genres_df['unsold_supply_calc'] = self.genres_df['sum_request'] - self.genres_df['sum_response']
            total_unsold_supply = self.genres_df['unsold_supply_calc'].sum()
            
            self.genres_df['x_calc'] = (self.genres_df['unsold_supply_calc'] / total_unsold_supply) * 100
            self.genres_df['y_calc'] = np.where(
                self.genres_df['sum_request'] == 0,
                0,
                (self.genres_df['unsold_supply_calc'] / self.genres_df['sum_request']) * 100
            )
            
            print(f"📊 Calculated x and y values for all genres")
            print(f"📈 x range: {self.genres_df['x_calc'].min():.4f} - {self.genres_df['x_calc'].max():.4f}")
            print(f"📈 y range: {self.genres_df['y_calc'].min():.4f} - {self.genres_df['y_calc'].max():.4f}")
            
            # Clean and prepare genre data
            self.genres_df = self.genres_df.dropna(subset=['genre'])
            self.genres_df['genre'] = self.genres_df['genre'].astype(str).str.strip()
            
            # Remove duplicates based on genre name
            self.genres_df = self.genres_df.drop_duplicates(subset=['genre'], keep='first')
            
            # Create list of genres for embedding
            self.genres_list = self.genres_df['genre'].tolist()
            
            # Load descriptions from Excel
            if 'description' in self.genres_df.columns:
                for _, row in self.genres_df.iterrows():
                    genre = row['genre']
                    description = str(row['description']).strip()
                    if description and description != 'nan':
                        self.genre_descriptions[genre] = description
                    else:
                        # Generate fallback description if empty
                        self.genre_descriptions[genre] = self._generate_fallback_description(genre)
            else:
                # Generate fallback descriptions for all genres
                for genre in self.genres_list:
                    self.genre_descriptions[genre] = self._generate_fallback_description(genre)
            
            print(f"✅ Successfully loaded {len(self.genres_list)} unique genres from {self.excel_file_path}")
            print(f"📝 Loaded {len(self.genre_descriptions)} descriptions from Excel file")
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            sys.exit(1)
    
    def _initialize_bedrock_client(self):
        """Initialize the Amazon Bedrock runtime client."""
        try:
            print(f"🤖 Initializing Amazon Bedrock client for model: {self.model_id}")
            
            # boto3 client will automatically use credentials from environment variables
            # or ~/.aws/credentials, or IAM role if running on EC2.
            # saml2aws typically populates ~/.aws/credentials.
            self.bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')
            
            print(f"✅ Amazon Bedrock client initialized successfully for {self.model_id}")
            
        except Exception as e:
            print(f"❌ Error initializing Bedrock client: {e}")
            print("💡 Ensure AWS credentials are configured (e.g., via saml2aws) and boto3 is installed: pip install boto3")
            sys.exit(1)

    def _get_titan_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a single text string using Amazon Titan Text Embeddings V2.
        
        Args:
            text (str): The input text to embed.
            
        Returns:
            List[float]: The embedding vector.
        """
        try:
            # Titan Text Embeddings V2 expects a specific JSON structure
            # For embedding a single string, we use "inputText"
            body = json.dumps({
                "inputText": text
            })
            
            response = self.bedrock_runtime.invoke_model(
                modelId=self.model_id,
                body=body,
                contentType="application/json",
                accept="application/json"
            )
            
            response_body = json.loads(response.get('body').read())
            embedding = response_body.get('embedding')
            
            if not embedding:
                raise ValueError("No embedding found in Titan response.")
                
            return embedding
            
        except Exception as e:
            print(f"❌ Error getting Titan embedding for text '{text}': {e}")
            # Potentially re-raise or handle more gracefully depending on requirements
            # For now, returning an empty list might cause issues downstream, so logging and re-raising
            raise
    
    def _generate_genre_description(self, genre_name: str, max_retries: int = 3) -> str:
        """
        Generate a one-line description for a genre using CrewAI agent.
        Only used for new input genres not in the dataset.
        
        Args:
            genre_name (str): The genre to generate a description for
            max_retries (int): Maximum number of retry attempts
            
        Returns:
            str: Generated one-line description
        """
        # Update the task description with the specific genre
        generate_description_task.description = f"""
        Generate a concise, meaningful one-line description for the genre '{genre_name}'.
        
        REQUIREMENTS:
        1. EXACTLY ONE SENTENCE: Your response must be a single, complete sentence
        2. ESSENCE CAPTURE: Focus on the core identity and key characteristics that define this genre
        3. CONCISENESS: Be informative but brief - aim for 15-25 words maximum
        4. ACCESSIBILITY: Make the description understandable to general audiences
        5. NEUTRALITY: Maintain objective, factual tone without subjective judgments
        6. NO FORMATTING: Do not use quotes, bullet points, or any special formatting
        
        Focus on what makes '{genre_name}' unique and distinctive as a genre.
        Provide only the single sentence description with no additional text or explanation.
        """
        
        # Implement retry logic
        for attempt in range(max_retries):
            try:
                print(f"🤖 Generating description for input genre '{genre_name}' (attempt {attempt + 1}/{max_retries})")
                
                # Run the crew
                result = description_crew.kickoff()
                
                # Extract the description from the result
                if hasattr(result, 'raw'):
                    description = result.raw.strip()
                else:
                    description = str(result).strip()
                
                # Clean up the description
                description = self._clean_description(description, genre_name)
                
                if description:
                    print(f"✅ Generated description: {description}")
                    return description
                else:
                    raise Exception("Empty description generated")
                    
            except Exception as e:
                error_msg = str(e)
                print(f"⚠️  Attempt {attempt + 1} failed: {error_msg}")
                
                if attempt < max_retries - 1:
                    print(f"🔄 Retrying in 2 seconds...")
                    time.sleep(2)
                    continue
                else:
                    print(f"❌ All retry attempts exhausted for '{genre_name}'")
                    # Return a fallback description
                    return self._generate_fallback_description(genre_name)
        
        # This shouldn't be reached, but just in case
        return self._generate_fallback_description(genre_name)
    
    def _clean_description(self, description: str, genre_name: str) -> str:
        """
        Clean and validate the generated description.
        
        Args:
            description (str): Raw description from CrewAI
            genre_name (str): The genre name for fallback
            
        Returns:
            str: Cleaned description or fallback if invalid
        """
        if not description:
            return self._generate_fallback_description(genre_name)
        
        # Remove quotes if present
        description = description.strip('"\'')
        
        # Remove common prefixes/suffixes
        description = description.replace("Description:", "").strip()
        description = description.replace("The genre", "").strip()
        
        # Ensure it's a single sentence (basic check)
        if '.' in description[:-1]:  # Allow period at the end
            # Take only the first complete sentence
            description = description.split('.')[0] + '.'
        
        # Validate length
        if len(description) < 10 or len(description) > 200:
            return self._generate_fallback_description(genre_name)
        
        return description
    
    def _generate_fallback_description(self, genre_name: str) -> str:
        """
        Generate a simple fallback description when CrewAI fails or no description exists.
        
        Args:
            genre_name (str): The genre name
            
        Returns:
            str: Fallback description
        """
        return f"A {genre_name} genre characterized by its distinctive {genre_name} elements and style."
    
    def _generate_embeddings(self):
        """Generate embeddings for all genre names and their descriptions combined using Titan."""
        try:
            print("🔄 Generating embeddings for all genres and their descriptions using Amazon Titan Text Embeddings V2...")
            
            # Combine genre name with description for each genre
            combined_texts = []
            for genre in self.genres_list:
                description = self.genre_descriptions.get(genre, self._generate_fallback_description(genre))
                combined_text = f"{genre}: {description}"
                combined_texts.append(combined_text)
            
            # Generate embeddings for combined genre+description texts using Titan
            embeddings_list = []
            for i, combined_text in enumerate(combined_texts):
                logging.info(f"Processing genre {i+1}/{len(combined_texts)}: {self.genres_list[i]}")
                embedding = self._get_titan_embedding(combined_text)
                embeddings_list.append(embedding)
            
            self.genre_embeddings = np.array(embeddings_list)
            
            print(f"✅ Generated embeddings for {len(self.genres_list)} genres with descriptions using Titan")
            print(f"📊 Embedding dimension: {self.genre_embeddings.shape[1]}")
            
        except Exception as e:
            print(f"❌ Error generating embeddings with Titan: {e}")
            sys.exit(1)
    
    def calculate_similarity(self, input_genre: str, top_k: int = None) -> List[Dict]:
        """
        Calculate similarity scores between input genre and all genres in dataset.
        
        Args:
            input_genre (str): The genre to find similar genres for
            top_k (int): Number of top similar genres to return. If None, returns all.
            
        Returns:
            List[Dict]: List of dictionaries containing genre info and similarity scores
        """
        try:
            # Check if input genre exists in our dataset
            if input_genre in self.genre_descriptions:
                input_description = self.genre_descriptions[input_genre]
                print(f"📝 Using existing description for '{input_genre}'")
            else:
                # Generate description for new input genre
                print(f"🤖 Generating description for new input genre: '{input_genre}'")
                input_description = self._generate_genre_description(input_genre)
            
            # Combine input genre with its description for embedding
            input_combined_text = f"{input_genre}: {input_description}"
            
            # Generate embedding for combined genre+description text using Titan
            input_embedding_list = self._get_titan_embedding(input_combined_text)
            input_embedding = np.array([input_embedding_list])  # Reshape for cosine_similarity
            
            # Calculate cosine similarity with all genre+description embeddings
            similarities = cosine_similarity(input_embedding, self.genre_embeddings)[0]
            
            # Create results list with genre info and similarity scores
            results = []
            for i, similarity_score in enumerate(similarities):
                genre_row = self.genres_df.iloc[i]
                genre_name = genre_row['genre']
                genre_description = self.genre_descriptions.get(genre_name, "")
                
                # Skip if it's the exact same genre (similarity = 1.0)
                if similarity_score >= 0.999 and genre_name.lower() == input_genre.lower():
                    continue
                
                result = {
                    'id': int(genre_row['id']),
                    'genre': genre_name,
                    'description': genre_description,
                    'similarity_score': float(similarity_score),
                    'x': float(genre_row['x_calc']),
                    'y': float(genre_row['y_calc'])
                }
                results.append(result)
            
            # Sort by similarity score (descending)
            results.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            # Return top K results if specified, otherwise return all
            if top_k is not None:
                return results[:top_k]
            else:
                return results
            
        except Exception as e:
            print(f"❌ Error calculating similarity: {e}")
            return []
    
    def find_similar_genres(self, input_genre: str, top_k: int = None, 
                          min_similarity: float = 0.3) -> List[Dict]:
        """
        Find genres similar to the input genre with filtering options.
        
        Args:
            input_genre (str): The genre to find similar genres for
            top_k (int): Maximum number of results to return. If None, returns all matching genres.
            min_similarity (float): Minimum similarity threshold (0.0 to 1.0)
            
        Returns:
            List[Dict]: Filtered and ranked list of similar genres
        """
        # Determine how many genres to fetch initially for similarity calculation
        num_to_fetch = None
        if top_k is not None and top_k > 0:
            num_to_fetch = top_k * 2  # Fetch more to allow for filtering
        
        # Get similarity scores
        similar_genres = self.calculate_similarity(input_genre, top_k=num_to_fetch)
        
        # Filter by minimum similarity threshold
        filtered_genres = [
            genre for genre in similar_genres 
            if genre['similarity_score'] >= min_similarity
        ]
        
        # Return top K after filtering if top_k is specified and positive, otherwise return all filtered
        if top_k is not None and top_k > 0:
            return filtered_genres[:top_k]
        else:
            return filtered_genres
    
    def get_genre_suggestions(self, limit: int = 15) -> List[str]:
        """
        Get a list of available genres for user reference.
        
        Args:
            limit (int): Maximum number of suggestions to return
            
        Returns:
            List[str]: List of genre names
        """
        return self.genres_list[:limit]
    
    def validate_genre_exists(self, input_genre: str) -> Tuple[bool, List[str]]:
        """
        Check if input genre exists in dataset and suggest similar ones if not.
        
        Args:
            input_genre (str): Genre to validate
            
        Returns:
            Tuple[bool, List[str]]: (exists, similar_genres_list)
        """
        # Exact match (case insensitive)
        exact_matches = [genre for genre in self.genres_list 
                        if genre.lower() == input_genre.lower()]
        
        if exact_matches:
            return True, []
        
        # Partial matches
        partial_matches = [genre for genre in self.genres_list 
                          if input_genre.lower() in genre.lower()]
        
        return False, partial_matches[:5]  # Return up to 5 suggestions
    
    def compute_weighted_average_ranking(self, input_genre: str, top_k: int = 10) -> List[Dict]:
        """
        Get similar genres using dynamic threshold and compute weighted average of x and y values.
        
        Dynamic threshold = highest_similarity - 30%
        Example: If highest similarity is 80%, threshold becomes 50%
        
        Only returns genres that meet the threshold (no fallback logic).
        If more than top_k genres meet threshold, returns top_k by weighted average.
        
        Args:
            input_genre (str): The genre to find similar genres for
            top_k (int): Maximum number of results to return (default: 10)
            
        Returns:
            List[Dict]: List of genres with weighted averages, ranked by weighted average
        """
        # Get ALL similarity results first (no top_k limit, no min_similarity filter)
        print(f"🔍 Getting all similarity scores for '{input_genre}'...")
        all_similar_genres = self.calculate_similarity(input_genre, top_k=None)
        
        if not all_similar_genres:
            print(f"❌ No similar genres found for '{input_genre}'")
            return []
        
        # Find the highest similarity score
        max_similarity = max(genre['similarity_score'] for genre in all_similar_genres)
        
        # Calculate dynamic threshold: highest_similarity - 30%
        dynamic_threshold = max_similarity - 0.3
        
        print(f"📊 Highest similarity: {max_similarity:.4f} ({max_similarity*100:.2f}%)")
        print(f"🎯 Dynamic threshold: {dynamic_threshold:.4f} ({dynamic_threshold*100:.2f}%)")
        
        # Filter genres based on dynamic threshold (no fallback)
        filtered_genres = [
            genre for genre in all_similar_genres
            if genre['similarity_score'] >= dynamic_threshold
        ]
        
        if not filtered_genres:
            print(f"❌ No genres found above dynamic threshold {dynamic_threshold:.4f}")
            return []
        
        print(f"✅ Found {len(filtered_genres)} genres above dynamic threshold")
        
        # Calculate weighted average (50% x + 50% y) for each filtered genre
        weighted_results = []
        for genre_data in filtered_genres:
            weighted_average = (genre_data['x'] * 0.5) + (genre_data['y'] * 0.5)
            
            result = {
                'id': genre_data['id'],
                'genre': genre_data['genre'],
                'similarity_score': genre_data['similarity_score'],
                'x': genre_data['x'],
                'y': genre_data['y'],
                'weighted_average': weighted_average
            }
            weighted_results.append(result)
        
        # Sort by weighted average (descending - highest first)
        weighted_results.sort(key=lambda x: x['weighted_average'], reverse=True)
        
        # Return top K results (or all if fewer than top_k)
        final_results = weighted_results[:top_k]
        
        print(f"🎯 Computed weighted averages for {len(filtered_genres)} genres")
        print(f"📊 Returning {len(final_results)} genres ranked by weighted average")
        if final_results:
            print(f"📈 Weighted average range: {final_results[-1]['weighted_average']:.4f} - {final_results[0]['weighted_average']:.4f}")
        
        return final_results

def print_results(input_genre: str, results: List[Dict]):
    """
    Print formatted results of genre similarity analysis.
    
    Args:
        input_genre (str): The input genre that was analyzed
        results (List[Dict]): List of similar genres with scores and descriptions
    """
    print("\n" + "="*80)
    print(f"🎯 GENRE SIMILARITY RESULTS FOR: '{input_genre.upper()}'")
    print("📊 Ranked by Semantic Similarity (Genre + Description Embeddings)")
    print("📝 Using combined genre names and descriptions from Excel file")
    print("="*80)
    
    if not results:
        print("❌ No similar genres found.")
        return
    
    print(f"{'Rank':<4} {'Genre':<25} {'Similarity':<12} {'Description'}")
    print("-" * 80)
    
    for i, result in enumerate(results, 1):
        similarity_pct = result['similarity_score'] * 100
        description = result.get('description', '')[:45] + '...' if len(result.get('description', '')) > 45 else result.get('description', '')
        print(f"{i:<4} {result['genre']:<25} {similarity_pct:>8.2f}%  {description}")
    
    print("-" * 80)
    print(f"📈 Found {len(results)} similar genres")
    print(f"🔝 Top match: '{results[0]['genre']}' ({results[0]['similarity_score']*100:.2f}% similarity)")

def print_weighted_average_results(input_genre: str, results: List[Dict]):
    """
    Print formatted results of weighted average analysis.
    
    Args:
        input_genre (str): The input genre that was analyzed
        results (List[Dict]): List of genres with weighted averages, ranked by weighted average
    """
    print("\n" + "="*90)
    print(f"🎯 WEIGHTED AVERAGE RANKING FOR: '{input_genre.upper()}'")
    print("📊 Genres Filtered by Dynamic Threshold (Highest Similarity - 30%)")
    print("🏆 Results Ranked by Weighted Average (50% x + 50% y)")
    print("📝 x = % of total unsold supply, y = % unsold supply relative to requests")
    print("="*90)
    
    if not results:
        print("❌ No genres found for weighted average calculation.")
        return
    
    print(f"{'Rank':<4} {'ID':<4} {'Genre':<25} {'Similarity':<12} {'x':<12} {'y':<12} {'Weighted Avg':<12}")
    print("-" * 100)
    
    for i, result in enumerate(results, 1):
        similarity_pct = result['similarity_score'] * 100
        print(f"{i:<4} {result['id']:<4} {result['genre']:<25} {similarity_pct:>8.2f}%  "
              f"{result['x']:>10.6f}%  {result['y']:>10.6f}%  {result['weighted_average']:>10.6f}%")
    
    print("-" * 100)
    print(f"📈 Analyzed {len(results)} genres")
    print(f"🏆 Highest weighted average: '{results[0]['genre']}' ({results[0]['weighted_average']:.6f}%)")
    print(f"🎯 Formula: Weighted Average = (x × 0.5) + (y × 0.5)")

def main():
    """Main function to run the enhanced genre similarity scorer."""
    print("🎬 Enhanced Genre Similarity Scorer using Pre-existing Descriptions")
    print("🤖 Model: Amazon Titan Text Embeddings V2 + CrewAI (Claude 3.7 Sonnet) for new inputs")
    print("📝 Combined genre name and description semantic similarity analysis")
    print("="*70)
    
    try:
        # Initialize the scorer
        scorer = GenreSimilarityScorerWithDescriptions()
        
        print(f"\n✅ System ready! Loaded {len(scorer.genres_list)} genres with descriptions")
        print("🎯 Enter a genre to find semantically similar genres")
        
        while True:
            # Get user input
            user_input = input("\n🎭 Enter a genre (or 'quit' to exit): ").strip()
            
            # Check if user wants to exit
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n🎬 Thank you for using the Enhanced Genre Similarity Scorer!")
                break
            
            # Check if input is empty
            if not user_input:
                print("⚠️  Please enter a valid genre name.")
                continue
            
            # Validate genre exists or suggest alternatives
            genre_exists, similar_genres = scorer.validate_genre_exists(user_input)
            
            if not genre_exists and similar_genres:
                print(f"\n⚠️  '{user_input}' not found exactly. Similar genres in dataset:")
                for i, genre in enumerate(similar_genres, 1):
                    print(f"   {i}. {genre}")
                
                choice = input(f"\n💡 Use one of these? Enter number (1-{len(similar_genres)}) or 'n': ").strip()
                
                if choice.isdigit() and 1 <= int(choice) <= len(similar_genres):
                    user_input = similar_genres[int(choice) - 1]
                    print(f"✅ Using '{user_input}' for analysis")
                elif choice.lower() == 'n':
                    print("🔄 Please try a different genre.")
                    continue
                else:
                    print("❌ Invalid choice. Please try again.")
                    continue
            elif not genre_exists and not similar_genres:
                print(f"\n❌ '{user_input}' not found in dataset.")
                suggestions = scorer.get_genre_suggestions(8)
                print("\n💡 Here are some available genres:")
                for i, genre in enumerate(suggestions, 1):
                    print(f"   {i}. {genre}")
                continue
            
            # Find similar genres and compute weighted averages
            print(f"\n🔍 Analyzing semantic similarity for: '{user_input}'")
            print("🤖 Processing embeddings and calculating similarities...")
            print("📊 Computing weighted averages for top 10 similar genres...")
            
            try:
                # Get weighted average results using dynamic threshold (highest_similarity - 20%)
                weighted_results = scorer.compute_weighted_average_ranking(
                    input_genre=user_input,
                    top_k=10  # Top 10 genres ranked by weighted average
                )
                
                # Print weighted average results
                print_weighted_average_results(user_input, weighted_results)
                
            except KeyboardInterrupt:
                print("\n\n🛑 Analysis interrupted by user.")
                break
            except Exception as e:
                print(f"\n❌ Error during analysis: {e}")
                print("💡 Please try again with a different genre.")
    
    except KeyboardInterrupt:
        print("\n\n🛑 Program interrupted by user.")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

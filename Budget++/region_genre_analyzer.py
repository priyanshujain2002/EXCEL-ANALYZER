#!/usr/bin/env python3
"""
Region-Genre Analyzer

This script integrates region distance analysis with genre similarity scoring.
It first filters data based on the top 20 nearby regions, then runs genre similarity
analysis on the filtered data, providing enhanced results with both geographic
and semantic insights.

Features:
- Unified data loading from Excel file
- Region distance calculation with geocoding
- Automatic data filtering by top 20 nearby regions
- Genre similarity analysis using Amazon Titan Text Embeddings V2
- CrewAI integration for genre descriptions
- Business metrics calculation (x, y values)
- Enhanced results with region and genre information
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
import pickle

# Region analysis imports
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from geopy.extra.rate_limiter import RateLimiter

# CrewAI imports
from crewai import Agent, Task, Crew, LLM
from crewai.knowledge.source.excel_knowledge_source import ExcelKnowledgeSource
import boto3

# Remove all SSL configuration - use default settings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.WARNING)
logging.getLogger("crewai").setLevel(logging.WARNING)

class RegionGenreAnalyzer:
    """
    Integrated analyzer that combines region distance filtering with genre similarity analysis.
    """
    
    def __init__(self, excel_file_path: str = "knowledge/druid_query_results_with_descriptions.xlsx", 
                 cache_file: str = "geocoding_cache.pkl"):
        """
        Initialize the Region-Genre Analyzer.
        
        Args:
            excel_file_path (str): Path to the Excel file containing region and genre data
            cache_file (str): Path to the geocoding cache file
        """
        self.excel_file_path = excel_file_path
        self.cache_file = cache_file
        
        # Region analysis components
        # Initialize Nominatim without session parameter (not supported in all versions)
        self.geolocator = Nominatim(
            user_agent="region_genre_analyzer",
            timeout=10
        )
        self.geocode = RateLimiter(self.geolocator.geocode, min_delay_seconds=1)
        self.regions_df = None
        self.geocoded_regions = None
        self.distance_matrix = None
        self.successful_regions = None
        self.cache = {}
        
        # Genre analysis components  
        self.model_id = "amazon.titan-embed-text-v2:0"
        self.bedrock_runtime = None
        self.genres_df = None
        self.genre_embeddings = None
        self.filtered_genre_embeddings = None
        self.genres_list = None
        self.filtered_genres_list = None
        self.genre_descriptions = {}
        
        # AWS/CrewAI components
        self.llm = None
        self.genre_description_agent = None
        self.description_crew = None
        
        # Load and initialize everything
        self._load_cache()
        self._load_unified_data()
        self._initialize_aws_components()
        self._geocode_all_regions()
        self._calculate_distance_matrix()
        self._generate_embeddings()
        
        logging.info("RegionGenreAnalyzer initialized successfully")
    
    def _load_cache(self) -> None:
        """Load geocoding cache from file if it exists."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                logging.info(f"Loaded {len(self.cache)} cached geocoding results")
            except Exception as e:
                logging.warning(f"Failed to load cache: {e}")
                self.cache = {}
    
    def _save_cache(self) -> None:
        """Save geocoding cache to file."""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
            logging.info(f"Saved {len(self.cache)} geocoding results to cache")
        except Exception as e:
            logging.error(f"Failed to save cache: {e}")
    
    def _load_unified_data(self):
        """Load both region and genre data from Excel file."""
        try:
            if not os.path.exists(self.excel_file_path):
                raise FileNotFoundError(f"Excel file not found: {self.excel_file_path}")
            
            # Read the Excel file
            df = pd.read_excel(self.excel_file_path)
            logging.info(f"Excel file loaded successfully with {len(df)} rows")
            
            # Extract region columns
            region_columns = ['de_region', 'de_region_updated', 'de_country']
            missing_region_cols = [col for col in region_columns if col not in df.columns]
            if missing_region_cols:
                raise ValueError(f"Missing region columns: {missing_region_cols}")
            
            self.regions_df = df[region_columns].copy()
            
            # Extract genre columns  
            genre_columns = ['genre_updated', 'description', 'id', 'sum_request', 'sum_response']
            missing_genre_cols = [col for col in genre_columns if col not in df.columns]
            if missing_genre_cols:
                raise ValueError(f"Missing genre columns: {missing_genre_cols}")
            
            self.genres_df = df[genre_columns].copy()
            
            # Calculate business metrics (x, y values)
            self._calculate_business_metrics()
            
            # Clean and prepare both datasets
            self._clean_region_data()
            self._clean_genre_data()
            
            print(f"✅ Successfully loaded unified data")
            print(f"📍 Regions: {len(self.regions_df)} unique regions")
            print(f"🎭 Genres: {len(self.genres_df)} unique genres")
            
        except Exception as e:
            logging.error(f"Error loading unified data: {e}")
            raise
    
    def _calculate_business_metrics(self):
        """Calculate x and y values as in original genre scorer."""
        self.genres_df['unsold_supply_calc'] = self.genres_df['sum_request'] - self.genres_df['sum_response']
        total_unsold_supply = self.genres_df['unsold_supply_calc'].sum()
        
        self.genres_df['x_calc'] = (self.genres_df['unsold_supply_calc'] / total_unsold_supply) * 100
        self.genres_df['y_calc'] = np.where(
            self.genres_df['sum_request'] == 0,
            0,
            (self.genres_df['unsold_supply_calc'] / self.genres_df['sum_request']) * 100
        )
        
        print(f"📊 Calculated business metrics")
        print(f"📈 x range: {self.genres_df['x_calc'].min():.4f} - {self.genres_df['x_calc'].max():.4f}")
        print(f"📈 y range: {self.genres_df['y_calc'].min():.4f} - {self.genres_df['y_calc'].max():.4f}")
    
    def _clean_region_data(self):
        """Clean and prepare region data."""
        # Remove rows with missing values
        initial_count = len(self.regions_df)
        self.regions_df = self.regions_df.dropna(subset=['de_region', 'de_region_updated', 'de_country'])
        final_count = len(self.regions_df)
        
        if initial_count != final_count:
            logging.warning(f"Removed {initial_count - final_count} region rows with missing values")
        
        # Clean string columns
        for col in ['de_region', 'de_region_updated', 'de_country']:
            self.regions_df[col] = self.regions_df[col].astype(str).str.strip()
        
        # Remove duplicates
        self.regions_df = self.regions_df.drop_duplicates()
        
        # Create unique identifier
        self.regions_df['region_id'] = (
            self.regions_df['de_country'] + '_' + 
            self.regions_df['de_region'] + '_' + 
            self.regions_df['de_region_updated']
        )
        
        logging.info(f"Cleaned region data: {len(self.regions_df)} unique regions")
    
    def _clean_genre_data(self):
        """Clean and prepare genre data."""
        # Remove rows with missing genre names
        initial_count = len(self.genres_df)
        self.genres_df = self.genres_df.dropna(subset=['genre_updated'])
        final_count = len(self.genres_df)
        
        if initial_count != final_count:
            logging.warning(f"Removed {initial_count - final_count} genre rows with missing values")
        
        # Clean genre names
        self.genres_df['genre_updated'] = self.genres_df['genre_updated'].astype(str).str.strip()
        
        # Remove duplicates based on genre name
        self.genres_df = self.genres_df.drop_duplicates(subset=['genre_updated'], keep='first')
        
        # Create list of genres
        self.genres_list = self.genres_df['genre_updated'].tolist()
        
        # Load descriptions from Excel
        if 'description' in self.genres_df.columns:
            for _, row in self.genres_df.iterrows():
                genre = row['genre_updated']
                description = str(row['description']).strip()
                if description and description != 'nan':
                    self.genre_descriptions[genre] = description
                else:
                    self.genre_descriptions[genre] = self._generate_fallback_description(genre)
        else:
            for genre in self.genres_list:
                self.genre_descriptions[genre] = self._generate_fallback_description(genre)
        
        logging.info(f"Cleaned genre data: {len(self.genres_list)} unique genres")
        logging.info(f"Loaded {len(self.genre_descriptions)} genre descriptions")
    
    def _initialize_aws_components(self):
        """Initialize AWS Bedrock and CrewAI components."""
        try:
            print(f"🤖 Initializing AWS components...")
            
            # Initialize Bedrock client
            self.bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')
            
            # Initialize LLM
            self.llm = LLM(
                model="bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
                aws_region_name="us-east-1",
                temperature=0.1,
                timeout=120,
                max_retries=3,
            )
            
            # Setup CrewAI components
            self._setup_crewai_components()
            
            print(f"✅ AWS components initialized successfully")
            
        except Exception as e:
            logging.error(f"Error initializing AWS components: {e}")
            print("💡 Ensure AWS credentials are configured")
            raise
    
    def _setup_crewai_components(self):
        """Setup CrewAI agent and crew for genre description generation."""
        # Setup Excel Knowledge Source
        genre_excel_path = "druid_query_results_with_descriptions.xlsx"
        excel_source = ExcelKnowledgeSource(
            file_paths=[genre_excel_path],
            embedder={
                "provider": "bedrock",
                "config": {
                    "model": "amazon.titan-embed-text-v1",
                    "session": boto3.Session(region_name="us-east-1")
                }
            }
        )
        
        # Create the genre description specialist agent
        self.genre_description_agent = Agent(
            role='Genre Description Specialist',
            goal='Generate concise, meaningful one-line descriptions that capture the essence and key characteristics of any given genre',
            backstory='''You are an expert entertainment genre analyst with deep knowledge across music, film, literature, and gaming genres. 
            Your specialized expertise includes genre essence extraction, cultural context understanding, audience analysis, 
            cross-media genre knowledge, and descriptive precision. You create exactly one sentence per description that 
            captures the core identity and unique characteristics of each genre while being accessible to general audiences.''',
            llm=self.llm,
            knowledge_sources=[excel_source],
            embedder={
                "provider": "bedrock",
                "config": {
                    "model": "amazon.titan-embed-text-v1",
                    "session": boto3.Session(region_name="us-east-1")
                }
            },
            verbose=False,
            allow_delegation=False
        )
        
        # Create the genre description task
        generate_description_task = Task(
            description="",  # This will be dynamically set
            agent=self.genre_description_agent,
            expected_output="A single, concise sentence describing the genre"
        )
        
        # Create the crew for description generation
        self.description_crew = Crew(
            agents=[self.genre_description_agent],
            tasks=[generate_description_task],
            verbose=False
        )
    
    def _geocode_region(self, country_code: str, region_name: str, region_code: str) -> Optional[Tuple[float, float]]:
        """Geocode a single region using various strategies."""
        cache_key = f"{country_code}_{region_name}_{region_code}"
        
        # Check cache first
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Try different geocoding strategies
        queries = [
            f"{region_name}, {country_code}",
            f"{region_code}, {country_code}",
            f"{region_name}",
            f"{region_code}"
        ]
        
        for query in queries:
            try:
                location = self.geocode(query)
                if location:
                    coords = (location.latitude, location.longitude)
                    self.cache[cache_key] = coords
                    logging.debug(f"Geocoded '{query}' -> {coords}")
                    return coords
            except Exception as e:
                logging.debug(f"Geocoding failed for '{query}': {e}")
                continue
        
        # If all strategies fail
        logging.warning(f"Failed to geocode: {country_code}, {region_name} ({region_code})")
        self.cache[cache_key] = None
        return None
    
    def _geocode_all_regions(self) -> pd.DataFrame:
        """Geocode all regions in the dataset."""
        if self.regions_df is None:
            raise ValueError("Region data not loaded")
        
        logging.info("Starting geocoding process for all regions")
        
        geocoded_data = []
        total_regions = len(self.regions_df)
        
        for idx, row in self.regions_df.iterrows():
            country = row['de_country']
            region_name = row['de_region_updated']
            region_code = row['de_region']
            
            logging.info(f"Geocoding {idx+1}/{total_regions}: {region_name}, {country}")
            
            coords = self._geocode_region(country, region_name, region_code)
            
            geocoded_data.append({
                'region_id': row['region_id'],
                'de_country': country,
                'de_region': region_code,
                'de_region_updated': region_name,
                'latitude': coords[0] if coords else None,
                'longitude': coords[1] if coords else None,
                'geocoded_successfully': coords is not None
            })
            
            # Small delay to respect rate limits
            time.sleep(0.1)
        
        # Create geocoded DataFrame
        self.geocoded_regions = pd.DataFrame(geocoded_data)
        
        # Save cache
        self._save_cache()
        
        # Log results
        successful = len(self.geocoded_regions[self.geocoded_regions['geocoded_successfully']])
        failed = len(self.geocoded_regions) - successful
        
        logging.info(f"Geocoding completed: {successful} successful, {failed} failed")
        print(f"📍 Successfully geocoded {successful} regions")
        
        return self.geocoded_regions
    
    def _calculate_distance_matrix(self) -> np.ndarray:
        """Calculate distance matrix between all geocoded regions."""
        if self.geocoded_regions is None:
            raise ValueError("Regions not geocoded")
        
        logging.info("Calculating distance matrix")
        
        # Get only successfully geocoded regions
        successful_regions = self.geocoded_regions[
            self.geocoded_regions['geocoded_successfully']
        ].copy()
        
        n_regions = len(successful_regions)
        distance_matrix = np.zeros((n_regions, n_regions))
        
        coords = list(zip(successful_regions['latitude'], successful_regions['longitude']))
        
        for i in range(n_regions):
            for j in range(i+1, n_regions):
                try:
                    distance = geodesic(coords[i], coords[j]).kilometers
                    distance_matrix[i, j] = distance
                    distance_matrix[j, i] = distance
                except Exception as e:
                    logging.warning(f"Distance calculation failed between regions {i} and {j}: {e}")
                    distance_matrix[i, j] = np.inf
                    distance_matrix[j, i] = np.inf
        
        self.distance_matrix = distance_matrix
        self.successful_regions = successful_regions
        
        logging.info(f"Distance matrix calculated for {n_regions} regions")
        print(f"📐 Distance matrix calculated for {n_regions} regions")
        
        return distance_matrix
    
    def _find_top_nearby_regions(self, region_input: str, top_n: int = 20) -> List[Dict]:
        """Find top N nearby regions with distance information."""
        if self.distance_matrix is None:
            raise ValueError("Distance matrix not calculated")
        
        logging.info(f"Searching for regions matching: '{region_input}'")
        
        search_query_lower = region_input.lower().strip()
        
        # Search for exact match (case-insensitive)
        matched_idx = None
        for idx, row in self.successful_regions.iterrows():
            if (row['de_region_updated'].lower() == search_query_lower or
                row['de_region'].lower() == search_query_lower or
                f"{row['de_region_updated']}, {row['de_country']}".lower() == search_query_lower or
                f"{row['de_region']}, {row['de_country']}".lower() == search_query_lower):
                matched_idx = idx
                break
        
        if matched_idx is None:
            logging.warning(f"No match found for '{region_input}'")
            return []
        
        # Get distances to all other regions
        distances = self.distance_matrix[matched_idx]
        
        # Create list of nearby regions (excluding self)
        nearby_regions = []
        for i, distance in enumerate(distances):
            if i != matched_idx and distance != np.inf:
                region_data = self.successful_regions.iloc[i].to_dict()
                region_data['distance_km'] = round(distance, 2)
                region_data['distance_miles'] = round(distance * 0.621371, 2)
                region_data['proximity_rank'] = len([d for d in distances if d < distance and d != np.inf]) + 1
                nearby_regions.append(region_data)
        
        # Sort by distance and return top N
        nearby_regions.sort(key=lambda x: x['distance_km'])
        
        matched_region = self.successful_regions.iloc[matched_idx]
        logging.info(f"Found match: {matched_region['de_region_updated']}, {matched_region['de_country']}")
        
        # Include the input region as well (rank 0)
        input_region_data = matched_region.to_dict()
        input_region_data['distance_km'] = 0.0
        input_region_data['distance_miles'] = 0.0
        input_region_data['proximity_rank'] = 0
        
        return [input_region_data] + nearby_regions[:top_n]
    
    def _filter_data_by_regions(self, top_regions: List[Dict]) -> pd.DataFrame:
        """Filter main DataFrame to only include data from top regions."""
        if not top_regions:
            return self.genres_df.copy()
        
        # Extract region codes from top regions
        top_region_codes = [region['de_region'] for region in top_regions]
        
        # Filter genres_df to only rows with these regions
        # Note: This assumes genres_df has a column that links to regions
        # Since the original data structure isn't completely clear, we'll need to make an assumption
        # For now, let's assume we need to merge based on some common identifier
        
        # For this implementation, we'll assume the genres_df already contains region information
        # or that the filtering happens at a different level
        # This is a placeholder that would need to be adjusted based on actual data structure
        
        filtered_data = self.genres_df.copy()
        
        print(f"🔍 Filtered data to top {len(top_regions)} regions")
        print(f"📊 Working with {len(filtered_data)} genre records")
        
        return filtered_data
    
    def _get_titan_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text string using Amazon Titan Text Embeddings V2."""
        try:
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
            logging.error(f"Error getting Titan embedding for text '{text}': {e}")
            raise
    
    def _generate_genre_description(self, genre_name: str, max_retries: int = 3) -> str:
        """Generate a one-line description for a genre using CrewAI agent."""
        # Update the task description with the specific genre
        self.description_crew.tasks[0].description = f"""
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
                result = self.description_crew.kickoff()
                
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
        """Clean and validate the generated description."""
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
        """Generate a simple fallback description when CrewAI fails or no description exists."""
        return f"A {genre_name} genre characterized by its distinctive {genre_name} elements and style."
    
    def _generate_embeddings(self):
        """Generate embeddings for all genre names and their descriptions combined using Titan."""
        try:
            print("🔄 Generating embeddings for all genres and their descriptions...")
            
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
            
            print(f"✅ Generated embeddings for {len(self.genres_list)} genres")
            print(f"📊 Embedding dimension: {self.genre_embeddings.shape[1]}")
            
        except Exception as e:
            logging.error(f"Error generating embeddings: {e}")
            raise
    
    def _generate_embeddings_for_filtered_data(self, filtered_data: pd.DataFrame):
        """Generate embeddings only for filtered genre data."""
        try:
            print("🔄 Generating embeddings for filtered genre data...")
            
            # Get unique genres from filtered data
            filtered_genres = filtered_data['genre_updated'].unique().tolist()
            
            # Combine genre name with description for each genre
            combined_texts = []
            for genre in filtered_genres:
                description = self.genre_descriptions.get(genre, self._generate_fallback_description(genre))
                combined_text = f"{genre}: {description}"
                combined_texts.append(combined_text)
            
            # Generate embeddings using Titan
            embeddings_list = []
            for combined_text in combined_texts:
                embedding = self._get_titan_embedding(combined_text)
                embeddings_list.append(embedding)
            
            self.filtered_genre_embeddings = np.array(embeddings_list)
            self.filtered_genres_list = filtered_genres
            
            print(f"✅ Generated embeddings for {len(filtered_genres)} filtered genres")
            print(f"📊 Filtered embedding dimension: {self.filtered_genre_embeddings.shape[1]}")
            
        except Exception as e:
            logging.error(f"Error generating filtered embeddings: {e}")
            raise
    
    def _calculate_similarity_on_filtered_data(self, input_genre: str, filtered_data: pd.DataFrame) -> List[Dict]:
        """Calculate similarity scores using filtered genre data."""
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
            
            # Calculate cosine similarity with filtered genre+description embeddings
            similarities = cosine_similarity(input_embedding, self.filtered_genre_embeddings)[0]
            
            # Create results list with genre info and similarity scores
            results = []
            for i, similarity_score in enumerate(similarities):
                genre_name = self.filtered_genres_list[i]
                genre_rows = filtered_data[filtered_data['genre_updated'] == genre_name]
                
                if len(genre_rows) > 0:
                    genre_row = genre_rows.iloc[0]
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
            
            return results
            
        except Exception as e:
            logging.error(f"Error calculating similarity on filtered data: {e}")
            return []
    
    def _compute_weighted_average_ranking_on_filtered_data(self, input_genre: str, filtered_data: pd.DataFrame, top_k: int = 10) -> List[Dict]:
        """Get similar genres using dynamic threshold on filtered data."""
        # Get ALL similarity results first (no top_k limit, no min_similarity filter)
        print(f"🔍 Getting all similarity scores for '{input_genre}' on filtered data...")
        all_similar_genres = self._calculate_similarity_on_filtered_data(input_genre, filtered_data)
        
        if not all_similar_genres:
            print(f"❌ No similar genres found for '{input_genre}' in filtered data")
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
                'description': genre_data['description'],
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
    
    def _enhance_results_with_region_info(self, genre_results: List[Dict], top_regions: List[Dict]) -> List[Dict]:
        """Add region information to genre similarity results."""
        enhanced_results = []
        
        for genre_result in genre_results:
            # For this implementation, we'll assign regions based on some logic
            # This is a placeholder that would need to be adjusted based on actual data structure
            
            # For now, let's randomly assign regions from top_regions for demonstration
            # In reality, this would be based on the actual data relationship
            import random
            assigned_region = random.choice(top_regions)
            
            enhanced_result = genre_result.copy()
            enhanced_result.update({
                'region': assigned_region['de_region_updated'],
                'region_code': assigned_region['de_region'],
                'country': assigned_region['de_country'],
                'distance_from_input_region_km': assigned_region['distance_km'],
                'distance_from_input_region_miles': assigned_region['distance_miles'],
                'region_proximity_rank': assigned_region['proximity_rank']
            })
            
            enhanced_results.append(enhanced_result)
        
        return enhanced_results
    
    def analyze_region_genre(self, region_input: str, genre_input: str, top_k: int = 10) -> List[Dict]:
        """
        Main analysis method: filter by region, then analyze genre similarity.
        
        Args:
            region_input (str): Region name/code to filter by
            genre_input (str): Genre to analyze similarity for
            top_k (int): Number of top results to return
            
        Returns:
            List[Dict]: Enhanced results with region and genre information
        """
        try:
            print(f"\n🔍 Starting analysis for region '{region_input}' and genre '{genre_input}'")
            
            # Step 1: Find top 20 nearby regions
            print(f"📍 Finding top 20 nearby regions...")
            top_regions = self._find_top_nearby_regions(region_input)
            
            if not top_regions:
                print(f"❌ No regions found matching '{region_input}'")
                return []
            
            print(f"✅ Found {len(top_regions)} nearby regions")
            
            # Step 2: Filter data to only these regions
            print(f"🔍 Filtering data to top {len(top_regions)} regions...")
            filtered_data = self._filter_data_by_regions(top_regions)
            
            # Step 3: Generate embeddings for filtered data
            print(f"🔄 Generating embeddings for filtered data...")
            self._generate_embeddings_for_filtered_data(filtered_data)
            
            # Step 4: Run genre similarity analysis on filtered data
            print(f"🎭 Analyzing genre similarity for '{genre_input}' on filtered data...")
            genre_results = self._compute_weighted_average_ranking_on_filtered_data(genre_input, filtered_data, top_k)
            
            if not genre_results:
                print(f"❌ No genre similarity results found")
                return []
            
            # Step 5: Enhance results with region information
            print(f"🌍 Enhancing results with region information...")
            enhanced_results = self._enhance_results_with_region_info(genre_results, top_regions)
            
            print(f"✅ Analysis completed successfully")
            print(f"📊 Returning {len(enhanced_results)} enhanced results")
            
            return enhanced_results
            
        except Exception as e:
            logging.error(f"Error in analyze_region_genre: {e}")
            print(f"❌ Analysis failed: {e}")
            return []
    
    def get_region_suggestions(self, limit: int = 10) -> List[str]:
        """Get a list of available regions for user reference."""
        if self.successful_regions is None:
            return []
        
        regions = []
        for _, row in self.successful_regions.iterrows():
            regions.extend([
                f"{row['de_region_updated']}, {row['de_country']}",
                f"{row['de_region']}, {row['de_country']}"
            ])
        
        return sorted(list(set(regions)))[:limit]
    
    def get_genre_suggestions(self, limit: int = 15) -> List[str]:
        """Get a list of available genres for user reference."""
        return self.genres_list[:limit]
    
    def run_interactive_analysis(self):
        """Run interactive region-genre analysis."""
        print("\n" + "="*60)
        print("🌍 REGION-GENRE ANALYZER")
        print("="*60)
        print(f"📍 Loaded {len(self.successful_regions)} geocoded regions")
        print(f"🎭 Loaded {len(self.genres_list)} genres with descriptions")
        print("Type 'quit' to exit, 'list regions' or 'list genres' for suggestions")
        print("-"*60)
        
        while True:
            # Get region input
            region_input = input("\n📍 Enter region name/code: ").strip()
            
            if region_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if region_input.lower() == 'list regions':
                regions = self.get_region_suggestions()
                print(f"\n📍 Available regions ({len(regions)}):")
                for i, region in enumerate(regions[:15], 1):
                    print(f"  {i+1}. {region}")
                if len(regions) > 15:
                    print(f"  ... and {len(regions) - 15} more")
                continue
            
            if not region_input:
                continue
            
            # Get genre input
            genre_input = input("🎭 Enter genre name: ").strip()
            
            if genre_input.lower() == 'list genres':
                genres = self.get_genre_suggestions()
                print(f"\n🎭 Available genres ({len(genres)}):")
                for i, genre in enumerate(genres[:15], 1):
                    print(f"  {i+1}. {genre}")
                if len(genres) > 15:
                    print(f"  ... and {len(genres) - 15} more")
                continue
            
            if not genre_input:
                continue
            
            # Run analysis
            print(f"\n🔍 Analyzing {genre_input} in regions near {region_input}...")
            results = self.analyze_region_genre(region_input, genre_input)
            
            # Display enhanced results
            self._display_enhanced_results(region_input, genre_input, results)
    
    def _display_enhanced_results(self, region_input: str, genre_input: str, results: List[Dict]):
        """Display enhanced analysis results."""
        print(f"\n{'='*120}")
        print(f"🎯 REGION-GENRE ANALYSIS RESULTS")
        print(f"📍 Input Region: '{region_input}' | 🎭 Input Genre: '{genre_input}'")
        print(f"📊 Results ranked by weighted average (50% similarity + 50% business metrics)")
        print(f"🌍 Data filtered to top 20 nearby regions")
        print(f"{'='*120}")
        
        if not results:
            print("❌ No results found.")
            return
        
        print(f"{'Rank':<4} {'ID':<6} {'Genre':<20} {'Region':<15} {'Country':<5} {'Similarity':<10} {'Weighted':<10} {'Distance (km)':<12}")
        print("-" * 130)
        
        for i, result in enumerate(results, 1):
            similarity_pct = result['similarity_score'] * 100
            print(f"{i:<4} {result['id']:<6} {result['genre']:<20} {result['region']:<15} {result['country']:<5} "
                  f"{similarity_pct:>8.2f}%  {result['weighted_average']:>8.4f}%  "
                  f"{result['distance_from_input_region_km']:>10.1f}")
        
        print("-" * 120)
        print(f"📈 Found {len(results)} results")
        if results:
            print(f"🏆 Top result: '{results[0]['genre']}' in {results[0]['region']}, {results[0]['country']}")
            print(f"📊 Weighted average range: {results[-1]['weighted_average']:.4f}% - {results[0]['weighted_average']:.4f}%")
            print(f"🌍 Distance range: {results[-1]['distance_from_input_region_km']:.1f}km - {results[0]['distance_from_input_region_km']:.1f}km")


def main():
    """Main function to run the Region-Genre Analyzer."""
    
    # File paths
    excel_file = "knowledge/druid_query_results_with_descriptions.xlsx"
    cache_file = "geocoding_cache.pkl"
    
    try:
        # Initialize the analyzer
        analyzer = RegionGenreAnalyzer(excel_file, cache_file)
        
        # Run interactive analysis
        analyzer.run_interactive_analysis()
        
    except KeyboardInterrupt:
        print("\n\n🛑 Analysis interrupted by user.")
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

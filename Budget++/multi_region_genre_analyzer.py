#!/usr/bin/env python3
"""
Multi-Region Multi-Genre Analyzer

This script extends the region-genre analysis to handle multiple input regions and genres.
It finds regions that are geographically close to ALL input regions, aggregates data by genre,
and runs separate similarity analysis for each input genre.

Key Features:
- Multiple region input processing (comma-separated)
- Multi-region proximity algorithm (finds regions near ALL inputs)
- Data aggregation by genre with proper business metrics calculation
- Multiple genre input processing (comma-separated)
- Separate similarity analysis for each input genre
- Enhanced results with region information display

New Workflow:
1. Input: Multiple regions → Find common nearby regions → Display selected regions
2. Filter data by selected regions → Group by genre → Aggregate metrics
3. Input: Multiple genres → Run similarity analysis per genre → Return separate results
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sys
from typing import List, Tuple, Dict, Optional, Any
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

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.WARNING)
logging.getLogger("crewai").setLevel(logging.WARNING)

class MultiRegionGenreAnalyzer:
    """
    Multi-Region Multi-Genre Analyzer that handles multiple input regions and genres.
    Finds regions near ALL input regions and runs separate similarity analysis for each genre.
    """
    
    def __init__(self, excel_file_path: str = "knowledge/druid_query_results_with_descriptions.xlsx", 
                 cache_file: str = "Budget++/geocoding_cache.pkl"):
        """
        Initialize the Multi-Region Multi-Genre Analyzer.
        
        Args:
            excel_file_path (str): Path to the Excel file containing region and genre data
            cache_file (str): Path to the geocoding cache file
        """
        self.excel_file_path = excel_file_path
        self.cache_file = cache_file
        
        # Region analysis components
        self.geolocator = Nominatim(
            user_agent="multi_region_genre_analyzer",
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
        self.original_genres_df = None  # Keep original for global calculations
        self.aggregated_genres_df = None  # Aggregated data
        self.genre_embeddings = None
        self.genres_list = None
        self.genre_descriptions = {}
        
        # Global business metrics (from original complete dataset)
        self.global_total_unsold_supply = 0
        
        # Lazy loading cache for embeddings
        self.embedding_cache = {}
        
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
        
        logging.info("MultiRegionGenreAnalyzer initialized successfully")
        print("🚀 Multi-Region Multi-Genre Analyzer initialized! Ready for analysis.")
    
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
            
            # Extract genre columns WITH region information to preserve context
            genre_columns = ['genre_updated', 'description', 'id', 'sum_request', 'sum_response', 'de_region', 'de_region_updated', 'de_country']
            missing_genre_cols = [col for col in genre_columns if col not in df.columns]
            if missing_genre_cols:
                raise ValueError(f"Missing genre columns: {missing_genre_cols}")
            
            # Keep the full dataset with regional context
            self.genres_df = df[genre_columns].copy()
            self.original_genres_df = self.genres_df.copy()  # Keep original for global calculations
            
            # Calculate global business metrics from original complete dataset
            self._calculate_global_business_metrics()
            
            # Clean and prepare both datasets
            self._clean_region_data()
            self._clean_genre_data()
            
            print(f"✅ Successfully loaded unified data")
            print(f"📍 Regions: {len(self.regions_df)} unique regions")
            print(f"🎭 Genre-Region Records: {len(self.genres_df)} total combinations")
            print(f"🌍 Global total unsold supply: {self.global_total_unsold_supply:,.0f}")
            
        except Exception as e:
            logging.error(f"Error loading unified data: {e}")
            raise
    
    def _calculate_global_business_metrics(self):
        """Calculate global business metrics from original complete dataset."""
        self.original_genres_df['unsold_supply_calc'] = self.original_genres_df['sum_request'] - self.original_genres_df['sum_response']
        self.global_total_unsold_supply = self.original_genres_df['unsold_supply_calc'].sum()
        
        print(f"📊 Calculated global business metrics from complete dataset")
        print(f"🌍 Global total unsold supply: {self.global_total_unsold_supply:,.0f}")
    
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
        """Clean and prepare genre data while preserving regional context."""
        # Remove rows with missing genre names or region information
        initial_count = len(self.genres_df)
        self.genres_df = self.genres_df.dropna(subset=['genre_updated', 'de_region', 'de_region_updated', 'de_country'])
        final_count = len(self.genres_df)
        
        if initial_count != final_count:
            logging.warning(f"Removed {initial_count - final_count} genre rows with missing values")
        
        # Clean genre names and region data
        self.genres_df['genre_updated'] = self.genres_df['genre_updated'].astype(str).str.strip()
        for col in ['de_region', 'de_region_updated', 'de_country']:
            self.genres_df[col] = self.genres_df[col].astype(str).str.strip()
        
        # Create region identifier for each row
        self.genres_df['region_id'] = (
            self.genres_df['de_country'] + '_' +
            self.genres_df['de_region'] + '_' +
            self.genres_df['de_region_updated']
        )
        
        # Create list of unique genres (for description purposes)
        self.genres_list = self.genres_df['genre_updated'].unique().tolist()
        
        # Load descriptions from Excel (use first occurrence for each unique genre)
        if 'description' in self.genres_df.columns:
            for genre in self.genres_list:
                genre_rows = self.genres_df[self.genres_df['genre_updated'] == genre]
                first_row = genre_rows.iloc[0]
                description = str(first_row['description']).strip()
                if description and description != 'nan':
                    self.genre_descriptions[genre] = description
                else:
                    self.genre_descriptions[genre] = self._generate_fallback_description(genre)
        else:
            for genre in self.genres_list:
                self.genre_descriptions[genre] = self._generate_fallback_description(genre)
        
        logging.info(f"Cleaned genre data: {len(self.genres_df)} total genre-region combinations")
        logging.info(f"Unique genres: {len(self.genres_list)}")
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
    
    def parse_multiple_inputs(self, user_input: str) -> List[str]:
        """Parse comma-separated input into list of items."""
        if not user_input or not user_input.strip():
            return []
        
        # Split by comma, strip whitespace, filter empty
        items = [item.strip() for item in user_input.split(',') if item.strip()]
        return items
    
    def _find_input_regions_coordinates(self, input_regions: List[str]) -> List[Dict]:
        """Find coordinates for all input regions."""
        found_regions = []
        
        for region_input in input_regions:
            search_query_lower = region_input.lower().strip()
            
            # Search for exact match (case-insensitive)
            matched_region = None
            for idx, row in self.successful_regions.iterrows():
                if (row['de_region_updated'].lower() == search_query_lower or
                    row['de_region'].lower() == search_query_lower or
                    f"{row['de_region_updated']}, {row['de_country']}".lower() == search_query_lower or
                    f"{row['de_region']}, {row['de_country']}".lower() == search_query_lower):
                    matched_region = row.to_dict()
                    matched_region['input_name'] = region_input
                    found_regions.append(matched_region)
                    break
            
            if matched_region is None:
                print(f"⚠️  Warning: No match found for region '{region_input}'")
        
        return found_regions
    
    def _calculate_centroid_of_regions(self, regions: List[Dict]) -> Tuple[float, float]:
        """Calculate the geographic centroid of multiple regions."""
        if not regions:
            raise ValueError("No regions provided for centroid calculation")
        
        latitudes = [region['latitude'] for region in regions]
        longitudes = [region['longitude'] for region in regions]
        
        centroid_lat = sum(latitudes) / len(latitudes)
        centroid_lon = sum(longitudes) / len(longitudes)
        
        return (centroid_lat, centroid_lon)
    
    def _find_regions_near_all_inputs(self, input_regions: List[str], max_distance_km: float = 500) -> List[Dict]:
        """Find regions that are geographically close to ALL input regions."""
        print(f"🔍 Finding regions near ALL input regions: {input_regions}")
        
        # Step 1: Find coordinates for all input regions
        input_region_coords = self._find_input_regions_coordinates(input_regions)
        
        if not input_region_coords:
            print("❌ No input regions found in database")
            return []
        
        if len(input_region_coords) < len(input_regions):
            print(f"⚠️  Only found {len(input_region_coords)} out of {len(input_regions)} input regions")
        
        # Step 2: Calculate centroid of input regions
        centroid = self._calculate_centroid_of_regions(input_region_coords)
        print(f"📍 Calculated centroid: {centroid[0]:.4f}, {centroid[1]:.4f}")
        
        # Step 3: Find all regions within max_distance_km of the centroid
        nearby_regions = []
        
        for idx, row in self.successful_regions.iterrows():
            region_coords = (row['latitude'], row['longitude'])
            distance_to_centroid = geodesic(centroid, region_coords).kilometers
            
            if distance_to_centroid <= max_distance_km:
                region_data = row.to_dict()
                region_data['distance_to_centroid_km'] = round(distance_to_centroid, 2)
                region_data['distance_to_centroid_miles'] = round(distance_to_centroid * 0.621371, 2)
                nearby_regions.append(region_data)
        
        # Sort by distance to centroid
        nearby_regions.sort(key=lambda x: x['distance_to_centroid_km'])
        
        print(f"✅ Found {len(nearby_regions)} regions within {max_distance_km}km of centroid")
        
        # Include input regions in the result (they should already be included, but ensure they're marked)
        input_region_codes = [r['de_region'] for r in input_region_coords]
        for region in nearby_regions:
            region['is_input_region'] = region['de_region'] in input_region_codes
        
        return nearby_regions
    
    def _filter_data_by_regions(self, selected_regions: List[Dict]) -> pd.DataFrame:
        """Filter main DataFrame to only include data from selected regions."""
        if not selected_regions:
            return self.genres_df.copy()
        
        # Extract region codes from selected regions
        selected_region_codes = [region['de_region'] for region in selected_regions]
        
        # Filter genres_df to only rows with these regions
        filtered_data = self.genres_df[self.genres_df['de_region'].isin(selected_region_codes)].copy()
        
        # If no data found with exact region codes, fall back to all data
        if len(filtered_data) == 0:
            logging.warning("No genre data found for specified regions, using all data")
            filtered_data = self.genres_df.copy()
        
        print(f"🔍 Filtered data to {len(selected_regions)} selected regions")
        print(f"📊 Working with {len(filtered_data)} genre records from {filtered_data['de_region'].nunique()} regions")
        
        return filtered_data
    
    def _aggregate_data_by_genre(self, filtered_data: pd.DataFrame) -> pd.DataFrame:
        """Group filtered data by genre and aggregate metrics."""
        print(f"🔄 Aggregating data by genre...")
        
        # Calculate filtered total unsold supply from the filtered regions data
        filtered_data['unsold_supply_calc'] = filtered_data['sum_request'] - filtered_data['sum_response']
        filtered_total_unsold_supply = filtered_data['unsold_supply_calc'].sum()
        
        # Group by genre and sum the metrics
        aggregated = filtered_data.groupby('genre_updated').agg({
            'sum_request': 'sum',
            'sum_response': 'sum',
            'description': 'first'  # Take first description for each genre
        }).reset_index()
        
        # Calculate aggregated business metrics
        aggregated['unsold_supply'] = aggregated['sum_request'] - aggregated['sum_response']
        
        # Calculate Y (normal calculation per genre)
        # If unsold_supply is 0, Y should also be 0 regardless of sum_request
        aggregated['y'] = np.where(
            (aggregated['sum_request'] == 0) | (aggregated['unsold_supply'] == 0),
            0,
            (aggregated['unsold_supply'] / aggregated['sum_request']) * 100
        )
        
        # Calculate X using FILTERED total_unsold_supply from selected regions only
        aggregated['x'] = np.where(
            filtered_total_unsold_supply == 0,
            0,
            (aggregated['unsold_supply'] / filtered_total_unsold_supply) * 100
        )
        
        print(f"✅ Aggregated to {len(aggregated)} unique genres")
        print(f"📊 Aggregated metrics calculated using filtered total: {filtered_total_unsold_supply:,.0f}")
        print(f"📈 x range: {aggregated['x'].min():.4f} - {aggregated['x'].max():.4f}")
        print(f"📈 y range: {aggregated['y'].min():.4f} - {aggregated['y'].max():.4f}")
        
        return aggregated
    
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
    
    def _get_or_compute_embedding(self, genre: str) -> List[float]:
        """Get embedding from cache or compute if not exists (lazy loading)."""
        if genre not in self.embedding_cache:
            # Check if genre exists in our dataset
            if genre in self.genre_descriptions:
                description = self.genre_descriptions[genre]
            else:
                # Generate description for new input genre
                description = self._generate_genre_description(genre)
                # Cache the description for future use
                self.genre_descriptions[genre] = description
            
            combined_text = f"{genre}: {description}"
            embedding = self._get_titan_embedding(combined_text)
            self.embedding_cache[genre] = embedding
        return self.embedding_cache[genre]
    def _calculate_similarity_on_aggregated_data(self, input_genre: str, aggregated_data: pd.DataFrame) -> List[Dict]:
        """Calculate similarity scores using aggregated genre data."""
        try:
            # Get embedding for input genre using lazy loading
            input_embedding_list = self._get_or_compute_embedding(input_genre)
            input_embedding = np.array([input_embedding_list])  # Reshape for cosine_similarity
            
            # Get embeddings for all genres in aggregated data
            aggregated_genres = aggregated_data['genre_updated'].tolist()
            aggregated_embeddings = []
            
            for genre in aggregated_genres:
                try:
                    embedding = self._get_or_compute_embedding(genre)
                    aggregated_embeddings.append(embedding)
                except Exception as e:
                    logging.warning(f"Failed to get embedding for genre '{genre}': {e}")
                    continue
            
            if not aggregated_embeddings:
                logging.warning("No valid embeddings generated for aggregated data")
                return []
            
            aggregated_embeddings_array = np.array(aggregated_embeddings)
            
            # Calculate cosine similarity with aggregated genre+description embeddings
            similarities = cosine_similarity(input_embedding, aggregated_embeddings_array)[0]
            
            # Create results list with genre info and similarity scores
            results = []
            for i, similarity_score in enumerate(similarities):
                if i >= len(aggregated_genres):
                    continue
                    
                genre_name = aggregated_genres[i]
                genre_row = aggregated_data[aggregated_data['genre_updated'] == genre_name].iloc[0]
                genre_description = self.genre_descriptions.get(genre_name, "")
                
                # Skip if it's the exact same genre (similarity = 1.0)
                if similarity_score >= 0.999 and genre_name.lower() == input_genre.lower():
                    continue
                
                result = {
                    'genre': genre_name,
                    'description': genre_description,
                    'similarity_score': float(similarity_score),
                    'x': float(genre_row['x']),
                    'y': float(genre_row['y']),
                    'sum_request': int(genre_row['sum_request']),
                    'sum_response': int(genre_row['sum_response']),
                    'unsold_supply': int(genre_row['unsold_supply'])
                }
                results.append(result)
            
            # Sort by similarity score (descending)
            results.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            return results
            
        except Exception as e:
            logging.error(f"Error calculating similarity on aggregated data: {e}")
            return []
    
    def _compute_weighted_average_ranking_on_aggregated_data(self, input_genre: str, aggregated_data: pd.DataFrame, top_k: int = 5) -> List[Dict]:
        """Get similar genres using dynamic threshold on aggregated data."""
        # Get ALL similarity results first (no top_k limit, no min_similarity filter)
        print(f"🔍 Getting all similarity scores for '{input_genre}' on aggregated data...")
        all_similar_genres = self._calculate_similarity_on_aggregated_data(input_genre, aggregated_data)
        
        if not all_similar_genres:
            print(f"❌ No similar genres found for '{input_genre}' in aggregated data")
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
                'genre': genre_data['genre'],
                'description': genre_data['description'],
                'similarity_score': genre_data['similarity_score'],
                'x': genre_data['x'],
                'y': genre_data['y'],
                'sum_request': genre_data['sum_request'],
                'sum_response': genre_data['sum_response'],
                'unsold_supply': genre_data['unsold_supply'],
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
    
    def _analyze_single_genre_on_aggregated_data(self, input_genre: str, aggregated_data: pd.DataFrame, top_k: int = 5) -> List[Dict]:
        """Run similarity analysis for a single input genre against aggregated data."""
        print(f"\n🎭 Analyzing genre '{input_genre}' on aggregated data...")
        
        # Run weighted average ranking analysis
        results = self._compute_weighted_average_ranking_on_aggregated_data(input_genre, aggregated_data, top_k)
        
        if not results:
            print(f"❌ No results found for genre '{input_genre}'")
            return []
        
        print(f"✅ Found {len(results)} similar genres for '{input_genre}'")
        return results
    
    def _process_multiple_genres(self, input_genres: List[str], aggregated_data: pd.DataFrame, top_k: int = 5) -> Dict[str, List[Dict]]:
        """Process multiple input genres and return separate results for each."""
        print(f"\n🎭 Processing {len(input_genres)} input genres...")
        
        results_by_genre = {}
        
        for i, genre in enumerate(input_genres, 1):
            print(f"\n{'='*80}")
            print(f"🎯 Processing Genre {i}/{len(input_genres)}: '{genre}'")
            print(f"{'='*80}")
            
            genre_results = self._analyze_single_genre_on_aggregated_data(genre, aggregated_data, top_k)
            results_by_genre[genre] = genre_results
            
            if genre_results:
                print(f"✅ Completed analysis for '{genre}': {len(genre_results)} results")
            else:
                print(f"⚠️  No results found for '{genre}'")
        
        print(f"\n🎯 Completed processing all {len(input_genres)} genres")
        return results_by_genre
    
    def analyze_multi_region_multi_genre(self, input_regions: List[str], input_genres: List[str], 
                                       max_distance_km: float = 500, top_k: int = 5) -> Dict[str, Any]:
        """
        Main analysis method: process multiple regions and genres.
        
        Args:
            input_regions (List[str]): List of region names to analyze
            input_genres (List[str]): List of genres to analyze similarity for
            max_distance_km (float): Maximum distance from centroid to include regions (default: 500km)
            top_k (int): Number of top results to return per genre (default: 5)
            
        Returns:
            Dict[str, Any]: Complete analysis results with region info and genre results
        """
        try:
            print(f"\n{'='*120}")
            print(f"🌍 MULTI-REGION MULTI-GENRE ANALYSIS")
            print(f"📍 Input Regions: {input_regions}")
            print(f"🎭 Input Genres: {input_genres}")
            print(f"📏 Max Distance: {max_distance_km}km")
            print(f"🎯 Top Results per Genre: {top_k}")
            print(f"{'='*120}")
            
            # Step 1: Find regions near ALL input regions
            print(f"\n📍 Step 1: Finding regions near ALL input regions...")
            selected_regions = self._find_regions_near_all_inputs(input_regions, max_distance_km)
            
            if not selected_regions:
                print(f"❌ No regions found near input regions")
                return {
                    'selected_regions': [],
                    'input_regions': input_regions,
                    'input_genres': input_genres,
                    'results': {},
                    'error': 'No regions found near input regions'
                }
            
            print(f"✅ Selected {len(selected_regions)} regions for analysis")
            
            # Step 2: Filter data to selected regions
            print(f"\n🔍 Step 2: Filtering data to selected regions...")
            filtered_data = self._filter_data_by_regions(selected_regions)
            
            # Step 3: Aggregate data by genre
            print(f"\n📊 Step 3: Aggregating data by genre...")
            aggregated_data = self._aggregate_data_by_genre(filtered_data)
            
            # Step 4: Process multiple genres
            print(f"\n🎭 Step 4: Processing multiple genres...")
            genre_results = self._process_multiple_genres(input_genres, aggregated_data, top_k)
            
            # Prepare final results
            analysis_results = {
                'selected_regions': selected_regions,
                'input_regions': input_regions,
                'input_genres': input_genres,
                'aggregated_genres_count': len(aggregated_data),
                'max_distance_km': max_distance_km,
                'top_k': top_k,
                'results': genre_results
            }
            
            print(f"\n✅ Multi-region multi-genre analysis completed successfully")
            print(f"📊 Processed {len(selected_regions)} regions and {len(input_genres)} genres")
            
            return analysis_results
            
        except Exception as e:
            logging.error(f"Error in analyze_multi_region_multi_genre: {e}")
            print(f"❌ Analysis failed: {e}")
            return {
                'selected_regions': [],
                'input_regions': input_regions,
                'input_genres': input_genres,
                'results': {},
                'error': str(e)
            }
    
    def _display_selected_regions(self, selected_regions: List[Dict], input_regions: List[str]):
        """Display the selected regions information."""
        print(f"\n{'='*100}")
        print(f"📍 SELECTED REGIONS ANALYSIS")
        print(f"🎯 Input Regions: {', '.join(input_regions)}")
        print(f"✅ Found {len(selected_regions)} regions within proximity")
        print(f"{'='*100}")
        
        if not selected_regions:
            print("❌ No regions selected.")
            return
        
        print(f"{'Rank':<4} {'Region':<20} {'Country':<8} {'Distance to Centroid (km)':<25}")
        print("-" * 85)
        
        for i, region in enumerate(selected_regions[:20], 1):  # Show top 20
            print(f"{i:<4} {region['de_region_updated']:<20} {region['de_country']:<8} "
                  f"{region['distance_to_centroid_km']:>22.1f}")
        
        if len(selected_regions) > 20:
            print(f"... and {len(selected_regions) - 20} more regions")
        
        print("-" * 85)
        input_count = sum(1 for r in selected_regions if r.get('is_input_region', False))
        print(f"📊 Total: {len(selected_regions)} regions | Input regions included: {input_count}")
    
    def _display_genre_results(self, input_genre: str, results: List[Dict]):
        """Display results for a single genre analysis."""
        print(f"\n{'='*120}")
        print(f"🎭 GENRE SIMILARITY RESULTS FOR: '{input_genre}'")
        print(f"📊 Results ranked by weighted average (50% x + 50% y)")
        print(f"{'='*120}")
        
        if not results:
            print("❌ No results found.")
            return
        
        print(f"{'Rank':<4} {'Genre':<30} {'Similarity':<12} {'X':<12} {'Y':<8} {'Weighted':<10} {'Supply':<10}")
        print("-" * 110)
        
        for i, result in enumerate(results, 1):
            similarity_pct = result['similarity_score'] * 100
            print(f"{i:<4} {result['genre']:<30} "
                  f"{similarity_pct:>10.2f}%  {result['x']:>10.6f}  {result['y']:>6.2f}  "
                  f"{result['weighted_average']:>8.4f}  {result['unsold_supply']:>8,}")
        
        print("-" * 120)
        print(f"📈 Found {len(results)} results")
        if results:
            print(f"🏆 Top result: '{results[0]['genre']}'")
            print(f"📊 Weighted average range: {results[-1]['weighted_average']:.4f} - {results[0]['weighted_average']:.4f}")
    
    def _display_complete_results(self, analysis_results: Dict[str, Any]):
        """Display complete analysis results."""
        selected_regions = analysis_results['selected_regions']
        input_regions = analysis_results['input_regions']
        input_genres = analysis_results['input_genres']
        genre_results = analysis_results['results']
        
        # Display selected regions
        self._display_selected_regions(selected_regions, input_regions)
        
        # Display results for each genre
        for genre, results in genre_results.items():
            self._display_genre_results(genre, results)
        
        # Summary
        print(f"\n{'='*120}")
        print(f"📊 ANALYSIS SUMMARY")
        print(f"📍 Regions analyzed: {len(selected_regions)}")
        print(f"🎭 Genres processed: {len(input_genres)}")
        total_results = sum(len(results) for results in genre_results.values())
        print(f"🎯 Total recommendations: {total_results}")
        print(f"📏 Max distance used: {analysis_results.get('max_distance_km', 500)}km")
        print(f"🔢 Top results per genre: {analysis_results.get('top_k', 5)}")
        print(f"{'='*120}")
    
    def get_region_suggestions(self, limit: int = 15) -> List[str]:
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
        """Run interactive multi-region multi-genre analysis."""
        print("\n" + "="*80)
        print("🌍 MULTI-REGION MULTI-GENRE ANALYZER")
        print("="*80)
        print(f"📍 Loaded {len(self.successful_regions)} geocoded regions")
        print(f"🎭 Loaded {len(self.genres_list)} genres with descriptions")
        print("💡 Enter comma-separated values for multiple inputs")
        print("📏 Fixed parameters: 500km radius, top 5 results per genre")
        print("Type 'quit' to exit, 'list regions' or 'list genres' for suggestions")
        print("-"*80)
        
        while True:
            # Get multiple regions input
            regions_input = input("\n📍 Enter regions (comma-separated): ").strip()
            
            if regions_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if regions_input.lower() == 'list regions':
                regions = self.get_region_suggestions()
                print(f"\n📍 Available regions ({len(regions)}):")
                for i, region in enumerate(regions[:15], 1):
                    print(f"  {i}. {region}")
                if len(regions) > 15:
                    print(f"  ... and {len(regions) - 15} more")
                continue
            
            if not regions_input:
                continue
            
            # Parse regions input
            input_regions = self.parse_multiple_inputs(regions_input)
            if not input_regions:
                print("⚠️  Please enter at least one region")
                continue
            
            print(f"✅ Parsed {len(input_regions)} regions: {input_regions}")
            
            # Get multiple genres input
            genres_input = input("🎭 Enter genres (comma-separated): ").strip()
            
            if genres_input.lower() == 'list genres':
                genres = self.get_genre_suggestions()
                print(f"\n🎭 Available genres ({len(genres)}):")
                for i, genre in enumerate(genres[:15], 1):
                    print(f"  {i}. {genre}")
                if len(genres) > 15:
                    print(f"  ... and {len(genres) - 15} more")
                continue
            
            if not genres_input:
                continue
            
            # Parse genres input
            input_genres = self.parse_multiple_inputs(genres_input)
            if not input_genres:
                print("⚠️  Please enter at least one genre")
                continue
            
            print(f"✅ Parsed {len(input_genres)} genres: {input_genres}")
            
            # Fixed parameters - no user input required
            max_distance_km = 500  # Fixed radius
            top_k = 5  # Fixed number of results per genre
            
            # Run analysis
            print(f"\n🔍 Starting multi-region multi-genre analysis...")
            analysis_results = self.analyze_multi_region_multi_genre(
                input_regions, input_genres, max_distance_km, top_k
            )
            
            # Display results
            if 'error' not in analysis_results:
                self._display_complete_results(analysis_results)
            else:
                print(f"❌ Analysis failed: {analysis_results['error']}")


def main():
    """Main function to run the Multi-Region Multi-Genre Analyzer."""
    
    # File paths
    excel_file = "knowledge/druid_query_results_with_descriptions.xlsx"
    cache_file = "geocoding_cache.pkl"
    
    try:
        # Initialize the analyzer
        analyzer = MultiRegionGenreAnalyzer(excel_file, cache_file)
        
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

#!/usr/bin/env python3
"""
Region Distance Finder

This script reads region data from an Excel file, geocodes the regions to get
latitude/longitude coordinates, and allows users to find the top 20 nearby
regions based on distance from a given region.

Features:
- Excel data extraction and cleaning
- Geocoding with caching using OpenStreetMap Nominatim
- Distance calculation using Haversine formula
- Case-insensitive fuzzy search for region names
- Top 20 nearest regions recommendation
"""

import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from geopy.extra.rate_limiter import RateLimiter
import pickle
import os
import time
import warnings
from typing import List, Dict, Tuple, Optional
import logging

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RegionDistanceFinder:
    """
    Main class for handling region data extraction, geocoding, and distance calculations.
    """
    
    def __init__(self, excel_file_path: str, cache_file: str = "geocoding_cache.pkl"):
        """
        Initialize the RegionDistanceFinder.
        
        Args:
            excel_file_path (str): Path to the Excel file containing region data
            cache_file (str): Path to the geocoding cache file
        """
        self.excel_file_path = excel_file_path
        self.cache_file = cache_file
        self.geolocator = Nominatim(user_agent="region_distance_finder")
        self.geocode = RateLimiter(self.geolocator.geocode, min_delay_seconds=1)
        
        # Data storage
   
        self.regions_df = None
        self.geocoded_regions = None
        self.distance_matrix = None
        self.cache = {}
        
        # Load cache if exists
        self._load_cache()
        
        logger.info("RegionDistanceFinder initialized successfully")
    
    def _load_cache(self) -> None:
        """Load geocoding cache from file if it exists."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                logger.info(f"Loaded {len(self.cache)} cached geocoding results")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                self.cache = {}
    
    def _save_cache(self) -> None:
        """Save geocoding cache to file."""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
            logger.info(f"Saved {len(self.cache)} geocoding results to cache")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def load_excel_data(self) -> pd.DataFrame:
        """
        Load and clean region data from Excel file.
        
        Returns:
            pd.DataFrame: Cleaned region data
        """
        logger.info(f"Loading Excel data from {self.excel_file_path}")
        
        try:
            # Read Excel file
            df = pd.read_excel(self.excel_file_path)
            logger.info(f"Excel file loaded successfully with {len(df)} rows")
            
            # Check required columns
            required_columns = ['de_region', 'de_region_updated', 'de_country']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Extract and clean relevant columns
            regions_df = df[required_columns].copy()
            
            # Remove rows with missing values in key columns
            initial_count = len(regions_df)
            regions_df = regions_df.dropna(subset=required_columns)
            final_count = len(regions_df)
            
            if initial_count != final_count:
                logger.warning(f"Removed {initial_count - final_count} rows with missing values")
            
            # Clean string columns
            for col in required_columns:
                regions_df[col] = regions_df[col].astype(str).str.strip()
            
            # Remove duplicates
            regions_df = regions_df.drop_duplicates()
            
            # Create unique identifier for each region
            regions_df['region_id'] = (
                regions_df['de_country'] + '_' + 
                regions_df['de_region'] + '_' + 
                regions_df['de_region_updated']
            )
            
            self.regions_df = regions_df
            logger.info(f"Data cleaned successfully. {len(regions_df)} unique regions loaded")
            
            return regions_df
            
        except Exception as e:
            logger.error(f"Error loading Excel data: {e}")
            raise
    
    def _geocode_region(self, country_code: str, region_name: str, region_code: str) -> Optional[Tuple[float, float]]:
        """
        Geocode a single region using various strategies.
        
        Args:
            country_code (str): Country code (e.g., 'US')
            region_name (str): Full region name (e.g., 'California')
            region_code (str): Region code (e.g., 'CA')
            
        Returns:
            Optional[Tuple[float, float]]: (latitude, longitude) or None if not found
        """
        # Create cache key
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
                    logger.debug(f"Geocoded '{query}' -> {coords}")
                    return coords
            except Exception as e:
                logger.debug(f"Geocoding failed for '{query}': {e}")
                continue
        
        # If all strategies fail
        logger.warning(f"Failed to geocode: {country_code}, {region_name} ({region_code})")
        self.cache[cache_key] = None
        return None
    
    def geocode_all_regions(self) -> pd.DataFrame:
        """
        Geocode all regions in the dataset.
        
        Returns:
            pd.DataFrame: DataFrame with geocoding results
        """
        if self.regions_df is None:
            raise ValueError("Excel data not loaded. Call load_excel_data() first.")
        
        logger.info("Starting geocoding process for all regions")
        
        geocoded_data = []
        total_regions = len(self.regions_df)
        
        for idx, row in self.regions_df.iterrows():
            country = row['de_country']
            region_name = row['de_region_updated']
            region_code = row['de_region']
            
            logger.info(f"Geocoding {idx+1}/{total_regions}: {region_name}, {country}")
            
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
        
        logger.info(f"Geocoding completed: {successful} successful, {failed} failed")
        
        return self.geocoded_regions
    
    def calculate_distance_matrix(self) -> np.ndarray:
        """
        Calculate distance matrix between all geocoded regions.
        
        Returns:
            np.ndarray: Distance matrix in kilometers
        """
        if self.geocoded_regions is None:
            raise ValueError("Regions not geocoded. Call geocode_all_regions() first.")
        
        logger.info("Calculating distance matrix")
        
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
                    logger.warning(f"Distance calculation failed between regions {i} and {j}: {e}")
                    distance_matrix[i, j] = np.inf
                    distance_matrix[j, i] = np.inf
        
        self.distance_matrix = distance_matrix
        self.successful_regions = successful_regions
        
        logger.info(f"Distance matrix calculated for {n_regions} regions")
        
        return distance_matrix
    
    def find_nearby_regions(self, search_query: str, top_n: int = 20) -> List[Dict]:
        """
        Find the top N nearest regions to a given search query.
        
        Args:
            search_query (str): Region name or code to search for
            top_n (int): Number of nearest regions to return
            
        Returns:
            List[Dict]: List of nearby regions with distance information
        """
        if self.distance_matrix is None:
            raise ValueError("Distance matrix not calculated. Call calculate_distance_matrix() first.")
        
        logger.info(f"Searching for regions matching: '{search_query}'")
        
        search_query_lower = search_query.lower().strip()
        
        # Search for exact match (case-insensitive)
        matched_idx = None
        for idx, row in self.successful_regions.iterrows():
            # Check various possible matches
            if (row['de_region_updated'].lower() == search_query_lower or
                row['de_region'].lower() == search_query_lower or
                f"{row['de_region_updated']}, {row['de_country']}".lower() == search_query_lower or
                f"{row['de_region']}, {row['de_country']}".lower() == search_query_lower):
                matched_idx = idx
                break
        
        if matched_idx is None:
            logger.warning(f"No match found for '{search_query}'")
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
                nearby_regions.append(region_data)
        
        # Sort by distance and return top N
        nearby_regions.sort(key=lambda x: x['distance_km'])
        
        matched_region = self.successful_regions.iloc[matched_idx]
        logger.info(f"Found match: {matched_region['de_region_updated']}, {matched_region['de_country']}")
        
        return nearby_regions[:top_n]
    
    def get_all_regions(self) -> List[str]:
        """
        Get list of all available region names for reference.
        
        Returns:
            List[str]: List of region names
        """
        if self.successful_regions is None:
            return []
        
        regions = []
        for _, row in self.successful_regions.iterrows():
            regions.extend([
                f"{row['de_region_updated']}, {row['de_country']}",
                f"{row['de_region']}, {row['de_country']}"
            ])
        
        return sorted(list(set(regions)))
    
    def run_interactive_search(self) -> None:
        """Run interactive search interface."""
        print("\n" + "="*60)
        print("REGION DISTANCE FINDER")
        print("="*60)
        print(f"Loaded {len(self.successful_regions)} geocoded regions")
        print("Type 'quit' to exit, 'list' to see all regions")
        print("-"*60)
        
        while True:
            search_query = input("\nEnter region name or code: ").strip()
            
            if search_query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if search_query.lower() == 'list':
                regions = self.get_all_regions()
                print(f"\nAvailable regions ({len(regions)}):")
                for i, region in enumerate(regions[:20]):  # Show first 20
                    print(f"  {i+1}. {region}")
                if len(regions) > 20:
                    print(f"  ... and {len(regions) - 20} more")
                continue
            
            if not search_query:
                continue
            
            # Search for nearby regions
            nearby = self.find_nearby_regions(search_query, top_n=20)
            
            if not nearby:
                print(f"No regions found matching '{search_query}'")
                continue
            
            # Get the matched region for display
            search_query_lower = search_query.lower().strip()
            matched_idx = None
            for idx, row in self.successful_regions.iterrows():
                if (row['de_region_updated'].lower() == search_query_lower or
                    row['de_region'].lower() == search_query_lower or
                    f"{row['de_region_updated']}, {row['de_country']}".lower() == search_query_lower or
                    f"{row['de_region']}, {row['de_country']}".lower() == search_query_lower):
                    matched_idx = idx
                    break
            
            matched_region = self.successful_regions.iloc[matched_idx]
            
            print(f"\nTop 20 regions near {matched_region['de_region_updated']}, {matched_region['de_country']}:")
            print("-"*80)
            print(f"{'Rank':<5} {'Region Name':<25} {'Country':<5} {'Distance (km)':<15} {'Distance (miles)':<15}")
            print("-"*80)
            
            for i, region in enumerate(nearby, 1):
                print(f"{i:<5} {region['de_region_updated']:<25} {region['de_country']:<5} "
                      f"{region['distance_km']:<15} {region['distance_miles']:<15}")


def main():
    """Main function to run the Region Distance Finder."""
    
    # File paths
    excel_file = "Budget++/knowledge/druid_query_results_with_descriptions.xlsx"
    cache_file = "geocoding_cache.pkl"
    
    try:
        # Initialize the finder
        finder = RegionDistanceFinder(excel_file, cache_file)
        
        # Load and process data
        print("Loading Excel data...")
        regions_df = finder.load_excel_data()
        print(f"Loaded {len(regions_df)} regions from Excel")
        
        # Geocode regions
        print("\nGeocoding regions (this may take a while)...")
        geocoded_df = finder.geocode_all_regions()
        
        successful = len(geocoded_df[geocoded_df['geocoded_successfully']])
        print(f"Successfully geocoded {successful} regions")
        
        if successful == 0:
            print("No regions were successfully geocoded. Exiting.")
            return
        
        # Calculate distance matrix
        print("\nCalculating distance matrix...")
        distance_matrix = finder.calculate_distance_matrix()
        print("Distance matrix calculated successfully")
        
        # Run interactive search
        finder.run_interactive_search()
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()

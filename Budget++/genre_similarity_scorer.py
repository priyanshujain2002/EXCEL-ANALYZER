"""
Genre Similarity Scorer using Sentence Transformers

This script reads genres from the druid query results file and uses 
sentence transformer embeddings (all-MiniLM-L6-v2) to calculate 
relevance scores between an input genre and all genres in the dataset.

Features:
- Uses sentence-transformers library with all-MiniLM-L6-v2 model
- Calculates cosine similarity between genre embeddings
- Returns ranked list of similar genres with similarity scores
- Includes supply metrics (X and Y values) for business insights
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import sys
from typing import List, Tuple, Dict
import warnings
import ssl
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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

class GenreSimilarityScorer:
    """
    A class to calculate genre similarity using sentence transformer embeddings.
    """
    
    def __init__(self, excel_file_path: str = "knowledge/druid_query_results.xlsx"):
        """
        Initialize the GenreSimilarityScorer.
        
        Args:
            excel_file_path (str): Path to the Excel file containing genre data
        """
        self.excel_file_path = excel_file_path
        self.model_name = "all-MiniLM-L6-v2"
        self.model = None
        self.genres_df = None
        self.genre_embeddings = None
        self.genres_list = None
        
        # Load data and model
        self._load_data()
        self._load_model()
        self._generate_embeddings()
    
    def _load_data(self):
        """Load genre data from Excel file."""
        try:
            if not os.path.exists(self.excel_file_path):
                raise FileNotFoundError(f"Excel file not found: {self.excel_file_path}")
            
            # Read the Excel file
            self.genres_df = pd.read_excel(self.excel_file_path)
            
            # Verify required columns exist
            required_columns = ['genre']
            missing_columns = [col for col in required_columns if col not in self.genres_df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Clean and prepare genre data
            self.genres_df = self.genres_df.dropna(subset=['genre'])
            self.genres_df['genre'] = self.genres_df['genre'].astype(str).str.strip()
            
            # Remove duplicates based on genre name
            self.genres_df = self.genres_df.drop_duplicates(subset=['genre'], keep='first')
            
            # Create list of genres for embedding
            self.genres_list = self.genres_df['genre'].tolist()
            
            print(f"✅ Successfully loaded {len(self.genres_list)} unique genres from {self.excel_file_path}")
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            sys.exit(1)
    
    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            print(f"🤖 Loading sentence transformer model: {self.model_name}")
            print("📥 This may take a moment on first run (downloading model)...")
            
            self.model = SentenceTransformer(self.model_name)
            
            print(f"✅ Model {self.model_name} loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("� Make sure you have sentence-transformers installed: pip install sentence-transformers")
            sys.exit(1)
    
    def _generate_embeddings(self):
        """Generate embeddings for all genres in the dataset."""
        try:
            print("🔄 Generating embeddings for all genres...")
            
            # Generate embeddings for all genres
            self.genre_embeddings = self.model.encode(self.genres_list, convert_to_tensor=False)
            
            print(f"✅ Generated embeddings for {len(self.genres_list)} genres")
            print(f"📊 Embedding dimension: {self.genre_embeddings.shape[1]}")
            
        except Exception as e:
            print(f"❌ Error generating embeddings: {e}")
            sys.exit(1)
    
    def calculate_similarity(self, input_genre: str, top_k: int = 10) -> List[Dict]:
        """
        Calculate similarity scores between input genre and all genres in dataset.
        
        Args:
            input_genre (str): The genre to find similar genres for
            top_k (int): Number of top similar genres to return
            
        Returns:
            List[Dict]: List of dictionaries containing genre info and similarity scores
        """
        try:
            # Generate embedding for input genre
            input_embedding = self.model.encode([input_genre], convert_to_tensor=False)
            
            # Calculate cosine similarity with all genre embeddings
            similarities = cosine_similarity(input_embedding, self.genre_embeddings)[0]
            
            # Create results list with genre info and similarity scores
            results = []
            for i, similarity_score in enumerate(similarities):
                genre_row = self.genres_df.iloc[i]
                
                # Skip if it's the exact same genre (similarity = 1.0)
                if similarity_score >= 0.999 and genre_row['genre'].lower() == input_genre.lower():
                    continue
                
                result = {
                    'genre': genre_row['genre'],
                    'similarity_score': float(similarity_score)
                }
                results.append(result)
            
            # Sort by similarity score (descending)
            results.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            # Return top K results
            return results[:top_k]
            
        except Exception as e:
            print(f"❌ Error calculating similarity: {e}")
            return []
    
    def find_similar_genres(self, input_genre: str, top_k: int = 10, 
                          min_similarity: float = 0.3) -> List[Dict]:
        """
        Find genres similar to the input genre with filtering options.
        
        Args:
            input_genre (str): The genre to find similar genres for
            top_k (int): Maximum number of results to return
            min_similarity (float): Minimum similarity threshold (0.0 to 1.0)
            
        Returns:
            List[Dict]: Filtered and ranked list of similar genres
        """
        # Get similarity scores
        similar_genres = self.calculate_similarity(input_genre, top_k * 2)  # Get more to filter
        
        # Filter by minimum similarity threshold
        filtered_genres = [
            genre for genre in similar_genres 
            if genre['similarity_score'] >= min_similarity
        ]
        
        # Return top K after filtering
        return filtered_genres[:top_k]
    
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

def print_results(input_genre: str, results: List[Dict]):
    """
    Print formatted results of genre similarity analysis.
    
    Args:
        input_genre (str): The input genre that was analyzed
        results (List[Dict]): List of similar genres with scores
    """
    print("\n" + "="*80)
    print(f"🎯 GENRE SIMILARITY RESULTS FOR: '{input_genre.upper()}'")
    print("📊 Ranked by Semantic Similarity (Sentence Transformers)")
    print("="*80)
    
    if not results:
        print("❌ No similar genres found.")
        return
    
    print(f"{'Rank':<4} {'Genre':<30} {'Similarity':<12}")
    print("-" * 50)
    
    for i, result in enumerate(results, 1):
        similarity_pct = result['similarity_score'] * 100
        print(f"{i:<4} {result['genre']:<30} {similarity_pct:>8.2f}%")
    
    print("-" * 80)
    print(f"📈 Found {len(results)} similar genres")
    print(f"🔝 Top match: '{results[0]['genre']}' ({results[0]['similarity_score']*100:.2f}% similarity)")

def main():
    """Main function to run the genre similarity scorer."""
    print("🎬 Genre Similarity Scorer using Sentence Transformers")
    print("🤖 Model: all-MiniLM-L6-v2")
    print("="*60)
    
    try:
        # Initialize the scorer
        scorer = GenreSimilarityScorer()
        
        print(f"\n✅ System ready! Loaded {len(scorer.genres_list)} genres")
        print("🎯 Enter a genre to find semantically similar genres")
        
        while True:
            # Get user input
            user_input = input("\n🎭 Enter a genre (or 'quit' to exit): ").strip()
            
            # Check if user wants to exit
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n🎬 Thank you for using the Genre Similarity Scorer!")
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
            
            # Find similar genres
            print(f"\n🔍 Analyzing semantic similarity for: '{user_input}'")
            print("⏳ Processing embeddings...")
            
            try:
                # Get similarity results
                results = scorer.find_similar_genres(
                    input_genre=user_input,
                    top_k=10,
                    min_similarity=0.2  # 20% minimum similarity
                )
                
                # Print results
                print_results(user_input, results)
                
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

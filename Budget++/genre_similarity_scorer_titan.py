"""
Genre Similarity Scorer using Amazon Titan Text Embeddings V2

This script reads genres from the druid query results file and uses 
Amazon Titan Text Embeddings V2 to calculate 
relevance scores between an input genre and all genres in the dataset.

Features:
- Uses Amazon Bedrock with Titan Text Embeddings V2 model
- Calculates cosine similarity between genre embeddings
- Returns ranked list of similar genres with similarity scores
- Includes supply metrics (X and Y values) for business insights
"""

import pandas as pd
import numpy as np
import boto3
from sklearn.metrics.pairwise import cosine_similarity
import os
import sys
from typing import List, Tuple, Dict
import warnings
import json
import logging

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GenreSimilarityScorer:
    """
    A class to calculate genre similarity using Amazon Titan Text Embeddings V2.
    """
    
    def __init__(self, excel_file_path: str = "Budget++/knowledge/druid_query_results.xlsx"):
        """
        Initialize the GenreSimilarityScorer.
        
        Args:
            excel_file_path (str): Path to the Excel file containing genre data
        """
        self.excel_file_path = excel_file_path
        self.model_id = "amazon.titan-embed-text-v2:0" # Titan Text Embeddings V2 model ID
        self.bedrock_runtime = None
        self.genres_df = None
        self.genre_embeddings = None
        self.genres_list = None
        
        # Load data and model client
        self._load_data()
        self._initialize_bedrock_client()
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
    
    def _initialize_bedrock_client(self):
        """Initialize the Amazon Bedrock runtime client."""
        try:
            print(f"🤖 Initializing Amazon Bedrock client for model: {self.model_id}")
            
            # boto3 client will automatically use credentials from environment variables
            # or ~/.aws/credentials, or IAM role if running on EC2.
            # saml2aws typically populates ~/.aws/credentials.
            self.bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1') # Or your preferred region
            
            # Test client by listing models (optional, but good for verification)
            # bedrock = boto3.client('bedrock', region_name='us-east-1')
            # foundation_models = bedrock.list_foundation_models()
            # titan_model_info = next((model for model in foundation_models['modelSummaries'] if model['modelId'] == self.model_id), None)
            # if titan_model_info:
            #     print(f"✅ Confirmed model {self.model_id} is available.")
            # else:
            #     print(f"⚠️  Model {self.model_id} not found in list_foundation_models. Check region or model ID.")
            
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

    def _generate_embeddings(self):
        """Generate embeddings for all genres in the dataset using Titan."""
        try:
            print("🔄 Generating embeddings for all genres using Amazon Titan Text Embeddings V2...")
            
            embeddings_list = []
            for i, genre in enumerate(self.genres_list):
                logging.info(f"Processing genre {i+1}/{len(self.genres_list)}: {genre}")
                embedding = self._get_titan_embedding(genre)
                embeddings_list.append(embedding)
            
            self.genre_embeddings = np.array(embeddings_list)
            
            print(f"✅ Generated embeddings for {len(self.genres_list)} genres using Titan")
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
            # Generate embedding for input genre
            input_embedding_list = self._get_titan_embedding(input_genre)
            input_embedding = np.array([input_embedding_list]) # Reshape for cosine_similarity
            
            # Calculate cosine similarity with all genre embeddings
            similarities = cosine_similarity(input_embedding, self.genre_embeddings)[0]
            
            # Create results list with genre info and similarity scores
            results = []
            for i, similarity_score in enumerate(similarities):
                genre_row = self.genres_df.iloc[i]
                
                # Skip if it's the exact same genre (similarity = 1.0)
                # Note: Titan embeddings might not be exactly 1.0 for identical strings due to normalization,
                # but it's good practice to keep this check if exact matches are to be excluded.
                if similarity_score >= 0.999 and genre_row['genre'].lower() == input_genre.lower():
                    continue
                
                result = {
                    'genre': genre_row['genre'],
                    'similarity_score': float(similarity_score)
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
            num_to_fetch = top_k * 2
        
        # Get similarity scores
        similar_genres = self.calculate_similarity(input_genre, top_k=num_to_fetch)
        
        # Filter by minimum similarity threshold
        filtered_genres = [
            genre for genre in similar_genres 
            if genre['similarity_score'] >= min_similarity
        ]
        
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
        exact_matches = [genre for genre in self.genres_list 
                        if genre.lower() == input_genre.lower()]
        
        if exact_matches:
            return True, []
        
        partial_matches = [genre for genre in self.genres_list 
                          if input_genre.lower() in genre.lower()]
        
        return False, partial_matches[:5]

def print_results(input_genre: str, results: List[Dict]):
    """
    Print formatted results of genre similarity analysis.
    
    Args:
        input_genre (str): The input genre that was analyzed
        results (List[Dict]): List of similar genres with scores
    """
    print("\n" + "="*80)
    print(f"🎯 GENRE SIMILARITY RESULTS FOR: '{input_genre.upper()}'")
    print("📊 Ranked by Semantic Similarity (Amazon Titan Text Embeddings V2)")
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
    if results: # Ensure results is not empty before accessing index 0
        print(f"🔝 Top match: '{results[0]['genre']}' ({results[0]['similarity_score']*100:.2f}% similarity)")

def main():
    """Main function to run the genre similarity scorer."""
    print("🎬 Genre Similarity Scorer using Amazon Titan Text Embeddings V2")
    print("🤖 Model: amazon.titan-embed-text-v2:0")
    print("="*60)
    
    try:
        # Initialize the scorer
        scorer = GenreSimilarityScorer()
        
        print(f"\n✅ System ready! Loaded {len(scorer.genres_list)} genres")
        print("🎯 Enter a genre to find semantically similar genres")
        
        while True:
            user_input = input("\n🎭 Enter a genre (or 'quit' to exit): ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n🎬 Thank you for using the Genre Similarity Scorer!")
                break
            
            if not user_input:
                print("⚠️  Please enter a valid genre name.")
                continue
            
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
            
            print(f"\n🔍 Analyzing semantic similarity for: '{user_input}'")
            print("⏳ Processing embeddings with Amazon Titan...")
            
            try:
                results = scorer.find_similar_genres(
                    input_genre=user_input,
                    top_k=None,
                    min_similarity=0.2
                )
                
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

"""
Enhanced Genre Similarity Scorer using Multi-Component Analysis

This script reads genres from the druid query results file and uses 
a comprehensive approach combining semantic, structural, and contextual
similarity to calculate relevance scores between an input genre and 
all genres in the dataset.

Features:
- Uses sentence-transformers library with all-MiniLM-L6-v2 model for semantic similarity
- Implements structural similarity based on TV genre patterns and word overlap
- Adds contextual similarity for TV format, audience, and content type analysis
- Multi-component weighted scoring (60% semantic, 25% structural, 15% contextual)
- Returns ranked list of similar genres with detailed similarity breakdown
- Includes SSL handling for development environments
- Enhanced business insights with component-wise analysis

Scoring Weights:
- Semantic Similarity: 60% (Sentence-BERT embeddings)
- Structural Similarity: 25% (Genre patterns + word overlap)
- Contextual Similarity: 15% (TV format + audience + content type)
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

class EnhancedGenreSimilarityScorer:
    """
    An enhanced class to calculate genre similarity using multi-component analysis.
    Combines semantic, structural, and contextual similarity for TV genres.
    """
    
    def __init__(self, excel_file_path: str = "knowledge/druid_query_results.xlsx"):
        """
        Initialize the EnhancedGenreSimilarityScorer.
        
        Args:
            excel_file_path (str): Path to the Excel file containing genre data
        """
        self.excel_file_path = excel_file_path
        self.model_name = "all-MiniLM-L6-v2"
        self.model = None
        self.genres_df = None
        self.genre_embeddings = None
        self.genres_list = None
        
        # Initialize empty categories
        self.genre_patterns = {}
        self.scripted_genres = []
        self.unscripted_genres = []
        self.family_friendly = []
        self.mature = []
        self.episodic = []
        self.event_based = []
        
        # Load data and model
        self._load_data()
        self._build_dynamic_categories()
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
            print("💡 Make sure you have sentence-transformers installed: pip install sentence-transformers")
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
    
    def _build_dynamic_categories(self):
        """Build dynamic genre categories from the actual genre data."""
        print("🔄 Building dynamic genre categories from data...")
        
        # Initialize category lists
        self.genre_patterns = {}
        self.scripted_genres = []
        self.unscripted_genres = []
        self.family_friendly = []
        self.mature = []
        self.episodic = []
        self.event_based = []
        
        # Define keyword-based category mappings
        format_keywords = {
            'scripted': ['drama', 'comedy', 'sitcom', 'thriller', 'romance', 'mystery'],
            'unscripted': ['reality', 'documentary', 'news', 'game show', 'talk show'],
            'family_friendly': ['children', 'kids', 'family', 'animation', 'animated'],
            'mature': ['crime', 'thriller', 'horror', 'mystery', 'suspense'],
            'episodic': ['sitcom', 'drama', 'soap opera', 'series'],
            'event_based': ['sports event', 'awards', 'concert', 'event']
        }
        
        # Build genre patterns based on actual data
        base_genres = {}
        
        for genre in self.genres_list:
            genre_lower = genre.lower()
            
            # Extract base genre (first word or main concept)
            words = genre_lower.split()
            if words:
                base_genre = words[0]
                
                # Add to base genres dictionary
                if base_genre not in base_genres:
                    base_genres[base_genre] = []
                base_genres[base_genre].append(genre)
            
            # Categorize by format keywords
            for category, keywords in format_keywords.items():
                if any(keyword in genre_lower for keyword in keywords):
                    if category == 'scripted':
                        self.scripted_genres.append(genre)
                    elif category == 'unscripted':
                        self.unscripted_genres.append(genre)
                    elif category == 'family_friendly':
                        self.family_friendly.append(genre)
                    elif category == 'mature':
                        self.mature.append(genre)
                    elif category == 'episodic':
                        self.episodic.append(genre)
                    elif category == 'event_based':
                        self.event_based.append(genre)
                    break
        
        # Convert base genres to genre patterns
        for base_genre, genre_list in base_genres.items():
            if len(genre_list) > 1:  # Only create patterns for genres with variations
                self.genre_patterns[base_genre] = genre_list
        
        # Add some common compound genres as patterns
        compound_patterns = {
            'crime': [g for g in self.genres_list if 'crime' in g.lower()],
            'sports': [g for g in self.genres_list if 'sport' in g.lower()],
            'music': [g for g in self.genres_list if 'music' in g.lower()],
            'news': [g for g in self.genres_list if 'news' in g.lower()],
            'reality': [g for g in self.genres_list if 'reality' in g.lower()],
            'documentary': [g for g in self.genres_list if 'documentar' in g.lower()],
            'comedy': [g for g in self.genres_list if 'comedy' in g.lower()],
            'drama': [g for g in self.genres_list if 'drama' in g.lower()]
        }
        
        # Merge compound patterns
        for pattern_name, genre_list in compound_patterns.items():
            if len(genre_list) > 1 and pattern_name not in self.genre_patterns:
                self.genre_patterns[pattern_name] = genre_list
        
        print(f"✅ Built {len(self.genre_patterns)} genre patterns from data")
        print(f"📊 Format categories - Scripted: {len(self.scripted_genres)}, Unscripted: {len(self.unscripted_genres)}")
        print(f"👥 Audience categories - Family: {len(self.family_friendly)}, Mature: {len(self.mature)}")
        print(f"📺 Content types - Episodic: {len(self.episodic)}, Event-based: {len(self.event_based)}")
    
    def _semantic_similarity(self, input_genre: str, target_genre: str) -> float:
        """Calculate semantic similarity using Sentence-BERT embeddings."""
        try:
            # Generate embeddings for both genres
            input_embedding = self.model.encode([input_genre], convert_to_tensor=False)
            target_embedding = self.model.encode([target_genre], convert_to_tensor=False)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(input_embedding, target_embedding)[0][0]
            return float(similarity)
            
        except Exception as e:
            print(f"❌ Error calculating semantic similarity: {e}")
            return 0.0
    
    def _structural_similarity(self, genre1: str, genre2: str) -> float:
        """Calculate structural similarity based on genre patterns and word overlap."""
        score = 0.0
        
        # Check if genres share pattern categories
        for category, genres in self.genre_patterns.items():
            if genre1 in genres and genre2 in genres:
                score += 0.5  # Same category bonus
                break
        
        # Word overlap analysis using Jaccard similarity
        words1 = set(genre1.lower().replace('-', ' ').replace('/', ' ').split())
        words2 = set(genre2.lower().replace('-', ' ').replace('/', ' ').split())
        
        if words1 and words2:  # Avoid division by zero
            jaccard_sim = len(words1.intersection(words2)) / len(words1.union(words2))
            score += 0.5 * jaccard_sim
        
        return min(score, 1.0)
    
    def _contextual_similarity(self, genre1: str, genre2: str) -> float:
        """Calculate contextual similarity based on TV-specific characteristics."""
        score = 0.0
        
        # Format compatibility (scripted vs unscripted)
        if (genre1 in self.scripted_genres and genre2 in self.scripted_genres) or \
           (genre1 in self.unscripted_genres and genre2 in self.unscripted_genres):
            score += 0.4
        
        # Audience appropriateness (inferred from genre names)
        if (genre1 in self.family_friendly and genre2 in self.family_friendly) or \
           (genre1 in self.mature and genre2 in self.mature):
            score += 0.3
        
        # Content type similarity
        if (genre1 in self.episodic and genre2 in self.episodic) or \
           (genre1 in self.event_based and genre2 in self.event_based):
            score += 0.3
        
        return score
    
    def calculate_similarity(self, input_genre: str, top_k: int = 10) -> List[Dict]:
        """
        Calculate comprehensive similarity scores between input genre and all genres.
        
        Args:
            input_genre (str): The genre to find similar genres for
            top_k (int): Number of top similar genres to return
            
        Returns:
            List[Dict]: List of dictionaries containing genre info and similarity scores
        """
        try:
            # Generate embedding for input genre
            input_embedding = self.model.encode([input_genre], convert_to_tensor=False)
            
            # Calculate SEMANTIC similarity only (for the semantic component)
            semantic_similarities = cosine_similarity(input_embedding, self.genre_embeddings)[0]
            
            # Calculate COMPREHENSIVE similarity using all components
            comprehensive_similarities = []
            component_breakdown = []
            
            for i, target_genre in enumerate(self.genres_list):
                semantic_score = semantic_similarities[i]
                structural_score = self._structural_similarity(input_genre, target_genre)
                contextual_score = self._contextual_similarity(input_genre, target_genre)
                
                # Weighted combination
                total_score = (
                    0.60 * semantic_score +
                    0.25 * structural_score +
                    0.15 * contextual_score
                )
                
                comprehensive_similarities.append(total_score)
                component_breakdown.append({
                    'semantic': semantic_score,
                    'structural': structural_score,
                    'contextual': contextual_score
                })
            
            # Create results list with genre info and similarity scores
            results = []
            for i, similarity_score in enumerate(comprehensive_similarities):
                genre_row = self.genres_df.iloc[i]
                target_genre = genre_row['genre']
                
                # Skip if it's the exact same genre (case-insensitive)
                if target_genre.lower() == input_genre.lower():
                    continue
                
                result = {
                    'genre': target_genre,
                    'similarity_score': float(similarity_score),
                    'semantic_score': float(component_breakdown[i]['semantic']),
                    'structural_score': float(component_breakdown[i]['structural']),
                    'contextual_score': float(component_breakdown[i]['contextual'])
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
                          min_similarity: float = 0.2) -> List[Dict]:
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
    
    def analyze_genre_components(self, input_genre: str, target_genre: str) -> Dict:
        """
        Analyze individual similarity components between two genres.
        
        Args:
            input_genre (str): First genre for comparison
            target_genre (str): Second genre for comparison
            
        Returns:
            Dict: Component-wise similarity breakdown
        """
        semantic = self._semantic_similarity(input_genre, target_genre)
        structural = self._structural_similarity(input_genre, target_genre)
        contextual = self._contextual_similarity(input_genre, target_genre)
        
        total = (0.60 * semantic) + (0.25 * structural) + (0.15 * contextual)
        
        return {
            'input_genre': input_genre,
            'target_genre': target_genre,
            'semantic_similarity': semantic,
            'structural_similarity': structural,
            'contextual_similarity': contextual,
            'total_similarity': total,
            'weights': {
                'semantic': 0.60,
                'structural': 0.25,
                'contextual': 0.15
            }
        }

def print_results(input_genre: str, results: List[Dict]):
    """
    Print formatted results of enhanced genre similarity analysis.
    
    Args:
        input_genre (str): The input genre that was analyzed
        results (List[Dict]): List of similar genres with scores
    """
    print("\n" + "="*120)
    print(f"🎯 ENHANCED GENRE SIMILARITY RESULTS FOR: '{input_genre.upper()}'")
    print("📊 Multi-Component Analysis (Semantic + Structural + Contextual)")
    print("⚖️  Scoring Weights: Semantic 60% | Structural 25% | Contextual 15%")
    print("="*120)
    
    if not results:
        print("❌ No similar genres found.")
        return
    
    print(f"{'Rank':<4} {'Genre':<35} {'Total':<8} {'Semantic':<9} {'Structural':<11} {'Contextual':<11}")
    print("-" * 110)
    
    for i, result in enumerate(results, 1):
        total_pct = result['similarity_score'] * 100
        semantic_pct = result['semantic_score'] * 100
        structural_pct = result['structural_score'] * 100
        contextual_pct = result['contextual_score'] * 100
        
        print(f"{i:<4} {result['genre']:<35} {total_pct:>6.1f}%  {semantic_pct:>7.1f}%   {structural_pct:>9.1f}%    {contextual_pct:>9.1f}%")
    
    print("-" * 120)
    print(f"📈 Found {len(results)} similar genres")
    print(f"🔝 Top match: '{results[0]['genre']}' ({results[0]['similarity_score']*100:.2f}% similarity)")
    
    # Show component analysis for top match
    if results:
        top = results[0]
        print(f"\n🔍 Top Match Analysis: '{input_genre}' vs '{top['genre']}'")
        print(f"   🧠 Semantic: {top['semantic_score']*100:.1f}% (meaning and concepts)")
        print(f"   🏗️  Structural: {top['structural_score']*100:.1f}% (patterns and word overlap)")
        print(f"   🎬 Contextual: {top['contextual_score']*100:.1f}% (TV format and audience)")

def print_component_analysis(analysis: Dict):
    """
    Print detailed component analysis for two genres.
    
    Args:
        analysis (Dict): Component analysis result
    """
    print("\n" + "="*80)
    print(f"🔍 DETAILED COMPONENT ANALYSIS")
    print(f"📊 Comparing: '{analysis['input_genre']}' vs '{analysis['target_genre']}'")
    print("="*80)
    
    print(f"\n🧠 Semantic Similarity: {analysis['semantic_similarity']*100:.2f}%")
    print("   Based on Sentence-BERT embeddings and meaning understanding")
    
    print(f"\n🏗️  Structural Similarity: {analysis['structural_similarity']*100:.2f}%")
    print("   Based on TV genre patterns and word overlap analysis")
    
    print(f"\n🎬 Contextual Similarity: {analysis['contextual_similarity']*100:.2f}%")
    print("   Based on TV format, audience, and content type compatibility")
    
    print(f"\n⚖️  Total Similarity: {analysis['total_similarity']*100:.2f}%")
    print("   Weighted combination of all components")
    
    print(f"\n📋 Scoring Weights:")
    for component, weight in analysis['weights'].items():
        print(f"   {component.capitalize()}: {weight*100:.0f}%")

def main():
    """Main function to run the enhanced genre similarity scorer."""
    print("🎬 Enhanced Genre Similarity Scorer")
    print("🤖 Model: all-MiniLM-L6-v2")
    print("📊 Multi-Component Analysis: Semantic + Structural + Contextual")
    print("="*70)
    
    try:
        # Initialize the scorer
        scorer = EnhancedGenreSimilarityScorer()
        
        print(f"\n✅ System ready! Loaded {len(scorer.genres_list)} genres")
        print("🎯 Enter a genre to find similar genres using enhanced analysis")
        print("🔧 Features: Component breakdown, TV-specific logic, weighted scoring")
        
        while True:
            # Get user input
            user_input = input("\n🎭 Enter a genre (or 'quit' to exit, 'analyze' for component comparison): ").strip()
            
            # Check if user wants to exit
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n🎬 Thank you for using the Enhanced Genre Similarity Scorer!")
                break
            
            # Check for component analysis mode
            if user_input.lower() == 'analyze':
                genre1 = input("📋 Enter first genre: ").strip()
                genre2 = input("📋 Enter second genre: ").strip()
                
                if genre1 and genre2:
                    analysis = scorer.analyze_genre_components(genre1, genre2)
                    print_component_analysis(analysis)
                continue
            
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
            print(f"\n🔍 Analyzing enhanced similarity for: '{user_input}'")
            print("⏳ Processing multi-component analysis...")
            
            try:
                # Get similarity results
                results = scorer.find_similar_genres(
                    input_genre=user_input,
                    top_k=10,
                    min_similarity=0.15  # 15% minimum similarity for enhanced analysis
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

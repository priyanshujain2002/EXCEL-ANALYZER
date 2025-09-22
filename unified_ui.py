import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import os
import sys

# Import both backend systems
from High_Impact.idRecommender_highImpactRecommender import recommend_packages
import importlib.util
import sys

# Import Budget++ module with special characters in path
spec = importlib.util.spec_from_file_location(
    "multi_region_genre_analyzer", 
    "Budget++/multi_region_genre_analyzer.py"
)
budget_module = importlib.util.module_from_spec(spec)
sys.modules["multi_region_genre_analyzer"] = budget_module
spec.loader.exec_module(budget_module)
MultiRegionGenreAnalyzer = budget_module.MultiRegionGenreAnalyzer

# Initialize session state variables
def init_session_state():
    # Mode selection
    if 'analysis_mode' not in st.session_state:
        st.session_state['analysis_mode'] = 'High Impact'
    
    # High Impact session state
    if 'high_impact_packages_result' not in st.session_state:
        st.session_state['high_impact_packages_result'] = None
    if 'high_impact_feedback_options' not in st.session_state:
        st.session_state['high_impact_feedback_options'] = []
    if 'high_impact_selected_placements' not in st.session_state:
        st.session_state['high_impact_selected_placements'] = []
    if 'high_impact_recommendation_generated' not in st.session_state:
        st.session_state['high_impact_recommendation_generated'] = False
    if 'high_impact_custom_feedback' not in st.session_state:
        st.session_state['high_impact_custom_feedback'] = ""
    
    # Budget++ session state
    if 'budget_analyzer' not in st.session_state:
        st.session_state['budget_analyzer'] = None
    if 'budget_results' not in st.session_state:
        st.session_state['budget_results'] = None
    if 'budget_regions_input' not in st.session_state:
        st.session_state['budget_regions_input'] = ""
    if 'budget_genres_input' not in st.session_state:
        st.session_state['budget_genres_input'] = ""
    if 'budget_max_distance' not in st.session_state:
        st.session_state['budget_max_distance'] = 500
    if 'budget_top_k' not in st.session_state:
        st.session_state['budget_top_k'] = 5
    if 'budget_analysis_generated' not in st.session_state:
        st.session_state['budget_analysis_generated'] = False
    if 'budget_custom_feedback' not in st.session_state:
        st.session_state['budget_custom_feedback'] = ""

# High Impact functionality (copied from existing testUI.py)
def extract_package_names_only(recommendation_result):
    """
    Extract only package names from the recommendation result, removing reasoning and numbering
    """
    try:
        if not recommendation_result:
            return ""
        
        package_names = []
        lines = str(recommendation_result).strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line and ':' in line:
                # Split by colon and take the first part (package name with numbering)
                package_with_numbering = line.split(':')[0].strip()
                
                # Remove the numbering (e.g., "1. ", "2. ", "3. ")
                if '. ' in package_with_numbering:
                    package_name = package_with_numbering.split('. ', 1)[1].strip()
                else:
                    package_name = package_with_numbering.strip()
                
                if package_name:
                    package_names.append(package_name)
        
        return ", ".join(package_names)
    except Exception as e:
        # If parsing fails, return the original result
        return str(recommendation_result)

def store_high_impact_feedback(selected_placements, recommended_packages, feedback_options, custom_feedback=""):
    """
    Store High Impact feedback data to Unified_Feedback.xlsx
    """
    try:
        # Read existing Excel file
        try:
            df_existing = pd.read_excel("Unified_Feedback.xlsx", sheet_name="High_Impact")
            # Get the next Task ID (auto-increment)
            if df_existing.empty:
                next_task_id = 1
            else:
                next_task_id = df_existing['Task ID'].max() + 1
        except FileNotFoundError:
            # If file doesn't exist, create new DataFrame
            next_task_id = 1
            df_existing = pd.DataFrame(columns=['Task ID', 'Input Packages', 'Recommended Packages', 'Feedback Package', 'Custom Feedback'])
        
        # Format data as comma-separated strings
        input_package_str = ", ".join(selected_placements)
        feedback_package_str = ", ".join(feedback_options)
        
        # Create new row
        new_row = {
            'Task ID': next_task_id,
            'Input Packages': input_package_str,
            'Recommended Packages': recommended_packages,
            'Feedback Package': feedback_package_str,
            'Custom Feedback': custom_feedback
        }
        
        # Append new row to existing data
        df_new = pd.concat([df_existing, pd.DataFrame([new_row])], ignore_index=True)
        
        # Save back to Excel file with both sheets
        with pd.ExcelWriter("Unified_Feedback.xlsx", engine='openpyxl') as writer:
            df_new.to_excel(writer, sheet_name="High_Impact", index=False)
            
            # Also create/update Budget_Plus sheet if it doesn't exist
            try:
                df_budget = pd.read_excel("Unified_Feedback.xlsx", sheet_name="Budget_Plus")
                df_budget.to_excel(writer, sheet_name="Budget_Plus", index=False)
            except:
                # Create empty Budget_Plus sheet
                df_budget_empty = pd.DataFrame(columns=['Task ID', 'Input Regions', 'Input Genres', 'Analysis Summary', 'Custom Feedback'])
                df_budget_empty.to_excel(writer, sheet_name="Budget_Plus", index=False)
        
        return True
    except Exception as e:
        st.error(f"Failed to save High Impact feedback: {str(e)}")
        return False

def store_budget_plus_feedback(input_regions, input_genres, analysis_summary, custom_feedback=""):
    """
    Store Budget++ feedback data to Unified_Feedback.xlsx
    """
    try:
        # Read existing Excel file
        try:
            df_existing = pd.read_excel("Unified_Feedback.xlsx", sheet_name="Budget_Plus")
            # Get the next Task ID (auto-increment)
            if df_existing.empty:
                next_task_id = 1
            else:
                next_task_id = df_existing['Task ID'].max() + 1
        except FileNotFoundError:
            # If file doesn't exist, create new DataFrame
            next_task_id = 1
            df_existing = pd.DataFrame(columns=['Task ID', 'Input Regions', 'Input Genres', 'Analysis Summary', 'Custom Feedback'])
        
        # Format data as comma-separated strings
        input_regions_str = ", ".join(input_regions)
        input_genres_str = ", ".join(input_genres)
        
        # Create new row
        new_row = {
            'Task ID': next_task_id,
            'Input Regions': input_regions_str,
            'Input Genres': input_genres_str,
            'Analysis Summary': analysis_summary,
            'Custom Feedback': custom_feedback
        }
        
        # Append new row to existing data
        df_new = pd.concat([df_existing, pd.DataFrame([new_row])], ignore_index=True)
        
        # Save back to Excel file with both sheets
        with pd.ExcelWriter("Unified_Feedback.xlsx", engine='openpyxl') as writer:
            df_new.to_excel(writer, sheet_name="Budget_Plus", index=False)
            
            # Also maintain High_Impact sheet
            try:
                df_high_impact = pd.read_excel("Unified_Feedback.xlsx", sheet_name="High_Impact")
                df_high_impact.to_excel(writer, sheet_name="High_Impact", index=False)
            except:
                # Create empty High_Impact sheet
                df_hi_empty = pd.DataFrame(columns=['Task ID', 'Input Packages', 'Recommended Packages', 'Feedback Package', 'Custom Feedback'])
                df_hi_empty.to_excel(writer, sheet_name="High_Impact", index=False)
        
        return True
    except Exception as e:
        st.error(f"Failed to save Budget++ feedback: {str(e)}")
        return False

# Hardcoded placement names for High Impact
PLACEMENT_NAMES = [
    'First Screen Masthead - US',
    'Universal Guide Masthead - US',
    'Universal Guide Masthead - US (Weekend Heavy-Up)',
    'Apps Store Masthead - US',
    'Addressable Program Guide - US',
    'First Screen Masthead Roadblock - US',
    'Addressable Program Guide Roadblock - US',
    'Gaming Hub Hero Ad Roadblock - US',
    'Gaming Hub Hero Ad - US',
    'Sponsored Row Roadblock - US',
    'Sponsored Row - US',
    'Roadblock Game Console - US',
    'Universal Guide Masthead Roadblock - US',
    'First Screen All Years - US',
    'First Screen All Years Roadblock - US',
    'Apps Store Masthead Roadblock - US',
    'First Screen All Years Audience Takeover - US',
    'CTV - US',
    'TV Plus RON (15sec) - US',
    'TV Plus RON (30sec) - US',
    'CTV Pharma - US'
]

# Hardcoded high impact package names for feedback
HIGH_IMPACT_PACKAGES = [
    "No change required",
    "Spotlight Row Roadblock",
    "First Screen Roadblock",
    "First Screen Immersive Roadblock",
]

# Budget++ UI functions
def initialize_budget_analyzer():
    """Initialize Budget++ analyzer with loading spinner"""
    if st.session_state.budget_analyzer is None:
        with st.spinner("Initializing Budget++ Multi-Region Genre Analyzer..."):
            try:
                excel_file = "Budget++/knowledge/druid_query_results_with_descriptions.xlsx"
                cache_file = "Budget++/geocoding_cache.pkl"
                st.session_state.budget_analyzer = MultiRegionGenreAnalyzer(excel_file, cache_file)
                st.success("✅ Budget++ analyzer initialized successfully!")
            except Exception as e:
                st.error(f"❌ Failed to initialize Budget++ analyzer: {str(e)}")
                st.session_state.budget_analyzer = None

def run_budget_analysis():
    """Run Budget++ analysis with current inputs"""
    if st.session_state.budget_analyzer is None:
        st.error("Budget++ analyzer not initialized")
        return
    
    # Parse inputs
    input_regions = st.session_state.budget_analyzer.parse_multiple_inputs(st.session_state.budget_regions_input)
    input_genres = st.session_state.budget_analyzer.parse_multiple_inputs(st.session_state.budget_genres_input)
    
    if not input_regions:
        st.error("Please enter at least one region")
        return
    
    if not input_genres:
        st.error("Please enter at least one genre")
        return
    
    # Run analysis
    with st.spinner("Running multi-region multi-genre analysis..."):
        try:
            results = st.session_state.budget_analyzer.analyze_multi_region_multi_genre(
                input_regions=input_regions,
                input_genres=input_genres,
                max_distance_km=st.session_state.budget_max_distance,
                top_k=st.session_state.budget_top_k
            )
            
            st.session_state.budget_results = results
            st.session_state.budget_analysis_generated = True
            
            if 'error' in results and results['error']:
                st.error(f"❌ Analysis failed: {results['error']}")
            else:
                st.success("✅ Analysis completed successfully!")
                
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            st.session_state.budget_results = None
            st.session_state.budget_analysis_generated = False

def display_budget_results():
    """Display Budget++ analysis results"""
    if st.session_state.budget_results is None:
        return
    
    results = st.session_state.budget_results
    
    if 'error' in results and results['error']:
        st.error(f"Analysis failed: {results['error']}")
        return
    
    # Display selected regions
    st.subheader("📍 Selected Regions Analysis")
    selected_regions = results['selected_regions']
    input_regions = results['input_regions']
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Input Regions:** {', '.join(input_regions)}")
    with col2:
        st.write(f"**Regions Found:** {len(selected_regions)}")
    
    if selected_regions:
        # Create regions dataframe
        regions_data = []
        for i, region in enumerate(selected_regions[:20], 1):  # Show top 20
            regions_data.append({
                'Rank': i,
                'Region': region['de_region_updated'],
                'Country': region['de_country'],
                'Distance to Centroid (km)': region['distance_to_centroid_km'],
                'Input Region': '✓' if region.get('is_input_region', False) else ''
            })
        
        regions_df = pd.DataFrame(regions_data)
        st.dataframe(regions_df, use_container_width=True)
        
        if len(selected_regions) > 20:
            st.write(f"... and {len(selected_regions) - 20} more regions")
    
    # Display genre results
    st.subheader("🎭 Genre Similarity Results")
    genre_results = results['results']
    
    for input_genre, genre_list in genre_results.items():
        st.write(f"**Results for '{input_genre}':**")
        
        if not genre_list:
            st.write("No similar genres found.")
            continue
        
        # Create results dataframe
        results_data = []
        for i, result in enumerate(genre_list, 1):
            results_data.append({
                'Rank': i,
                'Genre': result['genre'],
                'Description': result['description'][:100] + '...' if len(result['description']) > 100 else result['description'],
                'Similarity (%)': f"{result['similarity_score']*100:.2f}",
                'X Metric': f"{result['x']:.4f}",
                'Y Metric': f"{result['y']:.2f}",
                'Weighted Avg': f"{result['weighted_average']:.4f}",
                'Unsold Supply': result['unsold_supply']
            })
        
        results_df = pd.DataFrame(results_data)
        st.dataframe(results_df, use_container_width=True)
    
    # Display summary
    st.subheader("📊 Analysis Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Regions Analyzed", len(results['selected_regions']))
    with col2:
        st.metric("Genres Processed", len(results['input_genres']))
    with col3:
        total_results = sum(len(genre_list) for genre_list in results['results'].values())
        st.metric("Total Recommendations", total_results)
    with col4:
        st.metric("Max Distance (km)", results['max_distance_km'])

def create_budget_analysis_summary(results):
    """Create a brief summary of Budget++ analysis for feedback"""
    if not results or 'error' in results:
        return "Analysis failed or no results available"
    
    selected_regions_count = len(results['selected_regions'])
    input_genres_count = len(results['input_genres'])
    total_recommendations = sum(len(genre_list) for genre_list in results['results'].values())
    
    return f"Analyzed {selected_regions_count} regions, {input_genres_count} genres. Found {total_recommendations} similar recommendations."

# Main app
def main():
    # Initialize session state
    init_session_state()
    
    # Set page configuration
    st.set_page_config(
        page_title="Unified Analysis System",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # App title and description
    st.title("🔍 Unified Analysis System")
    st.markdown("""
    This system provides two types of analysis:
    - **High Impact Package Recommendation**: Find relevant high-impact packages based on placement names
    - **Budget++ Multi-Region Genre Analysis**: Analyze genre similarities across multiple geographic regions
    """)
    
    # Mode selection
    st.subheader("Select Analysis Mode")
    analysis_mode = st.radio(
        "Choose the type of analysis you want to perform:",
        ["High Impact", "Budget++"],
        index=0 if st.session_state.analysis_mode == "High Impact" else 1,
        key="analysis_mode_radio"
    )
    
    # Update session state
    st.session_state.analysis_mode = analysis_mode
    
    # Display mode-specific interface
    if analysis_mode == "High Impact":
        display_high_impact_interface()
    else:
        display_budget_plus_interface()

def display_high_impact_interface():
    """Display High Impact Package Recommendation interface"""
    st.markdown("---")
    st.header("🎯 High Impact Package Recommendation")
    st.markdown("""
    This system helps you find the most relevant high-impact packages based on your selected placement names.
    Simply select the placement names you're interested in and click the Recommend button.
    """)
    
    # Sidebar with inputs
    st.sidebar.header("Input Parameters")
    
    # Placement names selector - sync with session state
    selected_placements = st.sidebar.multiselect(
        "Select Placement Names:",
        PLACEMENT_NAMES,
        default=st.session_state['high_impact_selected_placements']
    )
    
    # Update session state when selection changes
    if selected_placements != st.session_state['high_impact_selected_placements']:
        st.session_state['high_impact_selected_placements'] = selected_placements
    
    # Recommend button
    recommend_button = st.sidebar.button("Recommend Packages", type="primary")
    
    # Main content area
    if recommend_button:
        if not selected_placements:
            st.error("Please select at least one placement name.")
        else:
            # Show a spinner while processing
            with st.spinner("Analyzing placement names and recommending packages..."):
                try:
                    # Call the recommendation function
                    high_impact_packages_result = recommend_packages(selected_placements)
                    
                    # Store results in session state for feedback
                    st.session_state['high_impact_packages_result'] = high_impact_packages_result
                    st.session_state['high_impact_recommendation_generated'] = True
                    
                except Exception as e:
                    st.error(f"An error occurred during recommendation: {str(e)}")
    
    # Display recommendation results if they exist in session state
    if st.session_state['high_impact_packages_result'] is not None:
        st.subheader("Recommendation Results")
        
        # Format the placement names for display
        placement_names_str = ", ".join(st.session_state['high_impact_selected_placements'])
        
        # Display the high impact packages in markdown format
        st.markdown(f"""
For the given placement names, the most recommended high impact packages are:

{st.session_state['high_impact_packages_result']}
        """)
        
        if st.session_state['high_impact_recommendation_generated']:
            st.success("Recommendation completed successfully!")
    else:
        # Initial message
        st.info("Select placement names from the sidebar and click 'Recommend Packages' to get started.")
    
    # Feedback section (only show after recommendations are made)
    if st.session_state['high_impact_packages_result'] is not None:
        st.subheader("Feedback")
        st.markdown("Please provide feedback on the recommended high-impact packages:")
        
        # Feedback multiselect - sync with session state
        feedback_options = st.multiselect(
            "Select feedback options:",
            HIGH_IMPACT_PACKAGES,
            default=st.session_state['high_impact_feedback_options']
        )
        
        # Update session state when feedback selection changes
        if feedback_options != st.session_state['high_impact_feedback_options']:
            st.session_state['high_impact_feedback_options'] = feedback_options
        
        # Custom feedback text area
        custom_feedback = st.text_area(
            "Or provide your custom feedback:",
            value=st.session_state['high_impact_custom_feedback'],
            placeholder="Please share your thoughts, suggestions, or any other feedback about the recommended packages...",
            height=100
        )
        
        # Update session state when custom feedback changes
        if custom_feedback != st.session_state['high_impact_custom_feedback']:
            st.session_state['high_impact_custom_feedback'] = custom_feedback
        
        # Feedback submit button
        if st.button("Submit Feedback", key="high_impact_feedback"):
            # Validate that at least one type of feedback is provided
            if not feedback_options and not custom_feedback.strip():
                st.warning("Please provide feedback either by selecting options or writing custom feedback.")
            else:
                # Process feedback
                if "No change required" in feedback_options and len(feedback_options) == 1 and not custom_feedback.strip():
                    st.success("Thank you for your feedback! The recommendations met your expectations.")
                else:
                    # Extract package names only (without reasoning)
                    package_names_only = extract_package_names_only(st.session_state['high_impact_packages_result'])
                    
                    # Store feedback to Excel file
                    success = store_high_impact_feedback(
                        st.session_state['high_impact_selected_placements'], 
                        package_names_only, 
                        feedback_options,
                        custom_feedback.strip()
                    )
                    
                    if success:
                        st.success("Thank you for your feedback!")
                        # Clear feedback options and custom feedback after successful submission
                        st.session_state['high_impact_feedback_options'] = []
                        st.session_state['high_impact_custom_feedback'] = ""
                    else:
                        st.error("Failed to save feedback")

def display_budget_plus_interface():
    """Display Budget++ Multi-Region Genre Analysis interface"""
    st.markdown("---")
    st.header("🌍 Budget++ Multi-Region Genre Analysis")
    st.markdown("""
    This system analyzes genre similarities across multiple geographic regions. It finds regions geographically close to all your input regions,
    aggregates data by genre, and runs similarity analysis for each input genre.
    """)
    
    # Initialize analyzer
    initialize_budget_analyzer()
    
    if st.session_state.budget_analyzer is None:
        st.error("Budget++ analyzer could not be initialized. Please check your AWS credentials and data files.")
        return
    
    # Sidebar with inputs
    st.sidebar.header("Input Parameters")
    
    # Regions input
    regions_input = st.sidebar.text_input(
        "Enter regions (comma-separated):",
        value=st.session_state.budget_regions_input,
        placeholder="e.g., California, US, New York, US",
        help="Enter region names separated by commas"
    )
    st.session_state.budget_regions_input = regions_input
    
    # Get regions suggestions button
    if st.sidebar.button("Get Region Suggestions"):
        suggestions = st.session_state.budget_analyzer.get_region_suggestions(15)
        st.sidebar.write("**Available regions:**")
        for suggestion in suggestions:
            st.sidebar.write(f"- {suggestion}")
    
    # Genres input
    genres_input = st.sidebar.text_input(
        "Enter genres (comma-separated):",
        value=st.session_state.budget_genres_input,
        placeholder="e.g., Action, Drama, Comedy",
        help="Enter genre names separated by commas"
    )
    st.session_state.budget_genres_input = genres_input
    
    # Get genres suggestions button
    if st.sidebar.button("Get Genre Suggestions"):
        suggestions = st.session_state.budget_analyzer.get_genre_suggestions(15)
        st.sidebar.write("**Available genres:**")
        for suggestion in suggestions:
            st.sidebar.write(f"- {suggestion}")
    
    # Parameters
    st.sidebar.subheader("Analysis Parameters")
    max_distance = st.sidebar.slider(
        "Max distance from centroid (km):",
        min_value=100,
        max_value=1000,
        value=st.session_state.budget_max_distance
    )
    st.session_state.budget_max_distance = max_distance
    
    top_k = st.sidebar.number_input(
        "Number of results per genre:",
        min_value=1,
        max_value=10,
        value=st.session_state.budget_top_k
    )
    st.session_state.budget_top_k = top_k
    
    # Analyze button
    analyze_button = st.sidebar.button("Run Analysis", type="primary")
    
    # Main content area
    if analyze_button:
        run_budget_analysis()
    
    # Display results
    if st.session_state.budget_analysis_generated and st.session_state.budget_results is not None:
        display_budget_results()
    else:
        st.info("Enter regions and genres, then click 'Run Analysis' to get started.")
    
    # Feedback section (only show after analysis is completed)
    if st.session_state.budget_analysis_generated and st.session_state.budget_results is not None:
        if 'error' not in st.session_state.budget_results or not st.session_state.budget_results['error']:
            st.subheader("Feedback")
            st.markdown("Please provide your feedback on the analysis:")
            
            # Custom feedback text area
            custom_feedback = st.text_area(
                "Your feedback:",
                value=st.session_state.budget_custom_feedback,
                placeholder="Please share your thoughts, suggestions, or any feedback about the analysis results...",
                height=120
            )
            
            # Update session state when custom feedback changes
            if custom_feedback != st.session_state.budget_custom_feedback:
                st.session_state.budget_custom_feedback = custom_feedback
            
            # Feedback submit button
            if st.button("Submit Feedback", key="budget_feedback"):
                if not custom_feedback.strip():
                    st.warning("Please provide your feedback before submitting.")
                else:
                    # Create analysis summary
                    analysis_summary = create_budget_analysis_summary(st.session_state.budget_results)
                    
                    # Store feedback to Excel file
                    success = store_budget_plus_feedback(
                        st.session_state.budget_results['input_regions'],
                        st.session_state.budget_results['input_genres'],
                        analysis_summary,
                        custom_feedback.strip()
                    )
                    
                    if success:
                        st.success("Thank you for your feedback!")
                        # Clear feedback after successful submission
                        st.session_state.budget_custom_feedback = ""
                    else:
                        st.error("Failed to save feedback")

if __name__ == "__main__":
    main()

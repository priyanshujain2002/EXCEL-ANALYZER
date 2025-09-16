import streamlit as st
import pandas as pd
from idRecommender_highImpactRecommender import recommend_packages

# Initialize session state variables
def init_session_state():
    if 'high_impact_packages_result' not in st.session_state:
        st.session_state['high_impact_packages_result'] = None
    if 'feedback_options' not in st.session_state:
        st.session_state['feedback_options'] = []
    if 'selected_placements' not in st.session_state:
        st.session_state['selected_placements'] = []
    if 'recommendation_generated' not in st.session_state:
        st.session_state['recommendation_generated'] = False
    if 'custom_feedback' not in st.session_state:
        st.session_state['custom_feedback'] = ""

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

def store_feedback_to_excel(selected_placements, recommended_packages, feedback_options, custom_feedback=""):
    """
    Store feedback data to Feedback_File.xlsx with auto-incrementing Task ID
    """
    try:
        # Read existing Excel file
        try:
            df_existing = pd.read_excel("Feedback_File.xlsx")
            # Get the next Task ID (auto-increment)
            if df_existing.empty:
                next_task_id = 1
            else:
                next_task_id = df_existing['Task ID'].max() + 1
        except FileNotFoundError:
            # If file doesn't exist, create new DataFrame
            next_task_id = 1
            df_existing = pd.DataFrame(columns=['Task ID', 'Input package', 'Recommended packages', 'Feedback package', 'Custom Feedback'])
        
        # Format data as comma-separated strings
        input_package_str = ", ".join(selected_placements)
        feedback_package_str = ", ".join(feedback_options)
        
        # Create new row
        new_row = {
            'Task ID': next_task_id,
            'Input package': input_package_str,
            'Recommended packages': recommended_packages,
            'Feedback package': feedback_package_str,
            'Custom Feedback': custom_feedback
        }
        
        # Append new row to existing data
        df_new = pd.concat([df_existing, pd.DataFrame([new_row])], ignore_index=True)
        
        # Save back to Excel file
        df_new.to_excel("Feedback_File.xlsx", index=False)
        
        return True
    except Exception as e:
        return False

# Hardcoded placement names
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

# Initialize session state
init_session_state()

# Set page configuration
st.set_page_config(
    page_title="Package Recommendation System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("Package Recommendation System")
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
    default=st.session_state['selected_placements']
)

# Update session state when selection changes
if selected_placements != st.session_state['selected_placements']:
    st.session_state['selected_placements'] = selected_placements

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
                st.session_state['recommendation_generated'] = True
                
            except Exception as e:
                st.error(f"An error occurred during recommendation: {str(e)}")

# Display recommendation results if they exist in session state
if st.session_state['high_impact_packages_result'] is not None:
    st.subheader("Recommendation Results")
    
    # Format the placement names for display
    placement_names_str = ", ".join(st.session_state['selected_placements'])
    
    # Display the high impact packages in markdown format
    st.markdown(f"""
For the given placement names, the most recommended high impact packages are:

{st.session_state['high_impact_packages_result']}
    """)
    
    if st.session_state['recommendation_generated']:
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
        default=st.session_state['feedback_options']
    )
    
    # Update session state when feedback selection changes
    if feedback_options != st.session_state['feedback_options']:
        st.session_state['feedback_options'] = feedback_options
    
    # Custom feedback text area
    custom_feedback = st.text_area(
        "Or provide your custom feedback:",
        value=st.session_state['custom_feedback'],
        placeholder="Please share your thoughts, suggestions, or any other feedback about the recommended packages...",
        height=100
    )
    
    # Update session state when custom feedback changes
    if custom_feedback != st.session_state['custom_feedback']:
        st.session_state['custom_feedback'] = custom_feedback
    
    # Feedback submit button
    if st.button("Submit Feedback"):
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
                success= store_feedback_to_excel(
                    st.session_state['selected_placements'], 
                    package_names_only, 
                    feedback_options,
                    custom_feedback.strip()
                )
                
                if success:
                    st.success("Thank you for your feedback!")
                    # Clear feedback options and custom feedback after successful submission
                    st.session_state['feedback_options'] = []
                    st.session_state['custom_feedback'] = ""
                else:
                    st.error("Failed to save feedback")

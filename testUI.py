import streamlit as st
import pandas as pd
from idRecommender_highImpactRecommender import recommend_packages

def store_feedback_to_excel(selected_placements, recommended_packages, feedback_options):
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
            df_existing = pd.DataFrame(columns=['Task ID', 'Input package', 'Recommended packages', 'Feedback package'])
        
        # Format data as comma-separated strings
        input_package_str = ", ".join(selected_placements)
        feedback_package_str = ", ".join(feedback_options)
        
        # Create new row
        new_row = {
            'Task ID': next_task_id,
            'Input package': input_package_str,
            'Recommended packages': recommended_packages,
            'Feedback package': feedback_package_str
        }
        
        # Append new row to existing data
        df_new = pd.concat([df_existing, pd.DataFrame([new_row])], ignore_index=True)
        
        # Save back to Excel file
        df_new.to_excel("Feedback_File.xlsx", index=False)
        
        return True, next_task_id
    except Exception as e:
        return False, str(e)

# Hardcoded placement names
PLACEMENT_NAMES = [
    'First Screen Masthead - US', 
    'Universal Guide Masthead - US', 
    'Universal Guide Masthead - US (Weekend Heavy-Up)',
    'Apps Store Masthead - US'
]

# Hardcoded high impact package names for feedback
HIGH_IMPACT_PACKAGES = [
    "No change required",
    "Spotlight Row Roadblock",
    "First Screen (2017-2021) Roadblock",
    "First Screen Immersive (22+) Roadblock",
    "First Screen Rotational Roadblock",
    "First Screen Immersive Roadblock",
    "First Screen All Models ('16-'25) Roadblock",
    "First Screen Immersive Rotational Roadblock"
]

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

# Placement names selector
selected_placements = st.sidebar.multiselect(
    "Select Placement Names:",
    PLACEMENT_NAMES
)

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
                
                # Display the results
                st.subheader("Recommendation Results")
                
                # Format the placement names for display
                placement_names_str = ", ".join(selected_placements)
                
                # Display the high impact packages in markdown format
                st.markdown(f"""
For the given placement names, the most recommended high impact packages are:

{high_impact_packages_result}
                """)
                
                # Store results in session state for feedback
                st.session_state['high_impact_packages_result'] = high_impact_packages_result
                
                # Show success message
                st.success("Recommendation completed successfully!")
                
            except Exception as e:
                st.error(f"An error occurred during recommendation: {str(e)}")
else:
    # Initial message
    st.info("Select placement names from the sidebar and click 'Recommend Packages' to get started.")

# Feedback section (only show after recommendations are made)
if 'high_impact_packages_result' in st.session_state:
    st.subheader("Feedback")
    st.markdown("Please provide feedback on the recommended high-impact packages:")
    
    # Feedback multiselect
    feedback_options = st.multiselect(
        "Select your feedback:",
        HIGH_IMPACT_PACKAGES
    )
    
    # Feedback submit button
    if st.button("Submit Feedback"):
        if not feedback_options:
            st.warning("Please select at least one feedback option.")
        else:
            # Process feedback
            if "No change required" in feedback_options and len(feedback_options) == 1:
                st.success("Thank you for your feedback! The recommendations met your expectations.")
            else:
                # Store feedback to Excel file
                success, result = store_feedback_to_excel(
                    selected_placements, 
                    st.session_state['high_impact_packages_result'], 
                    feedback_options
                )
                
                if success:
                    st.success(f"Thank you for your feedback!")
                else:
                    st.error(f"Failed to save feedback")

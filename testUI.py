import streamlit as st
from idRecommender_highImpactRecommender import recommend_packages

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

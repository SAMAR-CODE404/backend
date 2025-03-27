import streamlit as st
import os
from pathlib import Path
from graph import create_sequential_workflow, MnAagentState
from RAG.rag_llama import RAG
import logging
from tempfile import NamedTemporaryFile
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="M&A Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

def save_uploaded_file(uploaded_file):
    """Save uploaded file and return the path with explicit encoding"""
    try:
        with NamedTemporaryFile(delete=False, suffix='.txt', mode='wb') as f:
            f.write(uploaded_file.getvalue())
            temp_path = f.name
        
        # Try reading with different encodings
        encodings_to_try = ['utf-8', 'latin-1', 'windows-1252']
        
        for encoding in encodings_to_try:
            try:
                with open(temp_path, 'r', encoding=encoding) as f:
                    f.read()
                # If successful, write the file with this encoding
                with open(temp_path, 'r', encoding=encoding) as f_read:
                    content = f_read.read()
                
                with open(temp_path, 'w', encoding='utf-8') as f_write:
                    f_write.write(content)
                
                return temp_path
            except UnicodeDecodeError:
                continue
        
        st.error("Could not decode the file with standard encodings")
        return None
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def load_and_display_report(file_path, report_type):
    """Load and display reports with download button"""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                content = file.read()
                st.text_area(f"{report_type} Report", value=content, height=300)
                st.download_button(
                    label=f"Download {report_type} Report",
                    data=content,
                    file_name=f"{report_type.lower()}_report.txt",
                    mime="text/plain"
                )
        else:
            st.warning(f"{report_type} report not yet generated.")
    except Exception as e:
        st.error(f"Error loading {report_type} report: {e}")

def main():
    # Initialize session state
    if 'current_stage' not in st.session_state:
        st.session_state.current_stage = 'upload'
    if 'workflow_state' not in st.session_state:
        st.session_state.workflow_state = None

    st.title("M&A Analysis Dashboard")
    st.markdown("---")

    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # File uploads
    st.sidebar.subheader("Upload Company Documents")
    company_a_file = st.sidebar.file_uploader("Company A Document (PDF/TXT)", type=['pdf', 'txt'])
    company_b_file = st.sidebar.file_uploader("Company B Document (PDF/TXT)", type=['pdf', 'txt'])

    # Company names input
    company_a_name = st.sidebar.text_input("Company A Name", "")
    company_b_name = st.sidebar.text_input("Company B Name", "")

    # Progress indicator in sidebar
    progress_text = {
        'upload': 'Document Upload Stage',
        'research': 'Research Analysis Stage (1/3)',
        'financial': 'Financial Analysis Stage (2/3)',
        'operations': 'Operations Analysis Stage (3/3)',
        'complete': 'Analysis Complete!'
    }
    st.sidebar.markdown("### Progress")
    st.sidebar.info(progress_text.get(st.session_state.current_stage, 'Starting...'))

    # Main content area
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Company A Details")
        if company_a_file:
            st.success(f"File uploaded: {company_a_file.name}")

    with col2:
        st.subheader("Company B Details")
        if company_b_file:
            st.success(f"File uploaded: {company_b_file.name}")

    # Initialize analysis
    if st.button("Start Analysis", 
                 disabled=not (company_a_file and company_b_file and company_a_name and company_b_name)):
        try:
            with st.spinner("Processing documents and initializing analysis..."):
                # Save uploaded files
                company_a_path = save_uploaded_file(company_a_file)
                company_b_path = save_uploaded_file(company_b_file)

                # Initialize RAG instances
                rag_instances = {}
                indexes = {}
                retrievers = {}

                company_docs = {
                    company_a_name: company_a_path,
                    company_b_name: company_b_path
                }

                for company, text_path in company_docs.items():
                    progress_text = f"Initializing RAG for {company}..."
                    st.text(progress_text)
                    rag_instances[company] = RAG(text_path)
                    indexes[company] = rag_instances[company].create_db(db_name=str(company))
                    retrievers[company] = indexes[company].as_retriever()

                # Initialize state
                initial_state = MnAagentState(
                    company_a_name=company_a_name,
                    company_b_name=company_b_name,
                    company_a_doc=company_a_path,
                    company_b_doc=company_b_path,
                    rag_instances=rag_instances,
                    indexes=indexes,
                    retrievers=retrievers
                )
                st.session_state.workflow_state = initial_state
                st.session_state.current_stage = 'research'
                st.rerun()

        except Exception as e:
            st.error(f"An error occurred during initialization: {e}")
            logger.error(f"Initialization error: {e}", exc_info=True)

    # Analysis stages
    if st.session_state.current_stage != 'upload' and st.session_state.workflow_state:
        analysis_container = st.container()
        
        with analysis_container:
            # Research Stage
            if st.session_state.current_stage == 'research':
                st.subheader("Research Analysis Stage")
                col1, col2 = st.columns(2)
                
                with col1:
                    research_a = st.radio(
                        f"Approve research analysis for {company_a_name}?",
                        options=["Pending", "Approve", "Reject"],
                        key="research_a"
                    )
                
                with col2:
                    research_b = st.radio(
                        f"Approve research analysis for {company_b_name}?",
                        options=["Pending", "Approve", "Reject"],
                        key="research_b"
                    )
                
                if research_a == "Approve" and research_b == "Approve":
                    if st.button("Proceed to Financial Analysis"):
                        st.session_state.workflow_state.current_step = "human_approval_confirmed"
                        st.session_state.current_stage = 'financial'
                        st.rerun()

            # Financial Stage
            elif st.session_state.current_stage == 'financial':
                st.subheader("Financial Analysis Stage")
                col1, col2 = st.columns(2)
                
                with col1:
                    financial_a = st.radio(
                        f"Approve financial analysis for {company_a_name}?",
                        options=["Pending", "Approve", "Reject"],
                        key="financial_a"
                    )
                
                with col2:
                    financial_b = st.radio(
                        f"Approve financial analysis for {company_b_name}?",
                        options=["Pending", "Approve", "Reject"],
                        key="financial_b"
                    )
                
                if financial_a == "Approve" and financial_b == "Approve":
                    if st.button("Proceed to Operations Analysis"):
                        st.session_state.workflow_state.current_step = "human_approval_confirmed"
                        st.session_state.current_stage = 'operations'
                        st.rerun()

            # Operations Stage
            elif st.session_state.current_stage == 'operations':
                st.subheader("Operations Analysis Stage")
                col1, col2 = st.columns(2)
                
                with col1:
                    operations_a = st.radio(
                        f"Approve operations analysis for {company_a_name}?",
                        options=["Pending", "Approve", "Reject"],
                        key="operations_a"
                    )
                
                with col2:
                    operations_b = st.radio(
                        f"Approve operations analysis for {company_b_name}?",
                        options=["Pending", "Approve", "Reject"],
                        key="operations_b"
                    )
                
                if operations_a == "Approve" and operations_b == "Approve":
                    if st.button("Complete Analysis"):
                        st.session_state.workflow_state.current_step = "human_approval_confirmed"
                        st.session_state.current_stage = 'complete'
                        
                        # Execute final workflow
                        research_graph = create_sequential_workflow(st.session_state.workflow_state)
                        final_state = research_graph.invoke(st.session_state.workflow_state, 
                                                          config={"recursion_limit": 1000})
                        
                        # Display results
                        display_results(final_state, company_a_name, company_b_name)

            # Display current results and reports
            if st.session_state.current_stage != 'upload':
                display_current_progress(company_a_name, company_b_name)

def display_results(final_state, company_a_name, company_b_name):
    st.success("Analysis completed successfully!")
    
    # Results section
    st.markdown("## Final Analysis Results")
    
    # Tabs for different reports
    report_tabs = st.tabs(["Research Reports", "Financial Reports", "Operations Reports"])
    
    with report_tabs[0]:
        st.subheader("Research Reports")
        for company in [company_a_name, company_b_name]:
            st.markdown(f"### {company}")
            load_and_display_report(
                f"reports/{company}_research_report.txt",
                f"{company} Research"
            )

    with report_tabs[1]:
        st.subheader("Financial Reports")
        for company in [company_a_name, company_b_name]:
            st.markdown(f"### {company}")
            load_and_display_report(
                f"reports/{company}_financial_report.txt",
                f"{company} Financial"
            )

    with report_tabs[2]:
        st.subheader("Operations Reports")
        for company in [company_a_name, company_b_name]:
            st.markdown(f"### {company}")
            load_and_display_report(
                f"reports/{company}_operations_report.txt",
                f"{company} Operations"
            )

    # Display graph visualization
    if os.path.exists("assets/comprehensive_agent_graph.png"):
        st.markdown("## Workflow Visualization")
        st.image("assets/comprehensive_agent_graph.png")

def display_current_progress(company_a_name, company_b_name):
    st.markdown("### Current Progress")
    for report_type in ['research', 'financial', 'operations']:
        if st.session_state.current_stage in ['complete', report_type]:
            for company in [company_a_name, company_b_name]:
                if os.path.exists(f"reports/{company}_{report_type}_report.txt"):
                    with st.expander(f"{company} {report_type.capitalize()} Report"):
                        load_and_display_report(
                            f"reports/{company}_{report_type}_report.txt",
                            f"{company} {report_type.capitalize()}"
                        )

if __name__ == "__main__":
    main()
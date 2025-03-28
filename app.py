import streamlit as st
import os
from Main import create_sequential_workflow, MnAagentState
from RAG.rag_llama import RAG
import logging
from datetime import datetime
import glob
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def main():
    st.title("M&A Analysis Workflow")
    st.write("Upload company documents and analyze merger potential")

    # Sidebar for company information
    st.sidebar.header("Company Information")

    # Company A inputs
    st.sidebar.subheader("Company A")
    company_a_name = st.sidebar.text_input(
        "Company A Name", "Reliance_Industries_Limited"
    )
    company_a_doc = st.sidebar.file_uploader("Upload Company A Document", type=["txt"])

    # Company B inputs
    st.sidebar.subheader("Company B")
    company_b_name = st.sidebar.text_input("Company B Name", "180_Degree_Consulting")
    company_b_doc = st.sidebar.file_uploader("Upload Company B Document", type=["txt"])

    if st.sidebar.button("Start Analysis"):
        if company_a_doc is not None and company_b_doc is not None:
            try:
                # Save uploaded files
                os.makedirs("temp", exist_ok=True)
                company_a_path = os.path.join("temp", f"{company_a_name}_doc.txt")
                company_b_path = os.path.join("temp", f"{company_b_name}_doc.txt")

                with open(company_a_path, "wb") as f:
                    f.write(company_a_doc.getvalue())
                with open(company_b_path, "wb") as f:
                    f.write(company_b_doc.getvalue())

                # Initialize RAG instances
                with st.spinner("Initializing RAG instances..."):
                    rag_instances = {}
                    indexes = {}
                    retrievers = {}

                    company_docs = {
                        company_a_name: company_a_path,
                        company_b_name: company_b_path,
                    }

                    for company, text_path in company_docs.items():
                        st.write(f"Processing {company}...")
                        rag_instances[company] = RAG(text_path)
                        indexes[company] = rag_instances[company].create_db(
                            db_name=str(company)
                        )
                        retrievers[company] = indexes[company].as_retriever()

                # Initialize state and create workflow
                initial_state = MnAagentState(
                    company_a_name=company_a_name,
                    company_b_name=company_b_name,
                    company_a_doc=company_a_path,
                    company_b_doc=company_b_path,
                    rag_instances=rag_instances,
                    indexes=indexes,
                    retrievers=retrievers,
                )

                with st.spinner("Creating and executing workflow..."):
                    research_graph = create_sequential_workflow(initial_state)
                    final_state = research_graph.invoke(
                        initial_state, config={"recursion_limit": 1000}
                    )

                # Display results
                st.success("Analysis completed!")

                # Display the workflow graph
                if os.path.exists("assets/comprehensive_agent_graph.png"):
                    st.subheader("Workflow Visualization")
                    st.image("assets/comprehensive_agent_graph.png")

                # Display reports and analysis results
                if hasattr(final_state, "reports"):
                    st.subheader("Analysis Reports")
                    for report_name, report_content in final_state.reports.items():
                        with st.expander(f"{report_name} Report"):
                            st.write(report_content)

                # Add download section for generated reports
                st.subheader("Download Generated Reports")

                # Check for files in the report directory
                report_dir = "report"
                if os.path.exists(report_dir):
                    report_files = glob.glob(os.path.join(report_dir, "*.txt"))
                    for file_path in report_files:
                        try:
                            with open(file_path, "r", encoding="utf-8") as f:
                                file_content = f.read()
                                filename = os.path.basename(file_path)
                                st.download_button(
                                    label=f"Download {filename}",
                                    data=file_content,
                                    file_name=filename,
                                    mime="text/plain",
                                )
                                # Show file size info
                                st.info(f"File size: {len(file_content)/1024:.2f} KB")
                        except Exception as e:
                            logger.error(f"Error reading file {file_path}: {e}")

                # Check for legal report in root directory
                legal_report_path = "legal_report.txt"
                if os.path.exists(legal_report_path):
                    try:
                        with open(legal_report_path, "r", encoding="utf-8") as f:
                            legal_content = f.read()
                            st.download_button(
                                label="Download Legal Analysis Report",
                                data=legal_content,
                                file_name="legal_report.txt",
                                mime="text/plain",
                            )
                            st.info(f"File size: {len(legal_content)/1024:.2f} KB")
                    except Exception as e:
                        logger.error(f"Error reading legal report: {e}")

                # Show message if no files found
                if not (
                    os.path.exists(report_dir)
                    and glob.glob(os.path.join(report_dir, "*.txt"))
                ) and not os.path.exists(legal_report_path):
                    st.info("No report files were generated during the analysis.")

                # Remove old code that's no longer needed
                # Create a directory for downloads if it doesn't exist
                os.makedirs("downloads", exist_ok=True)

                # Save reports to files for record keeping (optional)
                if hasattr(final_state, "reports"):
                    for report_name, report_content in final_state.reports.items():
                        report_filename = f"{report_name}.txt"
                        report_path = os.path.join("downloads", report_filename)
                        with open(report_path, "w", encoding="utf-8") as f:
                            f.write(str(report_content))

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                logger.error(f"Error during analysis: {str(e)}")

            finally:
                # Cleanup temporary files
                try:
                    os.remove(company_a_path)
                    os.remove(company_b_path)
                except:
                    pass

        else:
            st.warning("Please upload documents for both companies.")

    # Display instructions
    with st.expander("How to Use"):
        st.write(
            """
        1. Enter the names of both companies in the sidebar with no gaps(Eg reliance_industries_limited)
        2. Upload text documents containing company information
        3. Click 'Start Analysis' to begin the M&A analysis
        4. The system will process the documents and generate various reports
        5. View the workflow visualization and analysis results below
        6. Download the generated reports using the download buttons
        7. It will take 10-15 minutes to complete the analysis
        8. Try to give all .txt file as parser dont implement ocr
        """
        )


if __name__ == "__main__":
    main()

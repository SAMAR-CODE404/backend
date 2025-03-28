import os
import logging
from typing import List, Dict, Any
import yaml
import uuid
from datetime import datetime
from pydantic import BaseModel, Field
from RAG.rag_llama import RAG
import sys
import json
from agents.states import MnAagentState
from langgraph.graph import StateGraph, END

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class ReportAgentNodes:
    def __init__(self, state: MnAagentState):
        """
        Initialize LegalAgentNodes with the given state and load necessary resources

        Args:
            state (MnAagentState): The current state of the multi-agent research process
        """
        self.state = state

        # Load legal report templates and prompts
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        prompts_path = os.path.join(project_root, "utils", "prompts.yaml")

        try:
            with open(prompts_path, "r", encoding="utf-8") as file:
                self.prompts = yaml.safe_load(file).get("Legal_Report_Prompts", {})
        except UnicodeDecodeError:
            # Fallback encoding strategy
            with open(prompts_path, "r", encoding="latin-1") as file:
                self.prompts = yaml.safe_load(file).get("Legal_Report_Prompts", {})
        except Exception as e:
            self.prompts = {}

        # Unique identifier for this report generation
        self.report_id = str(uuid.uuid4())

    def report_structure_creator(self, state: MnAagentState) -> MnAagentState:
        """
        Create the initial structure for the legal report

        Args:
            state (MnAagentState): Current state of the research process

        Returns:
            MnAagentState: Updated state with report structure
        """
        # Define a standard legal report structure
        report_structure = {
            "report_id": 1,
            "generated_at": datetime.now().isoformat(),
            "companies": [state.company_a_name, state.company_b_name],
            "sections": [
                {"name": "Executive Summary", "content": ""},
                {"name": "Company Background", "content": ""},
                {
                    "name": "Legal Compliance Analysis",
                    "subsections": [
                        "Regulatory Compliance",
                        "Corporate Governance",
                        "Risk Assessment",
                    ],
                    "content": "",
                },
                {"name": "Comparative Analysis", "content": ""},
                {"name": "Recommendations", "content": ""},
            ],
        }

        state.legal_report_structure = report_structure
        state.current_step = "report_structure_created"

        return state

    def section_template_generator(self, state: MnAagentState) -> MnAagentState:
        """
        Generate section templates for the legal report based on RAG queries

        Args:
            state (MnAagentState): Current state of the research process

        Returns:
            MnAagentState: Updated state with section templates
        """
        # Retrieve relevant information from RAG for each company
        templates = {}
        for company_name in [state.company_a_name, state.company_b_name]:
            rag_instance = state.rag_instances[company_name]

            # Example RAG queries for each section
            section_queries = {
                "Company Background": f"Provide key historical and organizational details about {company_name}",
                "Legal Compliance": f"Extract legal compliance and governance details for {company_name}",
                "Risk Assessment": f"Identify major legal and regulatory risks for {company_name}",
            }

            templates[company_name] = {}
            for section, query in section_queries.items():
                try:
                    context = rag_instance.rag_query(query, state.retrievers[company])
                    # if isinstance(context, str):
                    #     try:
                    #         context = json.loads(context)
                    #     except json.JSONDecodeError:
                    #         context = {}
                    templates[company_name][section] = context["result"]
                except Exception as e:
                    logger.error(f"RAG query error for {company_name} - {section}: {e}")
                    templates[company_name][section] = "No relevant information found."

        state.section_templates = templates
        state.current_step = "section_templates_generated"

        return state

    def rag_summary_generator(self, state: MnAagentState) -> MnAagentState:
        """
        Generate comprehensive summaries using RAG for the legal report

        Args:
            state (MnAagentState): Current state of the research process

        Returns:
            MnAagentState: Updated state with RAG-generated summaries
        """
        # Perform comprehensive RAG-based summarization
        for idx, section in enumerate(state.legal_report_structure["sections"]):
            company_summaries = []

            for company_name in [state.company_a_name, state.company_b_name]:
                rag_instance = state.rag_instances[company_name]

                # Detailed RAG query for each section
                try:
                    summary = rag_instance.rag_query(
                        f"Provide a comprehensive summary for the '{section['name']}' section about {company_name}",
                        state.retrievers[company_name],
                    )
                    # if isinstance(summary, str):
                    #     try:
                    #         context = json.loads(summary)
                    #     except json.JSONDecodeError:
                    #         context = {}
                    company_summaries.append(
                        {"company": company_name, "summary": summary["result"]}
                    )
                except Exception as e:
                    logger.error(f"RAG summary generation error: {e}")
                    company_summaries.append(
                        {
                            "company": company_name,
                            "summary": "Summary generation failed.",
                        }
                    )

            # Store summaries in the report structure
            state.legal_report_structure["sections"][idx][
                "company_summaries"
            ] = company_summaries

        state.current_step = "rag_summaries_generated"
        return state

    def consistency_checker(self, state: MnAagentState) -> MnAagentState:
        """
        Perform consistency checks on the generated legal report

        Args:
            state (MnAagentState): Current state of the research process

        Returns:
            MnAagentState: Updated state after consistency checks
        """
        # Implement consistency checks
        issues = []

        # Check for missing or incomplete sections
        for section in state.legal_report_structure["sections"]:
            if not section.get("company_summaries"):
                issues.append(f"Missing summaries for section: {section['name']}")

        # Check RAG source consistency
        for company_name in [state.company_a_name, state.company_b_name]:
            rag_instance = state.rag_instances[company_name]
            if not rag_instance.text:
                issues.append(f"No text data available for {company_name}")

        state.consistency_issues = issues
        state.current_step = "consistency_checked"

        return state

    def report_formatter(self, state: MnAagentState) -> MnAagentState:
        """
        Format the final legal report with a professional layout

        Args:
            state (MnAagentState): Current state of the research process

        Returns:
            MnAagentState: Updated state with formatted report
        """
        # Create a plain text formatted report
        report_content = f"""LEGAL COMPARATIVE ANALYSIS REPORT

REPORT DETAILS
Report ID: {state.legal_report_structure['report_id']}
Generated At: {state.legal_report_structure['generated_at']}
Companies Analyzed: {', '.join(state.legal_report_structure['companies'])}

"""

        # Add each section with company-wise summaries
        for section in state.legal_report_structure["sections"]:
            report_content += (
                f"\n{section['name'].upper()}\n{'=' * len(section['name'])}\n\n"
            )

            for company_summary in section.get("company_summaries", []):
                report_content += f"{company_summary['company']}\n{'-' * len(company_summary['company'])}\n"
                report_content += f"{company_summary['summary']}\n\n"

        # Add consistency issues if any
        if state.consistency_issues:
            report_content += "CONSISTENCY ISSUES\n==================\n\n"
            for issue in state.consistency_issues:
                report_content += f"* {issue}\n"

        # Save the report
        report_filename = f"legal_report.txt"
        try:
            with open(report_filename, "w", encoding="utf-8") as f:
                f.write(report_content)
            logger.info(f"Report saved: {report_filename}")
        except Exception as e:
            logger.error(f"Error saving report: {e}")

        state.final_report = report_content
        state.current_step = "report_formatted"

        return state


def create_report_agent_graph(mn_agent_state: MnAagentState):
    """
    Create and compile the legal agent workflow

    Args:
        mn_agent_state (MnAagentState): Initial state for the legal agent

    Returns:
        Compiled workflow graph
    """
    # Initialize legal agent nodes
    report_agent = ReportAgentNodes(mn_agent_state)

    # Define the graph workflow
    workflow = StateGraph(MnAagentState)

    # Add nodes
    workflow.add_node("report_structure_creator", report_agent.report_structure_creator)
    workflow.add_node(
        "section_template_generator", report_agent.section_template_generator
    )
    workflow.add_node("rag_summary_generator", report_agent.rag_summary_generator)
    workflow.add_node("consistency_checker", report_agent.consistency_checker)
    workflow.add_node("report_formatter", report_agent.report_formatter)

    # Define workflow edges
    workflow.set_entry_point("report_structure_creator")
    workflow.add_edge("report_structure_creator", "section_template_generator")
    workflow.add_edge("section_template_generator", "rag_summary_generator")
    workflow.add_edge("rag_summary_generator", "consistency_checker")
    workflow.add_edge("consistency_checker", "report_formatter")
    workflow.add_edge("report_formatter", END)

    # Compile the workflow
    return workflow.compile()


if __name__ == "__main__":
    rag_instances = {}
    indexes = {}
    retrievers = {}

    company_docs = {
        "Reliance_Industries_Limited": "/home/naba/Desktop/backend/RIL-Integrated-Annual-Report-2023-24_parsed.txt",
        "180_Degree_Consulting": "/home/naba/Desktop/backend/dc.txt",
    }

    for company, text_path in company_docs.items():
        logger.info(f"Initializing RAG for {company} with document: {text_path}")
        rag_instances[company] = RAG(text_path)
        indexes[company] = rag_instances[company].create_db(db_name=str(company))
        retrievers[company] = indexes[company].as_retriever()

    # Initialize MnAagentState with RAG instances
    initial_state = MnAagentState(
        company_a_name="Reliance_Industries_Limited",
        company_b_name="180_Degree_Consulting",
        company_a_doc="/home/naba/Desktop/backend/RIL-Integrated-Annual-Report-2023-24_parsed.txt",
        company_b_doc="/home/naba/Desktop/backend/dc.txt",
        rag_instances=rag_instances,
        indexes=indexes,
        retrievers=retrievers,
    )

    # Create and invoke the sequential workflow
    research_graph = create_report_agent_graph(initial_state)
    final_state = research_graph.invoke(initial_state, config={"recursion_limit": 1000})

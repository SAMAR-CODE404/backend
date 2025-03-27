from pydantic import BaseModel, Field
from typing import TypedDict, Dict, List, Any, Optional
from agents.states import MnAagentState
from datetime import datetime
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
from RAG.rag_llama import RAG
import logging
import os
from langgraph.graph import StateGraph, END
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Log to console
    ]
)
logger = logging.getLogger(__name__)

class OpsAgentNodes:
    def __init__(self, state: MnAagentState, company: str, approval: bool):

        """
        Initialize Operations Agent for a specific company
        
        Args:
            state (MnAagentState): The current state of the multi-agent system
            company (str): Identifier for the company ('a' or 'b')
        """
        self.approval = approval
        self.state = state
        if company == 'a':
            self.company_name = state.company_a_name
        else:
            self.company_name = state.company_b_name
        
        # Load prompts
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        prompts_path = os.path.join(project_root, "utils", "prompts.yaml")
        with open(prompts_path, "r") as file:
            self.prompts = yaml.safe_load(file)["Ops_Agent_prompt"]
        logger.info(f"Loaded prompts from {prompts_path}")

    def supply_chain_analysis(self, state: MnAagentState) -> MnAagentState:
        """
        Perform in-depth supply chain analysis for the company
        
        Args:
            state (MnAagentState): Current state of the multi-agent system
        
        Returns:
            MnAagentState: Updated state with supply chain analysis
        """
        state.current_step = "supply_chain_analysis"
        state.supply_chain_analyst = {self.company_name: "Supply Chain Analysis:\n"}
        
        # Use RAG to extract supply chain insights
        response = self.state.rag_instances[self.company_name].rag_query(
            query_text=self.prompts["supply_chain_prompt"].format(company_name=self.company_name),
            retriever=self.state.retrievers[self.company_name]
        )
        
        state.supply_chain_analyst[self.company_name] += response["result"]
        return state

    def industry_positioning(self, state: MnAagentState) -> MnAagentState:
        """
        Analyze the company's positioning within its industry
        
        Args:
            state (MnAagentState): Current state of the multi-agent system
        
        Returns:
            MnAagentState: Updated state with industry positioning insights
        """
        state.current_step = "industry_positioning"
        state.industry_position = {self.company_name: "Industry Positioning Analysis:\n"}
        
        # Use RAG to extract industry positioning insights
        response = self.state.rag_instances[self.company_name].rag_query(
            query_text=self.prompts["industry_positioning_prompt"].format(company_name=self.company_name),
            retriever=self.state.retrievers[self.company_name]
        )
        
        state.industry_position[self.company_name] += response["result"]
        return state

    def operations_reporting(self, state: MnAagentState) -> MnAagentState:
        """
        Generate comprehensive operations report
        
        Args:
            state (MnAagentState): Current state of the multi-agent system
        
        Returns:
            MnAagentState: Updated state with operations report
        """
        try:
            # Combine supply chain and industry positioning data
            combined_data = '\n'.join([
                state.supply_chain_analyst.get(self.company_name, ''),
                state.industry_position.get(self.company_name, '')
            ])
            
            # Create RAG instance for report generation
            rag_instances = RAG(combined_data)
            
            indexes = rag_instances.create_db(db_name=str(self.company_name))
            retrievers = indexes.as_retriever()
            
            state.current_step = "operations_reporting"
            response = rag_instances.rag_query(
                query_text=self.prompts["operations_reporting_prompt"].format(company_name=self.company_name),
                retriever=retrievers
            )
            
            # Assign report based on company
            if self.company_name == state.company_a_name:
                state.ops_report_a = response["result"]
            else:
                state.ops_report_b = response["result"]
            
            # Create output directory and save report
            output_dir = "report"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{self.company_name}_operations_report.txt")
            
            with open(output_path, "w") as f:
                f.write(response["result"])
            
            return state
        
        except Exception as e:
            logger.error(f"Error in operations reporting for {self.company_name}: {e}")
            raise

    def human_approval(self, state: MnAagentState) -> MnAagentState:
        """
        Ask human for approval to proceed with the report
        
        Args:
            state (MnAagentState): Current state of the multi-agent system
        
        Returns:
            MnAagentState: Updated state based on human approval
        """
        proceed = self.approval
        
        if proceed == 1:
            state.current_step = "human_approval_confirmed"
            return state
        
        state.current_step = "human_approval_rejected"
        return state


def create_workflow(mn_agent_state: MnAagentState, company: str):
    """
    Create a workflow for the Operations Agent
    
    Args:
        mn_agent_state (MnAagentState): Initial multi-agent state
        company (str): Company identifier
    
    Returns:
        Compiled workflow graph
    """
    ops_agent = OpsAgentNodes(mn_agent_state, company)
    workflow = StateGraph(MnAagentState)
    
    # Add nodes
    workflow.add_node("supply_chain_analysis", ops_agent.supply_chain_analysis)
    workflow.add_node("human_approval", ops_agent.human_approval)
    workflow.add_node("industry_positioning", ops_agent.industry_positioning)
    workflow.add_node("operations_reporting", ops_agent.operations_reporting)
    
    # Define workflow edges
    workflow.set_entry_point("supply_chain_analysis")
    workflow.add_edge("supply_chain_analysis", "industry_positioning")
    workflow.add_edge("industry_positioning", "human_approval")
    
    # Add conditional edge for human approval
    workflow.add_conditional_edges(
        "human_approval", 
        lambda state: "save_report" if state.current_step == "human_approval_confirmed" else END,
        {
            "save_report": "operations_reporting",
            END: END
        }
    )
    
    # Final edge to END
    workflow.add_edge("operations_reporting", END)
    
    compiled_graph = workflow.compile()
    return compiled_graph

if __name__ == "__main__":
    rag_instances = {}
    indexes = {}
    retrievers = {}
    
    company_docs = {
        "Reliance_Industries_Limited": "/home/naba/Desktop/backend/RIL-Integrated-Annual-Report-2023-24_parsed.txt",
        "180_Degree_Consulting": "/home/naba/Desktop/backend/dc.txt"
    }
    
    for company, text_path in company_docs.items():
        logger.info(f"Initializing RAG for {company} with document: {text_path}")
        rag_instances[company] = RAG(text_path)
        indexes[company] = rag_instances[company].create_db(db_name=str(company))
        retrievers[company] = indexes[company].as_retriever()

    initial_state = MnAagentState(
        company_a_name="Reliance_Industries_Limited",
        company_b_name="180_Degree_Consulting",
        company_a_doc="/home/naba/Desktop/backend/RIL-Integrated-Annual-Report-2023-24_parsed.txt",
        company_b_doc="/home/naba/Desktop/backend/dc.txt",
        rag_instances=rag_instances,
        indexes=indexes,
        retrievers=retrievers
    )
    research_graph = create_workflow(initial_state, 'a')
    final_state = research_graph.invoke(initial_state, config={"recursion_limit": 1000})
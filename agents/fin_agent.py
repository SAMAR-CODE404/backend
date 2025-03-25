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

class FinAgentNodes:
    def __init__(self, state: MnAagentState, company: str):
        self.state = state
        if company == 'a':
            self.company_name = state.company_a_name
        else:
            self.company_name = state.company_b_name
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        prompts_path = os.path.join(project_root, "utils", "prompts.yaml")
        with open(prompts_path, "r") as file:
            self.prompts = yaml.safe_load(file)["Fin_Agent_prompt"]
        logger.info(f"Loaded prompts from {prompts_path}")

    def DCF_modelling(self, state: MnAagentState) -> MnAagentState:
        """do DCF modelling for the company"""
        self.state.current_step = "DCF_modelling"
        state.dcf_models[self.company_name] = "DCF model:\n"
        response = self.state.rag_instances[self.company_name].rag_query(query_text=self.prompts["dcf_prompt"].format(company_name=self.company_name),retriever=self.state.retrievers[self.company_name])
        state.dcf_models[self.company_name] = state.dcf_models[self.company_name].join(response["result"])
        return state
    
    def financial_ratios(self, state: MnAagentState) -> MnAagentState:
        """Calculate financial ratios for the company"""
        state.current_step = "financial_ratios"
        state.financial_ratios[self.company_name] = "Financial Ratios:\n"
        response = self.state.rag_instances[self.company_name].rag_query(query_text=self.prompts["financial_ratios_prompt"].format(company_name=self.company_name),retriever=self.state.retrievers[self.company_name])
        state.financial_ratios[self.company_name] = state.financial_ratios[self.company_name].join(response["result"])
        return state
    
    def financial_reporting(self, state: MnAagentState) -> MnAagentState:
        """Generate financial reports for the company"""
        try:
            # Combine financial ratios and DCF models safely
            combined_data = '\n'.join([
                state.financial_ratios.get(self.company_name, ''),
                state.dcf_models.get(self.company_name, '')
            ])
            
            # Create RAG instance safely
            rag_instances = RAG(combined_data)
            
            indexes = rag_instances.create_db(db_name=str(self.company_name))
            retrievers = indexes.as_retriever()
            
            state.current_step = "financial_reporting"
            response = rag_instances.rag_query(
                query_text=self.prompts["financial_reporting_prompt"].format(company_name=self.company_name),
                retriever=retrievers
            )
            
            # Assign report based on company
            if self.company_name == state.company_a_name:
                state.fin_report_a = response["result"]
            else:
                state.fin_report_b = response["result"]
            
            # Create output directory and save report
            output_dir = "report"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{self.company_name}_financial_report.txt")
            
            with open(output_path, "w") as f:
                f.write(response["result"])
            
            return state
        
        except Exception as e:
            logger.error(f"Error in financial reporting for {self.company_name}: {e}")
            # Optional: You might want to raise the exception or handle it differently
            raise
    
    def human_approval(self, state: MnAagentState) -> MnAagentState:
        """
        Ask human whether to proceed with search
        """
        proceed = input("\nDo you want to proceed with the report? (yes/no): ").lower().strip()
        
        if proceed == 'yes':
            state.current_step = "human_approval_confirmed"
            return state
        
        state.current_step = "human_approval_rejected"
        return state
    def should_continue(self, state: MnAagentState) -> str:
        """
        Determine if more queries need to be processed
        """
        return END
    
def create_workflow(mn_agent_state: MnAagentState, company: str):
    fin_agent = FinAgentNodes(mn_agent_state, company)
    workflow = StateGraph(MnAagentState)

    workflow.add_node("DCF_modelling", fin_agent.DCF_modelling)
    workflow.add_node("human_approval", fin_agent.human_approval)
    workflow.add_node("financial_ratio", fin_agent.financial_ratios)
    workflow.add_node("financial_reporting", fin_agent.financial_reporting)
    
    # Define edges
    workflow.set_entry_point("DCF_modelling")
    workflow.add_edge("DCF_modelling", "financial_ratio")
    workflow.add_edge("financial_ratio", "human_approval")
    workflow.add_conditional_edges(
        "human_approval", 
        lambda state: "save_report" if state.current_step == "human_approval_confirmed" else END,
        {
            "save_report": "financial_reporting",
            END: END
        }
    )
    workflow.add_conditional_edges(
        "financial_reporting",
        lambda state: END,
        {
            END: END
        }
    )
    compiled_graph = workflow.compile()
    try:
        output_dir = "assets"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "fin_agent_graph.png")
        graph_image = compiled_graph.get_graph().draw_mermaid_png(
            curve_style=CurveStyle.LINEAR,
            node_colors=NodeStyles(first="#ffdfba", last="#baffc9", default="#fad7de"),
            wrap_label_n_words=9,
            output_file_path=None,
            draw_method=MermaidDrawMethod.PYPPETEER,
            background_color="white",
            padding=10
        )
        with open(output_path, "wb") as f:
            f.write(graph_image)  
        logger.info(f"Graph visualization saved to {output_path}")
    except Exception as e:
        logger.warning(f"Could not save graph visualization: {e}")
    
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
        rag_instances=rag_instances,
        indexes=indexes,
        retrievers=retrievers
    )
    research_graph = create_workflow(initial_state, 'a')
    final_state = research_graph.invoke(initial_state, config={"recursion_limit": 1000})
from pydantic import BaseModel, Field
from typing import TypedDict, Dict, List, Any, Optional
from agents.states import MnAagentState
from datetime import datetime
from RAG.rag_llama import RAG
from utils.chat import Chat
import yaml
from langgraph.graph import StateGraph, END
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
import os
import logging
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Log to console
    ]
)
logger = logging.getLogger(__name__)

class LegalAgentNodes:
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
            self.prompts = yaml.safe_load(file)["Legal_Prompt"]
        logger.info(f"Loaded prompts from {prompts_path}")

    def take_legal_document(self, state: MnAagentState) -> MnAagentState:
        while True:
            try:
                company_a_legal = input(f"Enter the path to the legal document for {state.company_a_name}: ")
                company_b_legal = input(f"Enter the path to the legal document for {state.company_b_name}: ")
                
                # Validate file paths
                if not os.path.exists(company_a_legal) or not os.path.exists(company_b_legal):
                    print("Error: One or both document paths are invalid. Please try again.")
                    continue
                
                company_docs = {
                    f"{state.company_a_name}": f"{company_a_legal}",
                    f"{state.company_b_name}": f"{company_b_legal}"
                }
                
                rag_instances = {}
                indexes = {}
                retrievers = {}
                
                for company, text_path in company_docs.items():
                    logger.info(f"Initializing RAG for {company} with document: {text_path}")
                    rag_instances[company] = RAG(text_path)
                    indexes[company] = rag_instances[company].create_db(db_name=str(company))
                    retrievers[company] = indexes[company].as_retriever()
                
                state.indexes = indexes
                state.retrievers = retrievers
                state.rag_instances = rag_instances
                
                return state
            
            except Exception as e:
                print(f"An error occurred: {e}")
                retry = input("Do you want to try again? (yes/no): ").lower().strip()
                if retry != 'yes':
                    raise SystemExit("Document input cancelled by user.")

    def risk_assessment(self, state: MnAagentState) -> MnAagentState:
        # Existing risk assessment logic remains the same
        state.current_step = "risk_assessment"
        state.risk_assessment[self.company_name] = "Risk_assessment\n"
        response = self.state.rag_instances[self.company_name].rag_query(
            query_text=self.prompts["risk_message"],
            retriever=self.state.retrievers[self.company_name]
        )
        state.risk_assessment[self.company_name] = state.risk_assessment[self.company_name].join(response["result"])
        return state
    
    def antitrust_checker(self, state: MnAagentState) -> MnAagentState:
        # antitrust
        state.current_step = "anti_trust_assessment"
        state.risk_check[self.company_name] = "Anti_Trust_assessment\n"
        response = self.state.rag_instances[self.company_name].rag_query(query_text=self.prompts["anti_trust_message"],retriever=self.state.retrievers[self.company_name])
        state.risk_check[self.company_name] = state.risk_check[self.company_name].join(response["result"])
        return state
    
    def legal_report(self, state: MnAagentState) -> MnAagentState:
        # legal report
        try:
            # Combine financial ratios and DCF models safely
            combined_data = '\n'.join([
                state.financial_ratios.get(self.company_name, ''),
                state.dcf_models.get(self.company_name, '')
            ])

            rag_instances = RAG(combined_data)
            
            indexes = rag_instances.create_db(db_name=str(self.company_name))
            retrievers = indexes.as_retriever()
            
            state.current_step = "Legal_reporting"
            response = rag_instances.rag_query(
                query_text=self.prompts["legal_report"],
                retriever=retrievers
            )
            
            # Assign report based on company
            if self.company_name == state.company_a_name:
                state.Legal_report_a  = response["result"]
            else:
                state.Legal_report_b = response["result"]
            
            output_dir = "report"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{self.company_name}_Legal_report.txt")
            
            with open(output_path, "w") as f:
                f.write(response["result"])
            
            return state
        
        except Exception as e:
            logger.error(f"Error in Legal reporting for {self.company_name}: {e}")
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
    
    def should_continue(self, state: MnAagentState) -> Any:
        """
        Determine if more queries need to be processed
        """
        return END
    
def create_legal_workflow(mn_agent_state: MnAagentState, company: str):
    legal_agent = LegalAgentNodes(mn_agent_state, company)
    workflow = StateGraph(MnAagentState)

    workflow.add_node("take_legal_document", legal_agent.take_legal_document)
    workflow.add_node("risk_assessment", legal_agent.risk_assessment)
    workflow.add_node("antitrust_checker", legal_agent.antitrust_checker)
    workflow.add_node("legal_report", legal_agent.legal_report)
    workflow.add_node("human_approval", legal_agent.human_approval)
    
    # Modify edges to handle document input more robustly
    workflow.set_entry_point("take_legal_document")
    
    # Add a conditional edge after document input to ensure valid documents
    workflow.add_conditional_edges(
        "take_legal_document", 
        lambda state: "proceed" if state.rag_instances and len(state.rag_instances) == 2 else "retry",
        {
            "proceed": "risk_assessment",
            "retry": "take_legal_document"
        }
    )
    
    # Rest of the workflow remains the same
    workflow.add_edge("risk_assessment", "antitrust_checker")
    workflow.add_edge("antitrust_checker", "legal_report")
    workflow.add_edge("legal_report", "human_approval")
    
    workflow.add_conditional_edges(
        "human_approval", 
        lambda state: "finalize_report" if state.current_step == "human_approval_confirmed" else END,
        {
            "finalize_report": "legal_report",
            END: END
        }
    )
    
    compiled_graph = workflow.compile()
    
    # Existing graph visualization code remains the same
    try:
        output_dir = "assets"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "legal_agent_graph.png")
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
    state = MnAagentState(
        company_a_name="Company A",
        company_b_name="Company B"
    )
    compiled_graph = create_legal_workflow(state, "a")
    logger.info(f"Legal workflow graph created: {compiled_graph}")
    compiled_graph = create_legal_workflow(state, "b")
    logger.info(f"Legal workflow graph created: {compiled_graph}")
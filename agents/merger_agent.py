import os
import yaml
from typing import Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from RAG.rag_llama import RAG
from langgraph.graph import StateGraph, END
from agents.states import MnAagentState
import logging

logger = logging.getLogger(__name__)

class MergerValuationAgent:
    def __init__(self, state: MnAagentState):
        """
        Initialize Merger Valuation Agent with the merged state
        
        Args:
            state (MnAagentState): Merged state containing company information
        """
        self.state = state
        
        # Load prompts for merger valuation
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        prompts_path = os.path.join(project_root, "utils", "prompts.yaml")
        
        try:
            with open(prompts_path, "r") as file:
                self.prompts = yaml.safe_load(file)["Merger_Valuation_Agent_Prompts"]
        except Exception as e:
            logger.error(f"Error loading prompts: {e}")
            self.prompts = {}
    
    def validate_merger_feasibility(self, state: MnAagentState) -> MnAagentState:
        """
        Validate the feasibility of the merger based on financial, legal, and strategic criteria
        
        Args:
            state (MnAagentState): Current state of the merger process
        
        Returns:
            MnAagentState: Updated state with merger feasibility assessment
        """
        state.current_step = "merger_feasibility_assessment"
        
        # Combine financial reports for comprehensive analysis
        combined_financial_data = f"""
        Company A Financial Report: {state.fin_report_a or 'No report available'}
        Company B Financial Report: {state.fin_report_b or 'No report available'}
        """
        
        # Use RAG to assess merger feasibility
        merger_feasibility_rag = RAG(combined_financial_data)
        indexes = merger_feasibility_rag.create_db(db_name="merger_feasibility")
        retriever = indexes.as_retriever()
        
        try:
            feasibility_response = merger_feasibility_rag.rag_query(
                query_text=self.prompts.get("merger_feasibility_prompt", "Assess the feasibility of merger between the two companies"),
                retriever=retriever
            )
            
            state.merger_acquisition_details['feasibility_assessment'] = feasibility_response['result']
            
            # Save feasibility report
            output_dir = "merger_reports"
            os.makedirs(output_dir, exist_ok=True)
            feasibility_report_path = os.path.join(output_dir, "merger_feasibility_report.txt")
            
            with open(feasibility_report_path, "w") as f:
                f.write(feasibility_response['result'])
            
            state.merger_report = feasibility_report_path
            
        except Exception as e:
            logger.error(f"Error in merger feasibility assessment: {e}")
            state.error = f"Merger feasibility assessment failed: {str(e)}"
        
        return state
    
    def calculate_merger_valuation(self, state: MnAagentState) -> MnAagentState:
        """
        Calculate the merger valuation using advanced techniques
        
        Args:
            state (MnAagentState): Current state of the merger process
        
        Returns:
            MnAagentState: Updated state with merger valuation details
        """
        state.current_step = "merger_valuation_calculation"
        
        # Combine DCF models and financial ratios
        valuation_data = f"""
        Company A DCF Model: {state.dcf_models.get(state.company_a_name, 'No DCF model available')}
        Company B DCF Model: {state.dcf_models.get(state.company_b_name, 'No DCF model available')}
        Company A Financial Ratios: {state.financial_ratios.get(state.company_a_name, 'No financial ratios available')}
        Company B Financial Ratios: {state.financial_ratios.get(state.company_b_name, 'No financial ratios available')}
        """
        
        # Use RAG for advanced valuation analysis
        valuation_rag = RAG(valuation_data)
        indexes = valuation_rag.create_db(db_name="merger_valuation")
        retriever = indexes.as_retriever()
        
        try:
            valuation_response = valuation_rag.rag_query(
                query_text=self.prompts.get("merger_valuation_prompt", "Calculate comprehensive merger valuation"),
                retriever=retriever
            )
            
            state.merger_acquisition_details['valuation_details'] = valuation_response['result']
            
            # Save valuation report
            output_dir = "merger_reports"
            os.makedirs(output_dir, exist_ok=True)
            valuation_report_path = os.path.join(output_dir, "merger_valuation_report.txt")
            
            with open(valuation_report_path, "w") as f:
                f.write(valuation_response['result'])
            
            state.merger_report = valuation_report_path
            
        except Exception as e:
            logger.error(f"Error in merger valuation calculation: {e}")
            state.error = f"Merger valuation calculation failed: {str(e)}"
        
        return state
    
    def assess_integration_risks(self, state: MnAagentState) -> MnAagentState:
        """
        Assess potential risks in merger integration
        
        Args:
            state (MnAagentState): Current state of the merger process
        
        Returns:
            MnAagentState: Updated state with integration risk assessment
        """
        state.current_step = "integration_risk_assessment"
        
        # Combine supply chain and industry position data
        integration_risk_data = f"""
        Company A Supply Chain: {state.supply_chain_analyst.get(state.company_a_name, 'No supply chain data')}
        Company B Supply Chain: {state.supply_chain_analyst.get(state.company_b_name, 'No supply chain data')}
        Company A Industry Position: {state.industry_position.get(state.company_a_name, 'No industry position data')}
        Company B Industry Position: {state.industry_position.get(state.company_b_name, 'No industry position data')}
        """
        
        # Use RAG for integration risk analysis
        risk_rag = RAG(integration_risk_data)
        indexes = risk_rag.create_db(db_name="integration_risks")
        retriever = indexes.as_retriever()
        
        try:
            risk_response = risk_rag.rag_query(
                query_text=self.prompts.get("integration_risks_prompt", "Assess potential risks in merger integration"),
                retriever=retriever
            )
            
            state.risk_check['integration_risks'] = risk_response['result']
            
            # Save risk assessment report
            output_dir = "merger_reports"
            os.makedirs(output_dir, exist_ok=True)
            risk_report_path = os.path.join(output_dir, "integration_risks_report.txt")
            
            with open(risk_report_path, "w") as f:
                f.write(risk_response['result'])
            
            state.merger_report = risk_report_path
            
        except Exception as e:
            logger.error(f"Error in integration risk assessment: {e}")
            state.error = f"Integration risk assessment failed: {str(e)}"
        
        return state
   

def create_merger_valuation_workflow(merger_agent: MergerValuationAgent) -> StateGraph:
    """
    Create a comprehensive merger valuation workflow
    
    Args:
        merger_agent (MergerValuationAgent): Merger valuation agent instance
    
    Returns:
        StateGraph: Compiled workflow for merger valuation
    """
    workflow = StateGraph(MnAagentState)
    
    # Add nodes for each stage of merger valuation
    workflow.add_node("validate_merger_feasibility", merger_agent.validate_merger_feasibility)
    workflow.add_node("calculate_merger_valuation", merger_agent.calculate_merger_valuation)
    workflow.add_node("assess_integration_risks", merger_agent.assess_integration_risks)
    workflow.add_node("finalize_merger_report", lambda state: state)
    
    # Set entry point and define workflow
    workflow.set_entry_point("validate_merger_feasibility")
    
    # Define edges between nodes
    workflow.add_edge("validate_merger_feasibility", "calculate_merger_valuation")
    workflow.add_edge("calculate_merger_valuation", "assess_integration_risks")
    workflow.add_edge("assess_integration_risks", "finalize_merger_report")
    
    # Set finish point
    workflow.set_finish_point("finalize_merger_report")
    
    # Compile the workflow
    return workflow.compile()

if __name__ == "__main__":
    # Initialize logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    # Initialize RAG instances
    rag_instances = {}
    indexes = {}
    retrievers = {}
    
    # Define company documents
    company_docs = {
        "Reliance_Industries_Limited": "/home/naba/Desktop/backend/RIL-Integrated-Annual-Report-2023-24_parsed.txt",
        "180_Degree_Consulting": "/home/naba/Desktop/backend/dc.txt"
    }
    
    # Create RAG instances for each company
    for company, text_path in company_docs.items():
        logger.info(f"Initializing RAG for {company} with document: {text_path}")
        rag_instances[company] = RAG(text_path)
        indexes[company] = rag_instances[company].create_db(db_name=str(company))
        retrievers[company] = indexes[company].as_retriever()
    
    # Create initial state
    initial_state = MnAagentState(
        company_a_name="Reliance_Industries_Limited",
        company_b_name="180_Degree_Consulting",
        company_a_doc="/home/naba/Desktop/backend/RIL-Integrated-Annual-Report-2023-24_parsed.txt",
        company_b_doc="/home/naba/Desktop/backend/dc.txt",
        rag_instances=rag_instances,
        indexes=indexes,
        retrievers=retrievers
    )
    
    # Create merger agent and workflow
    merger_agent = MergerValuationAgent(initial_state)
    merger_valuation_workflow = create_merger_valuation_workflow(merger_agent)
    
    # Run merger valuation
    result = merger_valuation_workflow.invoke(initial_state, config={"recursion_limit": 1000})
    print("Merger Valuation Complete!")
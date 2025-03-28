from agents.research_agent import ResearchAgentNodes
from agents.fin_agent import FinAgentNodes
from agents.operations_agent import OpsAgentNodes
from agents.states import MnAagentState
from datetime import datetime
from langgraph.graph import StateGraph, END
from langgraph.channels.last_value import LastValue
import os
import logging 
from agents.merger_agent import MergerValuationAgent
from RAG.rag_llama import RAG
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
from agents.legal_agent import MergerLegalAgent
from agents.report_agent import ReportAgentNodes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Log to console
    ]
)
logger = logging.getLogger(__name__)


def create_sequential_workflow(mn_agent_state: MnAagentState):
    """
    Create a comprehensive workflow for multi-company analysis and merger valuation
    """
    # Initialize agent nodes for both companies
    research_agent_a = ResearchAgentNodes(mn_agent_state, 'a', approval=True)
    fin_agent_a = FinAgentNodes(mn_agent_state, 'a', approval=True)
    ops_agent_a = OpsAgentNodes(mn_agent_state, 'a', approval=True)
    
    research_agent_b = ResearchAgentNodes(mn_agent_state, 'b', approval=True)
    fin_agent_b = FinAgentNodes(mn_agent_state, 'b', approval=True)
    ops_agent_b = OpsAgentNodes(mn_agent_state, 'b', approval=True)
    
    # Create merger valuation agent
    merger_agent = MergerValuationAgent(mn_agent_state)
    legal_agent = MergerLegalAgent(mn_agent_state)
    report_agent = ReportAgentNodes(mn_agent_state)
    # Define the graph workflow
    workflow = StateGraph(MnAagentState)
    
    # Add nodes for comprehensive workflow
    # Company A Workflow Nodes
    workflow.add_node("generate_queries_a", research_agent_a.generate_queries)
    workflow.add_node("research_human_approval_a", research_agent_a.human_approval)
    workflow.add_node("web_search_a", research_agent_a.web_search)
    workflow.add_node("DCF_modelling_a", fin_agent_a.DCF_modelling)
    workflow.add_node("financial_ratio_a", fin_agent_a.financial_ratios)
    workflow.add_node("fin_human_approval_a", fin_agent_a.human_approval)
    workflow.add_node("financial_reporting_a", fin_agent_a.financial_reporting)
    workflow.add_node("supply_chain_analysis_a", ops_agent_a.supply_chain_analysis)
    workflow.add_node("industry_positioning_a", ops_agent_a.industry_positioning)
    workflow.add_node("ops_human_approval_a", ops_agent_a.human_approval)
    workflow.add_node("operations_reporting_a", ops_agent_a.operations_reporting)
    workflow.add_node("assess_regulatory_compliance", legal_agent.assess_regulatory_compliance)
    workflow.add_node("conduct_legal_due_diligence", legal_agent.conduct_legal_due_diligence)
    workflow.add_node("assess_potential_legal_risks", legal_agent.assess_potential_legal_risks)
    workflow.add_node("finalize_legal_report", lambda state: state)

    
    # Company B Workflow Nodes
    workflow.add_node("generate_queries_b", research_agent_b.generate_queries)
    workflow.add_node("research_human_approval_b", research_agent_b.human_approval)
    workflow.add_node("web_search_b", research_agent_b.web_search)
    workflow.add_node("DCF_modelling_b", fin_agent_b.DCF_modelling)
    workflow.add_node("financial_ratio_b", fin_agent_b.financial_ratios)
    workflow.add_node("fin_human_approval_b", fin_agent_b.human_approval)
    workflow.add_node("financial_reporting_b", fin_agent_b.financial_reporting)
    workflow.add_node("supply_chain_analysis_b", ops_agent_b.supply_chain_analysis)
    workflow.add_node("industry_positioning_b", ops_agent_b.industry_positioning)
    workflow.add_node("ops_human_approval_b", ops_agent_b.human_approval)
    workflow.add_node("operations_reporting_b", ops_agent_b.operations_reporting)
    
    # Merger Valuation Nodes
    workflow.add_node("validate_merger_feasibility", merger_agent.validate_merger_feasibility)
    workflow.add_node("calculate_merger_valuation", merger_agent.calculate_merger_valuation)
    workflow.add_node("assess_integration_risks", merger_agent.assess_integration_risks)
    workflow.add_node("finalize_merger_report", lambda state: state)
    workflow.add_node("report_structure_creator", report_agent.report_structure_creator)
    workflow.add_node("section_template_generator", report_agent.section_template_generator)
    workflow.add_node("rag_summary_generator", report_agent.rag_summary_generator)
    workflow.add_node("consistency_checker", report_agent.consistency_checker)
    workflow.add_node("report_formatter", report_agent.report_formatter)
    
    
    # Set entry point
    workflow.set_entry_point("generate_queries_a")
    
    # Define sequential workflow for Company A
    workflow.add_edge("generate_queries_a", "research_human_approval_a")
    
    workflow.add_conditional_edges(
        "research_human_approval_a",
        lambda state: "continue_search" if state.current_step == "human_approval_confirmed" else END,
        {
            "continue_search": "web_search_a",
            END: END
        }
    )
    
    def web_search_condition_a(state):
        if not hasattr(state, 'iteration_tracker'):
            state.iteration_tracker = {'a': 0, 'b': 0}
        
        state.iteration_tracker['a'] += 1
        
        search_result = research_agent_a.should_continue(state)
        
        if state.iteration_tracker['a'] >= 26:
            return "fully_completed"
        
        if search_result == END and state.iteration_tracker['a'] > 0:
            return "fully_completed"
        
        return search_result

    workflow.add_conditional_edges(
        "web_search_a",
        web_search_condition_a,
        {
            "continue_search": "web_search_a",
            "fully_completed": "DCF_modelling_a",
            END: END
        }
    )
    
    # Financial workflow for A
    workflow.add_edge("DCF_modelling_a", "financial_ratio_a")
    workflow.add_edge("financial_ratio_a", "fin_human_approval_a")
    
    workflow.add_conditional_edges(
        "fin_human_approval_a",
        lambda state: "save_report" if state.current_step == "human_approval_confirmed" else END,
        {
            "save_report": "financial_reporting_a",
            END: END
        }
    )
    
    # Operations workflow for A
    workflow.add_edge("financial_reporting_a", "supply_chain_analysis_a")
    workflow.add_edge("supply_chain_analysis_a", "industry_positioning_a")
    workflow.add_edge("industry_positioning_a", "ops_human_approval_a")
    
    workflow.add_conditional_edges(
        "ops_human_approval_a",
        lambda state: "save_report" if state.current_step == "human_approval_confirmed" else END,
        {
            "save_report": "operations_reporting_a",
            END: END
        }
    )
    
    # Transition from Company A to Company B workflow
    workflow.add_edge("operations_reporting_a", "generate_queries_b")
    
    # Workflow for Company B (similar structure to A)
    workflow.add_edge("generate_queries_b", "research_human_approval_b")
    
    workflow.add_conditional_edges(
        "research_human_approval_b",
        lambda state: "continue_search" if state.current_step == "human_approval_confirmed" else END,
        {
            "continue_search": "web_search_b",
            END: END
        }
    )
    
    def web_search_condition_b(state):
        if not hasattr(state, 'iteration_tracker'):
            state.iteration_tracker = {'a': 0, 'b': 0}
        
        state.iteration_tracker['b'] += 1
        
        search_result = research_agent_b.should_continue(state)
        
        if state.iteration_tracker['b'] >= 26:
            return "fully_completed"
        
        if search_result == END and state.iteration_tracker['b'] > 0:
            return "fully_completed"
        
        return search_result

    workflow.add_conditional_edges(
        "web_search_b",
        web_search_condition_b,
        {
            "continue_search": "web_search_b",
            "fully_completed": "DCF_modelling_b",
            END: END
        }
    )
    
    # Financial workflow for B
    workflow.add_edge("DCF_modelling_b", "financial_ratio_b")
    workflow.add_edge("financial_ratio_b", "fin_human_approval_b")
    
    workflow.add_conditional_edges(
        "fin_human_approval_b",
        lambda state: "save_report" if state.current_step == "human_approval_confirmed" else END,
        {
            "save_report": "financial_reporting_b",
            END: END
        }
    )
    
    # Operations workflow for B
    workflow.add_edge("financial_reporting_b", "supply_chain_analysis_b")
    workflow.add_edge("supply_chain_analysis_b", "industry_positioning_b")
    workflow.add_edge("industry_positioning_b", "ops_human_approval_b")
    
    workflow.add_conditional_edges(
        "ops_human_approval_b",
        lambda state: "save_report" if state.current_step == "human_approval_confirmed" else END,
        {
            "save_report": "operations_reporting_b",
            END: END
        }
    )
    
    workflow.add_edge("operations_reporting_b", "validate_merger_feasibility")
    workflow.add_edge("validate_merger_feasibility", "calculate_merger_valuation")
    workflow.add_edge("calculate_merger_valuation", "assess_integration_risks")
    workflow.add_edge("assess_integration_risks", "finalize_merger_report")
    workflow.add_edge("finalize_merger_report", "assess_regulatory_compliance")
    workflow.add_edge("assess_regulatory_compliance", "conduct_legal_due_diligence")
    workflow.add_edge("conduct_legal_due_diligence", "assess_potential_legal_risks")
    workflow.add_edge("assess_potential_legal_risks", "finalize_legal_report")
    # Define workflow edges
    workflow.add_edge("finalize_legal_report", "report_structure_creator")
    workflow.add_edge("report_structure_creator", "section_template_generator")
    workflow.add_edge("section_template_generator", "rag_summary_generator")
    workflow.add_edge("rag_summary_generator", "consistency_checker")
    workflow.add_edge("consistency_checker", "report_formatter")
    workflow.add_edge("report_formatter", END)
    
    # Compile the workflow
    compiled_graph =  workflow.compile()
    try:
        output_dir = "assets"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "comprehensive_agent_graph.png")
        graph_image = compiled_graph.get_graph().draw_mermaid_png(
            curve_style=CurveStyle.CARDINAL,
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

# Main execution remains the same as in your original script
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
    
    # Initialize MnAagentState with RAG instances
    initial_state = MnAagentState(
        company_a_name="Reliance_Industries_Limited",
        company_b_name="180_Degree_Consulting",
        company_a_doc="/home/naba/Desktop/backend/RIL-Integrated-Annual-Report-2023-24_parsed.txt",
        company_b_doc="/home/naba/Desktop/backend/dc.txt",
        rag_instances=rag_instances,
        indexes=indexes,
        retrievers=retrievers
    )
    
    # Create and invoke the sequential workflow
    research_graph = create_sequential_workflow(initial_state)
    final_state = research_graph.invoke(initial_state, config={"recursion_limit": 1000})
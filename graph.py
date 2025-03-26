from agents.research_agent import ResearchAgentNodes
from agents.fin_agent import FinAgentNodes
from agents.legal_agent import LegalAgentNodes
# from agents.report_agent import ReportAgentNodes
from agents.states import MnAagentState
from datetime import datetime
from langgraph.graph import StateGraph, END
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
import os
import logging 
from RAG.rag_llama import RAG
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Log to console
    ]
)
logger = logging.getLogger(__name__)

def create_comprehensive_workflow(mn_agent_state: MnAagentState, company: str):
    """
    Create and compile a comprehensive workflow combining research and financial analysis
    with guaranteed full completion of web search
    """
    # Initialize agent nodes
    research_agent = ResearchAgentNodes(mn_agent_state, company)
    fin_agent = FinAgentNodes(mn_agent_state, company)
    
    # Define the graph workflow
    workflow = StateGraph(MnAagentState)
    
    # Add nodes from both workflows
    # Research Phase Nodes
    workflow.add_node("generate_queries", research_agent.generate_queries)
    workflow.add_node("research_human_approval", research_agent.human_approval)
    workflow.add_node("web_search", research_agent.web_search)
    
    # Financial Analysis Nodes
    workflow.add_node("DCF_modelling", fin_agent.DCF_modelling)
    workflow.add_node("financial_ratio", fin_agent.financial_ratios)
    workflow.add_node("fin_human_approval", fin_agent.human_approval)
    workflow.add_node("financial_reporting", fin_agent.financial_reporting)
    
    # Define entry point and initial workflow
    workflow.set_entry_point("generate_queries")
    
    # Research Phase Edges
    workflow.add_edge("generate_queries", "research_human_approval")
    
    workflow.add_conditional_edges(
        "research_human_approval",
        lambda state: "continue_search" if state.current_step == "human_approval_confirmed" else END,
        {
            "continue_search": "web_search",
            END: END
        }
    )
    
    def web_search_condition(state):
        search_result = research_agent.should_continue(state)
        if not hasattr(state, 'search_iterations'):
            state.search_iterations = 0
        state.search_iterations += 1
        
        # Add a max recursion limit (e.g., 3 iterations)
        if state.search_iterations >= 26:
            return "fully_completed"
        
        if search_result == END and state.search_iterations > 0:
            return "fully_completed"
        
        return search_result

    workflow.add_conditional_edges(
        "web_search",
        web_search_condition,
        {
            "continue_search": "web_search",
            "fully_completed": "DCF_modelling",
            END: END  # Change this to prevent bypassing the workflow
        }
    )
    
    workflow.add_edge("DCF_modelling", "financial_ratio")
    workflow.add_edge("financial_ratio", "fin_human_approval")
    
    workflow.add_conditional_edges(
        "fin_human_approval",
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
    
    # Compile the graph
    compiled_graph = workflow.compile()
    
    # Graph visualization
    try:
        output_dir = "assets"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "comprehensive_agent_graph.png")
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
    
    # Initialize MnAagentState with RAG instances
    initial_state = MnAagentState(
        company_a_name="Reliance_Industries_Limited",
        company_b_name="180_Degree_Consulting",
        rag_instances=rag_instances,
        indexes=indexes,
        retrievers=retrievers
    )
    research_graph = create_comprehensive_workflow(initial_state, 'a')
    final_state = research_graph.invoke(initial_state, config={"recursion_limit": 1000})

from typing import Dict, Any
import os
import yaml
import logging
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
from pydantic import BaseModel, Field
from agents.states import MnAagentState
from tools.websearcher import TavilySearchTool
from RAG.rag_llama import RAG
import uuid
import sys
sys.setrecursionlimit(10000)
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class ResearchAgentNodes:
    def __init__(self, state: MnAagentState, company: str):
        self.state = state
        self.company = company
        self.company_name = state.company_a_name if company == 'a' else state.company_b_name
        self.search_tool = TavilySearchTool()
        
        # Load prompts
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        prompts_path = os.path.join(project_root, "utils", "prompts.yaml")
        with open(prompts_path, "r") as file:
            self.prompts = yaml.safe_load(file)["Researcher_prompt"]
        logger.info(f"Loaded prompts from {prompts_path}")

    def generate_queries(self, state: MnAagentState) -> MnAagentState:
        """
        Generate search queries for the company
        """
        queries = []
        for line in self.prompts["web_Fin_prompt"].strip().split('\n'):
            queries.append(line.format(company_name = str(self.company_name)))
        
        # Determine which search results list to update based on company
        search_results_key = 'search_results_a' if self.company == 'a' else 'search_results_b'
        
        # Update state with queries and reset search results
        setattr(state, search_results_key, [])
        state.current_step = "generate_queries"
        state.queries = queries
        
        # Print queries for human review
        print("\n--- Generated Queries ---")
        for i, query in enumerate(queries, 1):
            print(f"{i}. {query}")
        
        return state

    def human_approval(self, state: MnAagentState) -> MnAagentState:
        """
        Ask human whether to proceed with search
        """
        proceed = input("\nDo you want to proceed with these queries? (yes/no): ").lower().strip()
        
        if proceed == 'yes':
            state.current_step = "human_approval_confirmed"
            return state
        
        state.current_step = "human_approval_rejected"
        return state

    def should_continue(self, state: MnAagentState) -> str:
        """
        Determine if more queries need to be processed
        """
        if state.queries:
            return "continue_search"
        return END

    def web_search(self, state: MnAagentState) -> MnAagentState:
        """
        Perform web search for current query and update vector DB
        """
        # Determine which search results list to update based on company
        search_results_key = 'search_results_a' if self.company == 'a' else 'search_results_b'
        
        # Get current search results
        search_results = getattr(state, search_results_key)
        
        # Perform search if queries exist
        if state.queries:
            current_query = state.queries.pop(0)
            print(f"\nSearching: {current_query}")
            
            # Perform web search
            response = self.search_tool.invoke_tool(current_query)
            search_results.append({"query": current_query, "result": response})
            
            # Update search results in state
            setattr(state, search_results_key, search_results)
            
            # Update RAG index
            # rag_instance = self.state.rag_instances[self.company_name]
            print((response))
            self.state.rag_instances[self.company_name].text = self.state.rag_instances[self.company_name].text + response
            self.state.rag_instances[self.company_name].update_db(
                db_name=self.company_name, 
                new_text=response
            )
        state = self.state
        state.current_step = "web_search"
        return state

def create_research_agent_graph(mn_agent_state: MnAagentState, company: str):
    """
    Create and compile the research agent workflow
    """
    # Initialize research agent nodes
    research_agent = ResearchAgentNodes(mn_agent_state, company)
    
    # Define the graph workflow
    workflow = StateGraph(MnAagentState)
    
    # Add nodes
    workflow.add_node("generate_queries", research_agent.generate_queries)
    workflow.add_node("human_approval", research_agent.human_approval)
    workflow.add_node("web_search", research_agent.web_search)
    
    # Define edges
    workflow.set_entry_point("generate_queries")
    workflow.add_edge("generate_queries", "human_approval")
    workflow.add_conditional_edges(
        "human_approval", 
        lambda state: "continue_search" if state.current_step == "human_approval_confirmed" else END,
        {
            "continue_search": "web_search",
            END: END
        }
    )
    workflow.add_conditional_edges(
        "web_search", 
        research_agent.should_continue,
        {
            "continue_search": "web_search",
            END: END
        }
    )
    
    compiled_graph = workflow.compile()
    try:
        output_dir = "assets"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "fact_checker_graph.png")
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

# Main execution would look like this in the main script
if __name__ == "__main__":
    # RAG initialization (assumed to be done before creating the graph)
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
    research_graph = create_research_agent_graph(initial_state, 'a')
    final_state = research_graph.invoke(initial_state, config={"recursion_limit": 1000})
    
    print("\n--- Vector DB Update Complete ---")

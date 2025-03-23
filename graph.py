from langgraph.graph import END, StateGraph
from agents.states import WebScraperState
from agents.webscrapper import WebScraperNodes
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
from utils.chat import Chat
import os


def create_webscraper_graph(llm):
    graph = StateGraph(WebScraperState)
    nodes = WebScraperNodes(llm)
    graph.add_node("initialize_scraper", nodes.initialize_scraper)
    graph.add_node("get_human_input", nodes.get_human_input)
    graph.add_node("scrape_website", nodes.scrape_website)
    graph.add_node("compile_scraped_data", nodes.compile_scraped_data)
    def route_based_on_human_input(state):
        if state["is_exit"]:
            return "compile_scraped_data"
        else:
            return "scrape_website"
    graph.add_edge("initialize_scraper", "get_human_input")
    graph.add_conditional_edges("get_human_input", route_based_on_human_input)
    graph.add_edge("scrape_website", "get_human_input")  # Loop back for more input
    graph.add_edge("compile_scraped_data", END)
    graph.set_entry_point("initialize_scraper")
    workflow = graph.compile()
    try:
            output_dir = "assets"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "webscrapper_graph.png")
            graph_image = workflow.get_graph().draw_mermaid_png(
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
    except Exception as e:      
        print(f"Error generating graph image: {e}")
    return workflow

if __name__ == "__main__":
    create_webscraper_graph(Chat.llm())
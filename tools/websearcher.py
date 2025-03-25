from langchain_community.tools import TavilySearchResults
from dotenv import load_dotenv
import json
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
class TavilySearchTool:
    def __init__(self, max_results=5, search_depth="advanced", include_answer=True, 
                 include_raw_content=True, include_images=True):
        load_dotenv()
        self.tool = TavilySearchResults(
            max_results=max_results,
            search_depth=search_depth,
            include_answer=include_answer,
            include_raw_content=include_raw_content,
            include_images=include_images,
        )
    
    def invoke_tool(self, query, tool_id="1", tool_name="tavily", tool_type="tool_call"):
        logger.info(f"Invoking Tavily search with query: '{query}', tool_id: '{tool_id}', tool_name: '{tool_name}'")
        answer = self.tool.invoke(query)
        logger.info(f"Successfully received search results: '{answer.content[:100]}...'")
        return answer
    
if __name__ == "__main__":
    # Test Tavily search
    search_tool = TavilySearchTool()
    tool_msg = search_tool.invoke_tool("sun rises in the west")
    try:
        results = json.loads(tool_msg.content)  
        for result in results:
            if isinstance(result, dict):  
                title = result.get("title", "No Title")
                url = result.get("url", "No URL")
                content = result.get("content", "No Content")
                print(f"Title: {title}\nURL: {url}\nContent: {content}\n")
            else:
                print(f"Unexpected result format: {result}")
    except json.JSONDecodeError:
        print("Error: Could not parse JSON content.") 
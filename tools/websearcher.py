import logging
from langchain_community.tools import TavilySearchResults
from dotenv import load_dotenv
import json

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
    def __init__(self, max_results=1, search_depth="advanced", include_answer=True,
                 include_raw_content=False, include_images=False):
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
        try:
            # Directly print the search results for debugging
            search_results = self.tool.invoke(query)
            print("Raw Search Results:", search_results)
            
            # Check if search_results is a list
            if not isinstance(search_results, list):
                logger.error(f"Unexpected search results type: {type(search_results)}")
                return "Error: Unexpected search results format"
            
            # Extract text from results
            result_texts = [
                result.get('content', '') or result.get('text', '') 
                for result in search_results
            ]
            
            # Print result texts for debugging
            # print("Result Texts:", result_texts)
            return '\n'.join(result_texts)
        except Exception as e:
            logger.error(f"Web search error for query '{query}': {e}")
            return f"Error in web search: {e}"

if __name__ == "__main__":
    # Test Tavily search
    search_tool = TavilySearchTool()
    tool_msg = search_tool.invoke_tool("recent scientific discoveries")
    
    # Print the entire response for debugging
    print("Full Tool Response:", tool_msg)
    
    # Try parsing if it looks like JSON
    try:
        results = json.loads(tool_msg)
        for result in results:
            if isinstance(result, dict):
                title = result.get("title", "No Title")
                url = result.get("url", "No URL")
                content = result.get("content", "No Content")
                print(f"Title: {title}\nURL: {url}\nContent: {content}\n")
            else:
                print(f"Unexpected result format: {result}")
    except (json.JSONDecodeError, TypeError):
        # If it's not JSON, just print the response
        print("Response is not JSON, printing as-is:", tool_msg)
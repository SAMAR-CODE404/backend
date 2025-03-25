from pydantic import BaseModel, Field
from typing import TypedDict, Dict, List, Any, Optional
from agents.states import MnAagentState
from datetime import datetime
from RAG.rag_llama import RAG
from utils.chat import Chat
from tools.websearcher import TavilySearchTool
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

class ResearchAgentNodes:
    def __init__(self, state: MnAagentState):
        self.state = state

    def query_for_search(self):
        # query
        pass
    def web_search_researcher(self):
        # web search
        pass
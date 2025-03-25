from pydantic import BaseModel, Field
from typing import TypedDict, Dict, List, Any, Optional
from agents.states import MnAagentState
from datetime import datetime
from RAG.rag_llama import RAG
from utils.chat import Chat
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
    def __init__(self, state: MnAagentState):
        self.state = state

    def report_structure_creator(self):
        # structure
        pass
    def section_template_generator(self):
        # section
        pass
    def rag_summary_generator(self):
        # summary generator
        pass
    def consistency_checker(self):
        # consistency check
        pass
    def report_formatter(self):
        # report formatter
        pass
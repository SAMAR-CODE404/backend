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
    def __init__(self, state: MnAagentState, company: str):
        self.state = state
        if company == 'a':
            self.company_name = state.company_a_name
        else:
            self.company_name = state.company_b_name

    def take_legal_document(self):
        pass
    def risk_assessment(self):
        # risk
        pass
    def reg_compliance(self):
        # Financial Analysis
        pass
    def antitrust_checker(self):
        # Financial Forecasting
        pass
    def negotiation_simulator(self):
        # Financial Reporting
        pass
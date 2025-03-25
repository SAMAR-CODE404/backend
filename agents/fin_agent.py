from pydantic import BaseModel, Field
from typing import TypedDict, Dict, List, Any, Optional
from agents.states import MnAagentState
from datetime import datetime
from RAG.rag_llama import RAG
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

class FinAgentNodes:
    def __init__(self, state: MnAagentState, company: str):
        self.state = state
        if company == 'a':
            self.company_name = state.company_a_name
        else:
            self.company_name = state.company_b_name
    def DCF_modelling(self):
        # DCF Modelling
        pass

    def financial_ratios(self):
        # Financial Ratios
        pass
    def financial_analysis(self):
        # Financial Analysis
        pass
    def financial_forecasting(self):
        # Financial Forecasting
        pass
    def financial_reporting(self):
        # Financial Reporting
        pass
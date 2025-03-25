from agents.comp_agent import CompAgentNodes
from agents.fin_agent import FinAgentNodes
from agents.legal_agent import LegalAgentNodes
from agents.report_agent import LegalAgentNodes
from agents.states import MnAagentState
from agents.research_agent import ResearchAgentNodes
from datetime import datetime
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

class MnAagent:
    pass
from pydantic import BaseModel, Field
from typing import TypedDict, Dict, List, Any, Optional
from agents.states import MnAagentState
from langchain_core.messages import AIMessage, HumanMessage
from datetime import datetime
from RAG.rag_llama import RAG
from utils.chat import Chat
import logging
import yaml
import os
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Log to console
    ]
)
logger = logging.getLogger(__name__)

class MergerAgentNodes:
    def __init__(self, state: MnAagentState):
        self.state = state
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        prompts_path = os.path.join(project_root, "utils", "prompts.yaml")
        self.rag_a =
        self.rag_b =  
        with open(prompts_path, "r") as file:
            self.prompts = yaml.safe_load(file)["Merger_agent_prompt"]
        logger.info(f"Loaded prompts from {prompts_path}")

    def Check_merger_eligibility(self, state: MnAagentState):# I'll create a vector db which will use the company's data to check if the merger is eligible
        # checks the eligibility of the merger
        # chatgroq call -> query -> response
        # check if the merger is eligible   
         
        for i in range(10):
            response, _, _ = Chat().invoke_llm_langchain(messages=[HumanMessage(content=self.prompts['Mergebility_prompt'].format(company_a_name=state.company_a_name, company_b_name=state.company_b_name))])
            response = response[-1].content

            # print(response[-1].content)
        return state
    
    def Operations_pipeline(self):
        
        pass
    def post_merger_competition_forecast(self):

        pass
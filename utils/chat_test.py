import os
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Tuple, Union
import time
from utils.message import HumanMessage, AIMessage

class Chat:
    def __init__(self):
        load_dotenv()
        self.client = OpenAI(
                        api_key = os.getenv("DATABRICKS_TOKEN"),
                        base_url="https://adb-2855448551482176.16.azuredatabricks.net/serving-endpoints"
                        )

    def _convert_messages(self, messages: List[Union[HumanMessage, AIMessage]]) -> List[dict]:
        """
        Convert custom message objects to OpenAI's message format
        """
        return [{"role": msg.role, "content": msg.content} for msg in messages]

    def invoke_llm_langchain(self, 
                              messages: List[Union[HumanMessage, AIMessage]], 
                              model="databricks-meta-llama-3-3-70b-instruct", 
                              temperature=0.7, 
                              max_tokens=5000) -> Tuple[List[Union[HumanMessage, AIMessage]], int, int]:
        """
        Invoke the LLM with the given messages
        """
        net_input = 0
        net_output = 0
        
        # Convert messages to OpenAI format
        openai_messages = self._convert_messages(messages)
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=openai_messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
        except Exception as e:
            print(f"Error in invoking LLM, sending LLM to sleep for 10 seconds")
            time.sleep(10)
            print(f"Retrying now")
            response = self.client.chat.completions.create(
                model=model,
                messages=openai_messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
        
        try:
            content = response.choices[0].message.content
            input_tokens = response.usage.prompt_tokens if response.usage else net_input
            output_tokens = response.usage.completion_tokens if response.usage else net_output
        except Exception as e:
            content = str(response)
            input_tokens = net_input
            output_tokens = net_output
        
        # Append the AI message to the messages
        ai_message = AIMessage(content=content)
        messages.append(ai_message)
        
        return messages, input_tokens, output_tokens

    def llm(self, model="databricks-meta-llama-3-3-70b-instruct", temperature=0.2):
        """
        Create an OpenAI client with specified model and temperature
        """
        return OpenAI(
            api_key = os.getenv("DATABRICKS_TOKEN"),
            base_url="https://adb-2855448551482176.16.azuredatabricks.net/serving-endpoints",
            model=model,
            temperature=temperature
        )

# Sample usage
if __name__ == "__main__":
    llm = Chat()
    messages = [HumanMessage(content="Why hotstar and Reliance Merger is so good?")]
    response, input_tokens, output_tokens = llm.invoke_llm_langchain(messages)
    print(response[-1].content)
    print(f"Input Tokens: {input_tokens}, Output Tokens: {output_tokens}")

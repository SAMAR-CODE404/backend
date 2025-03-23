from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import time

class Chat:
    def __init__(self):
        load_dotenv()

    def invoke_llm_langchain(self,messages, model="mistral-saba-24b", temperature=0.7, max_tokens=5000):
        """
        Invoke the LLM with the given messages
        """
        llm = ChatGroq(model=model, temperature=temperature, max_tokens=max_tokens)
        net_input = 0
        net_output = 0

        try:
            response = llm.invoke(messages)
        except Exception as e:
            print(f"Error in invoking LLM, sending LLM to sleep for 10 seconds")
            time.sleep(10)
            print(f"Retrying now")
            response = llm.invoke(messages)

        try:
            content = response.content
            input_tokens = response.usage_metadata["input_tokens"]
            output_tokens = response.usage_metadata["output_tokens"]
        except Exception as e:
            content = response
            input_tokens = net_input
            output_tokens = net_output

        messages.append(AIMessage(content=content)) 

        return messages, input_tokens, output_tokens

    def llm(model="mistral-saba-24b", temperature=0.2):
        load_dotenv()
        llm_ = ChatGroq(model=model, 
                        temperature=temperature, 
                        streaming=True,
                        callbacks=[StreamingStdOutCallbackHandler()],)
        return llm_

# sample usage
if __name__ == "__main__":
    llm = Chat()
    messages = [HumanMessage(content="Why hotstar and Reliance Merger is so good?")]
    response, _, _ = llm.invoke_llm_langchain(messages)
    print(response[-1].content)
import warnings
import logging
import uuid
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from utils.chat import Chat
import yaml
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.storage_context import StorageContext
from langchain_community.embeddings import HuggingFaceEmbeddings 
from llama_index.core.schema import Document
from llama_index.core.node_parser import SentenceSplitter
import chromadb
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

class RAG:
    def __init__(self, text_or_path):
        logger.info("Initializing RAG")
        
        if os.path.isfile(text_or_path):  
            try:
                with open(text_or_path, 'r', encoding='utf-8') as file:
                    self.text = file.read()
                logger.info(f"Loaded text from file: {text_or_path}")
            except FileNotFoundError:
                logger.error(f"Text file not found at {text_or_path}")
                self.text = None
        else:
            self.text = text_or_path
            logger.info("Loaded text from string input")

        self.embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        logger.info("Initialized embedding model: sentence-transformers/all-MiniLM-L6-v2")
        Settings.embed_model = self.embed_model
        Settings.chunk_size = 1000
        Settings.chunk_overlap = 100
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir) 
        prompts_path = os.path.join(project_root, "utils", "prompts.yaml")
        with open(prompts_path, "r") as file:
            self.prompts = yaml.safe_load(file)["RAG_prompts"]
        logger.info(f"Loaded prompts from {prompts_path}")
    
    def prepare_documents_from_text(self, text):
        logger.info("Preparing documents from text")
        documents = []
        doc_id = str(uuid.uuid4())
        section_id = 1
        documents.append({
            "content": text,
            "metadata": {
                "source": "text", 
                "section_id": section_id,
                "doc_id": doc_id
            }
        })
        logger.info(f"Created {len(documents)} documents from text")
        return documents
    
    def process_documents(self, docs):
       
        logger.info(f"Processing {len(docs)} documents into LlamaIndex format")
       
        splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=100)
        llama_nodes = []
        for doc in docs:
            llama_doc = Document(
                text=doc["content"],
                metadata=doc["metadata"]
            )
            nodes = splitter.get_nodes_from_documents([llama_doc])
            for i, node in enumerate(nodes):
                node.metadata.update({
                    "chunk_id": f"{doc['metadata']['doc_id']}-chunk-{i}",
                    "chunk_index": i,
                    "total_chunks": len(nodes)
                })
            
            llama_nodes.extend(nodes)
        
        logger.info(f"Created {len(llama_nodes)} total nodes from documents")
        return llama_nodes
    
    def create_db(self, db_name):
        logger.info(f"Creating vector database: {db_name}")

        documents = self.prepare_documents_from_text(self.text)
        llama_nodes = self.process_documents(documents)

        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        chroma_collection = chroma_client.get_or_create_collection(db_name)

        logger.info(f"Created Chroma collection: {db_name}")

        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index = VectorStoreIndex(
            nodes=llama_nodes,
            storage_context=storage_context
        )

        logger.info(f"Successfully created vector index {db_name} with {len(llama_nodes)} nodes")
        return index

    def update_db(self, db_name, new_text):
        logger.info(f"Updating vector database: {db_name}")

        new_documents = self.prepare_documents_from_text(new_text)
        new_nodes = self.process_documents(new_documents)

        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        chroma_collection = chroma_client.get_collection(db_name)

        logger.info(f"Updating existing Chroma collection: {db_name}")

        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Get the existing index
        existing_index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context
        )

        # Use insert_nodes method instead of insert for each node
        existing_index.insert_nodes(new_nodes)

        logger.info(f"Added {len(new_nodes)} new nodes to {db_name}")
        return existing_index

    def create_retriever(self, db_name):
        logger.info(f"Creating retriever for database: {db_name}")
        
        # If the input is already a VectorStoreIndex, use it directly
        if isinstance(db_name, VectorStoreIndex):
            logger.info(f"Using provided index as retriever")
            return db_name.as_retriever()
        
        # Otherwise, try to load from the database
        try:
            chroma_client = chromadb.PersistentClient(path="./chroma_db")
            chroma_collection = chroma_client.get_collection(name=str(db_name))
            
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            index = VectorStoreIndex.from_vector_store(vector_store)
            
            logger.info(f"Successfully created retriever from database: {db_name}")
            return index.as_retriever()
        except Exception as e:
            logger.error(f"Error creating retriever for {db_name}: {e}")
            raise

    def rag_query(self, query_text, retriever):
        query_id = str(uuid.uuid4())
        logger.info(f"Processing query: {query_id} - '{query_text}'")
        
        retrieval_result = retriever.retrieve(query_text)
        logger.info(f"Retrieved {len(retrieval_result)} relevant nodes")
        
        context_parts = []
        source_documents = []
        
        # Sort nodes by their chunk index to maintain original document order
        sorted_nodes = sorted(retrieval_result, key=lambda node: node.metadata.get('chunk_index', 0))
        
        for i, node in enumerate(sorted_nodes):
            source_type = node.metadata.get('source', 'unknown')
            doc_id = node.metadata.get('doc_id', 'unknown')
            chunk_index = node.metadata.get('chunk_index', 'N/A')
            total_chunks = node.metadata.get('total_chunks', 'N/A')
            
            context_parts.append(
                f"[Document {doc_id} - Chunk {chunk_index}/{total_chunks}] {source_type.capitalize()}: {node.text}"
            )
            
            source_documents.append({
                "page_content": node.text,
                "metadata": node.metadata
            })
        
        context = "\n\n".join(context_parts)
        prompt = self.prompts['human_message'].format(query_text=query_text, context=context)
        logger.debug(f"Generated prompt with context from {len(context_parts)} documents")
        message = [HumanMessage(content=prompt)]
        logger.info("Invoking LLM for response generation")
        llm = Chat()
        updated_messages, input_tokens, output_tokens = llm.invoke_llm_langchain(messages=message)
        logger.info(f"Generated response: {input_tokens} input tokens, {output_tokens} output tokens")
        
        return {
            "query_id": query_id,
            "query": query_text,
            "result": updated_messages[-1].content,
            "source_documents": source_documents,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }

if __name__ == "__main__":
    company_docs = {
        "company_a": "/home/naba/Desktop/backend/RIL-Integrated-Annual-Report-2023-24_parsed.txt",
        "company_b": "/home/naba/Desktop/backend/dc.txt"
    }
    
    rag_instances = {}
    indexes = {}
    retrievers = {}
    
    for company, text_path in company_docs.items():
        logger.info(f"Initializing RAG for {company} with document: {text_path}")
        rag_instances[company] = RAG(text_path)
        indexes[company] = rag_instances[company].create_db(db_name=str(company))
        # Get the retriever from the index
        retrievers[company] = indexes[company].as_retriever()
    
    while True:
        action = input("Enter 'query' to ask a question, 'update' to add new data, or 'q' to quit: ").strip().lower()
        
        if action == 'q':
            break
        
        elif action == 'update':
            company = input("Enter company name (company_a/company_b): ").strip().lower()
            if company not in rag_instances:
                print("Invalid company name. Please enter 'company_a' or 'company_b'.")
                continue
            
            update_type = input("Enter 'file' to update from a file or 'text' to enter raw text: ").strip().lower()
            
            if update_type == 'file':
                new_doc_path = input("Enter the path of the new document to add: ").strip()
                try:
                    with open(new_doc_path, 'r', encoding='utf-8') as file:
                        new_text = file.read()
                    logger.info(f"Successfully read file: {new_doc_path}")
                except Exception as e:
                    print(f"Error reading file: {e}")
                    continue
            elif update_type == 'text':
                print("Enter your text (type '###END###' on a new line when finished):")
                lines = []
                while True:
                    line = input()
                    if line.strip() == "###END###":
                        break
                    lines.append(line)
                new_text = "\n".join(lines)
                logger.info("Successfully captured raw text input")
            else:
                print("Invalid option. Please enter 'file' or 'text'.")
                continue
            
            logger.info(f"Updating database for {company} with new content")
            rag_instances[company].text = new_text
            indexes[company] = rag_instances[company].update_db(db_name=str(company), new_text=new_text)
            retrievers[company] = indexes[company].as_retriever()
            print(f"Updated vector database for {company}.")
        
        elif action == 'query':
            company = input("Enter company name (company_a/company_b): ").strip().lower()
            if company not in retrievers:
                print("Invalid company name. Please enter 'company_a' or 'company_b'.")
                continue
            
            query_text = input("Enter your query: ").strip()
            if not query_text:
                print("Query cannot be empty.")
                continue
            
            logger.info(f"User query for {company}: {query_text}")
            response = rag_instances[company].rag_query(query_text, retrievers[company])
            print(response["result"])
        
        else:
            print("Invalid option. Please enter 'query', 'update', or 'q' to quit.")
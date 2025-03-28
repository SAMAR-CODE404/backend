import os
import yaml
import logging
from typing import Dict, Any, List  
from RAG.rag_llama import RAG
from langgraph.graph import StateGraph, END
from agents.states import MnAagentState
import PyPDF2
import docx

logger = logging.getLogger(__name__)

class MergerLegalAgent:
    def __init__(self, state: MnAagentState):
        """
        Initialize Merger Legal Agent with the merged state
        
        Args:
            state (MnAagentState): Merged state containing company information
        """
        self.state = state
        
        # Load prompts for legal assessment
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        prompts_path = os.path.join(project_root, "utils", "prompts.yaml")
        
        try:
            with open(prompts_path, "r") as file:
                self.prompts = yaml.safe_load(file)["Merger_Legal_Agent_Prompts"]
        except Exception as e:
            logger.error(f"Error loading prompts: {e}")
            self.prompts = {}
        
        # Initialize legal document processing
        self.legal_documents = {}
    
    def load_legal_documents(self, document_paths: Dict[str, List[str]]) -> None:
        """
        Load and process legal documents for both companies
        
        Args:
            document_paths (Dict[str, List[str]]): Dictionary of company names to lists of document paths
        """
        for company, docs in document_paths.items():
            company_docs = []
            for doc_path in docs:
                try:
                    extracted_text = self._extract_document_text(doc_path)
                    company_docs.append({
                        'path': doc_path,
                        'text': extracted_text
                    })
                except Exception as e:
                    logger.error(f"Error processing document {doc_path} for {company}: {e}")
            
            self.legal_documents[company] = company_docs
        
        # Update state with loaded documents
        if self.state.company_a_name in self.legal_documents:
            self.state.legal_docs[self.state.company_a_name] = [
                doc['text'] for doc in self.legal_documents[self.state.company_a_name]
            ]
        if self.state.company_b_name in self.legal_documents:
            self.state.legal_docs[self.state.company_b_name] = [
                doc['text'] for doc in self.legal_documents[self.state.company_b_name]
            ]
    
    def _extract_document_text(self, file_path: str) -> str:
        """
        Extract text from various document types
        
        Args:
            file_path (str): Path to the document file
        
        Returns:
            str: Extracted text from the document
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_extension == '.pdf':
                return self._extract_pdf_text(file_path)
            elif file_extension in ['.docx', '.doc']:
                return self._extract_docx_text(file_path)
            elif file_extension == '.txt':
                with open(file_path, 'r', encoding='utf-8') as file:
                    return file.read()
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            return ""
    
    def _extract_pdf_text(self, pdf_path: str) -> str:
        """
        Extract text from PDF document
        
        Args:
            pdf_path (str): Path to PDF file
        
        Returns:
            str: Extracted text from PDF
        """
        text = []
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text.append(page.extract_text())
        return "\n".join(text)
    
    def _extract_docx_text(self, docx_path: str) -> str:
        """
        Extract text from Word document
        
        Args:
            docx_path (str): Path to Word document
        
        Returns:
            str: Extracted text from Word document
        """
        doc = docx.Document(docx_path)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])
    
    def assess_regulatory_compliance(self, state: MnAagentState) -> MnAagentState:
        """
        Assess regulatory compliance for the proposed merger
        
        Args:
            state (MnAagentState): Current state of the merger process
        
        Returns:
            MnAagentState: Updated state with regulatory compliance assessment
        """
        state.current_step = "regulatory_compliance_assessment"
        
        # Combine regulatory documents
        compliance_data = "\n\n".join([
            f"Company A Legal Documents: {' '.join(state.legal_docs.get(state.company_a_name, ['No documents']))}",
            f"Company B Legal Documents: {' '.join(state.legal_docs.get(state.company_b_name, ['No documents']))}",
            f"Merger Jurisdiction: {state.merger_jurisdiction or 'Not specified'}"
        ])
        
        # Use RAG for regulatory compliance analysis
        compliance_rag = RAG(compliance_data)
        indexes = compliance_rag.create_db(db_name="regulatory_compliance")
        retriever = indexes.as_retriever()
        
        try:
            compliance_response = compliance_rag.rag_query(
                query_text=self.prompts.get("regulatory_compliance_prompt", "Assess regulatory compliance for proposed merger"),
                retriever=retriever
            )
            
            state.legal_check['regulatory_compliance'] = compliance_response['result']
            
            # Save compliance report
            output_dir = "merger_reports"
            os.makedirs(output_dir, exist_ok=True)
            compliance_report_path = os.path.join(output_dir, "regulatory_compliance_report.txt")
            
            with open(compliance_report_path, "w") as f:
                f.write(compliance_response['result'])
            
            state.merger_report = compliance_report_path
            
        except Exception as e:
            logger.error(f"Error in regulatory compliance assessment: {e}")
            state.error = f"Regulatory compliance assessment failed: {str(e)}"
        
        return state
    
    def conduct_legal_due_diligence(self, state: MnAagentState) -> MnAagentState:
        """
        Conduct comprehensive legal due diligence for the merger
        
        Args:
            state (MnAagentState): Current state of the merger process
        
        Returns:
            MnAagentState: Updated state with legal due diligence findings
        """
        state.current_step = "legal_due_diligence"
        
        # Combine legal documents and corporate structures
        due_diligence_data = f"""
        Company A Legal Documents: {state.legal_docs.get(state.company_a_name, 'No legal documents available')}
        Company B Legal Documents: {state.legal_docs.get(state.company_b_name, 'No legal documents available')}
        Company A Corporate Structure: {state.corporate_structure.get(state.company_a_name, 'No corporate structure data')}
        Company B Corporate Structure: {state.corporate_structure.get(state.company_b_name, 'No corporate structure data')}
        """
        
        # Use RAG for legal due diligence analysis
        due_diligence_rag = RAG(due_diligence_data)
        indexes = due_diligence_rag.create_db(db_name="legal_due_diligence")
        retriever = indexes.as_retriever()
        
        try:
            due_diligence_response = due_diligence_rag.rag_query(
                query_text=self.prompts.get("legal_due_diligence_prompt", "Conduct comprehensive legal due diligence for merger"),
                retriever=retriever
            )
            
            state.legal_check['due_diligence_findings'] = due_diligence_response['result']
            
            # Save due diligence report
            output_dir = "merger_reports"
            os.makedirs(output_dir, exist_ok=True)
            due_diligence_report_path = os.path.join(output_dir, "legal_due_diligence_report.txt")
            
            with open(due_diligence_report_path, "w") as f:
                f.write(due_diligence_response['result'])
            
            state.merger_report = due_diligence_report_path
            
        except Exception as e:
            logger.error(f"Error in legal due diligence: {e}")
            state.error = f"Legal due diligence failed: {str(e)}"
        
        return state
    
    def assess_potential_legal_risks(self, state: MnAagentState) -> MnAagentState:
        """
        Assess potential legal risks associated with the merger
        
        Args:
            state (MnAagentState): Current state of the merger process
        
        Returns:
            MnAagentState: Updated state with legal risk assessment
        """
        state.current_step = "legal_risk_assessment"
        
        # Combine litigation history and potential conflict areas
        legal_risk_data = f"""
        Company A Litigation History: {state.litigation_history.get(state.company_a_name, 'No litigation history available')}
        Company B Litigation History: {state.litigation_history.get(state.company_b_name, 'No litigation history available')}
        Potential Conflict Areas: {state.potential_conflicts or 'No specific conflicts identified'}
        """
        
        # Use RAG for legal risk analysis
        risk_rag = RAG(legal_risk_data)
        indexes = risk_rag.create_db(db_name="legal_risks")
        retriever = indexes.as_retriever()
        
        try:
            risk_response = risk_rag.rag_query(
                query_text=self.prompts.get("legal_risks_prompt", "Assess potential legal risks in proposed merger"),
                retriever=retriever
            )
            
            state.legal_check['potential_legal_risks'] = risk_response['result']
            
            # Save legal risks report
            output_dir = "merger_reports"
            os.makedirs(output_dir, exist_ok=True)
            legal_risks_report_path = os.path.join(output_dir, "legal_risks_report.txt")
            
            with open(legal_risks_report_path, "w", encoding="utf-8") as f:
                f.write(risk_response['result'])
            
            state.merger_report = legal_risks_report_path
            
        except Exception as e:
            logger.error(f"Error in legal risk assessment: {e}")
            state.error = f"Legal risk assessment failed: {str(e)}"
        
        return state

def create_merger_legal_workflow(legal_agent: MergerLegalAgent) -> StateGraph:
    """
    Create a comprehensive merger legal workflow
    
    Args:
        legal_agent (MergerLegalAgent): Merger legal agent instance
    
    Returns:
        StateGraph: Compiled workflow for legal assessment
    """
    workflow = StateGraph(MnAagentState)
    
    # Add nodes for each stage of legal assessment
    workflow.add_node("assess_regulatory_compliance", legal_agent.assess_regulatory_compliance)
    workflow.add_node("conduct_legal_due_diligence", legal_agent.conduct_legal_due_diligence)
    workflow.add_node("assess_potential_legal_risks", legal_agent.assess_potential_legal_risks)
    workflow.add_node("finalize_legal_report", lambda state: state)
    
    # Set entry point and define workflow
    workflow.set_entry_point("assess_regulatory_compliance")
    
    # Define edges between nodes
    workflow.add_edge("assess_regulatory_compliance", "conduct_legal_due_diligence")
    workflow.add_edge("conduct_legal_due_diligence", "assess_potential_legal_risks")
    workflow.add_edge("assess_potential_legal_risks", "finalize_legal_report")
    
    # Set finish point
    workflow.set_finish_point("finalize_legal_report")
    
    # Compile the workflow
    return workflow.compile()

if __name__ == "__main__":
    # Initialize logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    # Create initial state
    initial_state = MnAagentState(
    company_a_name="Reliance_Industries_Limited",
    company_b_name="180_Degree_Consulting",
    company_a_doc="/path/to/reliance/legal/document1.pdf",  # Provide a document path
    company_b_doc="/path/to/180dc/legal/document1.txt"     # Provide a document path
)

    # Create legal agent
    legal_agent = MergerLegalAgent(initial_state)
    
    # Load legal documents
    legal_documents = {
        "Reliance_Industries_Limited": [
            "/path/to/reliance/legal/document1.pdf",
            "/path/to/reliance/legal/document2.docx"
        ],
        "180_Degree_Consulting": [
            "/path/to/180dc/legal/document1.txt",
            "/path/to/180dc/legal/document2.pdf"
        ]
    }
    
    # Process legal documents
    legal_agent.load_legal_documents(legal_documents)
    
    # Create and run merger legal workflow
    merger_legal_workflow = create_merger_legal_workflow(legal_agent)
    result = merger_legal_workflow.invoke(initial_state, config={"recursion_limit": 1000})
    print("Merger Legal Assessment Complete!")
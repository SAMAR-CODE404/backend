from typing import TypedDict, Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from RAG.rag_llama import RAG


class WebScraperStateRequired(TypedDict):
    company_name: str
    is_exit: bool


class WebScraperState(WebScraperStateRequired, total=False):
    final_report: str
    scraped_data: List[Dict[str, str]]
    current_url: Optional[str]
    human_input: str
    compiled_report: str
    is_exit: bool


class MnAagentState(BaseModel):
    # Company Identification
    company_a_name: str = Field(description="Name of the first company being analyzed")
    company_b_name: str = Field(description="Name of the second company being analyzed")
    company_a_doc: str = Field(description="Document for company A")
    company_b_doc: str = Field(description="Document for company B")

    # Workflow Tracking
    current_step: Optional[str] = Field(
        default=None, description="Current workflow step"
    )
    prev_step: Optional[str] = Field(default=None, description="Previous workflow step")
    search_iterations: int = Field(default=0, description="Current search iteration")
    iteration_tracker: Dict[str, int] = Field(
        default_factory=lambda: {"a": 0, "b": 0},
        description="Tracker for search iterations per company",
    )

    # Progress Tracking
    progress_callback: Any = Field(
        default=None, description="Callback for progress updates"
    )

    # Search and Retrieval
    rag_instances: Dict[str, RAG] = Field(
        default_factory=dict, description="RAG instances for each company"
    )
    indexes: Dict[str, Any] = Field(
        default_factory=dict, description="Indexes for each company"
    )
    retrievers: Dict[str, Any] = Field(
        default_factory=dict, description="Retrievers for each company"
    )
    queries: List[str] = Field(
        default_factory=list, description="Queries for each company"
    )
    search_results_a: List[Dict[str, Any]] = Field(
        default_factory=list, description="Additional search results for company A"
    )
    search_results_b: List[Dict[str, Any]] = Field(
        default_factory=list, description="Additional search results for company B"
    )

    # Reports
    fin_report_a: Optional[str] = Field(
        default=None, description="Filename of the final report for company A"
    )
    fin_report_b: Optional[str] = Field(
        default=None, description="Filename of the final report for company B"
    )
    ops_report_a: Optional[str] = Field(
        default=None, description="Operations report for company A"
    )
    ops_report_b: Optional[str] = Field(
        default=None, description="Operations report for company B"
    )
    legal_report_a: Optional[str] = Field(
        default=None, description="Filename of the legal report for company A"
    )
    legal_report_b: Optional[str] = Field(
        default=None, description="Filename of the legal report for company B"
    )
    merger_report: Optional[str] = Field(
        default=None, description="Filename of the merger report"
    )
    competition_report: Optional[str] = Field(
        default=None, description="Filename of the competition report"
    )

    # Financial Analysis
    dcf_models: Dict[str, Any] = Field(
        default_factory=dict, description="Discounted Cash Flow models for each company"
    )
    financial_ratios: Dict[str, Any] = Field(
        default_factory=dict, description="Financial ratios for each company"
    )

    # Business Analysis
    supply_chain_analyst: Dict[str, str] = Field(
        default_factory=dict, description="Supply chain analyst data"
    )
    industry_position: Dict[str, str] = Field(
        default_factory=dict, description="Industry position data"
    )

    merger_acquisition_details: Dict[str, Any] = Field(
        default_factory=dict, description="Merger and acquisition details"
    )
    risk_check: Dict[str, Any] = Field(
        default_factory=dict, description="Risk assessment for each company"
    )
    antitrust_assessment: Dict[str, Any] = Field(
        default_factory=dict, description="Antitrust assessment for each company"
    )

    # New fields to resolve previous errors
    legal_docs: Dict[str, List[str]] = Field(
        default_factory=dict, description="Legal documents for each company"
    )
    merger_jurisdiction: Optional[str] = Field(
        default=None, description="Jurisdiction for the merger"
    )
    legal_check: Dict[str, str] = Field(
        default_factory=dict, description="Legal check results"
    )

    # Error Handling
    error: Optional[str] = Field(
        default=None, description="Error message if any step fails"
    )

    # Model Configuration
    merger_structure: Optional[str] = Field(
        default=None, description="Merger structure"
    )
    merger_context: Optional[str] = Field(
        default=None, description="Additional context for the merger"
    )
    corporate_structure: Dict[str, Any] = Field(
        default_factory=dict, description="Corporate structure data"
    )
    litigation_history: Dict[str, Any] = Field(
        default_factory=dict, description="Litigation history data"
    )
    potential_conflicts: Dict[str, Any] = Field(
        default_factory=dict, description="Potential conflicts data"
    )
    legal_report_structure: Dict[str, Any] = Field(
        default_factory=dict, description="Legal report structure data"
    )
    consistency_issues: Dict[str, Any] = Field(
        default_factory=dict, description="Consistency issues data"
    )
    section_templates: Dict[str, Any] = Field(
        default_factory=dict, description="Section templates data"
    )
    final_report: Dict[str, Any] = Field(
        default_factory=dict, description="Final report structure data"
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # Allow complex types like RAG instances
        extra="ignore",  # Ignore extra fields not defined in the model
    )

    def update_progress(self):
        """Update progress if callback is available"""
        if self.progress_callback and self.current_step:
            self.progress_callback.update(self.current_step)

from typing import TypedDict, Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field

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
    company_a_name: str = Field(description="Name of the first company being analyzed")
    company_b_name: str = Field(description="Name of the second company being analyzed")
    merger_acquisition_details: Dict[str, Any] = Field(default_factory=dict, description="Merger and acquisition details")
    search_results_a: List[Dict[str, Any]] = Field(default_factory=list, description="Additional search results for company A")
    search_results_b: List[Dict[str, Any]] = Field(default_factory=list, description="Additional search results for company B")
    current_step: str = Field(default=None, description="Current workflow step")
    error: str = Field(default=None, description="Error message if any step fails")
    vector_index_a: Any = Field(default=None, description="Vector index for company A")
    vector_index_b: Any = Field(default=None, description="Vector index for company B")
    fin_report_a: str = Field(default=None, description="Filename of the final report for company A")
    fin_report_b: str = Field(default=None, description="Filename of the final report for company B")
    operations_report_a: str = Field(default=None, description="Filename of the operations report for company A")
    operations_report_b: str = Field(default=None, description="Filename of the operations report for company B")
    Legal_report_a: str = Field(default=None, description="Filename of the legal report for company A")
    Legal_report_b: str = Field(default=None, description="Filename of the legal report for company B")
    merger_report: str = Field(default=None, description="Filename of the merger report")
    competition_report: str = Field(default=None, description="Filename of the competition report")
    
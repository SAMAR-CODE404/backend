from typing import TypedDict, Dict, List, Any, Optional

class WebScraperStateRequired(TypedDict):
    company_name: str
    is_exit: bool

class WebScraperState(WebScraperStateRequired, total=False):
    final_report: str
    scraped_data: List[Dict[str, str]]
    current_url: Optional[str]
    human_input: str
    compiled_report: str
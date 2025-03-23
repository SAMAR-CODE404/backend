import os
from datetime import datetime
import yaml
from dotenv import load_dotenv
from utils.chat import Chat
from graph import create_webscraper_graph

class CompanyAnalyzer:
    def __init__(self):
        load_dotenv()
        # current_dir = os.path.dirname(os.path.abspath(__file__))
        # project_root = os.path.dirname(current_dir) 
        # prompts_path = os.path.join(project_root, "utils", "prompts.yaml")
        # prompts_path = os.path.abspath(r'utils/prompts.yaml')
        # with open(prompts_path, "r") as file:
        #     self.prompts = yaml.safe_load(file)["companyAnalyzer"]
        self.llm = Chat.llm()

    def analyze_with_web_scraping(self, company_name_):
        """Run company analysis followed by web scraping"""
        print(f"\nStarting analysis for {company_name_}...")
        print("Phase 1: Conducting company financial analysis")
        
        # First run the company analysis
        company_analysis = """tmkc"""
        
        print("\nCompany analysis complete!")
        print("Phase 2: Starting web scraping for additional information")
        print("You will be prompted to enter URLs to scrape. Type 'exit' when done.")
        webscraper_workflow = create_webscraper_graph(self.llm)
        initial_state = {
            "company_name": company_name_,
            "final_report": company_analysis,
            "scraped_data": [],
            "is_exit": False
        }
        scraper_result = webscraper_workflow.invoke(initial_state)
        return scraper_result["compiled_report"]
        
        
    
if __name__ == "__main__":
    company_name = "Adobe"
    analyzer = CompanyAnalyzer()
    combined_report = analyzer.analyze_with_web_scraping(company_name)
    message = combined_report
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{company_name.lower()}_analysis_{timestamp}.md"
    with open(filename, "w") as file:
        file.write(message)
    print(f"Analysis saved to {filename}")
    print(message)
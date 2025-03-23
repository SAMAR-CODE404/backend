import os
import requests
import re
import json
import time
from bs4 import BeautifulSoup
from datetime import datetime
from typing import Dict, Any, List, Optional
from utils.analyzer_utils import log_node, truncate_text
from langchain_core.messages import SystemMessage
from RAG.rag_llama import RAG
import yaml

class WebScraperNodes:
    def __init__(self, llm):
        self.llm = llm
        self.RAG = None
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir) 
        prompts_path = os.path.join(project_root, "utils", "prompts.yaml")
        with open(prompts_path, "r") as file:
            self.prompts = yaml.safe_load(file)["web_scrapper_prompt"]
        self.system_prompt = self.prompts["system_message"]
    
    @log_node("initialize_scraper")
    def initialize_scraper(self, state):
        """Initialize the web scraper state."""
        return {
            "scraped_data": [],
            "is_exit": False,
            "financial_metrics": {}  # Store structured financial data
        }
    
    @log_node("get_human_input")
    def get_human_input(self, state):
        """Get input from the human."""
        print("\n\n" + "="*50)
        print(f"Company Analysis Report is available. You can now scrape additional websites.")
        print("Enter a URL to scrape or type 'exit' to finish:")
        print("Tip: For financial sites like Yahoo Finance, use direct API endpoints for better results.")
        human_input = input("> ")
        is_exit = human_input.lower() == 'exit'
        
        # If this is a Yahoo Finance URL, convert it to use the API endpoint
        modified_url = self._modify_url_if_needed(human_input)
        if modified_url != human_input:
            print(f"Using optimized endpoint: {modified_url}")
        
        return {
            "human_input": human_input,
            "current_url": None if is_exit else modified_url,
            "original_url": None if is_exit else human_input,
            "is_exit": is_exit
        }
    
    def _modify_url_if_needed(self, url):
        """Modify URLs to use API endpoints when possible."""
        # Handle Yahoo Finance
        if 'finance.yahoo.com/quote/' in url:
            # Extract the ticker symbol
            match = re.search(r'finance\.yahoo\.com/quote/([A-Z0-9\.\-]+)', url)
            if match:
                ticker = match.group(1)
                
                # Check what type of page this is
                if '/financials' in url:
                    # Use Yahoo Finance API for financial data
                    return f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{ticker}?modules=incomeStatementHistory,balanceSheetHistory,cashflowStatementHistory,defaultKeyStatistics,financialData"
                else:
                    # Use Yahoo Finance API for quote data
                    return f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={ticker}"
        
        # Return the original URL if no modification needed
        return url
    
    @log_node("scrape_website")
    def scrape_website(self, state):
        """Scrape the website specified by the human with enhanced financial data extraction."""
        url = state["current_url"]
        original_url = state.get("original_url", url)
        if not url:
            return {}
        
        company_name = state.get("company_name", "")
        scraped_data = state.get("scraped_data", [])
        financial_metrics = state.get("financial_metrics", {})
        
        try:
            print(f"Scraping {url}...")
            
            # Different handling based on URL type
            if 'query1.finance.yahoo.com' in url:
                # This is a Yahoo Finance API endpoint
                extracted_data, content_text = self._scrape_yahoo_finance_api(url, original_url)
            else:
                # Standard web scraping approach
                extracted_data, content_text = self._scrape_standard_website(url)
            
            # If we have a company name, update the financial metrics
            if company_name and extracted_data:
                if company_name not in financial_metrics:
                    financial_metrics[company_name] = {}
                financial_metrics[company_name].update(extracted_data)
            
            # Use LLM to analyze and extract additional insights
            self.RAG = RAG(content_text)
            index = self.RAG.create_db()
            retriever = self.RAG.create_retriever(index)
            prompt = self.prompts['rag_prompt'].format(original_url = original_url, extracted_metrics = json.dumps(extracted_data, indent=2) if extracted_data else 'None')
            response = self.RAG.rag_query(prompt, retriever)          
            summary = response["result"]
            title = self._get_title_from_url(original_url)
            
            # Store the scraped data
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            scraped_item = {
                "url": original_url,
                "title": title,
                "summary": summary,
                "extracted_metrics": extracted_data,
                "timestamp": timestamp
            }
            
            # Add to the list of scraped data
            scraped_data.append(scraped_item)
            
            print(f"Successfully scraped {original_url}")
            if extracted_data:
                print(f"Found {len(extracted_data)} financial metrics")
            
            return {
                "scraped_data": scraped_data,
                "financial_metrics": financial_metrics
            }
            
        except Exception as e:
            print(f"Error scraping website {original_url}: {str(e)}")
            
            # Add error message to scraped data
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            scraped_item = {
                "url": original_url,
                "title": "Error",
                "summary": f"Failed to scrape this website: {str(e)}",
                "timestamp": timestamp
            }
            
            scraped_data.append(scraped_item)
            
            return {"scraped_data": scraped_data}
    
    def _get_title_from_url(self, url):
        """Generate a descriptive title from the URL."""
        # Extract ticker if present
        ticker_match = re.search(r'quote/([A-Z0-9\.\-]+)', url)
        ticker = ticker_match.group(1) if ticker_match else None
        
        # Determine content type
        content_type = "Summary"
        if '/financials' in url:
            content_type = "Financial Statements"
        elif '/balance-sheet' in url:
            content_type = "Balance Sheet"
        elif '/cash-flow' in url:
            content_type = "Cash Flow"
        elif '/income-statement' in url:
            content_type = "Income Statement"
        elif '/key-statistics' in url:
            content_type = "Key Statistics"
        elif '/profile' in url:
            content_type = "Company Profile"
        
        # Create title
        if ticker:
            return f"{ticker} - {content_type}"
        else:
            # Extract domain for other URLs
            domain_match = re.search(r'https?://(?:www\.)?([^/]+)', url)
            domain = domain_match.group(1) if domain_match else "Website"
            return f"Financial Data from {domain}"
    
    def _scrape_yahoo_finance_api(self, api_url, original_url):
        """Scrape financial data from Yahoo Finance API."""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(api_url, headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        extracted_data = {}
        
        # Process quote data (v7 endpoint)
        if 'quoteResponse' in data and 'result' in data['quoteResponse']:
            result = data['quoteResponse']['result'][0] if data['quoteResponse']['result'] else {}
            
            # Map API fields to our metrics
            field_mapping = {
                'regularMarketPrice': 'stock_price',
                'regularMarketChange': 'price_change',
                'regularMarketChangePercent': 'price_change_percent',
                'regularMarketVolume': 'volume',
                'marketCap': 'market_cap',
                'trailingPE': 'pe_ratio',
                'trailingAnnualDividendYield': 'dividend_yield',
                'epsTrailingTwelveMonths': 'eps',
                'fiftyTwoWeekHigh': '52_week_high',
                'fiftyTwoWeekLow': '52_week_low'
            }
            
            for api_field, metric_name in field_mapping.items():
                if api_field in result:
                    value = result[api_field]
                    if isinstance(value, (int, float)):
                        # Format numbers for readability
                        if api_field == 'marketCap':
                            value = self._format_large_number(value)
                        elif api_field == 'trailingAnnualDividendYield':
                            value = f"{value:.2%}"
                        elif api_field == 'regularMarketChangePercent':
                            value = f"{value:.2f}%"
                    extracted_data[metric_name] = value
            
            # Create 52-week range from high and low
            if '52_week_high' in extracted_data and '52_week_low' in extracted_data:
                extracted_data['52_week_range'] = f"{extracted_data['52_week_low']} - {extracted_data['52_week_high']}"
        
        # Process financial data (v10 endpoint)
        if 'quoteSummary' in data and 'result' in data['quoteSummary']:
            result = data['quoteSummary']['result'][0] if data['quoteSummary']['result'] else {}
            
            # Extract income statement data
            if 'incomeStatementHistory' in result:
                statements = result['incomeStatementHistory']['incomeStatementHistory']
                if statements:
                    latest = statements[0]
                    if 'totalRevenue' in latest:
                        extracted_data['total_revenue'] = self._format_large_number(latest['totalRevenue'].get('raw', 0))
                    if 'netIncome' in latest:
                        extracted_data['net_income'] = self._format_large_number(latest['netIncome'].get('raw', 0))
            
            # Extract balance sheet data
            if 'balanceSheetHistory' in result:
                statements = result['balanceSheetHistory']['balanceSheetStatements']
                if statements:
                    latest = statements[0]
                    if 'totalAssets' in latest:
                        extracted_data['total_assets'] = self._format_large_number(latest['totalAssets'].get('raw', 0))
                    if 'totalLiab' in latest:
                        extracted_data['total_liabilities'] = self._format_large_number(latest['totalLiab'].get('raw', 0))
            
            # Extract key statistics
            if 'defaultKeyStatistics' in result:
                stats = result['defaultKeyStatistics']
                if 'returnOnEquity' in stats:
                    extracted_data['return_on_equity'] = f"{stats['returnOnEquity'].get('raw', 0):.2%}"
                if 'profitMargins' in stats:
                    extracted_data['profit_margin'] = f"{stats['profitMargins'].get('raw', 0):.2%}"
            
            # Extract financial data
            if 'financialData' in result:
                financial = result['financialData']
                if 'operatingMargins' in financial:
                    extracted_data['operating_margin'] = f"{financial['operatingMargins'].get('raw', 0):.2%}"
                if 'grossMargins' in financial:
                    extracted_data['gross_margin'] = f"{financial['grossMargins'].get('raw', 0):.2%}"
                if 'ebitdaMargins' in financial:
                    extracted_data['ebitda_margin'] = f"{financial['ebitdaMargins'].get('raw', 0):.2%}"
        
        # Return the extracted data and the original JSON as text for the LLM to analyze
        return extracted_data, json.dumps(data, indent=2)
    
    def _format_large_number(self, number):
        """Format large numbers in a readable way (K, M, B)."""
        if number >= 1_000_000_000:
            return f"${number / 1_000_000_000:.2f}B"
        elif number >= 1_000_000:
            return f"${number / 1_000_000:.2f}M"
        elif number >= 1_000:
            return f"${number / 1_000:.2f}K"
        else:
            return f"${number:.2f}"
    
    def _scrape_standard_website(self, url):
        """Scrape a standard website using BeautifulSoup."""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept': 'text/html,application/xhtml+xml,application/xml',
            'Referer': 'https://www.google.com/'
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract financial data
        extracted_data = self._extract_financial_data(soup, url)
        
        # Extract main content text
        content_text = self._extract_main_content(soup)
        
        return extracted_data, content_text
    
    def _extract_main_content(self, soup):
        """Extract main content text from the webpage."""
        # Try to find main content areas first
        main_content_areas = soup.find_all(['article', 'main', 'div', 'section'], 
                                      class_=lambda c: c and any(term in str(c).lower() 
                                                               for term in ['content', 'article', 'main', 'body']))
        
        if main_content_areas:
            content = ""
            for area in main_content_areas:
                for element in area.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'table']):
                    if element.name == 'table':
                        # For tables, try to preserve structure
                        table_text = ""
                        for row in element.find_all('tr'):
                            row_cells = []
                            for cell in row.find_all(['th', 'td']):
                                row_cells.append(cell.get_text(strip=True))
                            if row_cells:
                                table_text += " | ".join(row_cells) + "\n"
                        content += table_text + "\n\n"
                    else:
                        text = element.get_text(strip=True)
                        if text:
                            content += text + "\n\n"
            return content
        
        # Fall back to extracting all paragraphs, headers, and tables
        content = ""
        for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'table']):
            if element.name == 'table':
                # Extract table content
                table_text = ""
                for row in element.find_all('tr'):
                    row_cells = []
                    for cell in row.find_all(['th', 'td']):
                        row_cells.append(cell.get_text(strip=True))
                    if row_cells:
                        table_text += " | ".join(row_cells) + "\n"
                content += table_text + "\n\n"
            else:
                text = element.get_text(strip=True)
                if text:
                    content += text + "\n\n"
        
        return content
    
    def _extract_financial_data(self, soup, url):
        """Extract financial metrics from the webpage."""
        metrics = {}
        
        # Check if this is a known financial website
        if 'finance.yahoo.com' in url:
            self._extract_yahoo_finance_html(soup, metrics)
        elif 'marketwatch.com' in url:
            self._extract_marketwatch(soup, metrics)
        elif 'investing.com' in url:
            self._extract_investing(soup, metrics)
        elif 'moneycontrol.com' in url:
            self._extract_moneycontrol(soup, metrics)
        else:
            # Generic extraction for other financial sites
            self._extract_generic_financial_data(soup, metrics)
        
        # Try to extract JSON-LD data that might contain financial information
        self._extract_json_ld(soup, metrics)
        
        # Check for any embedded JSON data in script tags
        self._extract_json_from_scripts(soup, metrics)
        
        return metrics
    
    def _extract_yahoo_finance_html(self, soup, metrics):
        """Extract financial data from Yahoo Finance HTML."""
        # Stock price
        price_element = soup.find('fin-streamer', {'data-field': 'regularMarketPrice'})
        if price_element:
            metrics['stock_price'] = price_element.text
        
        # Key metrics from summary table
        for label, key in [
            ('Previous Close', 'previous_close'),
            ('Open', 'open'),
            ('Bid', 'bid'),
            ('Ask', 'ask'),
            ("Day's Range", 'day_range'),
            ('52 Week Range', '52_week_range'),
            ('Volume', 'volume'),
            ('Avg. Volume', 'avg_volume'),
            ('Market Cap', 'market_cap'),
            ('Beta', 'beta'),
            ('PE Ratio', 'pe_ratio'),
            ('EPS', 'eps'),
            ('Forward Dividend & Yield', 'dividend_yield'),
            ('Ex-Dividend Date', 'ex_dividend_date'),
            ('1y Target Est', 'target_price')
        ]:
            try:
                td = soup.find('td', string=re.compile(f"^{re.escape(label)}"))
                if td:
                    value_td = td.find_next_sibling('td')
                    if value_td:
                        metrics[key] = value_td.text.strip()
            except Exception:
                pass
        
        # Extract financial statements data if on the financials page
        financial_tables = soup.find_all('div', {'class': 'D(tbr)'})
        if financial_tables:
            current_section = None
            for row in financial_tables:
                # Check if this is a section header
                section_el = row.find('span', {'class': 'Va(m)'})
                if section_el and section_el.text.strip():
                    current_section = section_el.text.strip().lower().replace(' ', '_')
                    continue
                
                # Extract row data
                if current_section:
                    cells = row.find_all('div', {'class': 'D(tbc)'})
                    if len(cells) >= 2:
                        metric_name = cells[0].text.strip().lower().replace(' ', '_')
                        value = cells[1].text.strip()
                        if metric_name and value:
                            metrics[f"{current_section}_{metric_name}"] = value
    
    def _extract_marketwatch(self, soup, metrics):
        """Extract financial data from MarketWatch."""
        # Stock price
        price_element = soup.find('h2', {'class': 'intraday__price'})
        if price_element:
            price_value = price_element.find('bg-quote')
            if price_value:
                metrics['stock_price'] = price_value.text
        
        # Extract key metrics
        key_metric_elements = soup.find_all('li', {'class': 'kv__item'})
        for element in key_metric_elements:
            label = element.find('small', {'class': 'label'})
            value = element.find('span', {'class': 'primary'})
            if label and value:
                label_text = label.text.strip().lower()
                if 'p/e' in label_text:
                    metrics['pe_ratio'] = value.text.strip()
                elif 'market cap' in label_text:
                    metrics['market_cap'] = value.text.strip()
                elif 'eps' in label_text:
                    metrics['eps'] = value.text.strip()
                elif 'dividend' in label_text and 'yield' in label_text:
                    metrics['dividend_yield'] = value.text.strip()
                elif 'revenue' in label_text:
                    metrics['revenue'] = value.text.strip()
    
    def _extract_investing(self, soup, metrics):
        """Extract financial data from Investing.com."""
        # Stock price
        price_element = soup.find('span', {'data-test': 'instrument-price-last'})
        if price_element:
            metrics['stock_price'] = price_element.text
        
        # Extract from overview table
        overview_table = soup.find('table', {'class': 'overview-table'})
        if overview_table:
            rows = overview_table.find_all('tr')
            for row in rows:
                cells = row.find_all('td')
                if len(cells) >= 2:
                    label = cells[0].text.strip().lower()
                    value = cells[1].text.strip()
                    if 'market cap' in label:
                        metrics['market_cap'] = value
                    elif 'p/e ratio' in label:
                        metrics['pe_ratio'] = value
                    elif 'eps' in label:
                        metrics['eps'] = value
                    elif 'dividend' in label and 'yield' in label:
                        metrics['dividend_yield'] = value
                    elif 'revenue' in label:
                        metrics['revenue'] = value
    
    def _extract_moneycontrol(self, soup, metrics):
        """Extract financial data from MoneyControl."""
        # Stock price
        price_element = soup.select_one('div.pcstks span.Bspr15.Bledg')
        if price_element:
            metrics['stock_price'] = price_element.text.strip()
        
        # Extract from financial tables
        tables = soup.select('table.mctable1')
        if tables:
            for table in tables:
                header_row = table.select_one('tr.lightblue')
                if header_row:
                    headers = [th.text.strip() for th in header_row.select('th')]
                    for row in table.select('tr:not(.lightblue)'):
                        cells = row.select('td')
                        if not cells:
                            continue
                        
                        # First cell is usually the metric name
                        if len(cells) > 0:
                            metric_name = cells[0].text.strip().lower().replace(' ', '_')
                            
                            # Skip empty rows or section headers
                            if not metric_name or len(cells) < 2 or not metric_name[0].isalpha():
                                continue
                            
                            # Get the most recent value (usually first data column)
                            if len(cells) > 1:
                                value = cells[1].text.strip()
                                if value and value != '--':
                                    # Map common financial metrics
                                    if 'profit' in metric_name:
                                        metrics['net_profit'] = value
                                    elif 'revenue' in metric_name or 'sales' in metric_name:
                                        metrics['revenue'] = value
                                    elif 'eps' in metric_name:
                                        metrics['eps'] = value
                                    else:
                                        metrics[metric_name] = value
    
    def _extract_generic_financial_data(self, soup, metrics):
        """Extract financial data using generic patterns."""
        # Common financial metric patterns
        metric_patterns = {
            r'Stock Price[:\s]*([\d\.,]+)': 'stock_price',
            r'Market Cap(?:italization)?[:\s]*([\d\.,]+\s*[KMBT]?\s*\$?|[\$\£\€]?\s*[\d\.,]+\s*[KMBT]?)': 'market_cap',
            r'P/E(?:\s*Ratio)?[:\s]*([\d\.,]+)': 'pe_ratio',
            r'Revenue[:\s]*([\d\.,]+\s*[KMBT]?\s*\$?|[\$\£\€]?\s*[\d\.,]+\s*[KMBT]?)': 'revenue',
            r'EPS[:\s]*([\d\.,\-]+)': 'eps',
            r'Dividend\s*Yield[:\s]*([\d\.,]+\%?)': 'dividend_yield',
            r'52(?:\-|\s+)Week\s*Range[:\s]*([^<>\n]*)': '52_week_range',
            r'Volume[:\s]*([\d\.,]+[KMBT]?)': 'volume',
            r'Net\s*(?:Income|Profit)[:\s]*([\d\.,]+\s*[KMBT]?\s*\$?|[\$\£\€]?\s*[\d\.,]+\s*[KMBT]?)': 'net_income'
        }
        
        # Extract metrics using regex patterns
        page_text = soup.get_text()
        for pattern, metric_name in metric_patterns.items():
            match = re.search(pattern, page_text, re.IGNORECASE)
            if match:
                metrics[metric_name] = match.group(1).strip()
        
        # Extract from tables that might contain financial data
        for table in soup.find_all('table'):
            # Check if table contains financial keywords
            table_text = table.get_text().lower()
            if any(keyword in table_text for keyword in ['price', 'market cap', 'p/e', 'eps', 'dividend', 'revenue']):
                for row in table.find_all('tr'):
                    cells = row.find_all(['th', 'td'])
                    if len(cells) >= 2:
                        label = cells[0].get_text(strip=True).lower()
                        value = cells[1].get_text(strip=True)
                        
                        # Map to standard metrics
                        if 'market cap' in label:
                            metrics['market_cap'] = value
                        elif 'p/e' in label or 'price to earning' in label:
                            metrics['pe_ratio'] = value
                        elif label == 'eps' or 'earning per share' in label:
                            metrics['eps'] = value
                        elif 'dividend' in label and ('yield' in label or '%' in value):
                            metrics['dividend_yield'] = value
                        elif 'revenue' in label or 'sales' in label:
                            metrics['revenue'] = value
                        elif '52' in label and 'week' in label:
                            metrics['52_week_range'] = value
                        elif 'volume' in label:
                            metrics['volume'] = value
    
    def _extract_json_ld(self, soup, metrics):
        """Extract financial data from JSON-LD structured data."""
        for script in soup.find_all('script', {'type': 'application/ld+json'}):
            if not script.string:
                continue
                
            try:
                json_data = json.loads(script.string)
                
                # Check for financial data in JSON-LD
                if isinstance(json_data, dict):
                    # Look for Corporation or Organization type
                    if '@type' in json_data and json_data['@type'] in ['Corporation', 'Organization']:
                        if 'tickerSymbol' in json_data:
                            metrics['ticker'] = json_data['tickerSymbol']
                        if 'name' in json_data:
                            metrics['company_name'] = json_data['name']
            except Exception:
                pass
    
    def _extract_json_from_scripts(self, soup, metrics):
        """Extract financial data from embedded JSON in script tags."""
        # Look for common patterns where financial data is stored in JavaScript objects
        for script in soup.find_all('script'):
            if not script.string:
                continue
                
            script_text = script.string
            
            # Look for financial data patterns
            try:
                # Yahoo Finance pattern
                yahoo_match = re.search(r'root\.App\.main = (.*?);', script_text)
                if yahoo_match:
                    try:
                        json_str = yahoo_match.group(1)
                        data = json.loads(json_str)
                        
                        # Navigate to the quote summary store if it exists
                        if 'context' in data and 'dispatcher' in data['context'] and 'stores' in data['context']['dispatcher']:
                            stores = data['context']['dispatcher']['stores']
                            if 'QuoteSummaryStore' in stores:
                                summary = stores['QuoteSummaryStore']
                                
                                # Extract price data
                                if 'price' in summary:
                                    price_data = summary['price']
                                    if 'regularMarketPrice' in price_data and 'raw' in price_data['regularMarketPrice']:
                                        metrics['stock_price'] = price_data['regularMarketPrice']['raw']
                                    if 'marketCap' in price_data and 'raw' in price_data['marketCap']:
                                        metrics['market_cap'] = self._format_large_number(price_data['marketCap']['raw'])
                                
                                # Extract summary detail data
                                if 'summaryDetail' in summary:
                                    detail = summary['summaryDetail']
                                    if 'trailingPE' in detail and 'raw' in detail['trailingPE']:
                                        metrics['pe_ratio'] = detail['trailingPE']['raw']
                                    if 'dividendYield' in detail and 'raw' in detail['dividendYield']:
                                        metrics['dividend_yield'] = f"{detail['dividendYield']['raw']:.2%}"
                                    if 'volume' in detail and 'raw' in detail['volume']:
                                        metrics['volume'] = detail['volume']['raw']
                                
                                # Extract financial data
                                if 'financialData' in summary:
                                    financial = summary['financialData']
                                    if 'profitMargins' in financial and 'raw' in financial['profitMargins']:
                                        metrics['profit_margin'] = f"{financial['profitMargins']['raw']:.2%}"
                                    if 'grossMargins' in financial and 'raw' in financial['grossMargins']:
                                        metrics['gross_margin'] = f"{financial['grossMargins']['raw']:.2%}"
                                    if 'totalRevenue' in financial and 'raw' in financial['totalRevenue']:
                                        metrics['revenue'] = self._format_large_number(financial['totalRevenue']['raw'])
                                
                                # Extract earnings data
                                if 'defaultKeyStatistics' in summary:
                                    stats = summary['defaultKeyStatistics']
                                    if 'trailingEps' in stats and 'raw' in stats['trailingEps']:
                                        metrics['eps'] = stats['trailingEps']['raw']
                    except Exception:
                        pass
                
                # Look for other financial data patterns
                # Pattern for direct JSON objects containing financial metrics
                json_objects = re.findall(r'(\{[^\{]*"price"[^\}]*\}|\{[^\{]*"marketCap"[^\}]*\})', script_text)
                for json_str in json_objects:
                    try:
                        if json_str.count('{') == json_str.count('}'):  # Ensure balanced braces
                            data = json.loads(json_str)
                            if 'price' in data and isinstance(data['price'], (int, float, str)):
                                metrics['stock_price'] = data['price']
                            if 'marketCap' in data and isinstance(data['marketCap'], (int, float, str)):
                                metrics['market_cap'] = data['marketCap']
                    except Exception:
                        pass
            except Exception:
                pass
    
    @log_node("compile_scraped_data")
    def compile_scraped_data(self, state):
        """Compile all scraped data into a formatted output."""
        scraped_data = state.get("scraped_data", [])
        company_name = state.get("company_name", "Unknown Company")
        financial_metrics = state.get("financial_metrics", {})
        
        # Start with company analysis if available
        compiled_output = ""
        if state.get("final_report"):
            compiled_output += state["final_report"] + "\n\n"
        
        # Add the web scraping results header
        compiled_output += f"# Web Scraping Results for {company_name}\n\n"
        compiled_output += f"*Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
        
        # Add consolidated financial metrics section if we have data
        if financial_metrics and company_name in financial_metrics:
            compiled_output += "## Consolidated Financial Metrics\n\n"
            metrics = financial_metrics[company_name]
            
            # Create a table of metrics
            compiled_output += "| Metric | Value |\n"
            compiled_output += "|--------|-------|\n"
            
            # Categories of financial metrics for better organization
            metric_categories = {
                "Stock Price Information": ["stock_price", "price_change", "price_change_percent", "previous_close", "52_week_range", "volume"],
                "Valuation Metrics": ["market_cap", "pe_ratio", "eps", "target_price"],
                "Dividend Information": ["dividend_yield", "ex_dividend_date"],
                "Financial Performance": ["revenue", "net_income", "profit_margin", "gross_margin", "operating_margin"],
                "Other Metrics": []
            }
            
            # Add metrics by category
            for category, metric_keys in metric_categories.items():
                category_metrics = {k: v for k, v in metrics.items() if any(k == key or k.endswith(f"_{key}") for key in metric_keys)}
                
                if category_metrics:
                    compiled_output += f"| **{category}** | |\n"
                    
                    for metric, value in category_metrics.items():
                        # Format the metric name for display
                        display_name = " ".join(word.capitalize() for word in metric.split('_'))
                        compiled_output += f"| {display_name} | {value} |\n"
            
            # Add any remaining metrics not in predefined categories
            other_metrics = {k: v for k, v in metrics.items() 
                            if not any(k == key or k.endswith(f"_{key}") for category_keys in metric_categories.values() for key in category_keys)}
            
            if other_metrics:
                compiled_output += f"| **Other Metrics** | |\n"
                for metric, value in other_metrics.items():
                    display_name = " ".join(word.capitalize() for word in metric.split('_'))
                    compiled_output += f"| {display_name} | {value} |\n"
                    
            compiled_output += "\n"
        
        # Add individual website summaries
        if not scraped_data:
            compiled_output += "No websites were scraped.\n\n"
        else:
            compiled_output += "## Website Summaries\n\n"
            
            for i, item in enumerate(scraped_data, 1):
                compiled_output += f"### {i}. {item['title']}\n\n"
                compiled_output += f"**URL:** {item['url']}\n\n"
                compiled_output += f"**Scraped on:** {item['timestamp']}\n\n"
                
                # Format extracted metrics as a bullet list
                if item.get('extracted_metrics'):
                    compiled_output += "**Extracted Metrics:**\n\n"
                    for metric, value in item['extracted_metrics'].items():
                        display_name = " ".join(word.capitalize() for word in metric.split('_'))
                        compiled_output += f"* {display_name}: {value}\n"
                    compiled_output += "\n"
                
                compiled_output += "**Summary:**\n\n"
                compiled_output += item['summary'] + "\n\n"
                compiled_output += "---\n\n"
        
        return {"compiled_report": compiled_output}
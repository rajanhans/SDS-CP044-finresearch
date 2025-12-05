"""
Manager Agent - Multi-Agent Orchestration
Coordinates all worker agents and manages the research workflow.
"""

from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from config.settings import get_settings
from memory.vector_store import VectorStore
from agents.researcher_agent import ResearcherAgent
from agents.analyst_agent import FinancialAnalystAgent
from agents.reporting_agent import ReportingAgent


class ManagerAgent:
    """
    Manager Agent orchestrates the multi-agent financial research system.
    Delegates tasks to worker agents and ensures quality control.
    """

    def __init__(self):
        """Initialize the Manager Agent and all worker agents."""
        self.settings = get_settings()
        self.name = "ManagerAgent"

        # Initialize shared vector memory
        self.vector_store = VectorStore()

        # Initialize worker agents
        self.researcher = ResearcherAgent(self.vector_store)
        self.analyst = FinancialAnalystAgent(self.vector_store)
        self.reporter = ReportingAgent(self.vector_store)

        print(f"[{self.name}] Initialized with all worker agents")

    def conduct_research(
        self,
        ticker: str,
        investor_mode: str = "neutral",
        parallel: bool = True
    ) -> Dict[str, Any]:
        """
        Conduct comprehensive financial research on a ticker.

        Args:
            ticker: Stock ticker symbol
            investor_mode: Report tone ('neutral', 'bullish', 'bearish')
            parallel: Whether to run researcher and analyst in parallel

        Returns:
            Dictionary containing complete research results and report
        """
        print(f"\n{'='*60}")
        print(f"[{self.name}] Starting research workflow for {ticker}")
        print(f"[{self.name}] Investor Mode: {investor_mode}")
        print(f"[{self.name}] Parallel Execution: {parallel}")
        print(f"{'='*60}\n")

        # Clear previous data for this ticker
        self._clear_previous_data(ticker)

        # Validate ticker first
        validation_result = self._validate_ticker(ticker)
        if not validation_result['valid']:
            return {
                'success': False,
                'error': validation_result['error'],
                'ticker': ticker
            }

        try:
            # Phase 1: Data gathering (Research + Analysis)
            if parallel:
                research_findings, analyst_findings = self._execute_parallel_research(ticker)
            else:
                research_findings, analyst_findings = self._execute_sequential_research(ticker)

            # Quality control check
            quality_check = self._quality_control(ticker, research_findings, analyst_findings)
            if not quality_check['passed']:
                print(f"[{self.name}] Warning: Quality control issues detected")
                print(f"[{self.name}] Issues: {quality_check['issues']}")

            # Phase 2: Report generation
            report = self._generate_report(ticker, research_findings, analyst_findings, investor_mode)

            # Final validation
            final_validation = self._validate_report(report)

            result = {
                'success': True,
                'ticker': ticker,
                'research_findings': research_findings,
                'analyst_findings': analyst_findings,
                'report': report,
                'quality_check': quality_check,
                'final_validation': final_validation
            }

            print(f"\n{'='*60}")
            print(f"[{self.name}] Research workflow completed successfully for {ticker}")
            print(f"{'='*60}\n")

            return result

        except Exception as e:
            error_msg = f"Error in research workflow: {str(e)}"
            print(f"[{self.name}] {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'ticker': ticker
            }

    def _validate_ticker(self, ticker: str) -> Dict[str, Any]:
        """
        Validate that the ticker exists and can be analyzed.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Validation result with valid flag and error message if applicable
        """
        print(f"[{self.name}] Validating ticker {ticker}...")

        try:
            # Try to fetch basic data using the analyst's client
            from utils.api_clients import FinancialAPIClient
            client = FinancialAPIClient()
            data = client.get_stock_data(ticker)

            if 'error' in data:
                return {
                    'valid': False,
                    'error': f"Invalid ticker or data unavailable: {data['error']}"
                }

            if not data.get('company_name') or data.get('current_price', 0) == 0:
                return {
                    'valid': False,
                    'error': f"Ticker {ticker} found but has insufficient data"
                }

            print(f"[{self.name}] Ticker {ticker} validated successfully")
            return {
                'valid': True,
                'company_name': data.get('company_name', ticker)
            }

        except Exception as e:
            return {
                'valid': False,
                'error': f"Ticker validation failed: {str(e)}"
            }

    def _clear_previous_data(self, ticker: str) -> None:
        """Clear previous research data for the ticker."""
        print(f"[{self.name}] Clearing previous data for {ticker}...")
        try:
            self.vector_store.clear_ticker(ticker)
            print(f"[{self.name}] Previous data cleared")
        except Exception as e:
            print(f"[{self.name}] Warning: Could not clear previous data: {str(e)}")

    def _execute_parallel_research(self, ticker: str) -> tuple:
        """
        Execute researcher and analyst in parallel.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Tuple of (research_findings, analyst_findings)
        """
        print(f"[{self.name}] Executing parallel research for {ticker}...")

        research_findings = None
        analyst_findings = None
        errors = []

        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both tasks
            future_research = executor.submit(self._execute_researcher, ticker)
            future_analyst = executor.submit(self._execute_analyst, ticker)

            # Collect results
            for future in as_completed([future_research, future_analyst]):
                try:
                    result = future.result()
                    if future == future_research:
                        research_findings = result
                    else:
                        analyst_findings = result
                except Exception as e:
                    errors.append(str(e))
                    print(f"[{self.name}] Error in parallel execution: {str(e)}")

        if errors:
            print(f"[{self.name}] Completed with errors: {errors}")

        return research_findings, analyst_findings

    def _execute_sequential_research(self, ticker: str) -> tuple:
        """
        Execute researcher and analyst sequentially.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Tuple of (research_findings, analyst_findings)
        """
        print(f"[{self.name}] Executing sequential research for {ticker}...")

        research_findings = self._execute_researcher(ticker)
        analyst_findings = self._execute_analyst(ticker)

        return research_findings, analyst_findings

    def _execute_researcher(self, ticker: str) -> Dict[str, Any]:
        """Execute the Researcher Agent."""
        print(f"[{self.name}] Delegating to Researcher Agent...")
        try:
            # Get company name from analyst data if available
            from utils.api_clients import FinancialAPIClient
            client = FinancialAPIClient()
            data = client.get_stock_data(ticker)
            company_name = data.get('company_name', '')

            return self.researcher.research(ticker, company_name)
        except Exception as e:
            print(f"[{self.name}] Researcher Agent failed: {str(e)}")
            return {
                'ticker': ticker,
                'error': str(e),
                'news_articles': [],
                'sentiment_analysis': {'overall_sentiment': 'neutral', 'sentiment_score': 0},
                'risk_analysis': {'risks': [], 'opportunities': []}
            }

    def _execute_analyst(self, ticker: str) -> Dict[str, Any]:
        """Execute the Financial Analyst Agent."""
        print(f"[{self.name}] Delegating to Financial Analyst Agent...")
        try:
            return self.analyst.analyze(ticker)
        except Exception as e:
            print(f"[{self.name}] Financial Analyst Agent failed: {str(e)}")
            return {
                'ticker': ticker,
                'error': str(e)
            }

    def _generate_report(
        self,
        ticker: str,
        research_findings: Dict[str, Any],
        analyst_findings: Dict[str, Any],
        investor_mode: str
    ) -> Dict[str, Any]:
        """Generate the final report."""
        print(f"[{self.name}] Delegating to Reporting Agent...")
        try:
            return self.reporter.generate_report(ticker, research_findings, analyst_findings, investor_mode)
        except Exception as e:
            print(f"[{self.name}] Reporting Agent failed: {str(e)}")
            return {
                'ticker': ticker,
                'error': str(e),
                'generated_date': 'N/A'
            }

    def _quality_control(
        self,
        ticker: str,
        research_findings: Dict[str, Any],
        analyst_findings: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform quality control checks on the findings.

        Args:
            ticker: Stock ticker symbol
            research_findings: Findings from Researcher Agent
            analyst_findings: Findings from Financial Analyst Agent

        Returns:
            Quality control result
        """
        print(f"[{self.name}] Running quality control checks...")

        issues = []

        # Check researcher findings
        if 'error' in research_findings:
            issues.append(f"Researcher error: {research_findings['error']}")
        elif not research_findings.get('news_articles'):
            issues.append("No news articles found")

        # Check analyst findings
        if 'error' in analyst_findings:
            issues.append(f"Analyst error: {analyst_findings['error']}")
        else:
            stock_data = analyst_findings.get('stock_data', {})
            if not stock_data.get('current_price'):
                issues.append("Missing current price data")
            if stock_data.get('pe_ratio', 0) == 0:
                issues.append("Missing P/E ratio")

        # Check data consistency
        if not issues:
            # Verify that data was stored in vector memory
            stats = self.vector_store.get_statistics()
            if ticker not in stats.get('tickers', []):
                issues.append("Data not properly stored in vector memory")

        passed = len(issues) == 0

        result = {
            'passed': passed,
            'issues': issues,
            'checks_performed': [
                'Researcher data validation',
                'Analyst data validation',
                'Vector memory storage verification',
                'Data consistency check'
            ]
        }

        if passed:
            print(f"[{self.name}] Quality control passed")
        else:
            print(f"[{self.name}] Quality control failed: {len(issues)} issue(s) found")

        return result

    def _validate_report(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the final report.

        Args:
            report: Generated report

        Returns:
            Validation result
        """
        print(f"[{self.name}] Validating final report...")

        required_sections = [
            'executive_summary',
            'company_snapshot',
            'financial_indicators',
            'news_sentiment',
            'bull_case',
            'bear_case',
            'final_perspective'
        ]

        missing_sections = [section for section in required_sections if not report.get(section)]

        valid = len(missing_sections) == 0 and 'error' not in report

        result = {
            'valid': valid,
            'missing_sections': missing_sections,
            'has_error': 'error' in report
        }

        if valid:
            print(f"[{self.name}] Report validation passed")
        else:
            print(f"[{self.name}] Report validation failed")

        return result

    def get_vector_store_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return self.vector_store.get_statistics()

    def clear_all_data(self) -> None:
        """Clear all data from vector store."""
        print(f"[{self.name}] Clearing all data from vector store...")
        self.vector_store.clear_all()
        print(f"[{self.name}] All data cleared")

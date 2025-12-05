"""
Financial Analyst Agent - Quantitative Analysis & Ratios
Fetches financial data and computes key metrics and ratios.
"""

from typing import Dict, Any
from openai import OpenAI
from config.settings import get_settings
from utils.api_clients import FinancialAPIClient
from utils.formatters import format_financial_data, format_metric_analysis
from memory.vector_store import VectorStore


class FinancialAnalystAgent:
    """
    Agent responsible for quantitative financial analysis.
    Computes ratios, analyzes trends, and identifies financial risks.
    """

    def __init__(self, vector_store: VectorStore):
        """
        Initialize the Financial Analyst Agent.

        Args:
            vector_store: Shared vector memory for storing findings
        """
        self.settings = get_settings()
        self.client = OpenAI(api_key=self.settings.openai_api_key)
        self.financial_client = FinancialAPIClient()
        self.vector_store = vector_store
        self.name = "FinancialAnalystAgent"

    def analyze(self, ticker: str) -> Dict[str, Any]:
        """
        Conduct comprehensive financial analysis on a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary containing financial analysis findings
        """
        print(f"[{self.name}] Starting analysis for {ticker}...")

        # Fetch financial data
        stock_data = self._fetch_stock_data(ticker)

        if 'error' in stock_data:
            return {
                'ticker': ticker,
                'error': stock_data['error']
            }

        # Analyze valuation
        valuation_analysis = self._analyze_valuation(ticker, stock_data)

        # Analyze financial health
        health_analysis = self._analyze_financial_health(ticker, stock_data)

        # Analyze growth metrics
        growth_analysis = self._analyze_growth(ticker, stock_data)

        # Analyze risk indicators
        risk_analysis = self._analyze_risk_indicators(ticker, stock_data)

        # Generate LLM-based insights
        llm_insights = self._generate_insights(ticker, stock_data)

        # Create comprehensive summary
        summary = self._create_summary(ticker, stock_data, valuation_analysis, health_analysis, growth_analysis, risk_analysis)

        # Store in vector memory
        self._store_findings(ticker, stock_data, valuation_analysis, health_analysis, growth_analysis, risk_analysis, llm_insights, summary)

        findings = {
            'ticker': ticker,
            'stock_data': stock_data,
            'valuation_analysis': valuation_analysis,
            'health_analysis': health_analysis,
            'growth_analysis': growth_analysis,
            'risk_analysis': risk_analysis,
            'llm_insights': llm_insights,
            'summary': summary
        }

        print(f"[{self.name}] Analysis completed for {ticker}")
        return findings

    def _fetch_stock_data(self, ticker: str) -> Dict[str, Any]:
        """Fetch comprehensive stock data."""
        print(f"[{self.name}] Fetching financial data for {ticker}...")
        return self.financial_client.get_stock_data(ticker)

    def _analyze_valuation(self, ticker: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze valuation metrics."""
        print(f"[{self.name}] Analyzing valuation for {ticker}...")

        pe_ratio = data.get('pe_ratio', 0)
        forward_pe = data.get('forward_pe', 0)
        peg_ratio = data.get('peg_ratio', 0)
        price_to_book = data.get('price_to_book', 0)

        # Determine valuation category
        valuation = 'Unknown'
        if pe_ratio > 0:
            if pe_ratio < 15:
                valuation = 'Undervalued'
            elif pe_ratio < 25:
                valuation = 'Fairly Valued'
            else:
                valuation = 'Overvalued'

        # PEG ratio interpretation
        peg_interpretation = 'Unknown'
        if peg_ratio > 0:
            if peg_ratio < 1:
                peg_interpretation = 'Potentially undervalued relative to growth'
            elif peg_ratio < 2:
                peg_interpretation = 'Fairly valued relative to growth'
            else:
                peg_interpretation = 'Expensive relative to growth'

        analysis = {
            'pe_ratio': pe_ratio,
            'forward_pe': forward_pe,
            'peg_ratio': peg_ratio,
            'price_to_book': price_to_book,
            'valuation_category': valuation,
            'peg_interpretation': peg_interpretation,
            'summary': f"Stock appears {valuation.lower()} with P/E of {pe_ratio:.2f}. {peg_interpretation}."
        }

        return analysis

    def _analyze_financial_health(self, ticker: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze financial health metrics."""
        print(f"[{self.name}] Analyzing financial health for {ticker}...")

        debt_to_equity = data.get('debt_to_equity', 0)
        current_ratio = data.get('current_ratio', 0)
        roe = data.get('roe', 0)

        # Debt analysis
        debt_category = 'Unknown'
        if debt_to_equity >= 0:
            if debt_to_equity < 0.5:
                debt_category = 'Conservative (Low leverage)'
            elif debt_to_equity < 1.0:
                debt_category = 'Moderate leverage'
            else:
                debt_category = 'High leverage (Higher risk)'

        # Liquidity analysis
        liquidity_category = 'Unknown'
        if current_ratio > 0:
            if current_ratio < 1.0:
                liquidity_category = 'Poor (Potential liquidity issues)'
            elif current_ratio < 2.0:
                liquidity_category = 'Adequate'
            else:
                liquidity_category = 'Strong'

        # Profitability analysis
        profitability_category = 'Unknown'
        if roe > 0:
            if roe < 0.10:
                profitability_category = 'Below average'
            elif roe < 0.20:
                profitability_category = 'Good'
            else:
                profitability_category = 'Excellent'

        analysis = {
            'debt_to_equity': debt_to_equity,
            'debt_category': debt_category,
            'current_ratio': current_ratio,
            'liquidity_category': liquidity_category,
            'roe': roe,
            'profitability_category': profitability_category,
            'summary': f"Financial health is {liquidity_category.lower()} with {debt_category.lower()}. Profitability is {profitability_category.lower()}."
        }

        return analysis

    def _analyze_growth(self, ticker: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze growth metrics."""
        print(f"[{self.name}] Analyzing growth for {ticker}...")

        revenue_growth = data.get('revenue_growth', 0)
        earnings_growth = data.get('earnings_growth', 0)
        price_changes = data.get('price_changes', {})

        # Growth category

        avg_growth = (revenue_growth + earnings_growth) / 2 if revenue_growth and earnings_growth else 0
        if avg_growth > 0.20:
            growth_category = 'High growth'
        elif avg_growth > 0.10:
            growth_category = 'Moderate growth'
        elif avg_growth > 0:
            growth_category = 'Slow growth'
        else:
            growth_category = 'Negative growth'

        # Price momentum
        year_change = price_changes.get('1_year', 0)

        if year_change > 20:
            momentum = 'Strong upward momentum'
        elif year_change > 0:
            momentum = 'Positive momentum'
        elif year_change > -20:
            momentum = 'Negative momentum'
        else:
            momentum = 'Strong downward momentum'

        analysis = {
            'revenue_growth': revenue_growth,
            'earnings_growth': earnings_growth,
            'growth_category': growth_category,
            'price_changes': price_changes,
            'momentum': momentum,
            'summary': f"Company shows {growth_category.lower()} with {momentum.lower()}."
        }

        return analysis

    def _analyze_risk_indicators(self, ticker: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze risk indicators."""
        print(f"[{self.name}] Analyzing risk indicators for {ticker}...")

        volatility = data.get('volatility', 0)
        beta = data.get('beta', 0)

        # Volatility assessment
        volatility_category = 'Unknown'
        if volatility > 0:
            if volatility < 20:
                volatility_category = 'Low volatility (Stable)'
            elif volatility < 40:
                volatility_category = 'Moderate volatility'
            else:
                volatility_category = 'High volatility (Risky)'

        # Beta assessment
        beta_category = 'Unknown'
        if beta > 0:
            if beta < 0.8:
                beta_category = 'Defensive (Less volatile than market)'
            elif beta < 1.2:
                beta_category = 'Market-correlated'
            else:
                beta_category = 'Aggressive (More volatile than market)'

        # Overall risk level
        risk_level = 'Medium'
        if volatility > 40 or beta > 1.5:
            risk_level = 'High'
        elif volatility < 20 and beta < 0.8:
            risk_level = 'Low'

        analysis = {
            'volatility': volatility,
            'volatility_category': volatility_category,
            'beta': beta,
            'beta_category': beta_category,
            'risk_level': risk_level,
            'summary': f"Risk level is {risk_level} with {volatility_category.lower()} and {beta_category.lower()} characteristics."
        }

        return analysis

    def _generate_insights(self, ticker: str, data: Dict[str, Any]) -> str:
        """Generate LLM-based insights from financial data."""
        print(f"[{self.name}] Generating insights for {ticker}...")

        # Prepare financial data context
        data_summary = format_financial_data(data)
        metric_analysis = format_metric_analysis(data)

        prompt = f"""Analyze the following financial data for {ticker} and provide key insights:

Financial Data:
{data_summary}

Metric Analysis:
{metric_analysis}

Provide:
1. 3-4 key strengths from a financial perspective
2. 3-4 key concerns or weaknesses
3. Overall financial assessment (2-3 sentences)

Be concise and focus on the most important financial indicators.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.settings.openai_model,
                messages=[
                    {"role": "system", "content": "You are a quantitative financial analyst."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.settings.temperature,
                max_tokens=1000
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"[{self.name}] Error generating insights: {str(e)}")
            return f"Error generating insights: {str(e)}"

    def _create_summary(
        self,
        ticker: str,
        stock_data: Dict[str, Any],
        valuation: Dict[str, Any],
        health: Dict[str, Any],
        growth: Dict[str, Any],
        risk: Dict[str, Any]
    ) -> str:
        """Create a comprehensive summary of financial analysis."""
        summary_parts = [
            f"Financial Analysis Summary for {ticker}:",
            f"- Current Price: ${stock_data.get('current_price', 0):.2f}",
            f"- Valuation: {valuation['valuation_category']} (P/E: {valuation['pe_ratio']:.2f})",
            f"- Financial Health: {health['liquidity_category']} liquidity, {health['debt_category']}",
            f"- Growth: {growth['growth_category']}",
            f"- Risk Level: {risk['risk_level']} ({risk['volatility_category']})"
        ]

        return "\n".join(summary_parts)

    def _store_findings(
        self,
        ticker: str,
        stock_data: Dict[str, Any],
        valuation: Dict[str, Any],
        health: Dict[str, Any],
        growth: Dict[str, Any],
        risk: Dict[str, Any],
        insights: str,
        summary: str
    ) -> None:
        """Store analysis findings in vector memory."""
        print(f"[{self.name}] Storing findings in vector memory...")

        # Store summary
        self.vector_store.add_document(
            content=summary,
            metadata={
                'ticker': ticker,
                'agent': self.name,
                'type': 'financial_summary'
            }
        )

        # Store valuation analysis
        self.vector_store.add_document(
            content=f"Valuation Analysis:\n{valuation['summary']}",
            metadata={
                'ticker': ticker,
                'agent': self.name,
                'type': 'valuation_analysis',
                'valuation_category': valuation['valuation_category']
            }
        )

        # Store financial health analysis
        self.vector_store.add_document(
            content=f"Financial Health:\n{health['summary']}",
            metadata={
                'ticker': ticker,
                'agent': self.name,
                'type': 'health_analysis'
            }
        )

        # Store growth analysis
        self.vector_store.add_document(
            content=f"Growth Analysis:\n{growth['summary']}",
            metadata={
                'ticker': ticker,
                'agent': self.name,
                'type': 'growth_analysis',
                'growth_category': growth['growth_category']
            }
        )

        # Store risk analysis
        self.vector_store.add_document(
            content=f"Risk Analysis:\n{risk['summary']}",
            metadata={
                'ticker': ticker,
                'agent': self.name,
                'type': 'risk_analysis',
                'risk_level': risk['risk_level']
            }
        )

        # Store LLM insights
        self.vector_store.add_document(
            content=f"Financial Insights:\n{insights}",
            metadata={
                'ticker': ticker,
                'agent': self.name,
                'type': 'llm_insights'
            }
        )

        print(f"[{self.name}] Findings stored successfully")

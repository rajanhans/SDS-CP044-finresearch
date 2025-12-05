"""
Reporting Agent - Synthesis & Formatting
Creates comprehensive financial research reports from agent findings.
"""

from typing import Dict, Any
from datetime import datetime
from openai import OpenAI
from config.settings import get_settings
from memory.vector_store import VectorStore
from utils.formatters import format_report


class ReportingAgent:
    """
    Agent responsible for synthesizing findings into a comprehensive report.
    Generates structured, investor-style financial reports.
    """

    def __init__(self, vector_store: VectorStore):
        """
        Initialize the Reporting Agent.

        Args:
            vector_store: Shared vector memory for retrieving findings
        """
        self.settings = get_settings()
        self.client = OpenAI(api_key=self.settings.openai_api_key)
        self.vector_store = vector_store
        self.name = "ReportingAgent"

    def generate_report(
        self,
        ticker: str,
        research_findings: Dict[str, Any],
        analyst_findings: Dict[str, Any],
        investor_mode: str = "neutral"
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive financial research report.

        Args:
            ticker: Stock ticker symbol
            research_findings: Findings from Researcher Agent
            analyst_findings: Findings from Financial Analyst Agent
            investor_mode: Tone of the report ('neutral', 'bullish', 'bearish')

        Returns:
            Dictionary containing the complete report
        """
        print(f"[{self.name}] Generating report for {ticker}...")

        # Retrieve additional context from vector memory
        context = self._retrieve_context(ticker)

        # Generate report sections
        executive_summary = self._generate_executive_summary(ticker, research_findings, analyst_findings, investor_mode)
        company_snapshot = self._generate_company_snapshot(ticker, analyst_findings)
        financial_indicators = self._generate_financial_indicators(ticker, analyst_findings)
        news_sentiment = self._generate_news_sentiment(ticker, research_findings)
        bull_case = self._generate_bull_case(ticker, research_findings, analyst_findings)
        bear_case = self._generate_bear_case(ticker, research_findings, analyst_findings)
        final_perspective = self._generate_final_perspective(ticker, research_findings, analyst_findings, investor_mode)

        # Compile report
        report = {
            'ticker': ticker,
            'generated_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'investor_mode': investor_mode,
            'executive_summary': executive_summary,
            'company_snapshot': company_snapshot,
            'financial_indicators': financial_indicators,
            'news_sentiment': news_sentiment,
            'bull_case': bull_case,
            'bear_case': bear_case,
            'final_perspective': final_perspective,
            'context_used': context
        }

        # Store report in vector memory
        self._store_report(ticker, report)

        print(f"[{self.name}] Report generated successfully for {ticker}")
        return report

    def _retrieve_context(self, ticker: str) -> str:
        """Retrieve all relevant context from vector memory."""
        print(f"[{self.name}] Retrieving context from vector memory...")
        return self.vector_store.get_context(ticker)

    def _generate_executive_summary(
        self,
        ticker: str,
        research: Dict[str, Any],
        analysis: Dict[str, Any],
        mode: str
    ) -> str:
        """Generate executive summary (≤150 words)."""
        print(f"[{self.name}] Generating executive summary...")

        stock_data = analysis.get('stock_data', {})
        company_name = stock_data.get('company_name', ticker)
        current_price = stock_data.get('current_price', 0)
        sentiment = research.get('sentiment_analysis', {})

        prompt = f"""Write a concise executive summary (≤150 words) for {company_name} ({ticker}).

Current Price: ${current_price:.2f}
Market Sentiment: {sentiment.get('overall_sentiment', 'neutral')}
Valuation: {analysis.get('valuation_analysis', {}).get('valuation_category', 'Unknown')}
Growth: {analysis.get('growth_analysis', {}).get('growth_category', 'Unknown')}
Risk Level: {analysis.get('risk_analysis', {}).get('risk_level', 'Unknown')}

Tone: {mode}

Focus on:
1. Current market position
2. Key financial metrics
3. Overall sentiment
4. Primary investment consideration

Keep it professional and concise.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.settings.openai_model,
                messages=[
                    {"role": "system", "content": "You are a professional financial report writer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.settings.temperature,
                max_tokens=300
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"[{self.name}] Error generating executive summary: {str(e)}")
            return f"Error generating summary: {str(e)}"

    def _generate_company_snapshot(self, ticker: str, analysis: Dict[str, Any]) -> str:
        """Generate company snapshot section."""
        print(f"[{self.name}] Generating company snapshot...")

        stock_data = analysis.get('stock_data', {})

        snapshot = f"""**Company:** {stock_data.get('company_name', ticker)}
**Ticker:** {ticker}
**Sector:** {stock_data.get('sector', 'N/A')}
**Industry:** {stock_data.get('industry', 'N/A')}
**Market Cap:** ${stock_data.get('market_cap', 0):,.0f}
**Current Price:** ${stock_data.get('current_price', 0):.2f}

**52-Week Range:** ${stock_data.get('52_week_low', 0):.2f} - ${stock_data.get('52_week_high', 0):.2f}
**Analyst Recommendation:** {stock_data.get('analyst_recommendation', 'N/A').upper()}
"""

        if stock_data.get('target_price', 0) > 0:
            snapshot += f"**Average Target Price:** ${stock_data.get('target_price', 0):.2f}\n"

        return snapshot

    def _generate_financial_indicators(self, ticker: str, analysis: Dict[str, Any]) -> str:
        """Generate financial indicators section."""
        print(f"[{self.name}] Generating financial indicators...")

        stock_data = analysis.get('stock_data', {})
        valuation = analysis.get('valuation_analysis', {})
        health = analysis.get('health_analysis', {})
        risk = analysis.get('risk_analysis', {})

        indicators = f"""### Price Performance
- **1 Day:** {stock_data.get('price_changes', {}).get('1_day', 0):+.2f}%
- **1 Week:** {stock_data.get('price_changes', {}).get('1_week', 0):+.2f}%
- **1 Month:** {stock_data.get('price_changes', {}).get('1_month', 0):+.2f}%
- **1 Year:** {stock_data.get('price_changes', {}).get('1_year', 0):+.2f}%

### Valuation Metrics
- **P/E Ratio:** {valuation.get('pe_ratio', 0):.2f} ({valuation.get('valuation_category', 'Unknown')})
- **Forward P/E:** {valuation.get('forward_pe', 0):.2f}
- **PEG Ratio:** {valuation.get('peg_ratio', 0):.2f}
- **Price to Book:** {valuation.get('price_to_book', 0):.2f}

### Financial Health
- **Debt to Equity:** {health.get('debt_to_equity', 0):.2f} ({health.get('debt_category', 'Unknown')})
- **Current Ratio:** {health.get('current_ratio', 0):.2f} ({health.get('liquidity_category', 'Unknown')})
- **ROE:** {health.get('roe', 0):.2%} ({health.get('profitability_category', 'Unknown')})

### Growth Metrics
- **Revenue Growth:** {stock_data.get('revenue_growth', 0):.2%}
- **Earnings Growth:** {stock_data.get('earnings_growth', 0):.2%}
- **EPS:** ${stock_data.get('eps', 0):.2f}

### Risk Indicators
- **Volatility:** {risk.get('volatility', 0):.2f}% ({risk.get('volatility_category', 'Unknown')})
- **Beta:** {risk.get('beta', 0):.2f} ({risk.get('beta_category', 'Unknown')})
- **Overall Risk Level:** {risk.get('risk_level', 'Unknown')}
"""

        return indicators

    def _generate_news_sentiment(self, ticker: str, research: Dict[str, Any]) -> str:
        """Generate news and sentiment section."""
        print(f"[{self.name}] Generating news sentiment...")

        sentiment = research.get('sentiment_analysis', {})
        news_articles = research.get('news_articles', [])

        section = f"""### Overall Sentiment
**{sentiment.get('overall_sentiment', 'neutral').upper()}** (Score: {sentiment.get('sentiment_score', 0)}/10)

{sentiment.get('explanation', 'No sentiment analysis available.')}

### Recent News Headlines
"""

        if news_articles:
            for i, article in enumerate(news_articles[:5], 1):
                section += f"{i}. [{article.get('title', 'N/A')}]({article.get('url', '#')})\n"
                section += f"   *{article.get('published_date', 'N/A')}*\n\n"
        else:
            section += "No recent news articles found.\n"

        return section

    def _generate_bull_case(self, ticker: str, research: Dict[str, Any], analysis: Dict[str, Any]) -> str:
        """Generate bull case (opportunities) section."""
        print(f"[{self.name}] Generating bull case...")

        opportunities = research.get('risk_analysis', {}).get('opportunities', [])
        llm_insights = analysis.get('llm_insights', '')

        prompt = f"""Based on the following information for {ticker}, write a compelling bull case (3-4 paragraphs):

Identified Opportunities:
{chr(10).join([f"- {o}" for o in opportunities]) if opportunities else "No specific opportunities identified"}

Financial Insights:
{llm_insights}

Focus on:
1. Growth potential
2. Competitive advantages
3. Positive catalysts
4. Strong financial metrics

Be balanced but optimistic.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.settings.openai_model,
                messages=[
                    {"role": "system", "content": "You are a financial analyst writing the bull case for an investment."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.settings.temperature,
                max_tokens=600
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"[{self.name}] Error generating bull case: {str(e)}")
            bull_case = "### Opportunities\n"
            if opportunities:
                for opp in opportunities:
                    bull_case += f"- {opp}\n"
            else:
                bull_case += "Opportunities analysis not available.\n"
            return bull_case

    def _generate_bear_case(self, ticker: str, research: Dict[str, Any], analysis: Dict[str, Any]) -> str:
        """Generate bear case (risks) section."""
        print(f"[{self.name}] Generating bear case...")

        risks = research.get('risk_analysis', {}).get('risks', [])
        risk_level = analysis.get('risk_analysis', {}).get('risk_level', 'Unknown')

        prompt = f"""Based on the following information for {ticker}, write a thorough bear case (3-4 paragraphs):

Identified Risks:
{chr(10).join([f"- {r}" for r in risks]) if risks else "No specific risks identified"}

Risk Level: {risk_level}

Focus on:
1. Potential headwinds
2. Competitive threats
3. Financial concerns
4. Market risks

Be balanced but cautious.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.settings.openai_model,
                messages=[
                    {"role": "system", "content": "You are a financial analyst writing the bear case for an investment."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.settings.temperature,
                max_tokens=600
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"[{self.name}] Error generating bear case: {str(e)}")
            bear_case = "### Risks\n"
            if risks:
                for risk in risks:
                    bear_case += f"- {risk}\n"
            else:
                bear_case += "Risk analysis not available.\n"
            return bear_case

    def _generate_final_perspective(
        self,
        ticker: str,
        research: Dict[str, Any],
        analysis: Dict[str, Any],
        mode: str
    ) -> str:
        """Generate final perspective section."""
        print(f"[{self.name}] Generating final perspective...")

        valuation = analysis.get('valuation_analysis', {}).get('valuation_category', 'Unknown')
        sentiment = research.get('sentiment_analysis', {}).get('overall_sentiment', 'neutral')
        risk_level = analysis.get('risk_analysis', {}).get('risk_level', 'Unknown')

        prompt = f"""Write a balanced final perspective (2-3 paragraphs) for {ticker}:

Valuation: {valuation}
Sentiment: {sentiment}
Risk Level: {risk_level}
Report Tone: {mode}

Provide:
1. Summary of key points
2. Who might find this investment suitable
3. What to watch for going forward

Be professional and balanced.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.settings.openai_model,
                messages=[
                    {"role": "system", "content": "You are a financial advisor providing a balanced perspective."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.settings.temperature,
                max_tokens=500
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"[{self.name}] Error generating final perspective: {str(e)}")
            return f"Based on the analysis, {ticker} presents a {valuation.lower()} opportunity with {sentiment} market sentiment and {risk_level.lower()} risk. Further analysis recommended."

    def _store_report(self, ticker: str, report: Dict[str, Any]) -> None:
        """Store the complete report in vector memory."""
        print(f"[{self.name}] Storing report in vector memory...")

        # Store full report
        report_text = format_report(report, format_type="markdown")
        self.vector_store.add_document(
            content=report_text,
            metadata={
                'ticker': ticker,
                'agent': self.name,
                'type': 'complete_report',
                'generated_date': report['generated_date']
            }
        )

        print(f"[{self.name}] Report stored successfully")

    def export_report(self, report: Dict[str, Any], format_type: str = "markdown") -> str:
        """
        Export report in specified format.

        Args:
            report: Report dictionary
            format_type: 'markdown' or 'json'

        Returns:
            Formatted report string
        """
        return format_report(report, format_type=format_type)

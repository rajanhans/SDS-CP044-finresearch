"""
API Client utilities for financial data and web search.
"""


from typing import List, Dict, Any
import yfinance as yf
from datetime import datetime
from config.settings import get_settings


class FinancialAPIClient:
    """Client for fetching financial data from various sources."""

    def __init__(self):
        self.settings = get_settings()

    def get_stock_data(self, ticker: str) -> Dict[str, Any]:
        """
        Fetch comprehensive stock data for a given ticker.

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', 'TSLA')

        Returns:
            Dictionary containing stock information and metrics
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period="1y")

            # Calculate price changes
            current_price = info.get('currentPrice', 0)
            day_change = ((current_price - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2] * 100) if len(hist) > 1 else 0
            week_change = ((current_price - hist['Close'].iloc[-5]) / hist['Close'].iloc[-5] * 100) if len(hist) > 5 else 0
            month_change = ((current_price - hist['Close'].iloc[-21]) / hist['Close'].iloc[-21] * 100) if len(hist) > 21 else 0
            year_change = ((current_price - hist['Close'].iloc[0]) / hist['Close'].iloc[0] * 100) if len(hist) > 0 else 0

            # Calculate volatility
            returns = hist['Close'].pct_change()
            volatility = returns.std() * (252 ** 0.5) * 100  # Annualized volatility

            return {
                'ticker': ticker,
                'company_name': info.get('longName', ticker),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'current_price': current_price,
                'market_cap': info.get('marketCap', 0),
                'price_changes': {
                    '1_day': round(day_change, 2),
                    '1_week': round(week_change, 2),
                    '1_month': round(month_change, 2),
                    '1_year': round(year_change, 2)
                },
                'volatility': round(volatility, 2),
                'pe_ratio': info.get('trailingPE', 0),
                'forward_pe': info.get('forwardPE', 0),
                'peg_ratio': info.get('pegRatio', 0),
                'price_to_book': info.get('priceToBook', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'current_ratio': info.get('currentRatio', 0),
                'roe': info.get('returnOnEquity', 0),
                'eps': info.get('trailingEps', 0),
                'revenue_growth': info.get('revenueGrowth', 0),
                'earnings_growth': info.get('earningsGrowth', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', 0),
                '52_week_high': info.get('fiftyTwoWeekHigh', 0),
                '52_week_low': info.get('fiftyTwoWeekLow', 0),
                'analyst_recommendation': info.get('recommendationKey', 'N/A'),
                'target_price': info.get('targetMeanPrice', 0)
            }
        except Exception as e:
            print(f"Error fetching data for {ticker}: {str(e)}")
            return {'ticker': ticker, 'error': str(e)}

    def get_financial_statements(self, ticker: str) -> Dict[str, Any]:
        """
        Fetch financial statements for a given ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary containing income statement, balance sheet, and cash flow
        """
        try:
            stock = yf.Ticker(ticker)
            return {
                'income_statement': stock.financials.to_dict() if hasattr(stock, 'financials') else {},
                'balance_sheet': stock.balance_sheet.to_dict() if hasattr(stock, 'balance_sheet') else {},
                'cash_flow': stock.cashflow.to_dict() if hasattr(stock, 'cashflow') else {}
            }
        except Exception as e:
            print(f"Error fetching financial statements for {ticker}: {str(e)}")
            return {'error': str(e)}


class SearchAPIClient:
    """Client for web search and news retrieval."""

    def __init__(self):
        self.settings = get_settings()

    def search_news(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for news articles related to the query.

        Args:
            query: Search query
            max_results: Maximum number of results to return

        Returns:
            List of news articles with title, url, snippet, and date
        """
        if self.settings.tavily_api_key:
            return self._search_with_tavily(query, max_results)
        elif self.settings.serpapi_api_key:
            return self._search_with_serpapi(query, max_results)
        else:
            return self._search_with_fallback(query, max_results)

    def _search_with_tavily(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search using Tavily API."""
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=self.settings.tavily_api_key)
            response = client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results,
                include_answer=True
            )

            results = []
            for result in response.get('results', []):
                results.append({
                    'title': result.get('title', ''),
                    'url': result.get('url', ''),
                    'snippet': result.get('content', ''),
                    'published_date': result.get('published_date', 'N/A')
                })
            return results
        except Exception as e:
            print(f"Error with Tavily search: {str(e)}")
            return []

    def _search_with_serpapi(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search using SerpAPI."""
        try:
            from serpapi import GoogleSearch
            params = {
                "q": query,
                "api_key": self.settings.serpapi_api_key,
                "num": max_results,
                "tbm": "nws"  # News search
            }
            search = GoogleSearch(params)
            results_data = search.get_dict()

            results = []
            for result in results_data.get('news_results', [])[:max_results]:
                results.append({
                    'title': result.get('title', ''),
                    'url': result.get('link', ''),
                    'snippet': result.get('snippet', ''),
                    'published_date': result.get('date', 'N/A')
                })
            return results
        except Exception as e:
            print(f"Error with SerpAPI search: {str(e)}")
            return []

    def _search_with_fallback(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Fallback search method using yfinance news."""
        try:
            # Extract ticker if present in query
            words = query.upper().split()
            ticker = None
            for word in words:
                if len(word) <= 5 and word.isalpha():
                    ticker = word
                    break

            if ticker:
                stock = yf.Ticker(ticker)
                news = stock.news[:max_results] if hasattr(stock, 'news') else []

                results = []
                for article in news:
                    results.append({
                        'title': article.get('title', ''),
                        'url': article.get('link', ''),
                        'snippet': article.get('summary', ''),
                        'published_date': datetime.fromtimestamp(
                            article.get('providerPublishTime', 0)
                        ).strftime('%Y-%m-%d') if article.get('providerPublishTime') else 'N/A'
                    })
                return results
            return []
        except Exception as e:
            print(f"Error with fallback search: {str(e)}")
            return []

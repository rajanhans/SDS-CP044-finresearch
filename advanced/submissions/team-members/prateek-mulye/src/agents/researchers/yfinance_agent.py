"""
This module contains the YFinanceAgent, which retrieves fundamental financial data
(valuation, margins, price history) using the yfinance library.
"""

import yfinance as yf
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from ...memory import VectorMemory
from ...state import AgentState

class YFinanceAgent:
    """
    Retrieves and analyzes fundamental stock data using YFinance.
    """
    def __init__(self):
        self.memory = VectorMemory()
        self.llm = ChatOpenAI(model="gpt-4o-mini")

    def run(self, state: AgentState):
        """
        Executes data retrieval, commentary generation, and storage.
        """
        ticker = state.get("ticker")
        
        # 1. Fetch Data
        stock = yf.Ticker(ticker)
        
        try:
            info = stock.info
            # Extract key metrics to minimize token usage / noise
            metrics = {
                "currentPrice": info.get("currentPrice"),
                "marketCap": info.get("marketCap"),
                "trailingPE": info.get("trailingPE"),
                "forwardPE": info.get("forwardPE"),
                "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
                "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
                "dividendYield": info.get("dividendYield"),
                "sector": info.get("sector"),
                "recommendationKey": info.get("recommendationKey"),
                "targetMeanPrice": info.get("targetMeanPrice")
            }
            metrics_json = json.dumps(metrics, indent=2)
        except Exception as e:
            print(f"YFinance fetch failed: {e}")
            metrics_json = "{}"

        # 2. Analyze with LLM (Data Commentary)
        system_prompt = """You are a Fundamental Data Researcher.
        Review the raw financial metrics for {ticker}.
        
        Provide a brief commentary on:
        1. Valuation (expensive/cheap based on P/E).
        2. Price position relative to 52-week range.
        3. Analyst consensus (target price vs current).
        
        Do not give a buy/sell recommendation, just interpret the data.
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "Financial Data:\n{data}")
        ])
        
        chain = prompt | self.llm
        analysis = chain.invoke({"ticker": ticker, "data": metrics_json})
        
        # 3. Store in Pinecone
        doc_raw = Document(
            page_content=f"Raw Financial Metrics:\n{metrics_json}",
            metadata={"source": "YFinance_Data", "ticker": ticker, "type": "raw_metrics"}
        )
        doc_analysis = Document(
            page_content=analysis.content,
            metadata={"source": "YFinance_Commentary", "ticker": ticker, "type": "data_analysis"}
        )
        
        self.memory.add_documents([doc_raw, doc_analysis], source="YFinance")
        
        return {
            "messages": [HumanMessage(content=f"YFinance Agent finished research for {ticker}.")]
        }

def yfinance_node(state: AgentState):
    agent = YFinanceAgent()
    return agent.run(state)

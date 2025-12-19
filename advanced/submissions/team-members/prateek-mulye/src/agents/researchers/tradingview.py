"""
This module contains the TradingViewAgent, which retrieves technical analysis indicators
(RSI, MACD, Moving Averages) using the tradingview-ta library.
"""

import json
from tradingview_ta import TA_Handler, Interval
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from ...memory import VectorMemory
from ...state import AgentState

class TradingViewAgent:
    """
    Retrieves and analyzes technical indicators using TradingView TA.
    """
    def __init__(self):
        self.memory = VectorMemory()
        self.llm = ChatOpenAI(model="gpt-4o-mini")

    def run(self, state: AgentState):
        """
        Executes technical analysis retrieval, interpretation, and storage.
        """
        ticker = state.get("ticker")
        
        # 1. Fetch Technicals
        try:
            handler = TA_Handler(
                symbol=ticker,
                screener="america",
                exchange="NASDAQ",
                interval=Interval.INTERVAL_1_DAY
            )
            analysis = handler.get_analysis()
            
            technicals = {
                "summary": analysis.summary,
                "indicators": {
                    "RSI": analysis.indicators.get("RSI"),
                    "MACD": analysis.indicators.get("MACD.macd"),
                    "SMA20": analysis.indicators.get("SMA20"),
                    "EMA20": analysis.indicators.get("EMA20"),
                    "Open": analysis.indicators.get("open"),
                    "Close": analysis.indicators.get("close")
                }
            }
            technicals_json = json.dumps(technicals, indent=2)
            
        except Exception:
            try:
                # Try NYSE fallback
                handler = TA_Handler(
                    symbol=ticker,
                    screener="america",
                    exchange="NYSE",
                    interval=Interval.INTERVAL_1_DAY
                )
                analysis = handler.get_analysis()
                technicals = {
                    "summary": analysis.summary,
                    "indicators": {
                        "RSI": analysis.indicators.get("RSI"),
                        "MACD": analysis.indicators.get("MACD.macd"),
                        "SMA20": analysis.indicators.get("SMA20")
                    }
                }
                technicals_json = json.dumps(technicals, indent=2)
            except Exception:
                technicals_json = "{}"

        # 2. Analyze with LLM
        system_prompt = """You are a Technical Analyst.
        Review the TradingView technical indicators for {ticker}.
        
        Provide a brief analysis of:
        1. Momentum (RSI, MACD).
        2. Trend (Moving Averages).
        3. Overall Technical Signal (Buy/Sell).
        
        Be precise about the signals.
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "Technical Data:\n{data}")
        ])
        
        chain = prompt | self.llm
        analysis_text = chain.invoke({"ticker": ticker, "data": technicals_json})
        
        # 3. Store in Pinecone
        doc_raw = Document(
            page_content=f"Raw Technicals:\n{technicals_json}",
            metadata={"source": "TradingView_Technicals", "ticker": ticker, "type": "raw_technicals"}
        )
        doc_analysis = Document(
            page_content=analysis_text.content,
            metadata={"source": "TradingView_Analysis", "ticker": ticker, "type": "technical_analysis"}
        )
        
        self.memory.add_documents([doc_raw, doc_analysis], source="TradingView")
        
        return {
            "messages": [HumanMessage(content=f"TradingView Agent finished research for {ticker}.")]
        }

def tradingview_node(state: AgentState):
    agent = TradingViewAgent()
    return agent.run(state)

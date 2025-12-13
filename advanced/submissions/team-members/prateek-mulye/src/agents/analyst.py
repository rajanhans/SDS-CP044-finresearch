import yfinance as yf
import json
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from ..memory import VectorMemory
from ..state import AgentState

class AnalystAgent:
    def __init__(self):
        self.memory = VectorMemory()
        self.llm = ChatOpenAI(model="gpt-5-nano")

    def run(self, state: AgentState):
        """
        Analyst Agent Node logic.
        """
        print("--- Financial Analyst Agent ---")
        ticker = state.get("ticker")
        
        # 1. Fetch Data
        print(f"Fetching data for: {ticker}")
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Extract key metrics
        current_price = info.get("currentPrice") or info.get("regularMarketPrice")
        pe_ratio = info.get("trailingPE")
        fifty_two_high = info.get("fiftyTwoWeekHigh")
        fifty_two_low = info.get("fiftyTwoWeekLow")
        market_cap = info.get("marketCap")
        
        financial_data = {
            "current_price": current_price,
            "pe_ratio": pe_ratio,
            "fifty_two_high": fifty_two_high,
            "fifty_two_low": fifty_two_low,
            "market_cap": market_cap
        }
        
        # 2. LLM Analysis
        # Generate a strategic insight based on the numbers
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a financial analyst. specificy in quantitative analysis. Given the following financial data for {ticker}, provide a brief analysis of its valuation and market position. Do not give financial advice, just analysis."),
            ("user", "Financial Data:\n{data}")
        ])
        
        chain = prompt | self.llm
        print("Generating analysis with LLM...")
        analysis = chain.invoke({"ticker": ticker, "data": json.dumps(financial_data, indent=2)})
        analysis_text = analysis.content
        
        # 3. Store in Pinecone as a Document
        # We create a document that contains BOTH the raw data (as context) and the analysis
        
        content = f"Financial Analysis for {ticker}:\n\n"
        content += f"Analysis:\n{analysis_text}\n\n"
        content += f"Raw Data:\n{json.dumps(financial_data, indent=2)}"
        
        doc = Document(
            page_content=content,
            metadata={
                "ticker": ticker,
                "type": "financial_analysis",
                "source": "analyst_agent"
            }
        )
        
        self.memory.add_documents([doc], source="analyst_agent")
        print(f"Stored structured financial analysis for {ticker} in Pinecone.")
        
        # 4. Update State
        return {
            "financial_data": financial_data,
            "messages": [HumanMessage(content=f"Analysis completed for {ticker}.")]
        }

def analyst_node(state: AgentState):
    agent = AnalystAgent()
    return agent.run(state)

from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from ..memory import VectorMemory
from ..state import AgentState

class ResearcherAgent:
    def __init__(self):
        self.search_tool = TavilySearchResults(max_results=5)
        self.memory = VectorMemory()
        self.llm = ChatOpenAI(model="gpt-5-nano")

    def run(self, state: AgentState):
        """
        Researcher Agent Node logic.
        """
        print("--- Researcher Agent ---")
        ticker = state.get("ticker")
        
        # 1. Search
        query = f"Financial news and market sentiment for {ticker} stock"
        print(f"Searching for: {query}")
        
        results = self.search_tool.invoke(query)
        
        # 2. LLM Summarization
        # We want to synthesize the search results before storing
        raw_content = ""
        for res in results:
            raw_content += f"Source: {res.get('url')}\nContent: {res.get('content')}\n\n"
            
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a specialized financial researcher. Summarize the following search results for {ticker} into a concise market update. Focus on sentiment, key events, and potential risks/opportunities."),
            ("user", "Search Results:\n{raw_content}")
        ])
        
        chain = prompt | self.llm
        print("Synthesizing research with LLM...")
        summary = chain.invoke({"ticker": ticker, "raw_content": raw_content})
        summary_text = summary.content
        
        # 3. Store in Pinecone
        # We store the synthesized summary as a high-value document
        doc = Document(
            page_content=summary_text, 
            metadata={
                "ticker": ticker, 
                "source": "researcher_agent_summary",
                "type": "market_summary"
            }
        )
        
        # We can also store the raw chunks if needed, but for now let's prioritize the insight
        self.memory.add_documents([doc], source="researcher_agent")
        print(f"Stored synthesized research summary in Pinecone.")
            
        # 4. Update State
        return {
            "research_summary": summary_text,
            "messages": [HumanMessage(content=f"Research completed for {ticker}.")]
        }

def researcher_node(state: AgentState):
    agent = ResearcherAgent()
    return agent.run(state)

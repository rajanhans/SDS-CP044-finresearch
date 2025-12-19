"""
This module contains the AnalystAgent, which is responsible for aggregating research data,
performing quality control, and generating a quantitative score and recommendation.
"""

import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from ..memory import VectorMemory
from ..state import AgentState

class AnalystAgent:
    """
    Acts as the Lead Analyst.
    Aggregates data from shared memory (Pinecone), performs QC, and scores the asset.
    """
    def __init__(self):
        self.memory = VectorMemory()
        self.llm = ChatOpenAI(model="gpt-4o-mini")

    def run(self, state: AgentState):
        """
        Executes the analysis logic:
        1. Retrieve context from memory.
        2. Synthesize and QC data.
        3. Generate Verdict (Score/Rec).
        4. Store result back to memory.
        """
        print("--- Analyst Agent (Aggregation & Scoring) ---")
        ticker = state.get("ticker")
        mode = state.get("investor_mode", "Neutral")
        
        # 1. Retrieve Context
        results = self.memory.similarity_search(f"financial data, news, and technicals for {ticker}", k=15)
        
        context_data = ""
        for doc in results:
            source = doc.metadata.get("source", "unknown")
            context_data += f"-- Source: {source} --\n{doc.page_content}\n\n"
            
        if not context_data:
            context_data = "No data available."

        # 2. Aggregation & QC
        qc_system_prompt = """You are a Lead Financial Analyst.
        Review the gathered research data for {ticker}.
        
        Tasks:
        1. Sanity Check: Are there conflicting price points or outdated data?
        2. Aggregation: Synthesize the Fundamental, Technical, and Sentiment data into a coherent view.
        """
        
        qc_prompt = ChatPromptTemplate.from_messages([
            ("system", qc_system_prompt),
            ("user", "Research Data:\n{data}")
        ])
        
        qc_chain = qc_prompt | self.llm
        aggregated_view = qc_chain.invoke({"ticker": ticker, "data": context_data}).content
        
        # 3. Scoring & Verdict
        score_system_prompt = """You are a Quantitative Analyst.
        Based on the Aggregated Research for {ticker} and the requested investor mode: {mode},
        generate a final rating.
        
        Output format must be strictly JSON:
        {{
            "score": <0-100 integer>,
            "recommendation": "<Buy|Sell|Hold>",
            "reasoning": "<Concise 2-sentence explanation>"
        }}
        """
        
        score_prompt = ChatPromptTemplate.from_messages([
            ("system", score_system_prompt),
            ("user", "Aggregated Research:\n{view}")
        ])
        
        score_chain = score_prompt | self.llm
        verdict = score_chain.invoke({"ticker": ticker, "mode": mode, "view": aggregated_view}).content
        
        # 4. Store Verdict
        doc_verdict = Document(
            page_content=f"Analyst Verdict:\n{verdict}\n\nAggregated View:\n{aggregated_view}",
            metadata={"source": "Analyst_Verdict", "ticker": ticker, "type": "final_verdict"}
        )
        self.memory.add_documents([doc_verdict], source="Analyst")
        
        # Parse verdict logic
        verdict_data = {}
        try:
            clean_verdict = verdict.replace("```json", "").replace("```", "").strip()
            verdict_data = json.loads(clean_verdict)
        except Exception as e:
            print(f"Error parsing verdict JSON: {e}")
        
        return {
            "analyst_verdict": verdict_data,
            "messages": [HumanMessage(content=f"Analyst Verdict: {verdict}")]
        }

def analyst_node(state: AgentState):
    agent = AnalystAgent()
    return agent.run(state)

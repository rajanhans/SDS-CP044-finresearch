"""
This module contains the ManagerAgent, which acts as the orchestrator of the system.
It checks for cached data and determines which research sub-agents to activate.
"""

import time
import json
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from ..state import AgentState
from ..memory import VectorMemory

class ManagerAgent:
    """
    The Manager Agent decides the execution plan based on cache freshness and user intent.
    """
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.memory = VectorMemory()

    def run(self, state: AgentState):
        """
        Analyzes the request, checks Pinecone cache, and determines the agents to run.
        """
        ticker = state.get("ticker")
        mode = state.get("investor_mode", "Neutral")
        
        print(f"--- Manager Agent: Orchestrating for {ticker} ({mode}) ---")
        
        # Check Cache
        query = f"Analyst Verdict for {ticker}"
        results = self.memory.similarity_search(query, k=1)
        
        cache_status = "No cache found."
        recency_minutes = 99999
        
        if results:
            doc = results[0]
            timestamp = doc.metadata.get("timestamp", 0)
            now = time.time()
            diff = now - timestamp
            recency_minutes = diff / 60
            
            if recency_minutes < 5:
                cache_status = f"FRESH_CACHE_FOUND ({recency_minutes:.1f} mins ago)."
            else:
                cache_status = f"STALE_CACHE_FOUND ({recency_minutes:.1f} mins ago)."
        
        print(f"Cache Status: {cache_status}")
        
        # LLM Decision
        system_prompt = """You are a Senior Research Manager.
        Your goal is to orchestrate a financial research team for {ticker}.
        User Mode: {mode}.
        Cache Context: {cache_status}
        
        Decide which agents to run.
        Available: ["tavily_researcher", "yfinance_researcher", "tradingview_researcher"]
        
        Strategy:
        1. If cache is FRESH (< 5 mins), return [] (empty list) to use existing data.
        2. If cache is STALE or NO CACHE, run ALL agents.
        
        Output strictly JSON:
        {{
            "plan": "Short strategic plan",
            "agents_to_run": ["agent_name"]
        }}
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", f"Orchestrate research for {ticker}.")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({"ticker": ticker, "mode": mode, "cache_status": cache_status})
        
        try:
            content = response.content.replace("```json", "").replace("```", "").strip()
            decision = json.loads(content)
            plan_text = decision.get("plan", "Proceeding with research.")
            agents_to_run = decision.get("agents_to_run", [])
        except Exception as e:
            print(f"Error parsing Manager JSON: {e}")
            plan_text = "Error parsing plan, defaulting to full run."
            agents_to_run = ["tavily_researcher", "yfinance_researcher", "tradingview_researcher"]
            
        print(f"Manager Decision: Run {agents_to_run}")
        
        return {
            "messages": [HumanMessage(content=f"Manager Plan: {plan_text}")],
            "agents_to_run": agents_to_run
        }

def manager_node(state: AgentState):
    agent = ManagerAgent()
    return agent.run(state)

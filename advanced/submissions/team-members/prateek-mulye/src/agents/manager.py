from langchain_core.messages import HumanMessage
from ..state import AgentState

class ManagerAgent:
    def run(self, state: AgentState):
        """
        Manager Agent Node logic.
        For Week 1, this simply acknowledges the request and prepares the flow.
        """
        print("--- Manager Agent ---")
        ticker = state.get("ticker")
        print(f"Manager received request for {ticker}. Delegating to Researcher and Analyst.")
        
        return {
            "messages": [HumanMessage(content=f"Manager: Starting investigation for {ticker}")]
        }

def manager_node(state: AgentState):
    agent = ManagerAgent()
    return agent.run(state)

from langgraph.graph import StateGraph, END
from .state import AgentState
from .agents.manager import manager_node
from .agents.researcher import researcher_node
from .agents.analyst import analyst_node

def create_graph():
    """
    Construct the FinResearch AI Multi-Agent Graph.
    Week 1 Flow: Manager -> Researcher -> Analyst -> END
    """
    workflow = StateGraph(AgentState)

    # 1. Add Nodes
    workflow.add_node("manager", manager_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("analyst", analyst_node)

    # 2. Add Edges
    # Start at Manager
    workflow.set_entry_point("manager")
    
    # Manager -> Researcher
    workflow.add_edge("manager", "researcher")
    
    # Researcher -> Analyst
    workflow.add_edge("researcher", "analyst")
    
    # Analyst -> END
    workflow.add_edge("analyst", END)

    # 3. Compile
    app = workflow.compile()
    return app

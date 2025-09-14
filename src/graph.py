# In src/graph.py

from typing import List, TypedDict
from langchain.schema import Document

class AgentState(TypedDict):
    question: str
    retrieved_texts: List
    retrieved_images: List[str]
    general_agent_response: str
    critical_text_info: str
    critical_image_info: str
    text_agent_response: str
    image_agent_response: str
    final_answer: str

    # In src/graph.py

from langgraph.graph import StateGraph, END
from src.agents import (
    general_agent_node,
    critical_agent_node,
    text_agent_node,
    image_agent_node,
    summarizing_agent_node
)

# This file already contains the AgentState TypedDict definition from Section 5

def build_mdoc_agent_graph():
    """
    Builds the LangGraph StateGraph for the MDocAgent workflow.
    """
    workflow = StateGraph(AgentState)

    # Add the nodes to the graph
    workflow.add_node("general_agent", general_agent_node)
    workflow.add_node("critical_agent", critical_agent_node)
    workflow.add_node("text_agent", text_agent_node)
    workflow.add_node("image_agent", image_agent_node)
    workflow.add_node("summarizing_agent", summarizing_agent_node)

    # Define the edges to control the flow
    workflow.set_entry_point("general_agent")
    workflow.add_edge("general_agent", "critical_agent")
    workflow.add_edge("critical_agent", "text_agent")
    workflow.add_edge("text_agent", "image_agent")
    workflow.add_edge("image_agent", "summarizing_agent")
    workflow.add_edge("summarizing_agent", END)

    # Compile the graph into a runnable application
    app = workflow.compile()
    
    print("MDocAgent graph compiled successfully.")
    
    return app
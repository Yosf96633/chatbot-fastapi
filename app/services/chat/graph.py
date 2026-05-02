# app/services/chat/graph.py
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from app.services.chat.model import chatState
from app.services.chat.nodes import chat_node, tool_node, should_use_tools 

# global variable — main.py stores the compiled workflow here
workflow: CompiledStateGraph

def build_graph(checkpointer):
    g = StateGraph(chatState)

    g.add_node("chat_node", chat_node)
    g.add_node("tools", tool_node)
    

    g.add_edge(START, "chat_node")
    # After chat_node: either call tools or finish
    g.add_conditional_edges(
        "chat_node",
        should_use_tools,
        {"tools": "tools", "end": END},
    )
    # After tools run, go back to chat_node so LLM can respond with results
    g.add_edge("tools", "chat_node")
    
    

    return g.compile(checkpointer=checkpointer)

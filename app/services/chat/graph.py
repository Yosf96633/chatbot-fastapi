from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from app.services.chat.model import chatState
from app.services.chat.nodes import chat_node, tool_node, should_use_tools, summarize_and_trim_messages, remember_node

workflow: CompiledStateGraph


def build_graph(checkpointer, store):
    g = StateGraph(chatState)

    g.add_node("remember_node", remember_node)                           # ← add
    g.add_node("chat_node", chat_node)
    g.add_node("tools", tool_node)
    g.add_node("summarize_and_trim_messages_node", summarize_and_trim_messages)

    g.add_edge(START, "remember_node")                                   # ← start here
    g.add_edge("remember_node", "summarize_and_trim_messages_node")      # ← then summarize
    g.add_edge("summarize_and_trim_messages_node", "chat_node")

    g.add_conditional_edges(
        "chat_node",
        should_use_tools,
        {"tools": "tools", "end": END},
    )

    g.add_edge("tools", "chat_node")

    return g.compile(
        checkpointer=checkpointer,
        store=store
    )
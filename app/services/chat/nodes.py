from dotenv import load_dotenv
load_dotenv()

from app.services.chat.model import chatState
from app.services.chat.tools import all_tools
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import ToolNode

# Initialize llm with tools bound
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.65, streaming=True)
llm_with_tools = llm.bind_tools(all_tools)

SYSTEM_MESSAGE = SystemMessage(content=(
    "You are a helpful assistant with access to tools for web search, "
    "weather, stock prices, and a calculator. Use tools when the user's "
    "question requires current or factual data. Think step by step."
))

# Prebuilt node that executes whatever tool calls the LLM requests
tool_node = ToolNode(all_tools)


def _sanitize_messages(messages: list) -> list:
    """
    Remove any AI messages that have tool_calls but are NOT followed by
    the corresponding ToolMessages. This handles corrupted checkpoint state.
    """
    # Collect all tool_call_ids that have a matching ToolMessage
    answered_ids = set()
    for msg in messages:
        if isinstance(msg, ToolMessage):
            answered_ids.add(msg.tool_call_id)

    cleaned = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            call_ids = {tc["id"] for tc in msg.tool_calls}
            if not call_ids.issubset(answered_ids):
                # This AI message has unanswered tool calls — drop it
                continue
        cleaned.append(msg)

    return cleaned


def _safe_trim(messages: list, limit: int = 20) -> list:
    """
    Trim to the last `limit` messages, but never break a tool_call /
    tool_result pair by cutting in the middle.
    Start from the oldest message that keeps us within the limit and
    doesn't start with a ToolMessage (which would be orphaned).
    """
    if len(messages) <= limit:
        return messages

    trimmed = messages[-limit:]

    # If the first message is a ToolMessage it is orphaned — drop messages
    # from the front until we start cleanly.
    while trimmed and isinstance(trimmed[0], ToolMessage):
        trimmed = trimmed[1:]

    return trimmed


async def chat_node(state: chatState):
    messages = _sanitize_messages(state['messages'])
    messages = _safe_trim(messages, limit=20)

    ai_response = await llm_with_tools.ainvoke([SYSTEM_MESSAGE] + messages)

    return {"messages": [ai_response]}


def should_use_tools(state: chatState) -> str:
    """Route to tool_node if the LLM requested tool calls, else END."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "end"

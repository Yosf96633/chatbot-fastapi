from langchain_core.messages import trim_messages, RemoveMessage
from langchain_core.output_parsers import StrOutputParser
import json
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.prebuilt import ToolNode
from langchain_core.messages import (
    AIMessage, SystemMessage, ToolMessage,
    BaseMessage, HumanMessage
)
from langchain_openai import ChatOpenAI
from app.services.chat.tools import all_tools
from app.services.chat.model import chatState
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from pydantic import BaseModel
from typing import Literal
from dotenv import load_dotenv
load_dotenv()
# ─────────────────────────────────────────────
#  COMPACT VIEW
# ─────────────────────────────────────────────


def print_messages(messages: list[BaseMessage], title: str = "MESSAGE HISTORY"):
    """Compact view — role + content + tool info only."""
    role_colors = {
        "system": "\033[90m",
        "human":  "\033[94m",
        "ai":     "\033[92m",
        "tool":   "\033[93m",
    }
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    print("\n" + "═" * 60)
    print(f"{BOLD}  {title}  ({len(messages)} messages){RESET}")
    print("═" * 60)

    for i, msg in enumerate(messages):
        role = msg.type
        color = role_colors.get(role, "\033[97m")

        print(f"\n{color}{BOLD}[{i}] {role.upper()}{RESET}")

        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"  {DIM}tool_call → {tc['name']}{RESET}")
                print(f"  {DIM}args      → {tc['args']}{RESET}")
                print(f"  {DIM}id        → {tc['id']}{RESET}")

        if isinstance(msg, ToolMessage):
            print(f"  {DIM}tool_call_id → {msg.tool_call_id}{RESET}")
            if hasattr(msg, "name") and msg.name:
                print(f"  {DIM}name         → {msg.name}{RESET}")

        content = msg.content
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    btype = block.get("type", "unknown")
                    if btype == "text":
                        print(f"  {block['text']}")
                    else:
                        print(f"  {DIM}[{btype} block]{RESET}")
        elif content:
            print(f"  {content}")
        else:
            print(f"  {DIM}(no content){RESET}")

    print("\n" + "═" * 60 + "\n")


# ─────────────────────────────────────────────
#  DETAILED VIEW
# ─────────────────────────────────────────────
def print_messages_detailed(messages: list[BaseMessage], title: str = "DETAILED MESSAGE INSPECTION"):
    """Verbose view — every field of every message, fully structured."""
    role_colors = {
        "system": "\033[90m",
        "human":  "\033[94m",
        "ai":     "\033[92m",
        "tool":   "\033[93m",
    }
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    RED = "\033[91m"

    def field(label: str, value, color=CYAN):
        print(f"    {DIM}{label:<20}{RESET}{color}{value}{RESET}")

    def section(label: str):
        print(f"  {YELLOW}{BOLD}▸ {label}{RESET}")

    print("\n" + "╔" + "═" * 62 + "╗")
    print(f"║{BOLD}  {title:<61}{RESET}║")
    print(f"║  Total messages: {len(messages):<43}║")
    print("╚" + "═" * 62 + "╝")

    for i, msg in enumerate(messages):
        role = msg.type
        color = role_colors.get(role, "\033[97m")

        print(f"\n  {color}{BOLD}{'─' * 56}{RESET}")
        print(f"  {color}{BOLD}  [{i}]  {role.upper():<50}{RESET}")
        print(f"  {color}{BOLD}{'─' * 56}{RESET}")

        section("Core")
        field("type",  msg.type)
        field("class", type(msg).__name__)
        field("id",    msg.id or "(none)")

        section("Content")
        content = msg.content
        if isinstance(content, list):
            field("content_type", f"list ({len(content)} blocks)")
            for j, block in enumerate(content):
                if isinstance(block, dict):
                    btype = block.get("type", "unknown")
                    print(f"      {DIM}block[{j}] type → {btype}{RESET}")
                    if btype == "text":
                        preview = block["text"][:120] + \
                            ("..." if len(block["text"]) > 120 else "")
                        print(f"      {DIM}block[{j}] text → {RESET}{preview}")
                    else:
                        print(
                            f"      {DIM}block[{j}] data → {json.dumps(block)[:80]}{RESET}")
        elif content:
            field("content_type", "str")
            words = content.split()
            line = ""
            prefix = f"    {'content':<20}"
            first = True
            for word in words:
                if len(line) + len(word) > 80:
                    if first:
                        print(f"{prefix}{CYAN}{line.strip()}{RESET}")
                        first = False
                    else:
                        print(f"    {' ' * 20}{CYAN}{line.strip()}{RESET}")
                    line = word + " "
                else:
                    line += word + " "
            if line.strip():
                if first:
                    print(f"{prefix}{CYAN}{line.strip()}{RESET}")
                else:
                    print(f"    {' ' * 20}{CYAN}{line.strip()}{RESET}")
        else:
            field("content", "(empty)", RED)

        if isinstance(msg, AIMessage):
            section("AI Specifics")
            if msg.tool_calls:
                field("tool_calls count", len(msg.tool_calls))
                for j, tc in enumerate(msg.tool_calls):
                    print(f"      {YELLOW}tool_call[{j}]{RESET}")
                    print(f"        {DIM}name  → {RESET}{tc.get('name')}")
                    print(f"        {DIM}id    → {RESET}{tc.get('id')}")
                    args_str = json.dumps(tc.get("args", {}), indent=2)
                    for ln in args_str.splitlines():
                        print(f"        {DIM}args  → {RESET}{ln}")
            else:
                field("tool_calls", "(none)")

            if hasattr(msg, "invalid_tool_calls") and msg.invalid_tool_calls:
                field("invalid_tool_calls", len(msg.invalid_tool_calls), RED)
            else:
                field("invalid_tool_calls", "(none)")

            if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                um = msg.usage_metadata
                field("usage.input_tokens",  um.get("input_tokens",  "—"))
                field("usage.output_tokens", um.get("output_tokens", "—"))
                field("usage.total_tokens",  um.get("total_tokens",  "—"))
            else:
                field("usage_metadata", "(none)")

            if hasattr(msg, "response_metadata") and msg.response_metadata:
                rm = msg.response_metadata
                field("finish_reason", rm.get("finish_reason", "—"))
                field("model_name",    rm.get("model_name",    "—"))
            else:
                field("response_metadata", "(none)")

        if isinstance(msg, ToolMessage):
            section("Tool Specifics")
            field("tool_call_id", msg.tool_call_id)
            field("name",         getattr(msg, "name", None) or "(none)")
            field("status",       getattr(msg, "status", "—"))
            field("artifact",     str(getattr(msg, "artifact", None) or "(none)"))

        if isinstance(msg, HumanMessage):
            section("Human Specifics")
            field("name",    getattr(msg, "name", None) or "(none)")
            field("example", getattr(msg, "example", False))

        if isinstance(msg, SystemMessage):
            section("System Specifics")
            field("name", getattr(msg, "name", None) or "(none)")

        section("additional_kwargs")
        if msg.additional_kwargs:
            for k, v in msg.additional_kwargs.items():
                field(k, str(v)[:80])
        else:
            field("(empty)", "", DIM)

    print("\n" + "╔" + "═" * 62 + "╗")
    print(f"║{BOLD}  END OF INSPECTION  —  {len(messages)} messages{' ' * 22}{RESET}║")
    print("╚" + "═" * 62 + "╝\n")



# ─────────────────────────────────────────────
#  LLM + TOOLS
# ─────────────────────────────────────────────
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.65, streaming=True)
llm_with_tools = llm.bind_tools(all_tools)

# ✅ Realistic limit — gpt-4o-mini has 128k context, keep last ~10k tokens
MAX_TOKEN = 10_000

class Memory(BaseModel):
    text: str
    action: Literal["add", "exists"]  # exists = already in store, skip

class MemoryList(BaseModel):
    memories: list[Memory]

tool_node = ToolNode(all_tools)


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def _sanitize_messages(messages: list) -> list:
    """
    Remove AI messages that have tool_calls not followed by
    corresponding ToolMessages (corrupted checkpoint state).
    """
    answered_ids = {
        msg.tool_call_id
        for msg in messages
        if isinstance(msg, ToolMessage)
    }
    cleaned = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            call_ids = {tc["id"] for tc in msg.tool_calls}
            if not call_ids.issubset(answered_ids):
                continue  # drop orphaned AI tool-call message
        cleaned.append(msg)
    return cleaned


def _safe_trim(messages: list, limit: int = 20) -> list:
    """
    Trim to the last `limit` messages without orphaning ToolMessages.
    """
    if len(messages) <= limit:
        return messages
    trimmed = messages[-limit:]
    while trimmed and isinstance(trimmed[0], ToolMessage):
        trimmed = trimmed[1:]
    return trimmed


# ─────────────────────────────────────────────
#  GRAPH NODES
# ─────────────────────────────────────────────

async def remember_node(state: chatState, config: RunnableConfig, *, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    namespace = ("users", user_id, "details")
    last_message = state["messages"][-1]

    # ── 1. fetch existing relevant memories from store
    existing_memories = await store.asearch(
        namespace,
        query=last_message.content,
        limit=10
    )

    # format existing memories into readable string
    existing_text = "\n".join([
        f"- {m.value.get('text', '')}"
        for m in existing_memories
        if m.value.get("text")
    ]) or "No memories yet."
    print(f'Existing memory : {existing_text}')
    # ── 2. extractor LLM with structured output
    extractor_llm = ChatOpenAI(model="gpt-4o-mini").with_structured_output(MemoryList)

    extraction_prompt = f"""
You are a memory manager. Your job is to extract important user information from the message below.

Important information includes:
- Name, age, location
- Role (teacher, admin, parent, student)
- School name
- Preferences or complaints
- Any personal details worth remembering

Already stored memories:
{existing_text}

New message:
{last_message.content}

Instructions:
- For each piece of important information you find, return it as a memory
- If the information is ALREADY in the stored memories above, mark action as "exists"
- If the information is NEW and not in stored memories, mark action as "add"
- If there is nothing important in the message, return an empty list
"""

    result: MemoryList = await extractor_llm.ainvoke([
        HumanMessage(content=extraction_prompt)
    ])

    # ── 3. iterate and only store new memories
    import time
    for memory in result.memories:
        if memory.action == "add":
            await store.aput(
                namespace,
                key=f"memory_{int(time.time() * 1000)}",  # unique timestamp key
                value={"text": memory.text},
            )

    # ── 4. pass existing memories to chat_node via state
    memory_context = existing_text

    return {"memory_context": memory_context}
    
    
    
    
async def chat_node(state: chatState, config: RunnableConfig):
    system_content = (
        "You are a helpful assistant with access to tools for web search, "
        "weather, stock prices, and a calculator. Use tools when the user's "
        "question requires current or factual data. Think step by step."
    )

    # ── inject long term memories
    memory_context = state.get("memory_context", "")
    if memory_context and memory_context != "No memories yet.":
        system_content += f"\n\nWhat you remember about this user:\n{memory_context}"

    # ── inject short term summary
    if state.get("summary"):
        system_content += f"\n\nContext from earlier in the conversation:\n{state['summary']}"

    SYSTEM_MESSAGE = SystemMessage(content=system_content)

    ai_response = await llm_with_tools.ainvoke([SYSTEM_MESSAGE] + state["messages"])
    return {"messages": [ai_response]}





async def summarize_and_trim_messages(state: chatState):
    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.65)
    parser = StrOutputParser()

    messages = state['messages']
    existing_summary = state.get('summary', None)
    print(f'Existing Summary : {existing_summary}')
    # only trigger when threshold is hit
    if len(messages) < 20:
        return {}  # no changes needed

    # split: old messages to summarize, recent to keep
    messages_to_summarize = messages[:-6]  # everything except last 4
    messages_to_keep = messages[-6:]       # last 4 kept verbatim

    # build summarization prompt based on whether summary already exists
    if existing_summary is None:
        # first time summarizing
        summarize_prompt = (
            "Summarize the following conversation concisely. "
            "Capture key topics, decisions, and important context:\n\n"
            + "\n".join([
                f"{m.type.upper()}: {m.content}"
                for m in messages_to_summarize
            ])
        )
    else:
        # extend existing summary
        summarize_prompt = (
            f"You have an existing summary of a conversation:\n{existing_summary}\n\n"
            "Now extend it by incorporating these newer messages. "
            "Keep it concise but preserve all important context:\n\n"
            + "\n".join([
                f"{m.type.upper()}: {m.content}"
                for m in messages_to_summarize
            ])
        )

    # call LLM to generate summary
    new_summary = await llm.ainvoke([HumanMessage(content=summarize_prompt)])
    new_summary_text = parser.invoke(new_summary)

    # delete old messages from state, keep only recent ones
    # RemoveMessage tells LangGraph to delete by message id
    messages_to_delete = [RemoveMessage(
        id=m.id) for m in messages_to_summarize if m.id is not None]

    return {
        "messages": messages_to_delete,  # LangGraph will remove these
        "summary": new_summary_text
    }


def should_use_tools(state: chatState) -> str:
    """Route to tool_node if the LLM requested tool calls, else END."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "end"

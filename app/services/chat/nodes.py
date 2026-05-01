from dotenv import load_dotenv
load_dotenv()

from app.services.chat.model import chatState
from app.services.chat.tools import all_tools
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    AIMessage, SystemMessage, ToolMessage,
    BaseMessage, HumanMessage
)
from langgraph.prebuilt import ToolNode
from langchain_core.messages import trim_messages
from langchain_core.messages.utils import count_tokens_approximately
import json


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
    BOLD  = "\033[1m"
    DIM   = "\033[2m"

    print("\n" + "═" * 60)
    print(f"{BOLD}  {title}  ({len(messages)} messages){RESET}")
    print("═" * 60)

    for i, msg in enumerate(messages):
        role  = msg.type
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
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    CYAN   = "\033[96m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"

    def field(label: str, value, color=CYAN):
        print(f"    {DIM}{label:<20}{RESET}{color}{value}{RESET}")

    def section(label: str):
        print(f"  {YELLOW}{BOLD}▸ {label}{RESET}")

    print("\n" + "╔" + "═" * 62 + "╗")
    print(f"║{BOLD}  {title:<61}{RESET}║")
    print(f"║  Total messages: {len(messages):<43}║")
    print("╚" + "═" * 62 + "╝")

    for i, msg in enumerate(messages):
        role  = msg.type
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
                        preview = block["text"][:120] + ("..." if len(block["text"]) > 120 else "")
                        print(f"      {DIM}block[{j}] text → {RESET}{preview}")
                    else:
                        print(f"      {DIM}block[{j}] data → {json.dumps(block)[:80]}{RESET}")
        elif content:
            field("content_type", "str")
            words  = content.split()
            line   = ""
            prefix = f"    {'content':<20}"
            first  = True
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

SYSTEM_MESSAGE = SystemMessage(content=(
    "You are a helpful assistant with access to tools for web search, "
    "weather, stock prices, and a calculator. Use tools when the user's "
    "question requires current or factual data. Think step by step."
))

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
async def chat_node(state: chatState):
    print_messages(state["messages"], title="Before Trimming")

    messages = trim_messages(
        messages=state["messages"],
        max_tokens=MAX_TOKEN,
        strategy="last",
        token_counter=count_tokens_approximately,
        include_system=False,   # we inject SYSTEM_MESSAGE manually
        start_on="human",       # ✅ never start with an orphaned AI reply
    )

    print_messages(messages, title="After Trimming")

    ai_response = await llm_with_tools.ainvoke([SYSTEM_MESSAGE] + messages)
    return {"messages": [ai_response]}


def should_use_tools(state: chatState) -> str:
    """Route to tool_node if the LLM requested tool calls, else END."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "end"
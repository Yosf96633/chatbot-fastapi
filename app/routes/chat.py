# app/routes/chat.py
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
import app.services.chat.graph as graph_module
import json
from langchain_openai import ChatOpenAI
from fastapi.exceptions import HTTPException
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import messages_to_dict
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.types import Command


router = APIRouter(prefix='/api/v1')


class RequestBody(BaseModel):
    query: str
    thread_id: str


class ResumeBody(BaseModel):
    thread_id: str
    decision: str  # "yes" or "no"


# ── Shared helper: extract & forward interrupt ───────────────────────────────
def _check_interrupt(event: dict):
    """
    If the event contains an __interrupt__ chunk, return the interrupt payload.
    Otherwise return None.
    """
    if event["event"] == "on_chain_stream":
        chunk = event["data"].get("chunk", {})
        if "__interrupt__" in chunk:
            interrupt_obj = chunk["__interrupt__"][0]
            return {
                "type": "interrupt",
                "id": interrupt_obj.id,
                "value": interrupt_obj.value,   # { question, subject, body, ... }
            }
    return None


# ── Shared helper: stream LLM + tool events ──────────────────────────────────
async def _stream_events(event_iter):
    """
    Consume an astream_events iterator and yield SSE-formatted strings.
    Stops and yields an interrupt event if one is detected.
    """
    async for event in event_iter:
        kind = event["event"]
        metadata = event.get("metadata", {})
        node = metadata.get("langgraph_node", "")

        # ── Interrupt ────────────────────────────────────────────────────────
        interrupt_payload = _check_interrupt(event)
        if interrupt_payload:
            yield f"data: {json.dumps(interrupt_payload)}\n\n"
            return  # stop stream — frontend must resume separately

        # ── Tool call started ─────────────────────────────────────────────────
        if kind == "on_tool_start":
            tool_name = event["name"]
            tool_input = event.get("data", {}).get("input", {})
            yield f"data: {json.dumps({'type': 'tool_start', 'tool': tool_name, 'input': tool_input})}\n\n"

        # ── Tool call finished ────────────────────────────────────────────────
        elif kind == "on_tool_end":
            tool_name = event["name"]
            yield f"data: {json.dumps({'type': 'tool_end', 'tool': tool_name})}\n\n"

        # ── LLM token stream ──────────────────────────────────────────────────
        elif kind == "on_chat_model_stream" and node == "chat_node":
            chunk = event["data"].get("chunk")
            if not chunk or chunk.tool_call_chunks:
                continue
            if chunk.content:
                yield f"data: {json.dumps({'type': 'text', 'content': chunk.content})}\n\n"


# ── /chat/completions ─────────────────────────────────────────────────────────
async def event_stream(query: str, thread_id: str, request: Request):
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    try:
        event_iter = graph_module.workflow.astream_events(
            {"messages": [HumanMessage(content=query)]},
            config=config,
            version="v2",
        )
        async for chunk in _stream_events(event_iter):
            yield chunk

    except Exception as e:
        print("🔥 STREAM ERROR:", e)
        yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
    finally:
        yield f"data: {json.dumps({'type': 'done'})}\n\n"


@router.post("/chat/completions")
async def chat(body: RequestBody, request: Request):
    return StreamingResponse(
        event_stream(body.query, body.thread_id, request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── /chat/resume ──────────────────────────────────────────────────────────────
async def resume_stream(decision: str, thread_id: str):
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    try:
        event_iter = graph_module.workflow.astream_events(
            Command(resume=decision),
            config=config,
            version="v2",
        )
        async for chunk in _stream_events(event_iter):
            yield chunk

    except Exception as e:
        print("🔥 RESUME STREAM ERROR:", e)
        yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
    finally:
        yield f"data: {json.dumps({'type': 'done'})}\n\n"


@router.post("/chat/resume")
async def resume_chat(body: ResumeBody):
    """
    Resume a paused graph after a human-in-the-loop interrupt.
    Body:
        thread_id : same thread_id used in /chat/completions
        decision  : "yes" to confirm, anything else cancels
    """
    return StreamingResponse(
        resume_stream(body.decision, body.thread_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── /threads/{thread_id}/messages ────────────────────────────────────────────
@router.get("/threads/{thread_id}/messages")
async def get_thread_messages(thread_id: str):
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    try:
        state = await graph_module.workflow.aget_state(config)
        messages = state.values.get("messages", [])
        print("*"*100)
        print(messages_to_dict(messages))
        print("*"*100)
        return {
            "thread_id": thread_id,
            "messages": messages_to_dict(messages),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── /thread-title-generator ───────────────────────────────────────────────────
class ThreadTitleRequest(BaseModel):
    user_query: str
    ai_response: str
    thread_id: str


llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.75)
prompt = PromptTemplate(
    template="""You are a helpful assistant that generates short, descriptive titles for support chat conversations.

Based on the conversation below, generate a concise title (4-6 words max) that summarizes the topic.

User: {user_message}
Assistant: {assistant_message}

Rules:
- Return ONLY the title, nothing else
- No quotes, no punctuation at the end
- Be specific, not generic (avoid titles like "User Query" or "Support Chat")
- Language: English

Title:""",
    input_variables=["user_message", "assistant_message"]
)
parser = StrOutputParser()


@router.post("/thread-title-generator")
async def thread_title_generator(body: ThreadTitleRequest):
    try:
        chain = prompt | llm | parser
        title = await chain.ainvoke({
            "user_message": body.user_query,
            "assistant_message": body.ai_response,
        })

        if not title:
            raise HTTPException(status_code=502, detail="LLM returned empty title")

        return {
            "success": True,
            "thread_id": body.thread_id,
            "title": title.strip(),
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
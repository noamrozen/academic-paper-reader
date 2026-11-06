"""The agentic chat loop: model call -> tools -> model call ... persisted step by step."""

import asyncio
import time
from typing import AsyncIterator

from . import db, usage
from .config import AGENT_TURN_SECONDS, MAX_AGENT_STEPS
from .prompts import SEARCH_SYSTEM_PROMPT, SYSTEM_PROMPT, elements_text, paper_message_text, question_text
from .providers import StepResult, TextDelta, ThinkingDelta, provider_for, stream_step
from .tools import TOOLS, run_tool


class NothingToResume(Exception):
    pass


def steps_since_question(history: list[dict]) -> int | None:
    """Model calls made since the last question, if the turn stopped after tool results (paused,
    see run_turn) and can be resumed; None otherwise."""
    if not history or history[-1]["role"] != "tool":
        return None
    steps = 0
    for m in reversed(history):
        if m["role"] == "user":
            return steps
        steps += m["role"] == "assistant"
    return None


def build_user_message(question: str, selection: dict | None, first: bool, refs: dict | None = None) -> dict:
    content = [{"type": "paper"}] if first else []
    if first and refs:
        # Stored with the message, so the list stays fixed (and cached) for the whole conversation.
        content.append({"type": "text", "text": elements_text(refs)})
    content.append({"type": "text", "text": question_text(question, selection)})
    return {"role": "user", "content": content,
            "display": {"question": question, "selection": selection}}


async def run_turn(user: dict, conv: dict, question: str, selection: dict | None,
                   resume: bool = False) -> AsyncIterator[dict]:
    """One turn of the conversation: model calls and tool calls until the model answers. A turn
    that runs long pauses between steps (a "paused" event; everything so far is saved), and
    resume=True continues it in a new request, so that no request outlives the platform's limit.
    The step limit counts the whole turn."""
    provider = provider_for(user["keys"], conv["provider"])
    # A search session (no paper_id) has no paper in context, only the search prompt.
    paper = await db.aio.get_paper(user["id"], conv["paper_id"], with_text=True) if conv["paper_id"] else None
    paper_text = paper_message_text(paper) if paper else ""
    system = SYSTEM_PROMPT if paper else SEARCH_SYSTEM_PROMPT

    history = await db.aio.get_messages(conv["id"])
    if resume:
        steps_done = steps_since_question(history)
        if steps_done is None:
            raise NothingToResume("This conversation has no paused answer to continue")
        messages = history
    else:
        steps_done = 0
        refs = None if history or not paper else \
            ((await db.aio.get_analysis(user["id"], paper["id"], "refs")) or {}).get("entries")
        user_msg = build_user_message(question, selection, first=bool(paper) and not history, refs=refs)
        await db.aio.add_messages(conv["id"], [user_msg])
        messages = history + [user_msg]

    started = time.monotonic()
    thinking: list[str] = []
    text: list[str] = []  # the current step's answer, until it is persisted
    try:
        for step in range(steps_done, MAX_AGENT_STEPS):
            if step > steps_done and time.monotonic() - started > AGENT_TURN_SECONDS:
                yield {"type": "paused"}
                yield {"type": "done"}
                return
            result: StepResult | None = None
            thinking.clear()
            async for ev in stream_step(
                provider, model=conv["model"], system=system, paper_text=paper_text, paper_id=paper and paper["id"],
                messages=messages, tools=TOOLS,
            ):
                if isinstance(ev, TextDelta):
                    text.append(ev.text)
                    yield {"type": "text", "text": ev.text}
                elif isinstance(ev, ThinkingDelta):
                    thinking.append(ev.text)
                    yield {"type": "thinking", "text": ev.text}
                else:
                    result = ev

            await usage.record(user["id"], conv["provider"], conv["model"], result.usage, kind="chat",
                               paper_id=conv["paper_id"], conversation_id=conv["id"])
            assistant = result.message
            assistant["display"] = {"thinking": "".join(thinking).strip(), "usage": result.usage,
                                    "model": conv["model"], "provider": conv["provider"],
                                    "stop_reason": result.stop_reason}
            yield {"type": "usage", "usage": result.usage, "stop_reason": result.stop_reason}

            tool_uses = [b for b in assistant["content"] if b["type"] == "tool_use"]
            if not tool_uses:
                await db.aio.add_messages(conv["id"], [assistant])
                text.clear()
                yield {"type": "done"}
                return

            for tu in tool_uses:
                yield {"type": "tool_call", "id": tu["id"], "name": tu["name"], "input": tu["input"]}
            results = await asyncio.gather(*(run_tool(user, tu["name"], tu["input"]) for tu in tool_uses))
            tool_msg = {"role": "tool", "content": []}
            for tu, res in zip(tool_uses, results):
                tool_msg["content"].append({
                    "type": "tool_result", "tool_use_id": tu["id"], "name": tu["name"],
                    "content": res.content, "is_error": res.is_error, "papers": res.papers,
                })
                yield {"type": "tool_result", "id": tu["id"], "name": tu["name"], "is_error": res.is_error,
                       "preview": res.content[:600], "papers": res.papers}

            # Persist the assistant turn together with its tool results, so history is always valid.
            await db.aio.add_messages(conv["id"], [assistant, tool_msg])
            text.clear()
            messages += [assistant, tool_msg]
            yield {"type": "step"}
    except (asyncio.CancelledError, GeneratorExit):
        # Stopped by the user (the client went away mid-answer): keep what was already said, so the
        # conversation shows it and the model sees it next turn. Text only (an unanswered tool call
        # would make the history invalid), and without "raw", so providers replay the neutral block.
        if "".join(text).strip():
            db.add_messages(conv["id"], [{
                "role": "assistant", "content": [{"type": "text", "text": "".join(text)}],
                "display": {"thinking": "".join(thinking).strip(), "model": conv["model"],
                            "provider": conv["provider"], "stopped": True},
            }])
        raise

    yield {"type": "error", "message": f"Stopped after {MAX_AGENT_STEPS} tool-use steps."}
    yield {"type": "done"}

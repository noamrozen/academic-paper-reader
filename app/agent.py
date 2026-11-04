"""The agentic chat loop: model call -> tools -> model call ... persisted step by step."""

import asyncio
from typing import AsyncIterator

from . import db, usage
from .config import MAX_AGENT_STEPS
from .prompts import SEARCH_SYSTEM_PROMPT, SYSTEM_PROMPT, elements_text, paper_message_text, question_text
from .providers import StepResult, TextDelta, ThinkingDelta, provider_for
from .tools import TOOLS, run_tool


def build_user_message(question: str, selection: dict | None, first: bool, refs: dict | None = None) -> dict:
    content = [{"type": "paper"}] if first else []
    if first and refs:
        # Stored with the message, so the list stays fixed (and cached) for the whole conversation.
        content.append({"type": "text", "text": elements_text(refs)})
    content.append({"type": "text", "text": question_text(question, selection)})
    return {"role": "user", "content": content,
            "display": {"question": question, "selection": selection}}


async def run_turn(user: dict, conv: dict, question: str, selection: dict | None) -> AsyncIterator[dict]:
    provider = provider_for(user["keys"], conv["provider"])
    # A search session (no paper_id) has no paper in context, only the search prompt.
    paper = db.get_paper(user["id"], conv["paper_id"], with_text=True) if conv["paper_id"] else None
    paper_text = paper_message_text(paper) if paper else ""
    system = SYSTEM_PROMPT if paper else SEARCH_SYSTEM_PROMPT

    history = db.get_messages(conv["id"])
    refs = None if history or not paper else (db.get_analysis(user["id"], paper["id"], "refs") or {}).get("entries")
    user_msg = build_user_message(question, selection, first=bool(paper) and not history, refs=refs)
    db.add_messages(conv["id"], [user_msg])
    messages = history + [user_msg]

    thinking: list[str] = []
    text: list[str] = []  # the current step's answer, until it is persisted
    try:
        for step in range(MAX_AGENT_STEPS):
            result: StepResult | None = None
            thinking.clear()
            async for ev in provider.stream_step(
                model=conv["model"], system=system, paper_text=paper_text, paper_id=paper and paper["id"],
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
                db.add_messages(conv["id"], [assistant])
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
            db.add_messages(conv["id"], [assistant, tool_msg])
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

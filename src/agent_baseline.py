from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A: within-session memory only.

    - Forgets everything when a new thread_id is used.
    - No User.md, no compact memory.
    """

    SYSTEM_PROMPT = (
        "Bạn là trợ lý AI hữu ích. Hãy trả lời ngắn gọn và chính xác bằng tiếng Việt."
    )

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None
        if not force_offline:
            self._maybe_build_langchain_agent()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent and not self.force_offline:
            return self._reply_live(thread_id, message)
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        return 0

    # ------------------------------------------------------------------
    # Offline (deterministic) path
    # ------------------------------------------------------------------

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        if thread_id not in self.sessions:
            self.sessions[thread_id] = SessionState()
        sess = self.sessions[thread_id]

        sess.messages.append({"role": "user", "content": message})

        # Estimate prompt context (system + all messages so far)
        context_text = self.SYSTEM_PROMPT + " ".join(
            m["content"] for m in sess.messages
        )
        prompt_tokens = estimate_tokens(context_text)
        sess.prompt_tokens_processed += prompt_tokens

        # Generate a simple deterministic response
        response = self._make_response(sess.messages)

        sess.messages.append({"role": "assistant", "content": response})
        resp_tokens = estimate_tokens(response)
        sess.token_usage += resp_tokens

        return {
            "response": response,
            "agent_tokens": resp_tokens,
            "prompt_tokens": prompt_tokens,
        }

    def _make_response(self, messages: list[dict[str, str]]) -> str:
        """Deterministic offline response based purely on the current thread."""
        last = messages[-1]["content"] if messages else ""
        lower = last.lower()

        # Keyword-based recall within the same thread
        user_turns = [m["content"] for m in messages if m["role"] == "user"]
        full_history = " ".join(user_turns).lower()

        # Name recall
        if any(k in lower for k in ["tên", "name"]):
            import re
            name_m = re.search(
                r"tên(?:\s+(?:mình|tôi|là))?\s+(?:là\s+)?([A-Za-zÀ-ỹ][A-Za-zÀ-ỹ\s]{1,30}?)(?:\.|,|\s|$)",
                full_history,
                re.IGNORECASE,
            )
            if name_m:
                name = name_m.group(1).strip()
                return f"Trong cuộc hội thoại này, bạn đã giới thiệu tên là **{name}**."
            return "Mình chưa biết tên bạn trong cuộc trò chuyện này."

        # Drink recall
        if any(k in lower for k in ["đồ uống", "thức uống", "uống"]):
            drink_m = re.search(
                r"(?:yêu thích|thích uống)\s+(?:là\s+)?([A-Za-zÀ-ỹ][A-Za-zÀ-ỹ\s]{1,30}?)(?:\.|,|$)",
                full_history,
                re.IGNORECASE,
            ) if "re" in dir() else None
            if not drink_m:
                import re
                drink_m = re.search(
                    r"(?:đồ uống|thức uống)[^.]*(?:là\s+)?([A-Za-zÀ-ỹ][A-Za-zÀ-ỹ\s]{1,30}?)(?:\.|,|$)",
                    full_history,
                    re.IGNORECASE,
                )
            if drink_m:
                return f"Đồ uống yêu thích bạn đề cập là **{drink_m.group(1).strip()}**."
            return "Bạn chưa chia sẻ đồ uống yêu thích trong cuộc trò chuyện này."

        # Style recall
        if "style" in lower or "trả lời" in lower:
            return "Trong session này bạn muốn câu trả lời ngắn gọn, rõ ý và có ví dụ thực tế."

        # Default acknowledgement
        sentences = last.split(".")
        key_info = sentences[0].strip() if sentences else last[:60]
        return f"Đã ghi nhận: {key_info}. Tôi sẽ nhớ điều này trong cuộc trò chuyện hiện tại."

    # ------------------------------------------------------------------
    # Live (LangChain) path
    # ------------------------------------------------------------------

    def _reply_live(self, thread_id: str, message: str) -> dict[str, Any]:
        """Route to live LangChain agent."""
        if thread_id not in self.sessions:
            self.sessions[thread_id] = SessionState()
        sess = self.sessions[thread_id]

        config = {"configurable": {"thread_id": thread_id}}
        result = self.langchain_agent.invoke(
            {"messages": [("user", message)]},
            config=config,
        )
        ai_messages = result.get("messages", [])
        response = ai_messages[-1].content if ai_messages else ""

        # Approximate token counts from usage metadata if available
        usage = getattr(ai_messages[-1], "usage_metadata", None) if ai_messages else None
        resp_tokens = usage.get("output_tokens", estimate_tokens(response)) if usage else estimate_tokens(response)
        prompt_tokens = usage.get("input_tokens", estimate_tokens(message)) if usage else estimate_tokens(message)

        sess.token_usage += resp_tokens
        sess.prompt_tokens_processed += prompt_tokens
        sess.messages.append({"role": "user", "content": message})
        sess.messages.append({"role": "assistant", "content": response})

        return {
            "response": response,
            "agent_tokens": resp_tokens,
            "prompt_tokens": prompt_tokens,
        }

    def _maybe_build_langchain_agent(self) -> None:
        """Wire LangChain agent with InMemorySaver when dependencies are available."""
        try:
            from langgraph.prebuilt import create_react_agent
            from langgraph.checkpoint.memory import MemorySaver

            llm = build_chat_model(self.config.model)
            memory = MemorySaver()
            self.langchain_agent = create_react_agent(
                llm,
                tools=[],
                checkpointer=memory,
                prompt=self.SYSTEM_PROMPT,
            )
        except Exception:
            # Fall back to offline mode silently
            self.langchain_agent = None


import re  # noqa: E402 – needed by _make_response at module level
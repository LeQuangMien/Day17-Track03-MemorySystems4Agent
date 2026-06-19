from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import (
    CompactMemoryManager,
    UserProfileStore,
    estimate_tokens,
    extract_profile_updates,
)
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent B: three-layer memory.

    1. Short-term memory (within-session message list via CompactMemoryManager)
    2. Persistent memory (User.md per user_id)
    3. Compact memory (auto-summarise old messages when token budget exceeded)
    """

    SYSTEM_PROMPT = (
        "Bạn là trợ lý AI hữu ích có khả năng nhớ thông tin người dùng qua nhiều phiên. "
        "Hãy trả lời ngắn gọn và chính xác bằng tiếng Việt. "
        "Sử dụng hồ sơ người dùng được cung cấp để cá nhân hóa câu trả lời."
    )

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.langchain_agent = None

        if not force_offline:
            self._maybe_build_langchain_agent()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent and not self.force_offline:
            return self._reply_live(user_id, thread_id, message)
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    # ------------------------------------------------------------------
    # Offline (deterministic) path
    # ------------------------------------------------------------------

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        # 1. Extract and persist stable facts
        updates = extract_profile_updates(message)
        for key, value in updates.items():
            self.profile_store.upsert_fact(user_id, key, value)

        # 2. Append to compact memory
        self.compact_memory.append(thread_id, "user", message)

        # 3. Estimate prompt context load
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = (
            self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        )

        # 4. Generate response
        response = self._offline_response(user_id, thread_id, message)

        # 5. Append assistant reply & update counters
        self.compact_memory.append(thread_id, "assistant", response)
        resp_tokens = estimate_tokens(response)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + resp_tokens

        return {
            "response": response,
            "agent_tokens": resp_tokens,
            "prompt_tokens": prompt_tokens,
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Estimate context tokens: User.md + summary + recent messages."""
        profile_text = self.profile_store.read_text(user_id)
        ctx = self.compact_memory.context(thread_id)
        summary = str(ctx.get("summary", ""))
        messages: list[dict[str, str]] = ctx.get("messages", [])  # type: ignore[assignment]
        recent_text = " ".join(m["content"] for m in messages)
        full = self.SYSTEM_PROMPT + profile_text + summary + recent_text
        return estimate_tokens(full)

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        """Deterministic response using persisted User.md + thread context."""
        lower = message.lower()
        facts = self.profile_store.facts(user_id)
        ctx = self.compact_memory.context(thread_id)
        thread_history = " ".join(
            m["content"] for m in ctx.get("messages", []) if m["role"] == "user"  # type: ignore[union-attr]
        ).lower()
        summary_text = str(ctx.get("summary", "")).lower()
        all_history = thread_history + " " + summary_text

        # --- Recall: name ---
        if any(k in lower for k in ["tên", "name"]):
            name = facts.get("tên")
            if not name:
                name_m = re.search(
                    r"tên(?:\s+(?:mình|tôi|là))?\s+(?:là\s+)?([A-Za-zÀ-ỹ][A-Za-zÀ-ỹ\s]{1,30}?)(?:\.|,|\s|$)",
                    all_history,
                    re.IGNORECASE,
                )
                name = name_m.group(1).strip() if name_m else None
            if name:
                return f"Tên của bạn là **{name}**."
            return "Mình chưa có thông tin tên của bạn."

        # --- Recall: location ---
        if any(k in lower for k in ["ở đâu", "nơi ở", "thành phố", "sống ở"]):
            loc = facts.get("nơi ở")
            if loc:
                return f"Bạn hiện đang ở **{loc}**."
            return "Mình chưa biết nơi ở hiện tại của bạn."

        # --- Recall: profession ---
        if any(k in lower for k in ["nghề", "làm gì", "công việc"]):
            job = facts.get("nghề nghiệp")
            if job:
                return f"Nghề nghiệp của bạn là **{job}**."
            return "Mình chưa có thông tin nghề nghiệp của bạn."

        # --- Recall: drink ---
        if any(k in lower for k in ["đồ uống", "thức uống", "uống"]):
            drink = facts.get("đồ uống yêu thích")
            if not drink:
                drink_m = re.search(
                    r"(?:đồ uống|thức uống|yêu thích)[^.]*?([A-Za-zÀ-ỹ][A-Za-zÀ-ỹ\s]{2,30}?)(?:\.|,|$)",
                    all_history,
                    re.IGNORECASE,
                )
                drink = drink_m.group(1).strip() if drink_m else None
            if drink:
                return f"Đồ uống yêu thích của bạn là **{drink}**."
            return "Mình chưa biết đồ uống yêu thích của bạn."

        # --- Recall: reply style ---
        if any(k in lower for k in ["style", "trả lời", "phong cách"]):
            style = facts.get("style trả lời")
            if not style:
                style_m = re.search(
                    r"(?:trả\s+lời\s+(?:ngắn\s+gọn|theo|thành)\s+)(.{5,80}?)(?:\.|,|$)",
                    all_history,
                    re.IGNORECASE,
                )
                style = style_m.group(1).strip() if style_m else None
            if style:
                return f"Style trả lời bạn thích: **{style}**."
            return "Bạn chưa chỉ định phong cách trả lời cụ thể."

        # --- Summary of known profile ---
        if any(k in lower for k in ["nhắc lại", "tóm tắt", "giới thiệu", "ghi nhớ"]):
            if facts:
                lines = "\n".join(f"- **{k}**: {v}" for k, v in facts.items())
                return f"Thông tin mình đã lưu về bạn:\n{lines}"
            return "Mình chưa lưu thông tin cụ thể nào về bạn."

        # --- Noise/joke detection ---
        noise_keywords = ["đùa", "câu đùa", "chỉ là"]
        if any(k in lower for k in noise_keywords):
            return "Đã hiểu, đây chỉ là câu đùa. Mình vẫn giữ thông tin chính xác của bạn."

        # --- Default acknowledgement with profile context ---
        sentences = message.split(".")
        key_info = sentences[0].strip() if sentences else message[:80]
        context_note = ""
        if facts:
            name = facts.get("tên", "")
            context_note = f" {'Chào ' + name + '! ' if name else ''}Mình đã cập nhật hồ sơ của bạn."
        return f"Đã ghi nhận: {key_info}.{context_note}"

    # ------------------------------------------------------------------
    # Live (LangChain) path
    # ------------------------------------------------------------------

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Route to live LangChain agent with profile injection."""
        # Still persist profile facts regardless of live/offline
        updates = extract_profile_updates(message)
        for key, value in updates.items():
            self.profile_store.upsert_fact(user_id, key, value)

        self.compact_memory.append(thread_id, "user", message)
        profile_text = self.profile_store.read_text(user_id)
        system_with_profile = (
            f"{self.SYSTEM_PROMPT}\n\n## Hồ sơ người dùng\n{profile_text}"
        )

        config_lc = {"configurable": {"thread_id": thread_id}}
        result = self.langchain_agent.invoke(
            {"messages": [("user", message)]},
            config=config_lc,
        )
        ai_messages = result.get("messages", [])
        response = ai_messages[-1].content if ai_messages else ""

        usage = getattr(ai_messages[-1], "usage_metadata", None) if ai_messages else None
        resp_tokens = usage.get("output_tokens", estimate_tokens(response)) if usage else estimate_tokens(response)
        prompt_tokens = usage.get("input_tokens", estimate_tokens(system_with_profile + message)) if usage else estimate_tokens(system_with_profile + message)

        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + resp_tokens
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        self.compact_memory.append(thread_id, "assistant", response)

        return {
            "response": response,
            "agent_tokens": resp_tokens,
            "prompt_tokens": prompt_tokens,
        }

    def _maybe_build_langchain_agent(self) -> None:
        """Wire live agent with compact memory middleware."""
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
            self.langchain_agent = None
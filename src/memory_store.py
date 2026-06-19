from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Simple heuristic token estimator (~4 chars per token)."""
    text = text.strip()
    if not text:
        return 0
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# UserProfileStore  (persistent User.md)
# ---------------------------------------------------------------------------

_DEFAULT_PROFILE = """\
# Hồ sơ người dùng

_Chưa có thông tin._
"""


@dataclass
class UserProfileStore:
    """Persistent storage for per-user User.md files."""

    root_dir: Path

    def __post_init__(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, user_id: str) -> Path:
        slug = re.sub(r"[^\w\-]", "_", user_id.strip().lower())
        return self.root_dir / f"{slug}.md"

    def read_text(self, user_id: str) -> str:
        p = self.path_for(user_id)
        if not p.exists():
            return _DEFAULT_PROFILE
        return p.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        p = self.path_for(user_id)
        p.write_text(content, encoding="utf-8")
        return p

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        current = self.read_text(user_id)
        if search_text not in current:
            return False
        updated = current.replace(search_text, replacement, 1)
        self.write_text(user_id, updated)
        return True

    def file_size(self, user_id: str) -> int:
        p = self.path_for(user_id)
        return p.stat().st_size if p.exists() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        """Parse simple `- key: value` lines from User.md."""
        result: dict[str, str] = {}
        for line in self.read_text(user_id).splitlines():
            m = re.match(r"^-\s+([^:]+):\s+(.+)$", line.strip())
            if m:
                result[m.group(1).strip()] = m.group(2).strip()
        return result

    def upsert_fact(self, user_id: str, key: str, value: str) -> None:
        """Insert or update a `- key: value` line in User.md."""
        current = self.read_text(user_id)
        pattern = re.compile(rf"^(- {re.escape(key)}:).+$", re.MULTILINE)
        new_line = f"- {key}: {value}"
        if pattern.search(current):
            updated = pattern.sub(new_line, current)
        else:
            if "_Chưa có thông tin._" in current:
                updated = f"# Hồ sơ người dùng\n\n{new_line}\n"
            else:
                updated = current.rstrip("\n") + f"\n{new_line}\n"
        self.write_text(user_id, updated)


# ---------------------------------------------------------------------------
# Profile extraction  (fixed)
# ---------------------------------------------------------------------------

# Known city/province names in Vietnam for location validation
_VN_LOCATIONS = {
    "hà nội", "tp hcm", "hồ chí minh", "đà nẵng", "hải phòng", "cần thơ",
    "huế", "nha trang", "vũng tàu", "đà lạt", "buôn ma thuột", "vinh",
    "quy nhơn", "pleiku", "phan thiết", "long xuyên", "mỹ tho", "thái nguyên",
    "nam định", "thanh hóa",
}

# Known profession keywords — require these to appear in the matched value
_PROFESSION_KEYWORDS = {
    "engineer", "developer", "designer", "manager", "director", "analyst",
    "scientist", "researcher", "teacher", "doctor", "nurse", "kỹ sư",
    "lập trình", "phát triển", "thiết kế", "quản lý", "giám đốc", "giáo viên",
    "bác sĩ", "nghiên cứu", "backend", "frontend", "fullstack", "devops",
    "mlops", "data", "product", "pm ",
}

# Noise words that should NEVER be a fact value
_NOISE_VALUES = {
    "gì", "không", "nào", "đâu", "sao", "thế", "vậy", "nhé", "ạ",
    "mình", "bạn", "tôi", "lại", "được", "và", "hoặc", "hay",
    "thật", "hiện tại", "này", "đó", "kia", "đây",
}

# Sentences that indicate a question rather than a fact statement — skip extraction
_QUESTION_INDICATORS = re.compile(
    r"(?:bạn\s+(?:có\s+thể|hãy|thử)|nhắc\s+lại|cho\s+mình\s+biết|"
    r"câu\s+hỏi|hỏi\s+tiếp|gì\s*\?|đâu\s*\?|như\s+thế\s+nào)",
    re.IGNORECASE,
)

# Sentences that indicate noise / jokes — skip extraction
_NOISE_INDICATORS = re.compile(
    r"(?:chỉ\s+là\s+câu\s+đùa|đùa\s+thôi|nói\s+đùa|đùa\s+với|"
    r"giả\s+sử|ví\s+dụ\s+như|không\s+phải\s+thật|tin\s+tức|"
    r"xác\s+suất|phần\s+trăm|năm\s+20[0-9]{2}|"
    r"nasa|wmo|el\s+nino|artemis|british\s+columbia)",
    re.IGNORECASE,
)


def _is_valid_value(value: str, min_len: int = 2) -> bool:
    """Return False if the extracted value looks like noise."""
    v = value.strip().lower()
    if len(v) < min_len:
        return False
    if v in _NOISE_VALUES:
        return False
    # Reject values that are mostly question words
    if re.search(r"\?$", v):
        return False
    return True


def extract_profile_updates(message: str) -> dict[str, str]:
    """Convert raw user text into stable profile facts.

    Rules:
    - Skip obvious question turns or noise/news context.
    - Each pattern is specific enough to avoid false positives.
    - Values are validated before returning.
    """
    stripped = message.strip()

    # Skip turns with question indicators or noise context
    if _QUESTION_INDICATORS.search(stripped):
        return {}
    if _NOISE_INDICATORS.search(stripped):
        return {}

    # Skip if the message is primarily a question (ends with ?)
    sentences = [s.strip() for s in stripped.split(".") if s.strip()]
    question_ratio = sum(1 for s in sentences if "?" in s) / max(len(sentences), 1)
    if question_ratio > 0.5:
        return {}

    facts: dict[str, str] = {}

    # ── Tên ────────────────────────────────────────────────────────────
    # Matches: "mình tên là X", "tên mình là X"
    # Negative lookbehind: skip "con X tên Y", "bé X tên Y" (pet/object names)
    _OBJECT_NOUNS = r"(?:con|bé|chú|cô|bạn|cái|chiếc|app|tool|bot|model|file|chó|mèo|thú)"
    _obj_before = re.search(
        rf"{_OBJECT_NOUNS}\s+\w+\s+tên\s+",
        stripped, re.IGNORECASE,
    )
    if not _obj_before:
        name_m = re.search(
            r"(?:mình\s+tên\s+(?:là\s+)?|tên\s+(?:mình\s+)?(?:là\s+)?)([A-Z][A-Za-zÀ-ỹ0-9]{1,20}(?:\s+[A-Z][A-Za-zÀ-ỹ]{1,15})?)\b",
            stripped,
        )
        if name_m:
            val = name_m.group(1).strip()
            if _is_valid_value(val, min_len=2):
                facts["tên"] = val

    # ── Nơi ở ──────────────────────────────────────────────────────────
    # Matches: "mình ở X", "mình đang ở X", "mình sống ở X"
    # Must match a known city OR be a capitalized proper noun followed by sentence end
    location_m = re.search(
        r"(?:mình\s+(?:đang\s+)?(?:ở|sống\s+ở|làm\s+việc\s+ở)\s+|"
        r"hiện\s+(?:đang\s+)?(?:ở|tại)\s+|"
        r"đang\s+làm\s+việc\s+ở\s+)"
        r"([A-ZÀ-Ỹ][A-Za-zÀ-ỹ\s]{1,20}?)(?:\s+(?:và|để|cho|trong|vài|từ|một|nhưng)|\.|,|$)",
        stripped,
        re.IGNORECASE,
    )
    if location_m:
        val = location_m.group(1).strip()
        # Only accept if it's a known city or looks like a proper noun (capitalized)
        if val.lower() in _VN_LOCATIONS or (val and val[0].isupper()):
            if _is_valid_value(val, min_len=3) and len(val.split()) <= 4:
                facts["nơi ở"] = val

    # ── Nghề nghiệp ────────────────────────────────────────────────────
    # Matches: "đang làm X cho ...", "mình làm X", "hiện đang làm X"
    # Uses lookahead so terminator is not consumed; requires profession keyword
    profession_m = re.search(
        r"(?:(?:mình\s+)?đang\s+làm\s+|mình\s+làm\s+(?:nghề\s+)?|hiện\s+(?:đang\s+)?làm\s+)"
        r"([A-Za-zÀ-ỹ][A-Za-zÀ-ỹ\s]{2,35}?)"
        r"(?=\s+(?:cho|tại|ở|trong|với)\b|[.,]|$)",
        stripped,
        re.IGNORECASE,
    )
    if profession_m:
        val = profession_m.group(1).strip()
        val_lower = val.lower()
        # Must contain a known profession keyword
        if any(kw in val_lower for kw in _PROFESSION_KEYWORDS):
            if _is_valid_value(val, min_len=4) and len(val.split()) <= 6:
                facts["nghề nghiệp"] = val

    # ── Đồ uống yêu thích ──────────────────────────────────────────────
    # Matches ONLY explicit statements: "đồ uống yêu thích là X" or "thích uống X"
    # Does NOT match questions like "đồ uống yêu thích của mình là gì"
    drink_m = re.search(
        r"(?:đồ\s+uống\s+yêu\s+thích\s+(?:(?:của\s+mình\s+)?là\s+)"
        r"(?!gì|không|nào|đâu))"
        r"([A-Za-zÀ-ỹ][A-Za-zÀ-ỹ\s]{1,30}?)(?:\.|,|$)",
        stripped,
        re.IGNORECASE,
    )
    if not drink_m:
        # "mình thích uống X" — explicit preference
        drink_m = re.search(
            r"mình\s+(?:vẫn\s+)?(?:thích\s+uống|hay\s+uống)\s+"
            r"([A-Za-zÀ-ỹ][A-Za-zÀ-ỹ\s]{1,30}?)"
            r"(?:\s+(?:như|lắm|lại)|\.|,|$)",
            stripped,
            re.IGNORECASE,
        )
    if drink_m:
        val = drink_m.group(1).strip()
        if _is_valid_value(val, min_len=3) and "gì" not in val and len(val.split()) <= 5:
            facts["đồ uống yêu thích"] = val

    # ── Style trả lời ──────────────────────────────────────────────────
    # Matches: "muốn bạn trả lời ngắn gọn", "trả lời thành 3 bullet"
    style_m = re.search(
        r"(?:muốn\s+bạn\s+trả\s+lời\s+|"
        r"trả\s+lời\s+(?:thành|theo|dạng)\s+)"
        r"(.{5,80}?)(?:\.|,\s+(?:rõ|có|và)|$)",
        stripped,
        re.IGNORECASE,
    )
    if style_m:
        val = style_m.group(1).strip().rstrip(".,")
        if _is_valid_value(val, min_len=5) and len(val.split()) <= 15:
            facts["style trả lời"] = val

    return facts


# ---------------------------------------------------------------------------
# Compact memory
# ---------------------------------------------------------------------------

def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Create a compact textual summary of older messages."""
    if not messages:
        return ""
    sample = messages[-max_items:] if len(messages) > max_items else messages
    parts: list[str] = []
    for msg in sample:
        role = "Người dùng" if msg.get("role") == "user" else "Trợ lý"
        content = msg.get("content", "").strip()
        if content:
            if len(content) > 200:
                content = content[:200] + "…"
            parts.append(f"[{role}] {content}")
    return "Tóm tắt hội thoại cũ:\n" + "\n".join(parts)


@dataclass
class CompactMemoryManager:
    """Compact memory for long threads."""

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def _init_thread(self, thread_id: str) -> None:
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0,
            }

    def append(self, thread_id: str, role: str, content: str) -> None:
        self._init_thread(thread_id)
        s = self.state[thread_id]
        s["messages"].append({"role": role, "content": content})  # type: ignore[index]
        self._maybe_compact(thread_id)

    def _maybe_compact(self, thread_id: str) -> None:
        s = self.state[thread_id]
        messages: list[dict[str, str]] = s["messages"]  # type: ignore[assignment]
        total_text = " ".join(m["content"] for m in messages) + str(s.get("summary", ""))
        if estimate_tokens(total_text) > self.threshold_tokens:
            keep = self.keep_messages
            old = messages[:-keep] if len(messages) > keep else []
            recent = messages[-keep:] if len(messages) > keep else messages[:]
            if old:
                prev_summary = str(s.get("summary", ""))
                s["summary"] = summarize_messages(
                    ([{"role": "summary", "content": prev_summary}] if prev_summary else []) + old
                )
                s["messages"] = recent
                s["compactions"] = int(s.get("compactions", 0)) + 1  # type: ignore[assignment]

    def context(self, thread_id: str) -> dict[str, object]:
        self._init_thread(thread_id)
        return self.state[thread_id]

    def compaction_count(self, thread_id: str) -> int:
        self._init_thread(thread_id)
        return int(self.state[thread_id].get("compactions", 0))
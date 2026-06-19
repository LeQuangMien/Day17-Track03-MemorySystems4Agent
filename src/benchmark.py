from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read JSON conversations from disk."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def recall_points(answer: str, expected: list[str]) -> float:
    """Return 0 / 0.5 / 1 depending on how many expected facts appear in answer."""
    if not expected:
        return 1.0
    lower = answer.lower()
    hits = sum(1 for e in expected if e.lower() in lower)
    ratio = hits / len(expected)
    if ratio >= 1.0:
        return 1.0
    if ratio >= 0.5:
        return 0.5
    return 0.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Lightweight quality score: penalise very short/empty answers."""
    if not answer or answer.strip() == "":
        return 0.0
    base = recall_points(answer, expected)
    length_bonus = min(0.2, len(answer) / 500)
    return min(1.0, base + length_bonus)


def run_agent_benchmark(
    agent_name: str,
    agent,
    conversations: list[dict[str, Any]],
    config,
    prefix: str = "",
) -> BenchmarkRow:
    """Evaluate one agent over a list of conversations."""
    total_agent_tokens = 0
    total_prompt_tokens = 0
    total_recall = 0.0
    total_quality = 0.0
    recall_count = 0
    total_compactions = 0
    memory_growth = 0

    for conv in conversations:
        user_id: str = conv.get("user_id", "default")
        conv_id: str = conv.get("id", "conv")
        thread_id = f"{prefix}{conv_id}-main"

        # Feed all turns
        for turn in conv.get("turns", []):
            result = agent.reply(user_id, thread_id, turn)
            total_agent_tokens += result.get("agent_tokens", 0)
            total_prompt_tokens += result.get("prompt_tokens", 0)

        total_compactions += agent.compaction_count(thread_id)

        # Memory growth (only advanced has this)
        if hasattr(agent, "memory_file_size"):
            memory_growth += agent.memory_file_size(user_id)

        # Recall questions in a FRESH thread
        recall_thread = f"{prefix}{conv_id}-recall"
        for rq in conv.get("recall_questions", []):
            question: str = rq.get("question", "")
            expected: list[str] = rq.get("expected_contains", [])

            result = agent.reply(user_id, recall_thread, question)
            answer = result.get("response", "")
            total_agent_tokens += result.get("agent_tokens", 0)
            total_prompt_tokens += result.get("prompt_tokens", 0)

            r = recall_points(answer, expected)
            q = heuristic_quality(answer, expected)
            total_recall += r
            total_quality += q
            recall_count += 1

    avg_recall = total_recall / recall_count if recall_count else 0.0
    avg_quality = total_quality / recall_count if recall_count else 0.0

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=total_agent_tokens,
        prompt_tokens_processed=total_prompt_tokens,
        recall_score=avg_recall,
        response_quality=avg_quality,
        memory_growth_bytes=memory_growth,
        compactions=total_compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Print a formatted markdown table."""
    try:
        from tabulate import tabulate

        headers = [
            "Agent",
            "Agent tokens only",
            "Prompt tokens processed",
            "Cross-session recall",
            "Response quality",
            "Memory growth (bytes)",
            "Compactions",
        ]
        table = [
            [
                r.agent_name,
                r.agent_tokens_only,
                r.prompt_tokens_processed,
                f"{r.recall_score:.2f}",
                f"{r.response_quality:.2f}",
                r.memory_growth_bytes,
                r.compactions,
            ]
            for r in rows
        ]
        return tabulate(table, headers=headers, tablefmt="github")
    except ImportError:
        # Fallback plain text
        lines = []
        for r in rows:
            lines.append(
                f"{r.agent_name}: recall={r.recall_score:.2f}, quality={r.response_quality:.2f}, "
                f"agent_tokens={r.agent_tokens_only}, prompt_tokens={r.prompt_tokens_processed}, "
                f"memory={r.memory_growth_bytes}B, compactions={r.compactions}"
            )
        return "\n".join(lines)


def main() -> None:
    """Run standard + long-context stress benchmarks."""
    config = load_config(Path(__file__).resolve().parent.parent)

    data_dir = config.data_dir
    conv_path = data_dir / "conversations.json"
    stress_path = data_dir / "advanced_long_context.json"

    print("=" * 70)
    print("DAY 17 – Memory Systems for AI Agent – Benchmark")
    print("=" * 70)

    # ── Standard Benchmark ─────────────────────────────────────────────
    print("\n## Standard Benchmark  (data/conversations.json)\n")
    std_convs = load_conversations(conv_path)

    baseline_std = BaselineAgent(config=config, force_offline=True)
    advanced_std = AdvancedAgent(config=config, force_offline=True)

    rows_std = [
        run_agent_benchmark("Baseline", baseline_std, std_convs, config, prefix="std-"),
        run_agent_benchmark("Advanced", advanced_std, std_convs, config, prefix="std-"),
    ]
    print(format_rows(rows_std))

    # ── Long-Context Stress Benchmark ──────────────────────────────────
    print("\n## Long-Context Stress Benchmark  (data/advanced_long_context.json)\n")
    stress_convs = load_conversations(stress_path)

    baseline_lc = BaselineAgent(config=config, force_offline=True)
    advanced_lc = AdvancedAgent(config=config, force_offline=True)

    rows_lc = [
        run_agent_benchmark("Baseline", baseline_lc, stress_convs, config, prefix="lc-"),
        run_agent_benchmark("Advanced", advanced_lc, stress_convs, config, prefix="lc-"),
    ]
    print(format_rows(rows_lc))

    # ── Analysis ───────────────────────────────────────────────────────
    print("""
## Phân tích kết quả

### Tại sao Advanced có recall tốt hơn Baseline?
- Advanced lưu facts ổn định (tên, nơi ở, nghề nghiệp, v.v.) vào User.md bền vững.
- Khi sang thread mới (recall questions), Advanced vẫn đọc được User.md; Baseline thì không có gì.

### Tại sao Advanced có thể tốn nhiều token hơn ở hội thoại ngắn?
- Mỗi lượt Advanced phải kéo theo User.md + summary vào prompt context.
- Ở hội thoại ngắn, overhead này lớn hơn lợi ích compact mang lại.

### Tại sao compact memory giúp Advanced ở hội thoại dài?
- Compact nén các messages cũ thành summary, tránh ngữ cảnh phình to.
- Baseline phải kéo toàn bộ lịch sử; prompt_tokens_processed tăng rất nhanh.

### Rủi ro khi memory file phình to:
- User.md có thể chứa facts sai nếu agent nhận nhầm câu đùa hoặc nhiễu.
- Cần confidence threshold và conflict handling để cập nhật đúng.
""")


if __name__ == "__main__":
    main()
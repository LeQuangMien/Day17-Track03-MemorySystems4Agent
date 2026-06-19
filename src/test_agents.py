from __future__ import annotations

import tempfile
from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig
from memory_store import CompactMemoryManager, UserProfileStore
from model_provider import ProviderConfig


def make_config(tmp_path: Path) -> LabConfig:
    """Build an isolated config for tests with a low compact threshold."""
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    dummy_provider = ProviderConfig(
        provider="anthropic",
        model_name="claude-haiku-4-5-20251001",
        temperature=0.0,
    )
    return LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        state_dir=state_dir,
        compact_threshold_tokens=50,   # Very low so tests trigger quickly
        compact_keep_messages=2,
        model=dummy_provider,
        judge_model=dummy_provider,
    )


# ---------------------------------------------------------------------------
# Test 1 – User.md read / write / edit
# ---------------------------------------------------------------------------

def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    user_id = "test_user"

    # Initial read returns default content
    content = store.read_text(user_id)
    assert "Hồ sơ" in content or "Chưa" in content, "Default profile should exist"

    # Write custom content
    store.write_text(user_id, "# Hồ sơ\n- tên: Alice\n")
    assert "Alice" in store.read_text(user_id)

    # Edit: replace fact
    changed = store.edit_text(user_id, "Alice", "Bob")
    assert changed, "edit_text should return True when replacement succeeds"
    assert "Bob" in store.read_text(user_id)
    assert "Alice" not in store.read_text(user_id)

    # upsert_fact
    store.upsert_fact(user_id, "nghề nghiệp", "kỹ sư")
    assert "kỹ sư" in store.read_text(user_id)

    # facts() should parse the key-value lines
    facts = store.facts(user_id)
    assert "tên" in facts
    assert facts["tên"] == "Bob"
    assert facts.get("nghề nghiệp") == "kỹ sư"

    # file_size
    assert store.file_size(user_id) > 0

    print("✅  test_user_markdown_read_write_edit passed")


# ---------------------------------------------------------------------------
# Test 2 – Compact trigger
# ---------------------------------------------------------------------------

def test_compact_trigger(tmp_path: Path) -> None:
    mgr = CompactMemoryManager(threshold_tokens=50, keep_messages=2)
    thread_id = "t1"

    # Add enough messages to exceed the threshold
    long_messages = [
        ("user", "Đây là đoạn văn bản rất dài để kiểm tra compact memory trigger " * 3),
        ("assistant", "Đã nhận thông tin, cảm ơn bạn đã chia sẻ " * 3),
        ("user", "Tiếp theo mình muốn nói về chủ đề memory trong agent AI " * 3),
        ("assistant", "Memory là thành phần quan trọng của agent thông minh " * 3),
        ("user", "Bạn hãy nhớ thông tin này để dùng ở các lượt sau nhé " * 3),
    ]

    for role, content in long_messages:
        mgr.append(thread_id, role, content)

    count = mgr.compaction_count(thread_id)
    assert count >= 1, f"Expected at least 1 compaction, got {count}"

    ctx = mgr.context(thread_id)
    assert ctx["summary"], "Summary should be non-empty after compaction"
    assert len(ctx["messages"]) <= 2, "Only keep_messages should remain"

    print(f"✅  test_compact_trigger passed (compactions={count})")


# ---------------------------------------------------------------------------
# Test 3 – Cross-session recall
# ---------------------------------------------------------------------------

def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    baseline = BaselineAgent(config=config, force_offline=True)
    advanced = AdvancedAgent(config=config, force_offline=True)

    user_id = "recall_user"
    session1 = "session-1"

    # Session 1: share facts
    intro_messages = [
        "Chào bạn, mình tên là AnhTuấn.",
        "Mình đang sống ở Hà Nội.",
        "Nghề nghiệp của mình là data scientist.",
        "Đồ uống yêu thích là trà đào.",
    ]
    for msg in intro_messages:
        baseline.reply(user_id, session1, msg)
        advanced.reply(user_id, session1, msg)

    # Session 2 (fresh thread): ask recall questions
    session2 = "session-2"

    result_b = baseline.reply(user_id, session2, "Mình tên gì?")
    result_a = advanced.reply(user_id, session2, "Mình tên gì?")

    baseline_answer = result_b["response"].lower()
    advanced_answer = result_a["response"].lower()

    # Advanced should recall the name; baseline should not
    assert "anhtuan" in advanced_answer.replace(" ", "") or "anh tuấn" in advanced_answer or "anhtấn" in advanced_answer or "tuấn" in advanced_answer, \
        f"Advanced should recall the name, got: {advanced_answer}"

    assert "anhtuan" not in baseline_answer.replace(" ", "") and "tuấn" not in baseline_answer, \
        f"Baseline should NOT recall the name cross-session, got: {baseline_answer}"

    print(f"✅  test_cross_session_recall passed")
    print(f"   Baseline: {result_b['response'][:80]}")
    print(f"   Advanced: {result_a['response'][:80]}")


# ---------------------------------------------------------------------------
# Test 4 – Compact reduces prompt load on long thread
# ---------------------------------------------------------------------------

def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    baseline = BaselineAgent(config=config, force_offline=True)
    advanced = AdvancedAgent(config=config, force_offline=True)

    user_id = "load_user"
    thread_id = "long-thread"

    long_turn = "Đây là một đoạn hội thoại rất dài để kiểm tra mức tăng token theo thời gian. " * 5
    for i in range(10):
        msg = f"[Turn {i}] {long_turn}"
        baseline.reply(user_id, thread_id, msg)
        advanced.reply(user_id, thread_id, msg)

    b_prompt = baseline.prompt_token_usage(thread_id)
    a_prompt = advanced.prompt_token_usage(thread_id)
    a_compactions = advanced.compaction_count(thread_id)

    print(f"   Baseline prompt tokens  : {b_prompt}")
    print(f"   Advanced prompt tokens  : {a_prompt}")
    print(f"   Advanced compactions    : {a_compactions}")

    # Baseline accumulates full history → should be higher than advanced after compaction
    assert a_compactions >= 1, "Advanced should have compacted at least once"
    assert b_prompt >= a_prompt, (
        f"Baseline ({b_prompt}) should process >= prompt tokens than Advanced ({a_prompt}) "
        "after compaction reduces context"
    )

    print("✅  test_compact_reduces_prompt_load_on_long_thread passed")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        tests = [
            test_user_markdown_read_write_edit,
            test_compact_trigger,
            test_cross_session_recall,
            test_compact_reduces_prompt_load_on_long_thread,
        ]
        failed = 0
        for test_fn in tests:
            print(f"\n▶  {test_fn.__name__}")
            try:
                test_fn(tmp)
            except AssertionError as e:
                print(f"❌  FAILED: {e}")
                failed += 1
            except Exception as e:
                print(f"❌  ERROR: {e}")
                failed += 1

        print("\n" + "=" * 50)
        passed = len(tests) - failed
        print(f"Results: {passed}/{len(tests)} tests passed")
        sys.exit(0 if failed == 0 else 1)
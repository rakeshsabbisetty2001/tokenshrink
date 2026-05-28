import pytest
from tokenshrink.counter import TokenCounter
from tokenshrink.compressors.conversation import compress_messages
from tests.fixtures.long_conversations import LONG_CONVERSATION


counter = TokenCounter()


def test_keeps_recent_turns_verbatim():
    keep = 4
    result = compress_messages(LONG_CONVERSATION, keep, counter, {})
    recent_original = LONG_CONVERSATION[-keep:]
    recent_result = result[-keep:]
    for orig, res in zip(recent_original, recent_result):
        assert orig["content"] == res["content"]


def test_compresses_old_turns():
    result = compress_messages(LONG_CONVERSATION, keep_recent=4, counter=counter, opts={})
    # Token count of all old turns should be <= original
    original_old_tokens = sum(
        counter.count(m["content"]) for m in LONG_CONVERSATION[:-4]
        if isinstance(m.get("content"), str)
    )
    result_old_tokens = sum(
        counter.count(m["content"]) for m in result[:-4]
        if isinstance(m.get("content"), str)
    )
    assert result_old_tokens <= original_old_tokens


def test_short_conversation_unchanged():
    short = LONG_CONVERSATION[-3:]
    result = compress_messages(short, keep_recent=5, counter=counter, opts={})
    assert len(result) == len(short)


def test_roles_preserved():
    result = compress_messages(LONG_CONVERSATION, keep_recent=4, counter=counter, opts={})
    for msg in result:
        assert msg["role"] in {"user", "assistant", "system"}

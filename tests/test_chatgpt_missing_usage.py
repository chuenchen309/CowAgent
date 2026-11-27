# encoding:utf-8
"""A 200 without `usage` must not throw away the answer it carries.

`open_ai_api_base` often points at a relay or an OpenAI-compatible gateway
that omits `usage`, the same case #4091 fixed for DeepSeek. Indexing it raised
KeyError, which cleared the session and replaced the answer with the "tired"
sentinel.
"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config as config_module
from config import Config


class _Client:
    def __init__(self, chat=None, responses=None):
        self._chat, self._responses = chat, responses

    def chat_completions(self, **kwargs):
        return self._chat

    def responses(self, **kwargs):
        return self._responses


def _bot(monkeypatch, client, api_type="chat"):
    monkeypatch.setattr(config_module, "config", Config({
        "bot_type": "chatGPT",
        "model": "gpt-5.5",
        "open_ai_api_key": "sk-test",
        "open_ai_api_base": "https://relay.example.com/v1",
        "open_ai_api_type": api_type,
    }))
    from models.chatgpt import chat_gpt_bot
    monkeypatch.setattr(chat_gpt_bot.time, "sleep", lambda s: None)
    bot = chat_gpt_bot.ChatGPTBot("chatGPT")
    bot._http_client = client
    return bot


def _session():
    return SimpleNamespace(session_id="s1", messages=[{"role": "user", "content": "hi"}])


def _chat(usage=None):
    payload = {"choices": [{"message": {"role": "assistant", "content": "real answer"},
                            "finish_reason": "stop"}]}
    if usage is not None:
        payload["usage"] = usage
    return payload


def test_chat_answer_survives_a_response_without_usage(monkeypatch):
    client = _Client(chat=_chat())

    reply = _bot(monkeypatch, client).reply_text(_session())

    assert reply["content"] == "real answer"
    assert reply["completion_tokens"] > 0, "reply() would report this as an error"


def test_responses_answer_survives_a_response_without_usage(monkeypatch):
    client = _Client(responses={
        "id": "resp_1", "model": "gpt-5.5", "status": "completed",
        "output": [{"type": "message", "content": [{"type": "output_text", "text": "real answer"}]}],
    })

    reply = _bot(monkeypatch, client, api_type="responses").reply_text(_session())

    assert reply["content"] == "real answer"
    assert reply["completion_tokens"] > 0, "reply() would report this as an error"


def test_reported_usage_is_used_as_is(monkeypatch):
    client = _Client(chat=_chat({"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}))

    reply = _bot(monkeypatch, client).reply_text(_session())

    assert reply == {"total_tokens": 12, "completion_tokens": 7, "content": "real answer"}

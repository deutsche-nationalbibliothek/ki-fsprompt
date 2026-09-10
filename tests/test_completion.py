"""Focused tests for LLMCompletion.predict_batch's chat-template rendering."""

import os
import sys
import unittest
from unittest import mock

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class FakeTokenizer:
    """Records apply_chat_template kwargs and renders a trivial fake prompt."""

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        self.calls.append(kwargs)
        rendered = "".join(m["content"] for m in messages)
        if tokenize:
            return list(range(len(rendered)))
        return rendered

    def __call__(self, text):
        return {"input_ids": list(range(len(text)))}

    def decode(self, tokens):
        return "x" * len(tokens)


class FakePromptBuilder:
    def build_messages(self, doc_id, text=""):
        return [{"role": "user", "content": text}]


class PredictBatchThinkingTest(unittest.TestCase):
    def test_enable_thinking_false_passed_to_apply_chat_template(self):
        from completion import LLMCompletion

        completer = object.__new__(LLMCompletion)
        completer.tokenizer = FakeTokenizer()
        completer.prompt_builder = FakePromptBuilder()
        completer.max_total_tokens = 100
        completer.max_new_tokens = 10
        completer.vllm_samplingparams = mock.Mock()
        completer.llm = mock.Mock()
        completer.llm.generate.return_value = []
        completer.debug = False

        documents = pd.DataFrame([{"doc_id": "doc1", "text": "some document text"}])
        completer.predict_batch(documents)

        self.assertTrue(completer.tokenizer.calls)
        for call_kwargs in completer.tokenizer.calls:
            self.assertEqual(call_kwargs.get("enable_thinking"), False)


if __name__ == "__main__":
    unittest.main()

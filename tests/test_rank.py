"""Focused tests for Ranker.rank_batches's chat-template rendering and score parsing."""

import os
import sys
import unittest
from unittest import mock

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rank import extract_score


class FakeTokenizer:
    """Records apply_chat_template kwargs and renders a trivial fake prompt."""

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        self.calls.append(kwargs)
        return "".join(m["content"] for m in messages)


class FakeOutput:
    def __init__(self, text):
        self.outputs = [mock.Mock(text=text)]


class RankBatchesTest(unittest.TestCase):
    def _make_ranker(self, enable_thinking):
        from rank import Ranker

        ranker = object.__new__(Ranker)
        ranker.tokenizer = FakeTokenizer()
        ranker.ranking_instruction = "Rate relevance 0-10."
        ranker.enable_thinking = enable_thinking
        ranker.max_suggestions = 10
        ranker.threshold_confidence = 0
        ranker.debug = False
        ranker.vllm_samplingparams = mock.Mock()
        ranker.llm = mock.Mock()
        return ranker

    def test_enable_thinking_and_thinking_kwargs_passed(self):
        ranker = self._make_ranker(enable_thinking=True)
        ranker.llm.generate.return_value = [FakeOutput("7")]

        preds = pd.DataFrame(
            {"text": ["Some document text"], "prelim_labels": [["Keyword A"]]},
            index=pd.Index(["doc1"], name="doc_id"),
        )
        ranker.rank_batches(preds)

        self.assertTrue(ranker.tokenizer.calls)
        for call_kwargs in ranker.tokenizer.calls:
            self.assertEqual(call_kwargs.get("enable_thinking"), True)
            self.assertEqual(call_kwargs.get("thinking"), True)

    def test_extract_score_strips_think_block(self):
        raw = "<think>\nlet me consider this\n</think>\n7"
        self.assertEqual(extract_score(raw), 7)

    def test_extract_score_without_think_block(self):
        self.assertEqual(extract_score(" 3 "), 3)

    def test_extract_score_no_digits_defaults_to_zero(self):
        self.assertEqual(extract_score("no number here"), 0)


if __name__ == "__main__":
    unittest.main()

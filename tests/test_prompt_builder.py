"""Focused tests for PromptBuilder.IndividualPromptBuilder.build_messages."""

import csv
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from PromptBuilder import IndividualPromptBuilder


def _write_prompt_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "doc_id",
                "text",
                "label_ids",
                "label_texts",
                "prompt_text",
                "prompt_labels",
                "prompt_label_texts",
                "similarity",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class BuildMessagesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False, mode="w"
        )
        self.tmp.close()
        _write_prompt_csv(
            self.tmp.name,
            [
                {
                    "doc_id": "doc1",
                    "text": "",
                    "label_ids": "",
                    "label_texts": "",
                    "prompt_text": "Example title one",
                    "prompt_labels": "",
                    "prompt_label_texts": "Keyword A; Keyword B",
                    "similarity": "0.9",
                },
                {
                    "doc_id": "doc1",
                    "text": "",
                    "label_ids": "",
                    "label_texts": "",
                    "prompt_text": "Example {title} two",
                    "prompt_labels": "",
                    "prompt_label_texts": "Keyword C",
                    "similarity": "0.8",
                },
            ],
        )

    def tearDown(self):
        os.remove(self.tmp.name)

    def test_message_role_ordering_with_system_instruction(self):
        builder = IndividualPromptBuilder(self.tmp.name, "Be concise.")
        messages = builder.build_messages("doc1", "Document text")
        roles = [m["role"] for m in messages]
        self.assertEqual(
            roles, ["system", "user", "assistant", "user", "assistant", "user"]
        )
        self.assertEqual(messages[-1]["content"], "Document text")

    def test_empty_system_instruction_is_omitted(self):
        builder = IndividualPromptBuilder(self.tmp.name, "")
        messages = builder.build_messages("doc1", "Document text")
        roles = [m["role"] for m in messages]
        self.assertNotIn("system", roles)
        self.assertEqual(roles[0], "user")

    def test_assistant_examples_are_structured_json(self):
        builder = IndividualPromptBuilder(self.tmp.name, "Be concise.")
        messages = builder.build_messages("doc1", "Document text")
        first_assistant = messages[2]["content"]
        self.assertEqual(first_assistant, '{"keywords": ["Keyword A", "Keyword B"]}')

    def test_brace_cleanup_on_examples_and_final_text(self):
        builder = IndividualPromptBuilder(self.tmp.name, "Be concise.")
        messages = builder.build_messages("doc1", "Text with {braces}")
        # second example's prompt_text originally contained "{title}"
        self.assertNotIn("{", messages[3]["content"])
        self.assertNotIn("}", messages[3]["content"])
        self.assertEqual(messages[-1]["content"], "Text with braces")


if __name__ == "__main__":
    unittest.main()

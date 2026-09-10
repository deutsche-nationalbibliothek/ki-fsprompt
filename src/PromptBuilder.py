import pandas as pd
import json
import re


class IndividualPromptBuilder:
    def __init__(
        self,
        parsed_prompt_file: str,
        custom_instruction: str,
        debug: bool = False,
    ):
        self.parsed_prompt_file = parsed_prompt_file
        self.custom_instruction = custom_instruction
        self.debug = debug

        self.prompts_by_id = self.parse_retrieved_prompts()

    def parse_retrieved_prompts(self) -> pd.DataFrame:
        """Parse the prompt examples retrieved with retrieve.py
        Args:
            parsed_prompt_file (str): Path to the parsed prompt file.
            text_type (str): Type of text to parse ('title' or 'ft).
        Returns:
            pd.DataFrame: DataFrame containing the parsed prompts.
        """
        prompts = pd.read_csv(self.parsed_prompt_file)
        # columns: doc_id, text, label_ids, label_texts, prompt_text, prompt_labels, prompt_label_texts, similarity
        prompts_gr = (
            prompts[["doc_id", "prompt_text", "prompt_label_texts"]]
            .groupby("doc_id")
            .agg(lambda x: list(x))
        )
        prompts_gr["prompt_examples"] = prompts_gr.apply(
            lambda row: list(zip(row["prompt_text"], row["prompt_label_texts"])), axis=1
        )
        return prompts_gr.to_dict(orient="index")

    def build_messages(self, doc_id: str, text: str = "") -> list[dict]:
        """Build a chat-message list for the given document.

        Args:
            doc_id (str): Id of the document to build the prompt for.
            text (str): The (possibly truncated) document text to append as the final user turn.
        Returns:
            list[dict]: Messages with roles system/user/assistant, ready for
                        `tokenizer.apply_chat_template()`.
        """
        prompt_examples = self.prompts_by_id[doc_id]["prompt_examples"]
        messages = []
        if self.custom_instruction != "":
            messages.append({"role": "system", "content": self.custom_instruction})
        for example_text, keywords in prompt_examples:
            example_text = re.sub(r"[{}]", "", example_text)
            if isinstance(keywords, (float, int)):
                keywords = str(keywords)
            elif keywords is None:
                keywords = ""
            keyword_list = [
                keyword.strip() for keyword in keywords.split(";") if keyword.strip()
            ]
            structured_keywords = json.dumps(
                {"keywords": keyword_list}, ensure_ascii=False
            )
            messages.append({"role": "user", "content": example_text})
            messages.append({"role": "assistant", "content": structured_keywords})
            if self.debug:
                print("Messages after example: ", messages)
        messages.append({"role": "user", "content": re.sub(r"[{}]", "", text)})
        return messages

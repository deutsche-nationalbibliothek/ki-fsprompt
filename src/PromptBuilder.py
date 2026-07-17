import pandas as pd
import json
import re


class IndividualPromptBuilder:
    def __init__(
        self,
        parsed_prompt_file: str,
        custom_instruction: str,
        template_file: str,
        debug: bool = False,
    ):
        self.parsed_prompt_file = parsed_prompt_file
        self.custom_instruction = custom_instruction
        self.template_file = template_file
        with open(self.template_file) as tf:
            self.template = json.load(tf)
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

    def build_prompt(self, doc_id: str) -> str:
        prompt_examples = self.prompts_by_id[doc_id]["prompt_examples"]
        if self.custom_instruction != "":
            prompt = self.template["instruction"].format(
                custom_instruction=self.custom_instruction
            )
        else:
            prompt = ""
        for text, keywords in prompt_examples:
            text = re.sub(r"[{}]", "", text)
            prompt += self.template["example"].format(text=text)
            if self.debug:
                print("Prompt after example: ", prompt)
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
            prompt += self.template["keywords"].format(keywords=structured_keywords)
        prompt += self.template["test_item"]
        return prompt

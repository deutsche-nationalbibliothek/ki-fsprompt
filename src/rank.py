"""
File name: rank.py
Description: Implementation of rank-stage
"""

import yaml
from argparse import ArgumentParser
import pandas as pd
from tqdm import tqdm
import json
import os
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from vllm.engine.arg_utils import EngineArgs
from dataclasses import asdict
import yaml
import re


def safe_int_conversion(rel):
    try:
        return int(rel)
    except (TypeError, ValueError):
        return 0


def extract_score(raw_text: str) -> int:
    """Extract the final relevance score, tolerating a leading <think>...</think> block."""
    content = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
    match = re.search(r"-?\d+", content)
    return int(match.group()) if match else 0


class Ranker:
    """Class to perform the ranking stage using vLLM.

    Methods:
        __init__: Initialize.
        rank: Starts the ranking and writes results.
        rank_batches: Performs the ranking over batches.
    """

    def __init__(
        self,
        dataset_file: str,
        predictions_file: str,
        output_file: str,
        custom_instructions: str,
        params_file: str,
    ):
        """Initialiazes the Ranker class.

        Args:
            dataset_file (str): Path to the dataset file (csv).
                                Needed to obtain the title/abstract of the texts.
            predictions_file (str): Path to the mapped predictions file (.arrow) from the mapping output.
            output_file (str): Arrow output file (for evaluation).
        """

        with open(params_file, "r") as file:
            self.shared_params = yaml.safe_load(file)
        # extract dir of the params file

        self.vllm_engineargs = self.shared_params["vllm"]["engineargs"]
        self.p_completion = self.shared_params["completion"]
        self.p_ranking = self.shared_params["ranking"]
        self.debug = False

        self.output_file = output_file

        self.global_samplingparams = self.shared_params["vllm"]["global_samplingparams"]
        self.score_param = self.p_ranking.get("score_param", "relevance")
        self.ranking_model = self.p_ranking.get(
            "model", "meta-llama/Meta-Llama-3.1-8B-Instruct"
        )
        self.temperature = self.p_ranking.get("temperature", 0)
        self.max_suggestions = self.p_ranking.get("max_suggestions", 10)
        self.min_confidence = self.p_ranking.get("min_confidence", 0)
        self.max_confidence = self.p_ranking.get("max_confidence", 10)
        self.threshold_confidence = self.p_ranking.get("threshold_confidence", 0.5)

        self.allowed_scores = [
            str(score)
            for score in range(self.min_confidence, self.max_confidence + 1)
        ]

        # opt-in chain-of-thought before the relevance score; off by default
        self.enable_thinking = self.p_ranking.get("enable_thinking", False)
        self.reasoning_parser = self.p_ranking.get("reasoning_parser")
        self.max_reasoning_tokens = self.p_ranking.get("max_reasoning_tokens", 128)

        max_tokens = 5  # small number of tokens to generate, so to fit only one number
        if self.enable_thinking:
            max_tokens += self.max_reasoning_tokens

        self.vllm_samplingparams = SamplingParams(
            max_tokens=max_tokens,
            min_tokens=1,
            temperature=self.temperature,
            presence_penalty=self.global_samplingparams.get("presence_penalty", 0),
            frequency_penalty=self.global_samplingparams.get("frequency_penalty", 0),
            repetition_penalty=self.global_samplingparams.get("repetition_penalty", 1),
            top_p=self.global_samplingparams.get("top_p", 1),
            structured_outputs=StructuredOutputsParams(choice=self.allowed_scores),
        )

        optional_engineargs = {}
        if self.enable_thinking and self.reasoning_parser:
            optional_engineargs["reasoning_parser"] = self.reasoning_parser

        self.vllm_engineargs = EngineArgs(
            model=self.ranking_model,
            gpu_memory_utilization=self.vllm_engineargs.get(
                "gpu_memory_utilization", 0.8
            ),
            tensor_parallel_size=self.vllm_engineargs.get("tensor_parallel_size", 2),
            dtype=self.vllm_engineargs.get("dtype", "auto"),
            **optional_engineargs,
        )
        self.llm = LLM.from_engine_args(self.vllm_engineargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.ranking_model, trust_remote_code=True
        )

        # Prepare prompt
        with open(custom_instructions, encoding="utf-8") as f:
            self.ranking_instruction = json.load(f)["instruction"]

        self.ranking_instruction = self.ranking_instruction.format(
            min_confidence=self.min_confidence, max_confidence=self.max_confidence
        )

        # Read dataset
        self.data = pd.read_csv(dataset_file)
        self.data = self.data[["doc_id", "text"]]
        self.data = self.data.set_index("doc_id")



        # Read predictions
        self.predictions = pd.read_csv(predictions_file)
        self.preds_for_ranking = self.predictions[["doc_id", "term"]]
        self.preds_for_ranking = self.preds_for_ranking.groupby("doc_id").agg(list)
        if self.debug:
            print("Preds for ranking head", self.preds_for_ranking.shape)
            print(self.preds_for_ranking.head(5))

        self.preds_for_ranking = self.preds_for_ranking.merge(self.data, on="doc_id")
        # Find rows where the 'content' column is empty
        empty_content_rows = self.preds_for_ranking[
            self.preds_for_ranking["text"].isna()
        ]
        empty_content_count = len(empty_content_rows)
        if self.debug:
            print(f"Number of rows with empty content: {empty_content_count}")
        if empty_content_count > 0:
            raise ValueError(f"Found {empty_content_count} rows with empty content.")
        if self.debug:
            print("Preds for ranking after join")
            print(self.preds_for_ranking.head())
        self.preds_for_ranking = self.preds_for_ranking.rename(
            columns={"term": "prelim_labels"}
        )
        if self.debug:
            self.preds_for_ranking = self.preds_for_ranking.head(5)
        self.pred_dict = {}
        self.predictions = self.predictions.set_index("doc_id")
        for doc_id, row in self.predictions.iterrows():
            if doc_id not in self.pred_dict:
                self.pred_dict[doc_id] = {}
            self.pred_dict[doc_id][row["term"]] = {
                "term": row["term"],
                "count": row["count"],
                "cosine_similarity": row["cosine_similarity"],
                "hybrid_score": row["hybrid_score"],
                "label_id": row["label_id"],
                "score": row["score"],
            }
        if self.debug:
            print("Predictions has this shape:", self.predictions.shape)
            keys_in_pred_dict = 0
            for key in self.pred_dict.keys():
                keys_in_pred_dict += len(self.pred_dict[key])
            print("Pred_dict has this shape:", keys_in_pred_dict)
            print("Pred_dict keys:", list(self.pred_dict.keys())[0:5])
            print("Pred_dict items: ", list(self.pred_dict.items())[0:5])
        del self.predictions

    def rank_batches(self, preds_for_ranking):
        """Rankes the predictions in batches."""
        results = []
        for doc_id, row in tqdm(
            preds_for_ranking.iterrows(),
            total=len(preds_for_ranking),
            desc="Reranking suggestions",
        ):
            text = row["text"]
            text = re.sub(r"[{}]", "", text)
            labels = sorted(list(set(row["prelim_labels"])))
            prompts = []
            for l in labels:
                l_sub = l
                if isinstance(l, (float, int)):
                    l_sub = str(l)  # Convert numeric types to string
                elif l is None:
                    l_sub = ""  # Handle None values
                l_sub = re.sub(r"[{}]", "", l_sub)
                messages = [
                    {"role": "system", "content": self.ranking_instruction},
                    {
                        "role": "user",
                        "content": "Text: {}\nSchlagwort:{}".format(text, l_sub),
                    },
                ]
                # enable_thinking/thinking cover Qwen3/Gemma4 vs. Granite/DeepSeek-V3.1 kwarg naming;
                # templates that don't declare either are unaffected
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self.enable_thinking,
                    thinking=self.enable_thinking,
                )
                prompts.append(prompt)

            response = self.llm.generate(prompts, self.vllm_samplingparams, use_tqdm=False)
            answers = [o.outputs[0].text for o in response]
            assert len(answers) == len(
                labels
            ), "Number of answers does not match number of labels"
            answers_ints = [extract_score(answer) for answer in answers]
            ranked_kw = list(zip(labels, answers_ints))

            if self.debug:
                print(
                    "This is the current item\nText:{}\nLabels:{}\nranked_kw:{}\n".format(
                        text, labels, ranked_kw
                    )
                )
                correctly_returned = [
                    x for x in labels if x in [kw for kw, r in ranked_kw]
                ]
                print(
                    "Returned {}/{} keywords".format(
                        len(correctly_returned), len(labels)
                    )
                )

            # More ranked keywords than possible
            if self.debug:
                print(
                    "Hallucinated keywords: ",
                    [kw for kw, r in ranked_kw if kw not in labels],
                )
                print("Original keywords: ", labels)
                print("Ranking output: ", ranked_kw)
            ranked_kw = [(kw, r) for kw, r in ranked_kw if kw in labels]

            accepted_kw = 0
            for keyword, relevance in ranked_kw:
                if float(relevance) >= self.threshold_confidence:
                    # enough relevance
                    results.append(
                        {"doc_id": doc_id, "keyword": keyword, "relevance": relevance}
                    )
                    accepted_kw += 1
                    if self.debug == True:
                        print(
                            "Appending item: ",
                            str(
                                {
                                    "doc_id": doc_id,
                                    "keyword": keyword,
                                    "relevance": relevance,
                                }
                            ),
                        )
                if accepted_kw == self.max_suggestions:
                    break

        return results

    def rank(self):
        """Starts the ranking and writes results"""

        prelim_results = self.rank_batches(self.preds_for_ranking)

        ranked_results = []
        for item in prelim_results:
            doc_id, term, relevance = item["doc_id"], item["keyword"], item["relevance"]
            res_entry = {"doc_id": doc_id, "term": term, "relevance": relevance}
            res_entry.update(self.pred_dict[doc_id][term])
            ranked_results.append(res_entry)

        ranked_results = pd.DataFrame(ranked_results)

        if self.debug:
            print(ranked_results)
            return
        if self.score_param == "score":
            # Keep the score from previous stages = do nothing
            pass
        else:
            ranked_results[self.score_param] = ranked_results[self.score_param].astype(
                float
            )
            ranked_results["score"] = ranked_results[self.score_param] / (
                ranked_results[self.score_param].max()
            )
        ranked_results = ranked_results[["doc_id", "label_id", "score"]]
        ranked_results.columns = ["doc_id", "label_id", "score"]
        # make sure each tuple doc_id label_id is unique
        ranked_results = (
            ranked_results.groupby(["doc_id", "label_id"])
            .agg({"score": "max"})
            .reset_index()
        )
        ranked_results.to_csv(self.output_file)


def execute():
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_file", help="Dataset Filename/Path", type=str, required=True
    )
    parser.add_argument(
        "--predictions_file", help="Predictions Filename/Path", type=str, required=True
    )
    parser.add_argument(
        "--output_file", help="Output Filename/Path", type=str, required=True
    )
    parser.add_argument(
        "--params_file", help="Path to the params file", type=str, required=True
    )
    parser.add_argument(
        "--custom_instructions", help="Custom Instructions for Ranking", type=str, required=False
    )

    args = parser.parse_args()
    ranker = Ranker(
        args.dataset_file,
        args.predictions_file,
        args.output_file,
        args.custom_instructions,
        args.params_file,
    )
    ranker.rank()


if __name__ == "__main__":
    execute()

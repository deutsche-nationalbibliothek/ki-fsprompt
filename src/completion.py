"""
File name: completion.py
Description: Script to complete the complete-stage of our pipeline.
"""

import json
import datetime
import logging
import os
import re
import pandas as pd
import logging
import datetime
from tqdm import tqdm

from pathlib import Path
import os
from transformers import AutoTokenizer
from argparse import ArgumentParser
from dataclasses import asdict

# from prompt_template import PromptBuilder
from PromptBuilder import IndividualPromptBuilder


from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.distributed.parallel_state import destroy_model_parallel
from vllm.sampling_params import StructuredOutputsParams

import torch
import gc

import yaml

# from huggingface_hub import login

# login(token=.os.environ["HF_TOKEN"])
# import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"


class LLMCompletion:
    """Class to perform the completion stage using vLLM.

    Methods:
        __init__: Initialize.
        predict: Starts the prediction and writes results.
        predict_batch: Predicts the completion for a batch of documents.
        post_process_results: Post-process the result returned by vLLM.
    """

    def __init__(
        self,
        hf_model_name: str,
        dataset_file: str,
        prompt_specification: str,
        completion_file: str = "completion.csv",
        params_file: str = "params.yaml",
        prompt_template_info: str | None = None,
        prompt_template_dir: str | None = None,
        custom_instructions: str | None = None,
    ):
        """Initializes the LLMCompletion class.

        Args:
            hf_model_name (str): Identifier of a HuggingFace model.
            dataset_file (str): Path to the csv-dataset file, created with preprocess.py-script.
            prompt_specification (str): Path to the prompt file containing the few-shot examples.
            completion_file (str): Desired filename of the completion file.
                                   Defaults to completions.csv"
        """

        with open(params_file, "r") as file:
            self.shared_params = yaml.safe_load(file)
        # extract dir of the params file
        self.base_dir = os.path.dirname(params_file)

        # self.p_general = self.shared_params["general"]
        self.vllm_engineargs = self.shared_params["vllm"]["engineargs"]
        self.p_completion = self.shared_params["completion"]
        self.completion_samplingparams = self.shared_params["completion"]["vllm"][
            "samplingparams"
        ]
        self.global_samplingparams = self.shared_params["vllm"]["global_samplingparams"]

        self.instruction_file = custom_instructions

        self.hf_model_name = hf_model_name
        model_id = self.hf_model_name.lower()
        explicit_tokenizer_mode = self.vllm_engineargs.get("tokenizer_mode")
        auto_tokenizer_mode = "mistral" if model_id.startswith("mistralai/") else "auto"
        selected_tokenizer_mode = explicit_tokenizer_mode or auto_tokenizer_mode
        if model_id.startswith("mistralai/"):
            optional_engineargs = {
                "config_format": "mistral",
                "load_format": "mistral",
                "reasoning_parser": "mistral",
            }
        else:
            optional_engineargs = {}

        self.vllm_engineargs = EngineArgs(
            model=self.hf_model_name,
            # task=self.vllm_engineargs.get("task", "generate"),
            gpu_memory_utilization=self.vllm_engineargs.get(
                "gpu_memory_utilization", 0.8
            ),
            tensor_parallel_size=self.vllm_engineargs.get("tensor_parallel_size", 2),
            dtype=self.vllm_engineargs.get("dtype", "auto"),
            tokenizer_mode=selected_tokenizer_mode,
            trust_remote_code=True,
            enforce_eager=self.vllm_engineargs.get("enforce_eager"),
            max_model_len=(
                self.vllm_engineargs.get("max_model_len", 15000)
                if self.hf_model_name
                in [
                    "meta-llama/Meta-Llama-3.1-70B-Instruct",
                    "mistralai/Mixtral-8x7B-Instruct-v0.1",
                ]
                else None
            ),
            **optional_engineargs,
        )

        keyword_schema = {
            "type": "object",
            "properties": {
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 20,
                }
            },
            "required": ["keywords"],
            "additionalProperties": False,
        }

        self.vllm_samplingparams = SamplingParams(
            min_tokens=self.completion_samplingparams.get("min_new_tokens", 1),
            max_tokens=self.completion_samplingparams.get("max_new_tokens", 64),
            temperature=self.completion_samplingparams.get("temperature", 0),
            presence_penalty=self.global_samplingparams.get("presence_penalty", 0),
            frequency_penalty=self.global_samplingparams.get("frequency_penalty", 0),
            repetition_penalty=self.global_samplingparams.get("repetition_penalty", 1),
            top_p=self.global_samplingparams.get("top_p", 1),
            structured_outputs=StructuredOutputsParams(json=keyword_schema),
        )

        self.llm = LLM.from_engine_args(self.vllm_engineargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hf_model_name, trust_remote_code=True
        )
        self.debug = False

        # self.write_dir = Path(self.p_general.get("write_directory", None))
        # if not os.path.exists(self.write_dir):
        #     os.makedirs(self.write_dir)

        self.n_processes = self.p_completion.get("n_processes", 1)
        self.batch_size = self.p_completion.get("batch_size", None)

        try:
            self.data = pd.read_csv(dataset_file)
            print("Successfully loaded dataset with shape ", self.data.shape)
        except FileNotFoundError:
            print(
                "Dataset file not found... expected to read in data in this path: ",
                dataset_file,
            )
            raise FileNotFoundError

        self.completion_file = completion_file
        self.max_new_tokens = self.completion_samplingparams.get("max_new_tokens")
        self.max_total_tokens = self.completion_samplingparams.get("max_total_tokens")

        with open(self.instruction_file, encoding="utf-8") as f:
            self.custom_instruction = f.read()
        self.prompt_file = prompt_specification
        self.prompt_template_info = prompt_template_info
        self.prompt_template_dir = prompt_template_dir
        with open(self.prompt_template_info) as f:
            content = json.load(f)
            self.prompt_template_file = content[self.hf_model_name]
        self.prompt_builder = IndividualPromptBuilder(
            self.prompt_file,
            self.custom_instruction,
            os.path.join(self.prompt_template_dir, self.prompt_template_file),
            self.debug,
        )

        if self.debug:
            print("Completion file: ", self.completion_file)
            print("Data: ", self.data.head())

    def post_process_results(self, response, prompt, doc_id):
        """Post-process the result returned by vLLM.

        This method is called by predict_batch.
        """
        raw_text = response.outputs[0].text.strip()
        result = []
        try:
            answer = json.loads(raw_text)
            if isinstance(answer, dict) and "keywords" in answer:
                result = list(set(answer["keywords"]))
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            logging.warning(
                f"Failed standard JSON parse for doc {doc_id}: {e}. Attempting robust fallback on: {raw_text!r}"
            )
            # Try repairing truncated JSON by appending closing quotes/brackets
            for suffix in ['"]}', '"]', '}', ']']:
                try:
                    repaired = json.loads(raw_text + suffix)
                    if isinstance(repaired, dict) and "keywords" in repaired:
                        result = list(set(repaired["keywords"]))
                        break
                except json.JSONDecodeError:
                    continue

            # If JSON repair did not work, extract fully completed quoted strings via regex
            if not result:
                keywords_part = raw_text.split('"keywords"', 1)[-1] if '"keywords"' in raw_text else raw_text
                matches = re.findall(r'"((?:[^"\\]|\\.)*)"', keywords_part)
                result = list(set(m for m in matches if m))

        if self.debug:
            print("Result:", result)

        if len(result) == 0:
            logging.info("Results for document {} entirely empty".format(doc_id))
            result = []
        if self.debug:
            print("Prompt: ", prompt)
            print("Response: ", response.outputs[0].text)
            print("Result: ", result)
            print("-------------")
        return result

    def predict_batch(self, documents):
        """Predicts the completion for a batch of documents.

        This method is called by the predict-method.
        """

        results = []
        prompts = []
        doc_ids = []
        for _, row in documents.iterrows():
            row_content = row.text
            prompt_frame = self.prompt_builder.build_prompt(row.doc_id)
            prompt_length = len(self.tokenizer(prompt_frame)["input_ids"])
            free_tokens = self.max_total_tokens - self.max_new_tokens - prompt_length
            tokens = self.tokenizer(row_content)["input_ids"][:free_tokens]
            chunk = self.tokenizer.decode(tokens)
            prompt = prompt_frame.replace("{text}", chunk)
            prompts.append(prompt)
            doc_ids.append(row.doc_id)

        responses = self.llm.generate(prompts, self.vllm_samplingparams, use_tqdm = True)
        candidate_lists = []
        for i, response in enumerate(responses):
            result = self.post_process_results(response, prompts[i], doc_ids[i])
            candidate_lists.append(result)
        j = 0
        for candidate_list in candidate_lists:
            for candidate in candidate_list:
                results.append({"doc_id": doc_ids[j], "candidate": candidate})
            j += 1
        results_df = pd.DataFrame(results)
        return results_df

    def predict(self):
        """Starts the prediction and writes results."""

        results = []
        df_documents = self.data
        if self.debug:
            df_documents = df_documents.head(10)

        logging.info(
            "Start computation at time {}".format(str(datetime.datetime.now()))
        )
        batch_size = self.batch_size if self.batch_size is not None else df_documents.shape[0]
        batches = [
            df_documents[i : i + batch_size]
            for i in range(0, df_documents.shape[0], batch_size)
        ]
        results = []
        for batch in batches:
            candidates = self.predict_batch(batch)
            results.append(candidates)

        candidates = pd.concat(results, ignore_index=True)

        if not self.debug:
            candidates.to_csv(
                self.completion_file,
                index=False,
                mode="a",
                header=not os.path.exists(self.completion_file),
            )
        logging.info(
            "End complete computation at time {}".format(str(datetime.datetime.now()))
        )
        destroy_model_parallel()
        del self.llm
        gc.collect()
        torch.cuda.empty_cache()
        try:
            torch.distributed.destroy_process_group()
        except AssertionError:
            pass  # Process group was not initialized or already destroyed
        logging.info(
            "Done with prediction! Find the results at {}".format(self.completion_file)
        )


def execute():
    """Builds an ArgumentParser and runs the completion stage."""
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_file", help="Dataset Filename/Path", type=str, required=True
    )
    parser.add_argument(
        "--hf_model_name",
        help="Identifier of hugging face model i.e. domain/model",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--completion_file", help="Completion Filename/Path", type=str, required=True
    )
    parser.add_argument(
        "--prompt_specification",
        help="Path to prompt specification file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--params_file", help="Path to the params file", type=str, required=True
    )
    parser.add_argument(
        "--prompt_template_info",
        help="Optional override for completion.prompt_template_info",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--prompt_template_dir",
        help="Optional override for completion.prompt_template_dir",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--custom_instructions",
        help="Optional override for completion.custom_instruction",
        type=str,
        required=False,
        default=None,
    )
    
    args = parser.parse_args()
    completer = LLMCompletion(
        hf_model_name=args.hf_model_name,
        dataset_file=args.dataset_file,
        prompt_specification=args.prompt_specification,
        completion_file=args.completion_file,
        params_file=args.params_file,
        prompt_template_info=args.prompt_template_info,
        prompt_template_dir=args.prompt_template_dir,
        custom_instructions=args.custom_instructions,
    )

    log_file_path = Path("logs/LLMCompletion.log")
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=log_file_path, filemode="w", level=logging.INFO)
    logging.basicConfig(format="%(asctime)s %(message)s")

    completer.predict()


if __name__ == "__main__":
    execute()

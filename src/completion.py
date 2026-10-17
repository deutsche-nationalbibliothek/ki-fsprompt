"""
File name: completion.py
Description: Script to complete the complete-stage of our pipeline.
"""

import json
import datetime
import logging
import os
import re
import time
import atexit
import subprocess
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
        optional_engineargs = {}
        if model_id.startswith("mistralai/"):
            optional_engineargs["config_format"] = "mistral"
            optional_engineargs["load_format"] = "mistral"

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
                    "mistralai/Mistral-7B-Instruct-v0.3",
                    "Qwen/Qwen3.8-27B-FP8", #diese zwei sind zum Testen
                    "RedHatAI/gemma-4-31B-it-FP8-block" ,
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

        # Resource accounting: model loading is timed separately from generation,
        # because it depends on the local HF cache rather than on the model itself.
        self.completion_file = completion_file
        self.generate_windows = []
        self.start_gpu_logger()
        load_start = time.perf_counter()
        self.llm = LLM.from_engine_args(self.vllm_engineargs)
        self.load_seconds = time.perf_counter() - load_start
        self.generate_seconds = 0.0
        self.prompt_tokens = 0
        self.output_tokens = 0
        # documents whose few-shot examples / own text had to be shortened to fit max_total_tokens
        self.n_examples_shortened = 0
        self.n_docs_truncated = 0
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
        self.prompt_builder = IndividualPromptBuilder(
            self.prompt_file,
            self.custom_instruction,
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

    def prompt_length(self, messages):
        """Number of tokens of the rendered chat prompt, including the assistant-turn prefix."""
        # enable_thinking=False keeps output JSON-only; templates without this kwarg ignore it
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        # transformers >= 5 returns a BatchEncoding instead of a list of token ids
        return len(encoded["input_ids"] if hasattr(encoded, "keys") else encoded)

    def shorten_examples(self, doc_id, text, budget):
        """Cut the few-shot examples to one common token cap so that the full document fits into budget.

        Only used when the complete prompt is too long: examples shorter than the cap stay untouched,
        longer ones are cut at the end to the same length. Returns None if the prompt already fits
        (examples unchanged), otherwise the shortened (text, keywords) pairs. If even empty examples
        leave too little room, the caller truncates the document.
        """
        examples = self.prompt_builder.prompts_by_id[doc_id]["prompt_examples"]
        overflow = self.prompt_length(self.prompt_builder.build_messages(doc_id, text)) - budget
        if overflow <= 0:
            return None
        self.n_examples_shortened += 1
        example_tokens = [self.tokenizer(str(t), add_special_tokens=False)["input_ids"] for t, _ in examples]
        lengths = [len(t) for t in example_tokens]
        shortened = examples
        # re-tokenizing the rendered prompt can differ by a few tokens, so lower the cap until it fits
        while overflow > 0:
            removable = lambda cap: sum(max(length - cap, 0) for length in lengths)
            low, high = 0, max(lengths)
            while low < high:  # largest cap that still removes at least `overflow` tokens
                mid = (low + high + 1) // 2
                if removable(mid) >= overflow:
                    low = mid
                else:
                    high = mid - 1
            cap = low
            shortened = [
                (self.tokenizer.decode(tokens[:cap]), keywords)
                for tokens, (_, keywords) in zip(example_tokens, examples)
            ]
            if cap == 0:
                break
            new_overflow = self.prompt_length(self.prompt_builder.build_messages(doc_id, text, shortened)) - budget
            if new_overflow <= 0:
                break
            overflow += new_overflow
        return shortened

    def predict_batch(self, documents):
        """Predicts the completion for a batch of documents.

        This method is called by the predict-method.
        """

        results = []
        prompts = []
        doc_ids = []
        budget = self.max_total_tokens - self.max_new_tokens
        for _, row in documents.iterrows():
            row_content = row.text
            # the document is the important part: shorten the examples first, the document only as a last resort
            examples = self.shorten_examples(row.doc_id, row_content, budget)
            empty_messages = self.prompt_builder.build_messages(row.doc_id, "", examples)
            free_tokens = max(budget - self.prompt_length(empty_messages), 0)
            tokens = self.tokenizer(row_content)["input_ids"]
            if len(tokens) > free_tokens:
                self.n_docs_truncated += 1
            chunk = self.tokenizer.decode(tokens[:free_tokens])
            messages = self.prompt_builder.build_messages(row.doc_id, chunk, examples)
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prompts.append(prompt)
            doc_ids.append(row.doc_id)

        generate_start = time.perf_counter()
        window_start = datetime.datetime.now()
        responses = self.llm.generate(prompts, self.vllm_samplingparams, use_tqdm = True)
        self.generate_windows.append((window_start, datetime.datetime.now()))
        self.generate_seconds += time.perf_counter() - generate_start
        self.prompt_tokens += sum(len(r.prompt_token_ids) for r in responses)
        self.output_tokens += sum(len(r.outputs[0].token_ids) for r in responses)
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
        self.stop_gpu_logger()
        if not self.debug:
            self.write_resource_stats(n_docs=df_documents.shape[0])
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

    def start_gpu_logger(self):
        """Samples the GPU once per second with nvidia-smi into gpu_log.csv next to the completion file.

        Note: nvidia-smi reports the whole device, so on a shared GPU the values include
        other users' processes.
        """
        self.gpu_device = (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0]
        self.gpu_log_file = os.path.join(os.path.dirname(self.completion_file) or ".", "gpu_log.csv")
        self.gpu_logger = None
        try:
            self.gpu_log_handle = open(self.gpu_log_file, "w")
            self.gpu_log_handle.write("timestamp,utilization_gpu_pct,memory_used_mib,power_draw_w\n")
            self.gpu_log_handle.flush()
            self.gpu_logger = subprocess.Popen(
                [
                    "nvidia-smi", "-i", self.gpu_device,
                    "--query-gpu=timestamp,utilization.gpu,memory.used,power.draw",
                    "--format=csv,noheader,nounits", "-l", "1",
                ],
                stdout=self.gpu_log_handle,
                stderr=subprocess.DEVNULL,
            )
            # make sure the logger does not outlive a crashed run
            atexit.register(self.stop_gpu_logger)
        except FileNotFoundError:
            logging.warning("nvidia-smi not found, no GPU log is written")

    def stop_gpu_logger(self):
        if self.gpu_logger is not None and self.gpu_logger.poll() is None:
            self.gpu_logger.terminate()
            self.gpu_logger.wait()
        if getattr(self, "gpu_log_handle", None) is not None and not self.gpu_log_handle.closed:
            self.gpu_log_handle.close()

    def summarize_gpu_log(self):
        """Energy (Wh), mean power and utilization from gpu_log.csv, overall and during generation."""
        try:
            log = pd.read_csv(self.gpu_log_file, skipinitialspace=True)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            return {}
        if log.empty:
            return {}
        log["timestamp"] = pd.to_datetime(log["timestamp"], format="%Y/%m/%d %H:%M:%S.%f", errors="coerce")
        for column in ["utilization_gpu_pct", "memory_used_mib", "power_draw_w"]:
            log[column] = pd.to_numeric(log[column], errors="coerce")
        log = log.dropna(subset=["timestamp"])
        # seconds until the next sample (last sample counts as 1 s)
        log["dt"] = log["timestamp"].diff().shift(-1).dt.total_seconds().fillna(1.0)

        in_generate = pd.Series(False, index=log.index)
        for start, end in self.generate_windows:
            in_generate |= (log["timestamp"] >= start) & (log["timestamp"] <= end)
        generate = log[in_generate]

        def energy_wh(frame):
            return round(float((frame["power_draw_w"] * frame["dt"]).sum()) / 3600, 2)

        return {
            "energy_wh_total": energy_wh(log),
            "energy_wh_generate": energy_wh(generate),
            "mean_power_w_generate": round(float(generate["power_draw_w"].mean()), 1) if not generate.empty else None,
            "mean_gpu_util_pct_generate": round(float(generate["utilization_gpu_pct"].mean()), 1) if not generate.empty else None,
            "max_memory_used_mib": int(log["memory_used_mib"].max()),
            "gpu_log_samples": int(len(log)),
        }

    def write_resource_stats(self, n_docs):
        """Writes runtime, token counts and GPU energy next to the completion file (resource_stats.json).

        GPU memory is taken from nvidia-smi, not from torch: vLLM pre-allocates according to
        gpu_memory_utilization, so torch's allocator statistics say nothing about the model.
        """
        stats = {
            "hf_model_name": self.hf_model_name,
            "n_docs": int(n_docs),
            "load_seconds": round(self.load_seconds, 1),
            "generate_seconds": round(self.generate_seconds, 1),
            "seconds_per_doc": round(self.generate_seconds / n_docs, 3) if n_docs else None,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "prompt_tokens_per_doc": round(self.prompt_tokens / n_docs, 1) if n_docs else None,
            "output_tokens_per_doc": round(self.output_tokens / n_docs, 1) if n_docs else None,
            "n_docs_examples_shortened": self.n_examples_shortened,
            "n_docs_truncated": self.n_docs_truncated,
            "tokens_per_second": (
                round((self.prompt_tokens + self.output_tokens) / self.generate_seconds, 1)
                if self.generate_seconds else None
            ),
            **self.summarize_gpu_log(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_memory_utilization": self.vllm_engineargs.gpu_memory_utilization,
            "enforce_eager": self.vllm_engineargs.enforce_eager,
            "finished_at": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        stats_file = os.path.join(os.path.dirname(self.completion_file), "resource_stats.json")
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)
        logging.info("Resource stats written to {}".format(stats_file))


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
        custom_instructions=args.custom_instructions,
    )

    log_file_path = Path("logs/LLMCompletion.log")
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=log_file_path, filemode="w", level=logging.INFO)
    logging.basicConfig(format="%(asctime)s %(message)s")

    completer.predict()


if __name__ == "__main__":
    execute()

# KI-FSPrompt: Knowledge Injected Few-Shot Prompting

This repository contains a research prototype for the task of automated subject
indexing, featuring a system developed for the shared task [LLMs4Subjects](https://sites.google.com/view/llms4subjects-germeval).

KI-FSPrompt core idea is a simple few-shot
prompting approach, where a generative LLM is presented
with a limited number of examples
of texts annotated with subject terms, and
prompted to identify the most relevant subject
terms for a new input text. To ensure alignment
with the library’s normed subject terms, we employ
a mapping approach based on embedding
similarity. We extend our [previous work](https://github.com/deutsche-nationalbibliothek/semeval25_llmensemble) by
incorporating a retrieval stage, which selects
relevant few-shot examples from the training
set to create knowledge-injected prompts, enabling
the LLM to provide more specific and
accurate keyword suggestions.

## Installation

### Step 1: Create virtual environment for ki-fsprompt

```bash
uv sync
```

Activate environment

```bash
source .venv/bin/activate
```

### Step 2: Start Docker services

The file `docker-compose.yaml` configures two services. Weaviate as a vector
storage and Hugging Face TEI as a service for embedding generation. 

```bash
docker compose --file docker-compose.yaml up
```

### Step 3: Create virtual R-environment for eval-stage

If you have a working R installation on your system, we recommend the following
packages:

```R
# fetch dependencies (all from CRAN)
install.packages("casimir", "optparse", "svglite", "rjson", "yaml")
# fetch polars (not from CRAN)
Sys.setenv(NOT_CRAN = "true") 
install.packages("polars", repos = "https://community.r-multiverse.org")
```

If you have no R installation, you may use conda to create a virtual environment
for your R dependencies. 

```bash
conda create --file r-environment.yml
conda activate r-eval
# install polars bindings for R
Rscript -e 'Sys.setenv(NOT_CRAN = "true"); install.packages("polars", repos = "https://community.r-multiverse.org")'
```

## Running the program

This repository relies on [Data Version Control](dvc.org) (DVC) to orchestrate
the dependencies of all steps necessary to run the complete program. 
 
In particular, there are three separate DVC-pipeline, executing the various
stages of the KI-FSPrompt System.

* `pipeline/preprocess/dvc.yaml`: Takes the corpora from raw input format to the required output format.
* `pipeline/train/dvc.yaml`: Creates the document collection for retrieving
  documents, indexes a small dev-set and trains the post-processing ranker on the
  dev-set.
* `pipeline/predict/dvc.yaml`: Indexing of a test set and evaluation

Each pipeline can be run with `dvc repro dvc.yaml` and has its own parameters
`params.yaml` as well as shared parameters in `pipelines/shared_params.yaml`
to configure the pipeline.

## Data input

This project is run with a dataset of German open access dissertations. The
data is currently not at public disclosure. 

## Citation

You can cite this work as:

```bibtex
@inproceedings{kahler-etal-2025-dnb,
    title = "{DNB}-{AI}-Project at the {G}erm{E}val-2025 {LLM}s4{S}ubjects Task: {KIFSP}rompt - Knowledge-Injected Few-Shot Prompting",
    author = {K{\"a}hler, Maximilian  and
      Kluge, Lisa  and
      Konermann, Katja},
    editor = "Wartena, Christian  and
      Heid, Ulrich",
    booktitle = "Proceedings of the 21st Conference on Natural Language Processing (KONVENS 2025): Workshops",
    month = sep,
    year = "2025",
    address = "Hannover, Germany",
    publisher = "HsH Applied Academics",
    url = "https://aclanthology.org/2025.konvens-2.42/",
    pages = "455--464"
}
```

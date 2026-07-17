from argparse import ArgumentParser
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
from tqdm import tqdm
import weaviate
import weaviate.classes as wvc
from transformers import AutoModel, AutoTokenizer
import logging

import torch

from data_adapters.constants import REQUIRED_COLUMNS


LOGGER = logging.getLogger(__name__)


def validate_args(args: Any, parser: ArgumentParser) -> None:
    """Validate cross-argument constraints before running the pipeline."""
    if args.text_type == "ft" and (args.chunk_size is None or args.chunk_size <= 0):
        parser.error("--chunk_size must be a positive integer when --text_type ft")
    if args.batch_size <= 0:
        parser.error("--batch_size must be a positive integer")
    if args.max_n_docs is not None and args.max_n_docs <= 0:
        parser.error("--max_n_docs must be positive when provided")


class CollectionCreator:
    def __init__(
        self,
        text_data,
        collection_name,
        text_type,
        chunk_size,
        batch_size,
        embedding_model,
        overwrite=False,
        weaviate_port=8087,
        manifest_out="logs/create_train_collection_manifest.json",
        max_n_docs=None,
        debug=False,
    ):
        self.collection_name = collection_name
        self.text_type = text_type
        self.chunk_size = chunk_size
        self.max_n_docs = max_n_docs
        self.debug = debug
        self.overwrite = overwrite
        self.weaviate_port = weaviate_port
        self.manifest_out = Path(manifest_out)

        self.data = pd.read_csv(text_data)
        missing_columns = [col for col in REQUIRED_COLUMNS if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Preprocessed data missing columns: {missing_columns}")

        if self.max_n_docs is not None:
            self.data = self.data.head(self.max_n_docs)
            LOGGER.info("Reduced data to max_n_docs. shape=%s", self.data.shape)
        if self.debug:
            LOGGER.info("Debug mode enabled. Using only 3 rows.")
            self.data = self.data.head(3)

        if self.text_type == "ft":
            self.data["text"] = self.chunk_texts()
            self.data["n_chunk"] = self.data["text"].apply(
                lambda x: list(range(1, len(x) + 1))
            )
            self.data = self.data.explode(["text", "n_chunk"])
            self.data["chunk_id"] = (
                self.data["doc_id"].astype(str) + "_" + self.data["n_chunk"].astype(str)
            )
        else:
            self.data["chunk_id"] = self.data["doc_id"].astype(str)
        self.batch_size = batch_size
        self.embedding_model = embedding_model



    def chunk_texts(self):
        LOGGER.info("Chunking texts with chunk_size=%s", self.chunk_size)
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", "\t", "."],
            chunk_size=self.chunk_size,
            chunk_overlap=50,
        )
        chunked_texts = [
            text_splitter.split_text(t) for t in tqdm(self.data.text, desc="Chunking")
        ]
        return chunked_texts

    def create_collection(self, overwrite=False):
        client = weaviate.connect_to_local(port=self.weaviate_port)
        LOGGER.info(
            f"Attempting to create collection {self.collection_name} in Weaviate"
        )
        if client.collections.exists(self.collection_name):
            LOGGER.info("Collection %s already exists", self.collection_name)
            if overwrite:
                client.collections.delete(self.collection_name)
                LOGGER.info("Deleted old collection %s", self.collection_name)
            else:
                LOGGER.info("Reusing existing collection %s", self.collection_name)
                return client, self.collection_name

        client.collections.create(
            name=self.collection_name,
            properties=[
                wvc.config.Property(
                    name="label_ids",
                    description="GND identifier for each concept",
                    data_type=wvc.config.DataType.TEXT,
                    tokenization=wvc.config.Tokenization.WORD,
                    vectorize_property_name=False,
                    skip_vectorization=True,
                    index_searchable=False,
                    index_filterable=True,
                ),
                wvc.config.Property(
                    name="doc_id",
                    description="Document idn",
                    data_type=wvc.config.DataType.TEXT,
                    vectorize_property_name=False,
                    tokenization=wvc.config.Tokenization.WORD,
                    index_searchable=True,
                    index_filterable=False,
                ),
                wvc.config.Property(
                    name="label_texts",
                    description="Label description (pref label or alt label)",
                    data_type=wvc.config.DataType.TEXT,
                    vectorize_property_name=False,
                    tokenization=wvc.config.Tokenization.WORD,
                    index_searchable=True,
                    index_filterable=False,
                ),
                wvc.config.Property(
                    name="doc_text",
                    description="Text content of the documents",
                    data_type=wvc.config.DataType.TEXT,
                    vectorize_property_name=False,
                    index_searchable=True,
                    index_filterable=False,
                ),
            ],
        )

        return client, self.collection_name

    def insert_docs(
        self,
        client: weaviate.Client,
        collection_name: str,
        text_data: pd.DataFrame,
        embeddings: torch.Tensor,
        phrase: str = None,
    ):
        if len(text_data) != embeddings.shape[0]:
            raise ValueError(
                "Embedding count mismatch. "
                f"rows={len(text_data)} embeddings={embeddings.shape[0]}"
            )

        LOGGER.info("Inserting %s documents into %s", len(text_data), collection_name)
        this_collection = client.collections.get(collection_name)
        with this_collection.batch.dynamic() as batch:
            # Loop through the data
            for pos, row in enumerate(
                tqdm(text_data.itertuples(index=False), total=len(text_data))
            ):

                # Build the object payload
                gnd_entity_obj = {
                    "label_ids": row.label_ids,
                    "label_texts": (
                        row.label_texts
                        if phrase is None
                        else f"{phrase}{row.label_texts}"
                    ),
                    "doc_text": row.text,
                    "doc_id": row.doc_id,
                }
                if self.debug:
                    LOGGER.info("Batch object: %s", gnd_entity_obj)
                    LOGGER.info("Vector: %s", embeddings[pos].tolist())

                # Add object to batch queue
                batch.add_object(
                    properties=gnd_entity_obj, vector=embeddings[pos].tolist()
                )

        # Check for failed objects
        failed_count = len(this_collection.batch.failed_objects)
        if failed_count > 0:
            LOGGER.warning("Failed to import %s objects", failed_count)
            LOGGER.warning("Example failed object: %s", this_collection.batch.failed_objects[0])

        return len(text_data), failed_count

    def gen_embeddings(self, texts):
        # Load the model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.embedding_model, trust_remote_code=True
        )
        model = AutoModel.from_pretrained(self.embedding_model, trust_remote_code=True)
        # device = "cpu"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move the model to the GPU and wrap it with DataParallel
        model = model.to(device)
        # if torch.cuda.device_count() > 1:
        #     print(f"Using {torch.cuda.device_count()} GPUs")
        #     model = torch.nn.DataParallel(model)
        model.eval()
        embeddings = []

        def cls_pooling(model_output):
            return model_output.last_hidden_state[:, 0]

        with torch.no_grad():
            for i in tqdm(
                range(0, len(texts), self.batch_size), desc="Generating embeddings"
            ):
                batch_texts = texts[i : i + self.batch_size]
                if self.debug:
                    LOGGER.info("Batch texts: %s", batch_texts)
                inputs = tokenizer(
                    batch_texts, padding=True, truncation=True, return_tensors="pt"
                ).to(device)
                outputs = model(**inputs)
                pooled_output = cls_pooling(outputs)
                norm = torch.norm(pooled_output, dim=1, keepdim=True)
                normalized_embeddings = pooled_output / norm
                embeddings.append(normalized_embeddings)
        return torch.cat(embeddings)

    def write_manifest(self, inserted_count, failed_count):
        """Persist run metadata so DVC can track this side-effect stage."""
        self.manifest_out.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "collection_name": self.collection_name,
            "embedding_model": self.embedding_model,
            "text_type": self.text_type,
            "rows_prepared": int(len(self.data)),
            "inserted_count": int(inserted_count),
            "failed_count": int(failed_count),
            "overwrite": bool(self.overwrite),
            "weaviate_port": int(self.weaviate_port),
        }
        with self.manifest_out.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        LOGGER.info("Wrote manifest to %s", self.manifest_out)

    def create_collection_docs(self):
        client, collection = self.create_collection(overwrite=self.overwrite)
        embeddings = self.gen_embeddings(self.data["text"].tolist())
        try:
            inserted_count, failed_count = self.insert_docs(
                client, collection, self.data, embeddings, phrase=None
            )
            self.write_manifest(inserted_count, failed_count)
        finally:
            client.close()


def execute():
    parser = ArgumentParser(
        description="Create a Weaviate collection for train retrieval documents."
    )
    parser.add_argument("--text_data", help="Preprocessed CSV input file", required=True)
    parser.add_argument("--collection_name", help="Collection name in Weaviate", required=True)
    parser.add_argument(
        "--text_type",
        help="Type of text in the file.",
        choices=["title", "ft"],
        required=True,
    )
    parser.add_argument(
        "--chunk_size",
        help="Chunk size of documents to send to Weaviate",
        required=False,
        type=int,
    )
    parser.add_argument(
        "--batch_size",
        help="Batch size for generating embeddings",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--embedding_model",
        help="Hugging Face model name for generating embeddings",
        default="BAAI/bge-m3",
    )
    parser.add_argument(
        "--max_n_docs",
        help="Maximum number of documents to process",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--weaviate_port",
        help="Port for local Weaviate instance",
        type=int,
        default=8087,
    )
    parser.add_argument(
        "--overwrite",
        help="Delete existing collection before creation",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--manifest_out",
        help="Path to write a DVC-tracked run manifest",
        default="logs/create_train_collection_manifest.json",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
    )

    args = parser.parse_args()
    validate_args(args, parser)
    log_file_path = Path("logs/create_train_collection.log")
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_file_path,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    LOGGER.info("Parsed args: %s", args)

    creator = CollectionCreator(
        args.text_data,
        args.collection_name,
        args.text_type,
        args.chunk_size,
        args.batch_size,
        args.embedding_model,
        args.overwrite,
        args.weaviate_port,
        args.manifest_out,
        args.max_n_docs,
        args.debug,
    )
    creator.create_collection_docs()


if __name__ == "__main__":
    execute()

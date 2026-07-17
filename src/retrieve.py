import argparse
import pandas as pd
import numpy as np
import requests
from weaviate.classes.query import MetadataQuery
import weaviate
from tqdm import tqdm


class ExampleRetriever:
    def __init__(
        self,
        input_text_data,
        output_file,
        n_examples,
        collection_name,
        host="8090",
        debug=False,
    ):
        self.data = pd.read_csv(input_text_data)
        
        required_columns = ["text", "doc_id", "label_ids", "label_texts"]
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        assert not missing_columns, (
            "Invalid dataset format. Missing columns: "
            + ", ".join(missing_columns)
            + ". Required columns are: text,doc_id,label_ids,label_texts"
        )
        
        self.output_file = output_file
        self.n_examples = n_examples
        self.collection_name = collection_name
        self.host = host
        self.debug = debug
        if self.debug:
            self.data = self.data.head(10)

    def retrieve_examples(self):
        weaviate_client = weaviate.connect_to_local(port=8087)
        chunks = weaviate_client.collections.get(self.collection_name)
        # Your code here to retrieve closest prompt examples using the provided arguments
        total_results = []
        for i, row in tqdm(self.data.iterrows()):
            # columns: text,doc_id,label_ids,label_texts

            embedding = list(
                np.array(
                    requests.post(
                        "http://127.0.0.1:{}/embed".format(self.host),
                        headers={"Content-Type": "application/json"},
                        json={"inputs": row["text"]},
                    ).json()
                ).reshape(-1)
            )
            response = chunks.query.near_vector(
                near_vector=embedding,
                limit=self.n_examples,
                include_vector=True,
                return_metadata=MetadataQuery(distance=True),
            )
            for resp in response.objects:
                if self.debug:
                    print(f"Response: {resp}")
                total_results.append(
                    {
                        "doc_id": row["doc_id"],
                        "text": row["text"],
                        "label_ids": row["label_ids"],
                        "label_texts": row["label_texts"],
                        "prompt_doc_id": resp.properties["doc_id"],
                        "prompt_text": resp.properties["doc_text"],
                        "prompt_labels": resp.properties["label_ids"],
                        "prompt_label_texts": resp.properties["label_texts"],
                        "similarity": resp.metadata.distance,
                    }
                )
            if self.debug:
                print(f"Processed {i + 1}/{len(self.data)} rows")
        weaviate_client.close()
        # Save the results to a CSV file
        results_df = pd.DataFrame(total_results)
        results_df.to_csv(self.output_file, index=False)
        print(f"Results saved to {self.output_file}")
        return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve closest prompt examples")

    parser.add_argument("--input_text_data", type=str, help="Path to the dataset file")
    parser.add_argument("--output_file", type=str, help="Path to the output file")
    parser.add_argument("--n_examples", type=int, help="Number of examples to retrieve")
    parser.add_argument("--collection_name", type=str, help="Name of the collection")
    parser.add_argument("--host", type=str, default="8090", help="Weaviate host")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    retriever = ExampleRetriever(
        args.input_text_data,
        args.output_file,
        args.n_examples,
        args.collection_name,
        args.host,
        args.debug,
    )
    retriever.retrieve_examples()

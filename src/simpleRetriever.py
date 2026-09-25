import argparse
import pandas as pd
import numpy as np
import requests
import unicodedata
from weaviate.classes.query import MetadataQuery
import weaviate
from tqdm import tqdm


class simpleRetriever:
    def __init__(
        self,
        input_text,
        n_examples,
        collection_name,
        tei_port="8090",
        weaviate_port="8087",
    ):
        # if input_text_data is a string, it is assumed to be a file path; 
        # otherwise, it is assumed to be a DataFrame
        self.input_text = input_text

        self.tei_port = tei_port
        self.weaviate_port = weaviate_port
        self.n_examples = n_examples
        self.collection_name = collection_name


    def retrieve_examples(self):
        weaviate_client = weaviate.connect_to_local(port=self.weaviate_port)
        chunks = weaviate_client.collections.get(self.collection_name)
        # Your code here to retrieve closest prompt examples using the provided arguments
        total_results = []
        normalized_text = unicodedata.normalize("NFC", self.input_text)

        embedding = list(
            np.array(
                requests.post(
                    "http://127.0.0.1:{}/embed".format(self.tei_port),
                    headers={"Content-Type": "application/json"},
                    json={"inputs": normalized_text},
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
            total_results.append(
                {
                    #"retrieved_id": resp.properties["doc_id"],
                    "retrieved_text": resp.properties["doc_text"],
                    #"retrieved_label_ids": resp.properties["label_ids"],
                    "retrieved_label_texts": resp.properties["label_texts"],
                    "distance": resp.metadata.distance,
                }
            )
        weaviate_client.close()
        # Save the results to a CSV file
        results_df = pd.DataFrame(total_results)
        # if self.output_file:
        #     results_df.to_csv(self.output_file, index=False)
        #     print(f"Results saved to {self.output_file}")
        return results_df

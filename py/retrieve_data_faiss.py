# Importing necessarry libraries and data
import numpy as np
from sentence_transformers import SentenceTransformer
from embeddings import embeddings, embedding_model
from faiss_vector_db import index
from time import perf_counter as timer
from dataset import access_logs_dict

def retrieve_relevant_resources(query: str,
                                embeddings: np.ndarray = embeddings,
                                model: SentenceTransformer = embedding_model,
                                n_resources_to_return: int = 5,
                                print_time: bool = True):
    """
    Embeds a query with model and returns top 'n_resources_to_return' distances and indices from embeddings.
    """

    # Embed the query
    query_embedding = model.encode(query, convert_to_tensor=True).cpu().numpy().astype("float32")

    if len(query_embedding.shape) == 1:
        query_embedding = query_embedding.reshape(1, -1)

    # Get the scores and time it
    start_time = timer()
    distances, indices = index.search(query_embedding, k = 5)
    end_time = timer()

    if print_time:
        print(f"[INFO] Time taken to get distances on {len(embeddings)} embeddings: {end_time - start_time:.5f} seconds.")

    return distances, indices

# A function to print the results
def print_top_results_and_scores(query: str,
                                 embeddings: np.ndarray = embeddings,
                                 access_logs_dict: list[dict] = access_logs_dict,
                                 n_resources_to_return: int = 5,
                                 print_time: bool = True):
    """
    Finds relevant passages given a query and prints them out along with their scores.
    """

    distances, indices = retrieve_relevant_resources(query=query,
                                                   embeddings = embeddings,
                                                   model=embedding_model,
                                                   n_resources_to_return=n_resources_to_return,
                                                   print_time=True)

    for distance, index in zip(distances[0], indices[0]):
        print(f"Distance : {distance:.4f}")
        print(f"LOG Context :")
        print(f"{access_logs_dict[index]['context']}")
        print(f"LOG ID : {index}")
        print(f"LOG IP : {access_logs_dict[index]['ip']}")
        print("********************************")
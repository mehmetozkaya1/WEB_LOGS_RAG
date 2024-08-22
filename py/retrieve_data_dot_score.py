# Importing necessary libraries and data
from time import perf_counter as timer
import numpy as np
from sentence_transformers import SentenceTransformer, util
from embeddings import embeddings, embedding_model
import torch
from dataset import access_logs_dict

def retrieve_relevant_resources_dot_score(query: str,
                                          embeddings: np.ndarray = embeddings,
                                          model: SentenceTransformer = embedding_model,
                                          n_resources_to_return = 5,
                                          print_time: bool = True):
    """
    Embeds a query with model and returns top 'n_resources_to_return' scores and indices from embeddings.
    """
    # Embed the query
    query_embedding = model.encode(query, convert_to_tensor = True)

    # Get dot scores
    start_time = timer()
    dot_scores = util.dot_score(a = query_embedding, b = embeddings)[0]
    end_time = timer()

    if print_time:
        print(f"[INFO] Time taken to get scores on {len(embeddings)} embeddings: {end_time - start_time:.5f} seconds.")

    whole = torch.topk(input = dot_scores,
                                k = n_resources_to_return),
  
    scores, indices = whole[0][0], whole[0][1]

    return scores, indices

def print_top_results_and_scores_dot_score(query: str,
                                 embeddings: torch.tensor,
                                 access_logs_dict: list[dict] = access_logs_dict,
                                 n_resources_to_return: int = 5,
                                 print_time: bool = True):
    """
    Finds relevant passages given a query and prints them out along with their scores.
    """
    scores, indices = retrieve_relevant_resources_dot_score(query = query, embeddings = embeddings, n_resources_to_return = n_resources_to_return, print_time = print_time)

    scores = scores.tolist()[0]
    indices = indices.tolist()[0]

    # Loop through zipped together scores and indices from torch.topk
    for score, idx in zip(scores, indices):
        print(f"Score : {score:.4f}")
        print(f"LOG Context :")
        print(f"{access_logs_dict[idx]['context']}")
        print(f"LOG ID : {idx}")
        print(f"LOG IP : {access_logs_dict[idx]['ip']}")
        print("********************************")
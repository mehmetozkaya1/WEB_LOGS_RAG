# Import necessary libraries and data
from sentence_transformers import SentenceTransformer
from dataset import access_logs_dict, dict_to_df
import torch
import numpy as np
import random
from tqdm.auto import tqdm

# Set the device and create the embedding model
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create the embedding model
embedding_model = SentenceTransformer(model_name_or_path = "all-mpnet-base-v2", device = device)

def embed_context(access_logs_dict):
    """
    A function to embed the context and add another 'embedding' key.
    """
    for item in tqdm(access_logs_dict):
        item["embedding"] = embedding_model.encode(item["context"])

def convert_np_array(access_logs_dict):
    """
    A function to convert embeddings into a numpy array.
    """
    for item in tqdm(access_logs_dict):
        item["embedding"] = np.array(item["embedding"])

def random_samples(access_logs_dict, num_examples):
    """
    A function to visualize some samples from the data.
    """
    examples = random.sample(access_logs_dict, k = num_examples)
    print(f"Random examples from the data :\n", examples)

embed_context(access_logs_dict)
convert_np_array(access_logs_dict)
print(random_samples(access_logs_dict, 1))
access_logs_dict_df = dict_to_df(access_logs_dict)

embeddings = torch.tensor(np.stack(access_logs_dict_df["embedding"].tolist(), axis=0), dtype = torch.float32).to("cuda").cpu().numpy().astype("float32")
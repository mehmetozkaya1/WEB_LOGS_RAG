# Import necessary libraries and data
import faiss
from embeddings import embeddings

# Creating global variables
VECTOR_DIM = embeddings.shape[1]

def to_cpu(embeddings):
    """
    A function to move embeddings from GPU to CPU.
    """
    embedding_vectors = embeddings

    return embedding_vectors

def create_faiss_index(embedding_vectors, vector_dimension):
    """
    A function to create a FAISS index.
    """
    index = faiss.IndexFlatL2(vector_dimension)
    index.add(embedding_vectors)
    print(f"Total number of vectors in the index is {index.ntotal}")

    return index

embedding_vectors = to_cpu(embeddings)
index = create_faiss_index(embedding_vectors, VECTOR_DIM)
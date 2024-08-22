# Importing necessary libraries
from dataset import drop_keys, access_logs_dict, DELETED_KEYS
from retrieve_data_faiss import retrieve_relevant_resources
from embeddings import embeddings
import textwrap
from llm import tokenizer, llm_model
from prompt_formatter import prompt, query
from prompt_formatter import prompt_formatter

# Drop unnecessarry keys
drop_keys(access_logs_dict, DELETED_KEYS)

def index_logs(access_logs_dict):
    """
    A function to index logs
    """
    indexed_logs = []

    for i, log in enumerate(access_logs_dict, start=1):
        indexed_log = {"Number": i, **log}
        indexed_logs.append(indexed_log)

    return indexed_logs

def print_wrapped(text, wrap_length = 80):
    """
    A function to wrap strings in the output.
    """
    wrapped_text = textwrap.fill(text, wrap_length)
    print(wrapped_text)

# Get relevant resources
def create_context(query, embeddings, n_resources_to_return):
    """
    A function to create context from the relevant resources
    """
    _, indices = retrieve_relevant_resources(query = query,
                                                embeddings = embeddings,
                                                n_resources_to_return = n_resources_to_return)

    retrieved_logs = []
    for idx in indices[0].tolist():
        retrieved_logs.append(access_logs_dict[idx])

    context = index_logs(retrieved_logs)
    return context

def ask(query: str,
        temperature: float = 0.7,
        max_new_tokens: int = 256,
        format_answer_text = True,
        return_answer_only = True):
    """
    Takes a query, finds relevant resources/content and generates an answer to the query based on the relevant resources
    """
    ## RETRIEVAL
    # Get just the scores and indices of top related results
    distances, indices = retrieve_relevant_resources(query = query,
                                                    embeddings = embeddings)

    # Create a list of context items
    contexts = [access_logs_dict[i] for i in indices[0].tolist()]
    context_items = index_logs(contexts)

    # Add distance to context item
    for i, item in enumerate(context_items):
        item["distance"] = distances[0][i]

    ## AUGMENTATION
    # Create the prompt and format it with context items
    prompt = prompt_formatter(query = query,
                                access_logs_list = context_items)
  
    ## GENERATION
    # Tokenize the prompt
    input_ids = tokenizer(prompt, return_tensors = "pt").to("cuda")

    # Generate an output of tokens
    output_tokens = llm_model.generate(**input_ids,
                                temperature = temperature,
                                do_sample = True,
                                max_new_tokens = max_new_tokens)
  
    # Decode the tokens into text
    output_text = tokenizer.decode(output_tokens[0])

    # Format the answer
    if format_answer_text:
        output_text = output_text.replace(prompt, '').replace("<bos>", '').replace("<eos>", '')

    # Only return the answer without context items
    if return_answer_only:
        return output_text

    return output_text, context_items

context = create_context(query = query, embeddings = embeddings, n_resources_to_return = 5)
print(f"Number of retrieved resources : {len(context)}")
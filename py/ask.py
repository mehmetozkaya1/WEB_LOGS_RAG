# Importing necessarry libraries and data
from llm import llm_model, tokenizer
from prompt_formatter import prompt, query
from augmentation import ask

def print_context_items(context_items: list[dict]):
    """
    A function to print context items
    """
    print("Context Items:\n")
    for item in context_items:
        print(item)

## Exampple output of llm
input_ids = tokenizer(prompt, return_tensors = "pt").to("cuda")

# Generate on output tokens
outputs = llm_model.generate(**input_ids,
                             temperature = 0.7, # from 0 to 1 and the lower the value, the more deterministic text, the higher the value, the more creative text.
                             do_sample = True, # whether or not to use sampling,
                             max_new_tokens = 512)

# Turn output tokens into texts
output_text = tokenizer.decode(outputs[0])
print(f"Query: {query}")
print(f"RAG answer:\n{output_text.replace(prompt, '').replace('<bos>', '').replace('<eos>', '')}")

## Example queries
query_list = [
    "Which logs viewed a product image using Chrome browser on Windows? What is the total size of the logs?",
    "Which logs accessed a CSS file using Chrome browser? What are the IP adresses of the logs?",
    "Which users pinged the site on a desktop device? What are the user agents of these logs?",
    "Which logs accessed an Instagram image on a mobile device on Android? What are the timestamps of these logs?",
    "Which logs viewed a product image on a mobile device? What are the ID numbers of the viewed products?",
    "Which logs using Firefox browser to access a product? What are the log indexes of the logs?"
]

for query in query_list:
    output_text, context_items = ask(query = query,
                        temperature = 0.7,
                        max_new_tokens = 256,
                        format_answer_text = True,
                        return_answer_only = False)
    print("Query : ", query)
    print_context_items(context_items)
    print("RAG answer :\n", output_text)
    print("***************************")
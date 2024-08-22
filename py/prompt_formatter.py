# Importing necessarry libraries and data
from dataset import extract_browser_and_os
from llm import tokenizer
from augmentation import context

# Example query
query = "Which logs viewed a product image using Chrome on Windows? What is the total size of the logs?"
print(f"Query: {query}")

def prompt_formatter(query: str,
                     access_logs_list: list[dict]) -> str:
    """
    A function to format our prompt.
    """

    formatted_logs = ',\n'.join(
        f"The LOG with the ID {log['Number']} belongs to a user with IP address {log['ip']} with request time {log['timestamp']} and size {log['size']}, {extract_browser_and_os(log['user_agent'])} as user agent. Context={log['context']}"
        for log in access_logs_list
    )

    base_prompt = f"""Based on the following web traffic logs, please answer the query.
    Give yourself room to think by extracting relevant passages from the logs before answering the query.
    Don't return the thinking, only return the answer.
    Make sure your answers are as explanatory as possible.
    Here are some examples:
    Example 1:
    User query: Which logs viewed a picture of a product using Chrome browser and what is the total request size of these logs?
    Answer: The logs with IDs 1,2,3,4 and 5 viewed a picture of a product using Chrome browser and the total request size is: 1024 + 54 + 65 + 12124 + 25 = 13292
    Example 2:
    Which logs used a mobile device to access the website using Firefox browser on Linux and what are their IP adresses?
    Answer: The logs used a mobile device to access the website using Firefox browser on Linux are the logs with IDs 1,2,3,4 and 5 and their IP adresses are:
    192.168.1.1
    254.256.32.15
    58.68.47.12
    56.98.2.4
    124.266.32.1
    \nNow use the following log entries to answer the user query:
    {formatted_logs}
    \nUser query: {query}
    Answer:"""

    # Create prompt template for instruction-tuned model
    dialogue_template = [
        {"role": "user",
        "content": base_prompt}
    ]

    # Apply the chat template
    prompt = tokenizer.apply_chat_template(conversation=dialogue_template,
                                            tokenize=False,
                                            add_generation_prompt=True)

    return prompt

# Format our prompt
prompt = prompt_formatter(query, context)
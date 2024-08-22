# Impporting necessarry libraries
import torch
from llm import llm_model, tokenizer, device

# Example : Input Text
INPUT_TEXT = "Which logs using GET method and loaded the page successfully and viewed a product image using Chrome on Linux?"

# Create the prompt template for instruction-tuned model
DIALOGUE_TEMPLATE = [
    {"role": "user",
     "content": INPUT_TEXT}
]

# A function to get number of model parameters
def get_model_num_params(model: torch.nn.Module):
    return sum([param.numel() for param in model.parameters()])

# A function to apply dialogue_template to the LLM model and generate a prompt for the model
def generate_prompt(tokenizer, template, input_text):
    prompt = tokenizer.apply_chat_template(conversation = template,
                                        tokenize = False,
                                        add_generation_prompt = True)

    return prompt

# A function to generate output with respect to the prompt
def generate_output_tokens(tokenizer, prompt, device, model):
    # Tokenize the input text (turn it into numbers and send it to the GPU)
    input_ids = tokenizer(prompt, return_tensors = "pt").to(device)

    # Generate outputs from local LLM
    output_tokens = model.generate(**input_ids,
                                max_new_tokens = 256)

    return output_tokens[0]

# A function to decode output tokens
def decode_output_tokens(tokenizer, output_tokens):
    output = tokenizer.decode(output_tokens)

    return output

num_params = get_model_num_params(model = llm_model)
print("LLM Model number of parameters : ", num_params)

prompt = generate_prompt(tokenizer = tokenizer,template = DIALOGUE_TEMPLATE, input_text = INPUT_TEXT)
print(f"Input text : \n", INPUT_TEXT)
print(f"Prompt (formatted):\n", prompt)

output_tokens = generate_output_tokens(tokenizer = tokenizer,
                                prompt = prompt,
                                device = device,
                                model = llm_model)

print(f"Input text : \n", INPUT_TEXT)
print(f"Model output (tokens): \n{output_tokens}")

output = decode_output_tokens(tokenizer = tokenizer,
                              output_tokens = output_tokens)

print(f"Input text : \n", INPUT_TEXT)
print(f"Model output (decoded): \n{output}")
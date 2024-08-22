# Importing necessarry libraries
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import is_flash_attn_2_available
from transformers import BitsAndBytesConfig

# Get device name
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Quantization config
quantization_config = BitsAndBytesConfig(load_in_4bit = True,
                                         bnb_4bit_compute_dtype = torch.float16)
use_quantization_config = False

# 2. Attention mechanism setup
if is_flash_attn_2_available() and torch.cuda.get_device_capability(0)[0] >= 8:
    attn_implementation = "flash_attention_2"
else:
    attn_implementation = "sdpa"

# 3. Model ID
model_id = "google/gemma-7b-it"

# 4. Instantiate tokenizer with token for gated model access
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = model_id, use_auth_token = True)

# 5. Instantiate model with quantization and memory settings
llm_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path = model_id, torch_dtype = torch.float16, quantization_config = quantization_config if use_quantization_config else None, low_cpu_mem_usage = False, use_auth_token = True)

# 6. Move to GPU if not using quantization
if not use_quantization_config:
    llm_model.to(device)

# (Optional) Check attention implementation and configure if possible
llm_model.config.attn_implementation = attn_implementation
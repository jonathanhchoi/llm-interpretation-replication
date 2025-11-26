print("Instruction-Tuned Models Comparison - Version 1.3.0") 

import sys
import io
from datetime import datetime
import os
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", message="Both `max_new_tokens` and `max_length`.*")
warnings.filterwarnings("ignore", message="`torch.backends.cuda.sdp_kernel.*")

# Define the base directory for all operations
BASE_DIR = "gdrive/My Drive/Computational/llm_interpretation/"
os.makedirs(BASE_DIR, exist_ok=True)

# Global variable to store all output
captured_output = []

def log_print(*args, **kwargs):
    """Function that both prints to console and saves to our global variable"""
    # Convert all arguments to strings and join them
    output_str = " ".join(str(arg) for arg in args)
    
    # Handle the 'end' parameter (default to newline if not specified)
    end = kwargs.get('end', '\n')
    output_str += end
    
    # Print to console
    print(*args, **kwargs)
    
    # Save to our global variable
    captured_output.append(output_str)

# Function to save captured output to file
def save_captured_output(filename):
    """Save all captured output to a file"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        f.writelines(captured_output)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
    __version__ as transformers_version
)
import gc
import psutil
import os
import shutil
from huggingface_hub import login
import logging
import contextlib
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd

def get_memory_usage():
    """Get current memory usage and disk space statistics"""
    process = psutil.Process()
    ram_usage = process.memory_info().rss / 1024 / 1024  # Convert to MB
    
    # Get disk space info
    total, used, free = shutil.disk_usage("/")
    disk_free_gb = free / (1024**3)
    
    if torch.cuda.is_available():
        gpu_usage = torch.cuda.memory_allocated() / 1024 / 1024  # Convert to MB
        gpu_cached = torch.cuda.memory_reserved() / 1024 / 1024  # Convert to MB
        return f"RAM: {ram_usage:.0f}MB, GPU Used: {gpu_usage:.0f}MB, GPU Cached: {gpu_cached:.0f}MB, Disk Free: {disk_free_gb:.1f}GB"
    return f"RAM: {ram_usage:.0f}MB, Disk Free: {disk_free_gb:.1f}GB"

def clear_memory(model_name=None):
    """Aggressively clear memory and disk cache"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # Triple GC collection for more thorough cleanup
    for _ in range(3):
        gc.collect()
        
    # Clear model cache if model_name is provided
    if model_name:
        try:
            cache_dir = os.path.expanduser(f"~/.cache/huggingface/hub/models--{model_name.replace('/', '--')}")
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir, ignore_errors=True)
                log_print(f"Cleared cache for {model_name}")
        except Exception as e:
            log_print(f"Warning: Could not clear cache for {model_name}: {e}")
    
    log_print(f"Memory after cleanup: {get_memory_usage()}")

def prepare_data_for_csv(outputs, models):
    """Prepare model outputs for CSV export"""
    data = []
    for model_name in models:
        # Extract model family name (e.g., 't5', 'llama', etc.)
        model_family = model_name.split('/')[1].split('-')[0].lower()
        
        for prompt in outputs.get(model_name, {}):
            result = outputs[model_name][prompt]
            data.append({
                'prompt': prompt,
                'model': model_name,
                'model_family': model_family,
                'model_output': result.get('completion', 'N/A'),
                'yes_prob': result.get('yes_prob', float('nan')),
                'no_prob': result.get('no_prob', float('nan')),
                'relative_prob': result.get('relative_prob', float('nan'))
            })
    return pd.DataFrame(data)

# Set up Hugging Face authentication
import os
from dotenv import load_dotenv

load_dotenv()
# Use provided HuggingFace token from environment variable
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

if HF_TOKEN:
    login(token=HF_TOKEN)
    print("Successfully logged in to HuggingFace")
else:
    print("Warning: HUGGINGFACE_TOKEN not found. Some models may not be accessible.")

device = "cuda" if torch.cuda.is_available() else "cpu"
log_print("Using device:", device)
log_print(f"Transformers version: {transformers_version}")

# Memory optimization settings
USE_8BIT = True  # Load models in 8-bit quantization
USE_4BIT = False  # Even more aggressive quantization (only works with some models)
LOW_CPU_MEM_USAGE = True  # Load model weights in chunks to reduce memory spikes
UNLOAD_UNUSED_MODELS = True  # Explicitly unload unused model components

# List of instruction-tuned models to compare
models = [
    "allenai/tk-instruct-3b-def",
    "baichuan-inc/Baichuan2-7B-Chat",
    "bigscience/bloomz-7b1",
    "bigscience/T0_3B",
    #"databricks/dolly-v2-7b",
    "facebook/opt-iml-1.3b",
    #"hakurei/instruct-12b",
    "h2oai/h2ogpt-oasst1-512-12b",
    #"lewtun/chip-12B-instruct-alpha",
    #"meta-llama/Llama-2-7b-chat-hf",
    #"mosaicml/mpt-7b-instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    #"OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5",
    "Qwen/Qwen-7B-Chat",
    #"stabilityai/stablelm-tuned-alpha-7b",
    "tiiuae/falcon-7b-instruct",
    #"togethercomputer/GPT-JT-6B-v1",
    "togethercomputer/RedPajama-INCITE-7B-Instruct",
    #"VMware/open-llama-7b-v2-open-instruct"
    #"THUDM/chatglm2-6b"
]

# Dictionary to store outputs
outputs = {}

def get_yes_no_logprobs(model, tokenizer, prompt, device):
    """Get log probabilities for 'Yes' and 'No' responses and generate completion"""
    MAX_LOOK_AHEAD = 10  # Maximum number of tokens to look ahead for Yes/No
    
    # Format input
    is_encoder_decoder = hasattr(model, 'encoder') and hasattr(model, 'decoder')
    
    if is_encoder_decoder:
        # For T5 models
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
        
        # Generate with scores
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=50,  # Use max_new_tokens instead of max_length
            num_return_sequences=1,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        # Get token IDs for Yes and No
        yes_token_id = tokenizer("Yes").input_ids[0]
        no_token_id = tokenizer("No").input_ids[0]
        
        # Look through each position's scores
        yes_no_found = False
        position_found = -1
        
        for pos, scores in enumerate(outputs.scores[:MAX_LOOK_AHEAD]):
            scores = scores[0]  # Get scores for first (only) sequence
            probs = torch.softmax(scores, dim=-1)
            
            # Get top token probabilities
            top_probs, top_tokens = torch.topk(probs, k=2)
            
            # Check if Yes or No is among top tokens
            if yes_token_id in top_tokens or no_token_id in top_tokens:
                yes_prob = probs[yes_token_id].item()
                no_prob = probs[no_token_id].item()
                yes_no_found = True
                position_found = pos
                break
        
        if not yes_no_found:
            # If no clear Yes/No found, use first position
            scores = outputs.scores[0][0]
            probs = torch.softmax(scores, dim=-1)
            yes_prob = probs[yes_token_id].item()
            no_prob = probs[no_token_id].item()
            position_found = 0
        
        # Decode the completion
        completion = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        
    else:
        # For decoder-only models
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        yes_tokens = tokenizer(" Yes").input_ids
        no_tokens = tokenizer(" No").input_ids
        yes_token_id = yes_tokens[0] if len(yes_tokens) > 0 else tokenizer.encode("Yes")[0]
        no_token_id = no_tokens[0] if len(no_tokens) > 0 else tokenizer.encode("No")[0]
        
        with torch.no_grad():
            # Generate completion
            completion_ids = model.generate(
                inputs.input_ids,
                max_new_tokens=50,  # Use max_new_tokens instead of max_length
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
            
            # Look through each position's scores
            yes_no_found = False
            position_found = -1
            
            # Get all logits for the sequence
            for pos, scores in enumerate(completion_ids.scores[:MAX_LOOK_AHEAD]):
                probs = torch.softmax(scores[0], dim=-1)
                
                # Get top token probabilities
                top_probs, top_tokens = torch.topk(probs, k=2)
                
                # Check if Yes or No is among top tokens
                if yes_token_id in top_tokens or no_token_id in top_tokens:
                    yes_prob = probs[yes_token_id].item()
                    no_prob = probs[no_token_id].item()
                    yes_no_found = True
                    position_found = pos
                    break
            
            if not yes_no_found:
                # If no clear Yes/No found, use first position
                scores = completion_ids.scores[0]
                probs = torch.softmax(scores[0], dim=-1)
                yes_prob = probs[yes_token_id].item()
                no_prob = probs[no_token_id].item()
                position_found = 0
            
            completion = tokenizer.decode(completion_ids.sequences[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    
    # Calculate log probabilities and relative probability
    yes_logprob = np.log(yes_prob) if yes_prob > 0 else float('-inf')
    no_logprob = np.log(no_prob) if no_prob > 0 else float('-inf')
    
    # Calculate relative probability: P(Yes) / (P(Yes) + P(No))
    relative_prob = yes_prob / (yes_prob + no_prob) if (yes_prob + no_prob) > 0 else float('nan')
    
    return {
        'yes_logprob': yes_logprob,
        'no_logprob': no_logprob,
        'yes_prob': yes_prob,
        'no_prob': no_prob,
        'relative_prob': relative_prob,
        'prompt': prompt,
        'completion': completion.strip(),
        'position_found': position_found,
        'yes_no_found': yes_no_found
    }

# List of questions to ask each model (Survey 2 - Part 2)
prompts = [
    'Is "typing" a form of "speech"?',
    'Is "charging" an electric vehicle "refueling" it?',
    'Is a "billboard" a "structure"?',
    'Is "littering" a form of "vandalism"?',
    'Is "memorizing" a type of "learning"?',
    'Is "paraphrasing" text "copying" it?',
    'Is a "vending machine" a "retailer"?',
    'Is a "windmill" a "machine"?',
    'Is "biodegradable plastic" an "organic material"?',
    'Is a "cave" a "structure"?',
    'Is a "canyon" a "valley"?',
    'Is "bartering" a type of "sale"?',
    'Is a "birdcage" a "habitat"?',
    'Is "chewing gum" a "food"?',
    'Is a "hammock" a "bed"?',
    'Is a "deepfake video" a "recording"?',
    'Is a "moat" a "barrier"?',
    'Is a "donation" a "payment"?',
    'Is a "bridge" a "road"?',
    'Is a "pond" a "lake"?',
    'Is a "hoverboard" a "vehicle"?',
    'Is "volunteering" a form of "donation"?',
    'Is a "voice memo" a "document"?',
    'Is a "podcast" a "publication"?',
    'Is a "meme" a "statement"?',
    'Is a "sundial" a type of "clock"?',
    'Is "stretching" a form of "exercise"?',
    'Is a "basement" a "story" of a building?',
    'Is a "virus" a "living organism"?',
    'Is "fasting" a form of "diet"?',
    'Is "composting" a form of "waste disposal"?',
    'Is a "garden gnome" a form of "art"?',
    'Is "silence" a form of "communication"?',
    'Is a "picnic table" an item of "furniture"?',
    'Is a "carport" a "building"?',
    'Is a "parking lot" a "facility"?',
    'Is a "raft" a type of "boat"?',
    'Is "wearing" a smartwatch "carrying" it?',
    'Is a "jigsaw puzzle" a "toy"?',
    'Is a "Fitbit" a "medical device"?',
    'Is a "livestream" an "event"?',
    'Is "virtual reality" a "place"?',
    'Is "hibernation" a form of "sleep"?',
    'Is "knitting" a form of "sewing"?',
    'Is a "smart thermostat" a "home appliance"?',
    'Is a "dock" a "structure"?',
    'Is a "skylight" a "window"?',
    'Is a "digital avatar" a "likeness"?',
    'Is a "chatroom" a "public place"?',
    'Is an "algorithmic recommendation" an "editorial decision"?'
]

#prompts = ["Is a 'screenshot' a 'photograph'?"]

# Context manager to disable HF progress bars
@contextlib.contextmanager
def suppress_progress_bars():
    """Context manager to suppress HuggingFace progress bar outputs"""
    # Disable logging for transformers
    log = logging.getLogger("transformers")
    log.setLevel(logging.ERROR)
    
    # Disable progress bars from tqdm
    original_tqdm = tqdm.__init__
    
    def silent_tqdm(*args, **kwargs):
        kwargs['disable'] = True
        return original_tqdm(*args, **kwargs)
    
    tqdm.__init__ = silent_tqdm
    
    try:
        yield
    finally:
        # Restore original tqdm
        tqdm.__init__ = original_tqdm
        # Restore logging level
        log.setLevel(logging.INFO)

# Process each model
for model_name in models:
    # Initialize nested dictionary for this model
    outputs[model_name] = {}
    
    log_print(f"\n=== Loading model: {model_name} ===")
    log_print(f"System state before loading: {get_memory_usage()}")
    model = None  # Initialize model as None
    try:
        # Skip MPT models if transformers version is incompatible
        if "mpt" in model_name.lower() and transformers_version < "4.33.0":
            log_print(f"Warning: MPT models require transformers >= 4.33.0, current version: {transformers_version}")
            log_print(f"Attempting to load {model_name} anyway, but it may fail.")

        # Check for xformers availability
        try:
            import xformers
            has_xformers = True
            log_print("Xformers is available and will be used for memory-efficient attention")
        except ImportError:
            has_xformers = False
            log_print("Xformers is not available. For better performance, consider installing it with: pip install xformers")

        # Load tokenizer and model with progress bars suppressed
        with suppress_progress_bars():
            # Load tokenizer with minimal memory usage
            tokenizer_kwargs = {
                "use_fast": True,
                "trust_remote_code": True,
                "low_cpu_mem_usage": LOW_CPU_MEM_USAGE
            }
            
            # Special handling for different model types
            if "xgen" in model_name.lower():
                # XGen models need special handling
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    low_cpu_mem_usage=LOW_CPU_MEM_USAGE
                )
            elif "chatglm" in model_name.lower():
                # ChatGLM models need special handling
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
            elif "baichuan" in model_name.lower():
                # Baichuan models need special handling
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    use_fast=False  # Use slow tokenizer for Baichuan
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    **tokenizer_kwargs
                )

            # Ensure tokenizer has pad_token
            if tokenizer.pad_token is None:
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    tokenizer.pad_token = tokenizer.unk_token

            # Common model loading arguments
            model_kwargs = {
                "device_map": "auto",
                "trust_remote_code": True,
                "low_cpu_mem_usage": LOW_CPU_MEM_USAGE
            }

            # Add quantization config if enabled
            if USE_8BIT:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True
                )
            elif USE_4BIT:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
            else:
                model_kwargs["torch_dtype"] = torch.float16

            # Special handling for Baichuan models
            if "baichuan" in model_name.lower():
                model_kwargs["use_cache"] = True
                # Baichuan models may need additional parameters
                if has_xformers:
                    model_kwargs["attn_implementation"] = "flash_attention_2" if torch.cuda.is_available() else "eager"
                
            # For T5-like models, use AutoModelForSeq2SeqLM
            if "t5" in model_name.lower() or "t0" in model_name.lower() or "tk-instruct" in model_name.lower():
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    **model_kwargs
                )
            else:
                # For most decoder-only LLMs
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    **model_kwargs
                )

        # Process each prompt
        for prompt in prompts:
            log_print(f"\nProcessing prompt: {prompt}")
            
            # Format prompt for instruction models - direct question without few-shot examples
            formatted_prompt = f"{prompt} Answer either 'Yes' or 'No', without any other text."
            
            # Special handling for Baichuan models
            if "baichuan" in model_name.lower():
                formatted_prompt = f"<human>: {prompt} Answer either 'Yes' or 'No', without any other text.\n<bot>:"
            
            log_print(f"Formatted prompt: {formatted_prompt}")
            
            # Get log probabilities
            try:
                logprobs = get_yes_no_logprobs(model, tokenizer, formatted_prompt, device)
                outputs[model_name][prompt] = logprobs
                log_print(f"Model completion: {logprobs['completion']}")
                log_print(f"Yes prob: {logprobs['yes_prob']:.3f}, No prob: {logprobs['no_prob']:.3f}, Relative prob: {logprobs['relative_prob']:.3f}")
            except Exception as e:
                log_print(f"Error getting logprobs: {e}")
                outputs[model_name][prompt] = {
                    'yes_prob': float('nan'),
                    'no_prob': float('nan'),
                    'relative_prob': float('nan'),
                    'prompt': formatted_prompt,  # Store the formatted prompt even if we get an error
                    'completion': 'N/A'
                }

    except Exception as e:
        log_print(f"Error running {model_name}: {e}")
        # Store error for all prompts
        for prompt in prompts:
            outputs[model_name][prompt] = {
                'yes_prob': float('nan'),
                'no_prob': float('nan'),
                'relative_prob': float('nan'),
                'prompt': prompt,  # Store the prompt even if we get an error
                'completion': 'N/A'
            }

    finally:
        # Aggressive cleanup
        if model is not None:
            model.cpu()  # Move model to CPU first
            del model
            if UNLOAD_UNUSED_MODELS:
                del tokenizer
                if 'output_ids' in locals():
                    del output_ids
                if 'input_ids' in locals():
                    del input_ids
                clear_memory(model_name)  # Pass model_name for cache cleanup
                log_print(f"System state after model unloading: {get_memory_usage()}")

# Save results to CSV
log_print("\nSaving results to CSV file...")
df = prepare_data_for_csv(outputs, models)
csv_filename = os.path.join(BASE_DIR, "instruct_model_comparison_results_survey2.csv")
df.to_csv(csv_filename, index=False)
log_print(f"Results saved to {csv_filename}")

# Print final summary
log_print("\n\n============= Final Model Responses =============\n")
for model_name in models:
    log_print(f"\nModel: {model_name}")
    log_print("=" * 80 + "\n")
    
    for prompt in prompts:
        model_resp = outputs.get(model_name, {}).get(prompt, {})
        log_print(f"Question: {prompt}")
        log_print(f"Input prompt: {model_resp.get('prompt', 'N/A')}")
        log_print(f"Model completion: {model_resp.get('completion', 'N/A')}")
        log_print(f"Yes prob: {model_resp.get('yes_prob', 'N/A'):.3f}, "
              f"No prob: {model_resp.get('no_prob', 'N/A'):.3f}, "
              f"Relative prob: {model_resp.get('relative_prob', 'N/A'):.3f}")
        log_print(f"Found Yes/No at position: {model_resp.get('position_found', 'N/A')}, "
              f"Clear Yes/No found: {model_resp.get('yes_no_found', 'N/A')}")
        log_print("-" * 80 + "\n")

# Save the captured output to a file
output_file = os.path.join(BASE_DIR, "instruct_model_comparison_output.txt")
save_captured_output(output_file)
log_print(f"\nComplete output log has been saved to: {output_file}") 
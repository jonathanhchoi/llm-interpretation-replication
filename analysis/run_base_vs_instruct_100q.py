"""
Run Base vs Instruction-Tuned Models on All 100 Questions

This script runs both base and instruction-tuned versions of multiple LLM families
on the full set of 100 questions (Survey 1 + Survey 2) and calculates MAE differences.

Designed to run on Google Colab with GPU support.

Usage:
    1. Upload to Google Colab
    2. Set your HuggingFace token in the HF_TOKEN variable
    3. Run all cells

Features:
    - Checkpointing: saves progress after each model to resume if interrupted
    - Memory management: clears GPU memory between models
    - 8-bit quantization: reduces memory requirements
    - Automatic analysis: calculates MAE with bootstrap CI and p-values after running
"""

print("=" * 80)
print("BASE vs INSTRUCTION-TUNED MODEL COMPARISON")
print("Running on 100 Questions (Survey 1 + Survey 2)")
print("=" * 80)

# =============================================================================
# SETUP AND IMPORTS
# =============================================================================

import sys
import os
from datetime import datetime

# Install required packages if running on Colab
try:
    import google.colab
    IN_COLAB = True
    print("Running in Google Colab")
    # Install required packages
    os.system("pip install -q transformers accelerate bitsandbytes scipy")
except ImportError:
    IN_COLAB = False
    print("Running locally")

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
    __version__ as transformers_version
)
import gc
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# HuggingFace Token - SET THIS!
# Get your token from: https://huggingface.co/settings/tokens
HF_TOKEN = ""  # <-- PUT YOUR TOKEN HERE

# Output directory (Google Drive path for Colab)
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    OUTPUT_DIR = "/content/drive/MyDrive/Computational/llm_interpretation"
else:
    OUTPUT_DIR = "G:/My Drive/Computational/llm_interpretation"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Checkpoint file for resuming interrupted runs
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "base_vs_instruct_100q_checkpoint.json")
RESULTS_FILE = os.path.join(OUTPUT_DIR, "base_vs_instruct_100q_results.csv")

# Memory settings
USE_8BIT = True  # 8-bit quantization (recommended for 7B models on Colab)
USE_4BIT = False  # 4-bit quantization (more aggressive, may reduce quality)

# Model pairs to run: (base_model, instruct_model, family_name)
MODEL_PAIRS = [
    # Falcon - KNOWN TO WORK WELL
    ("tiiuae/falcon-7b", "tiiuae/falcon-7b-instruct", "Falcon"),

    # StableLM - KNOWN TO WORK WELL
    ("stabilityai/stablelm-base-alpha-7b", "stabilityai/stablelm-tuned-alpha-7b", "StableLM"),

    # RedPajama - KNOWN TO WORK WELL
    ("togethercomputer/RedPajama-INCITE-7B-Base", "togethercomputer/RedPajama-INCITE-7B-Instruct", "RedPajama"),

    # BLOOM/BLOOMZ - should work
    ("bigscience/bloom-7b1", "bigscience/bloomz-7b1", "BLOOM"),

    # Pythia/Dolly - should work (different architectures but comparable)
    ("EleutherAI/pythia-6.9b", "databricks/dolly-v2-7b", "Pythia-Dolly"),

    # Mistral - needs testing (had issues with equal probabilities before)
    ("mistralai/Mistral-7B-v0.1", "mistralai/Mistral-7B-Instruct-v0.2", "Mistral"),

    # LLaMA-2 - requires license acceptance on HuggingFace
    # ("meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-7b-chat-hf", "LLaMA-2"),

    # Gemma - Google's newer model
    # ("google/gemma-7b", "google/gemma-7b-it", "Gemma"),

    # Phi-2 - Microsoft's small but capable model (2.7B)
    # ("microsoft/phi-2", "microsoft/phi-2", "Phi-2"),  # No instruct version
]

# =============================================================================
# ALL 100 QUESTIONS (Survey 1 + Survey 2)
# =============================================================================

# Survey 1: 50 questions (original)
SURVEY1_PROMPTS = [
    'Is a "screenshot" a "photograph"?',
    'Is "advising" someone "instructing" them?',
    'Is an "algorithm" a "procedure"?',
    'Is a "drone" an "aircraft"?',
    'Is "reading aloud" a form of "performance"?',
    'Is "training" an AI model "authoring" content?',
    'Is a "wedding" a "party"?',
    'Is "streaming" a video "broadcasting" that video?',
    'Is "braiding" hair a form of "weaving"?',
    'Is "digging" a form of "construction"?',
    'Is a "smartphone" a "computer"?',
    'Is a "cactus" a "tree"?',
    'Is a "bonus" a form of "wages"?',
    'Is "forwarding" an email "sending" that email?',
    'Is a "chatbot" a "service"?',
    'Is "plagiarism" a form of "theft"?',
    'Is "remote viewing" of an event "attending" it?',
    'Is "whistling" a form of "music"?',
    'Is "caching" data in computer memory "storing" that data?',
    'Is a "waterway" a form of "roadway"?',
    'Is a "deepfake" a "portrait"?',
    'Is "humming" a form of "singing"?',
    'Is "liking" a social media post "endorsing" it?',
    'Is "herding" animals a form of "transporting" them?',
    'Is an "NFT" a "security"?',
    'Is "sleeping" an "activity"?',
    'Is a "driverless car" a "motor vehicle operator"?',
    'Is a "subscription fee" a form of "purchase"?',
    'Is "mentoring" someone a form of "supervising" them?',
    'Is a "biometric scan" a form of "signature"?',
    'Is a "digital wallet" a "bank account"?',
    'Is "dictation" a form of "writing"?',
    'Is a "virtual tour" a form of "inspection"?',
    'Is "bartering" a form of "payment"?',
    'Is "listening" to an audiobook "reading" it?',
    'Is a "nest" a form of "dwelling"?',
    'Is a "QR code" a "document"?',
    'Is a "tent" a "building"?',
    'Is a "whisper" a form of "speech"?',
    'Is "hiking" a form of "travel"?',
    'Is a "recipe" a form of "instruction"?',
    'Is "daydreaming" a form of "thinking"?',
    'Is "gossip" a form of "news"?',
    'Is a "mountain" a form of "hill"?',
    'Is "walking" a form of "exercise"?',
    'Is a "candle" a "lamp"?',
    'Is a "trail" a "road"?',
    'Is "repainting" a house "repairing" it?',
    'Is "kneeling" a form of "sitting"?',
    'Is a "mask" a form of "clothing"?'
]

# Survey 2: 50 questions (part 2)
SURVEY2_PROMPTS = [
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

# Combine all prompts
ALL_PROMPTS = SURVEY1_PROMPTS + SURVEY2_PROMPTS
print(f"\nTotal questions: {len(ALL_PROMPTS)}")
print(f"  Survey 1: {len(SURVEY1_PROMPTS)}")
print(f"  Survey 2: {len(SURVEY2_PROMPTS)}")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def log_print(*args, **kwargs):
    """Print and flush immediately"""
    print(*args, **kwargs)
    sys.stdout.flush()


def get_memory_usage():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return f"GPU: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved"
    return "No GPU"


def clear_memory(model_name=None):
    """Clear GPU memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    for _ in range(3):
        gc.collect()
    log_print(f"Memory after cleanup: {get_memory_usage()}")


def load_checkpoint():
    """Load checkpoint if it exists"""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {'completed_models': [], 'results': []}


def save_checkpoint(checkpoint):
    """Save checkpoint"""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)


def get_yes_no_logprobs(model, tokenizer, prompt, device):
    """
    Get log probabilities for 'Yes' and 'No' responses.

    Returns dict with yes_prob, no_prob, relative_prob, completion
    """
    MAX_LOOK_AHEAD = 10

    is_encoder_decoder = hasattr(model, 'encoder') and hasattr(model, 'decoder')

    try:
        if is_encoder_decoder:
            # For T5-style models
            inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=50,
                    num_return_sequences=1,
                    output_scores=True,
                    return_dict_in_generate=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            yes_token_id = tokenizer("Yes").input_ids[0]
            no_token_id = tokenizer("No").input_ids[0]

            # Find Yes/No in output
            yes_no_found = False
            for pos, scores in enumerate(outputs.scores[:MAX_LOOK_AHEAD]):
                probs = torch.softmax(scores[0], dim=-1)
                top_probs, top_tokens = torch.topk(probs, k=5)

                if yes_token_id in top_tokens or no_token_id in top_tokens:
                    yes_prob = probs[yes_token_id].item()
                    no_prob = probs[no_token_id].item()
                    yes_no_found = True
                    break

            if not yes_no_found:
                probs = torch.softmax(outputs.scores[0][0], dim=-1)
                yes_prob = probs[yes_token_id].item()
                no_prob = probs[no_token_id].item()

            completion = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

        else:
            # For decoder-only models
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            # Get token IDs for Yes and No (with leading space for decoder models)
            yes_tokens = tokenizer(" Yes", add_special_tokens=False).input_ids
            no_tokens = tokenizer(" No", add_special_tokens=False).input_ids
            yes_token_id = yes_tokens[0] if yes_tokens else tokenizer.encode("Yes")[0]
            no_token_id = no_tokens[0] if no_tokens else tokenizer.encode("No")[0]

            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=50,
                    num_return_sequences=1,
                    output_scores=True,
                    return_dict_in_generate=True,
                    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            # Find Yes/No in output
            yes_no_found = False
            for pos, scores in enumerate(outputs.scores[:MAX_LOOK_AHEAD]):
                probs = torch.softmax(scores[0], dim=-1)
                top_probs, top_tokens = torch.topk(probs, k=5)

                if yes_token_id in top_tokens or no_token_id in top_tokens:
                    yes_prob = probs[yes_token_id].item()
                    no_prob = probs[no_token_id].item()
                    yes_no_found = True
                    break

            if not yes_no_found:
                probs = torch.softmax(outputs.scores[0][0], dim=-1)
                yes_prob = probs[yes_token_id].item()
                no_prob = probs[no_token_id].item()

            completion = tokenizer.decode(
                outputs.sequences[0][len(inputs.input_ids[0]):],
                skip_special_tokens=True
            )

        # Calculate relative probability
        if yes_prob + no_prob > 0:
            relative_prob = yes_prob / (yes_prob + no_prob)
        else:
            relative_prob = 0.5

        return {
            'yes_prob': yes_prob,
            'no_prob': no_prob,
            'relative_prob': relative_prob,
            'completion': completion.strip()[:100],  # Truncate long completions
            'success': True
        }

    except Exception as e:
        log_print(f"Error in get_yes_no_logprobs: {e}")
        return {
            'yes_prob': float('nan'),
            'no_prob': float('nan'),
            'relative_prob': float('nan'),
            'completion': f'ERROR: {str(e)[:50]}',
            'success': False
        }


def run_model_on_prompts(model_name, prompts, is_base_model=True):
    """
    Run a single model on all prompts.

    Args:
        model_name: HuggingFace model name
        prompts: List of prompts to run
        is_base_model: Whether this is a base model (affects prompt formatting)

    Returns:
        List of result dicts
    """
    log_print(f"\n{'='*60}")
    log_print(f"Loading model: {model_name}")
    log_print(f"Memory before: {get_memory_usage()}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []

    try:
        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=True
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Model loading kwargs
        model_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        # Quantization
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

        # Load model
        if "t5" in model_name.lower() or "t0" in model_name.lower():
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **model_kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        log_print(f"Model loaded. Memory: {get_memory_usage()}")

        # Few-shot examples for base models
        few_shot = '''Question: Is "soup" a "beverage"? Answer either 'Yes' or 'No', without any other text.
Answer: No.

Question: Is a "tweet" a "publication"? Answer either 'Yes' or 'No', without any other text.
Answer: Yes.

'''

        # Run on all prompts
        for i, prompt in enumerate(tqdm(prompts, desc=model_name.split('/')[-1])):
            # Format prompt based on model type
            if is_base_model:
                formatted = f"{few_shot}Question: {prompt} Answer either 'Yes' or 'No', without any other text.\nAnswer:"
            else:
                formatted = f"{prompt} Answer either 'Yes' or 'No', without any other text."

            # Get response
            result = get_yes_no_logprobs(model, tokenizer, formatted, device)
            result['prompt'] = prompt
            result['model'] = model_name
            result['formatted_prompt'] = formatted[:200]  # Truncate for storage
            results.append(result)

            # Print progress every 10 prompts
            if (i + 1) % 10 == 0:
                log_print(f"  Completed {i+1}/{len(prompts)}")

        log_print(f"Finished {model_name}. Processed {len(results)} prompts.")

    except Exception as e:
        log_print(f"ERROR loading/running {model_name}: {e}")
        # Return empty results for failed model
        for prompt in prompts:
            results.append({
                'prompt': prompt,
                'model': model_name,
                'yes_prob': float('nan'),
                'no_prob': float('nan'),
                'relative_prob': float('nan'),
                'completion': f'MODEL_ERROR: {str(e)[:50]}',
                'success': False
            })

    finally:
        # Cleanup
        try:
            del model
            del tokenizer
        except:
            pass
        clear_memory(model_name)

    return results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    # Login to HuggingFace
    if HF_TOKEN:
        from huggingface_hub import login
        login(token=HF_TOKEN)
        log_print("Logged in to HuggingFace")
    else:
        log_print("WARNING: No HF_TOKEN set. Some models may not be accessible.")

    # Check GPU
    log_print(f"\nDevice: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        log_print(f"GPU: {torch.cuda.get_device_name(0)}")
        log_print(f"Memory: {get_memory_usage()}")

    # Load checkpoint
    checkpoint = load_checkpoint()
    completed = set(checkpoint['completed_models'])
    all_results = checkpoint['results']

    if completed:
        log_print(f"\nResuming from checkpoint. Already completed: {completed}")

    # Run each model pair
    for base_model, instruct_model, family_name in MODEL_PAIRS:
        log_print(f"\n{'#'*60}")
        log_print(f"MODEL FAMILY: {family_name}")
        log_print(f"{'#'*60}")

        # Run base model
        if base_model not in completed:
            log_print(f"\nRunning BASE model: {base_model}")
            results = run_model_on_prompts(base_model, ALL_PROMPTS, is_base_model=True)
            for r in results:
                r['model_family'] = family_name
                r['base_or_instruct'] = 'base'
            all_results.extend(results)

            # Save checkpoint
            checkpoint['completed_models'].append(base_model)
            checkpoint['results'] = all_results
            save_checkpoint(checkpoint)
            log_print(f"Checkpoint saved after {base_model}")
        else:
            log_print(f"Skipping {base_model} (already completed)")

        # Run instruct model
        if instruct_model not in completed:
            log_print(f"\nRunning INSTRUCT model: {instruct_model}")
            results = run_model_on_prompts(instruct_model, ALL_PROMPTS, is_base_model=False)
            for r in results:
                r['model_family'] = family_name
                r['base_or_instruct'] = 'instruct'
            all_results.extend(results)

            # Save checkpoint
            checkpoint['completed_models'].append(instruct_model)
            checkpoint['results'] = all_results
            save_checkpoint(checkpoint)
            log_print(f"Checkpoint saved after {instruct_model}")
        else:
            log_print(f"Skipping {instruct_model} (already completed)")

    # Save final results to CSV
    log_print(f"\n{'='*60}")
    log_print("SAVING RESULTS")
    log_print(f"{'='*60}")

    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_FILE, index=False)
    log_print(f"Results saved to: {RESULTS_FILE}")
    log_print(f"Total rows: {len(df)}")

    # Print summary
    log_print("\nSUMMARY:")
    for family in df['model_family'].unique():
        family_df = df[df['model_family'] == family]
        base_df = family_df[family_df['base_or_instruct'] == 'base']
        inst_df = family_df[family_df['base_or_instruct'] == 'instruct']

        base_valid = base_df['relative_prob'].notna().sum()
        inst_valid = inst_df['relative_prob'].notna().sum()

        log_print(f"  {family}: Base={base_valid}/{len(base_df)} valid, Instruct={inst_valid}/{len(inst_df)} valid")

    return df


# =============================================================================
# RUN ANALYSIS
# =============================================================================

def analyze_results(df):
    """
    Analyze MAE differences between base and instruct models.
    """
    log_print(f"\n{'='*60}")
    log_print("ANALYSIS: MAE Differences")
    log_print(f"{'='*60}")

    # Bootstrap parameters
    N_BOOTSTRAP = 10000
    np.random.seed(42)

    results = {}
    all_family_errors = {}

    for family in df['model_family'].unique():
        family_df = df[df['model_family'] == family]
        base_df = family_df[family_df['base_or_instruct'] == 'base']
        inst_df = family_df[family_df['base_or_instruct'] == 'instruct']

        # Merge on prompt to get paired data
        merged = pd.merge(
            base_df[['prompt', 'relative_prob']],
            inst_df[['prompt', 'relative_prob']],
            on='prompt',
            suffixes=('_base', '_instruct')
        ).dropna()

        if len(merged) < 10:
            log_print(f"{family}: Insufficient data ({len(merged)} valid pairs)")
            continue

        # Calculate per-prompt absolute differences
        # Note: This is the difference in model predictions, not MAE vs human
        base_probs = merged['relative_prob_base'].values
        inst_probs = merged['relative_prob_instruct'].values
        diffs = inst_probs - base_probs  # Positive = instruct higher

        # Store for overall analysis
        all_family_errors[family] = {
            'base': base_probs,
            'instruct': inst_probs,
            'diffs': diffs
        }

        # Bootstrap the mean difference
        obs_mean_diff = np.mean(diffs)
        obs_mae = np.mean(np.abs(diffs))

        boot_diffs = []
        for _ in range(N_BOOTSTRAP):
            idx = np.random.choice(len(diffs), size=len(diffs), replace=True)
            boot_diffs.append(np.mean(diffs[idx]))

        boot_diffs = np.array(boot_diffs)
        ci_lower = np.percentile(boot_diffs, 2.5)
        ci_upper = np.percentile(boot_diffs, 97.5)

        # P-value
        if obs_mean_diff > 0:
            p_value = 2 * np.mean(boot_diffs <= 0)
        else:
            p_value = 2 * np.mean(boot_diffs >= 0)
        p_value = min(p_value, 1.0)

        results[family] = {
            'n_prompts': len(merged),
            'mean_diff': obs_mean_diff,
            'mae': obs_mae,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_value': p_value
        }

        stars = '***' if p_value < 0.01 else ('**' if p_value < 0.05 else ('*' if p_value < 0.1 else ''))
        log_print(f"{family}: Mean Diff = {obs_mean_diff:+.3f} [{ci_lower:+.3f}, {ci_upper:+.3f}] {stars} (n={len(merged)})")

    # Overall analysis
    if len(all_family_errors) >= 2:
        log_print(f"\nOVERALL (across {len(all_family_errors)} model families):")

        all_diffs = np.concatenate([v['diffs'] for v in all_family_errors.values()])
        obs_overall_mean = np.mean(all_diffs)
        obs_overall_mae = np.mean(np.abs(all_diffs))

        # Bootstrap overall
        boot_overall = []
        for _ in range(N_BOOTSTRAP):
            idx = np.random.choice(len(all_diffs), size=len(all_diffs), replace=True)
            boot_overall.append(np.mean(all_diffs[idx]))

        boot_overall = np.array(boot_overall)
        ci_lower = np.percentile(boot_overall, 2.5)
        ci_upper = np.percentile(boot_overall, 97.5)

        if obs_overall_mean > 0:
            p_value = 2 * np.mean(boot_overall <= 0)
        else:
            p_value = 2 * np.mean(boot_overall >= 0)
        p_value = min(p_value, 1.0)

        stars = '***' if p_value < 0.01 else ('**' if p_value < 0.05 else ('*' if p_value < 0.1 else ''))
        log_print(f"  Mean Diff = {obs_overall_mean:+.3f} [{ci_lower:+.3f}, {ci_upper:+.3f}] {stars}")
        log_print(f"  MAE = {obs_overall_mae:.3f}")
        log_print(f"  P-value = {p_value:.4f}")

    return results


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Run models
    df = main()

    # Analyze results
    if df is not None and len(df) > 0:
        analysis = analyze_results(df)

        log_print("\n" + "="*60)
        log_print("COMPLETE!")
        log_print("="*60)
        log_print(f"Results saved to: {RESULTS_FILE}")
        log_print(f"Checkpoint file: {CHECKPOINT_FILE}")

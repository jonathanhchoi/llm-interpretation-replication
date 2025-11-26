"""
Calculate similarity metrics between perturbed prompts.

This script loads existing perturbed prompts and calculates various similarity
metrics including:
- Embedding cosine similarity
- Levenshtein distance
- BM25 similarity
- TF-IDF cosine similarity
"""

import json
import os
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import combinations

# Similarity metrics
from Levenshtein import distance as levenshtein_distance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

# Optional: sentence-transformers for embeddings (will gracefully degrade if not available)
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Embedding similarity will be skipped.")

def load_perturbations(file_path: str = "data/perturbations.json") -> List[Dict]:
    """Load perturbation data from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def normalize_levenshtein(text1: str, text2: str) -> float:
    """
    Calculate normalized Levenshtein distance (0 = identical, 1 = completely different).
    Returns similarity score (1 - normalized_distance).
    """
    max_len = max(len(text1), len(text2))
    if max_len == 0:
        return 1.0
    dist = levenshtein_distance(text1, text2)
    normalized_dist = dist / max_len
    return 1.0 - normalized_dist  # Convert distance to similarity

def calculate_embedding_similarity(
    texts: List[str],
    model
) -> np.ndarray:
    """
    Calculate pairwise cosine similarity using sentence embeddings.
    Returns a symmetric matrix of similarities.
    """
    if not EMBEDDINGS_AVAILABLE or model is None:
        # Return identity matrix if embeddings not available
        return np.eye(len(texts))
    embeddings = model.encode(texts, show_progress_bar=False)
    similarity_matrix = cosine_similarity(embeddings)
    return similarity_matrix

def calculate_tfidf_similarity(texts: List[str]) -> Tuple[np.ndarray, TfidfVectorizer]:
    """
    Calculate pairwise TF-IDF cosine similarity.
    Returns similarity matrix and the fitted vectorizer.
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return similarity_matrix, vectorizer

def calculate_bm25_similarity(texts: List[str]) -> np.ndarray:
    """
    Calculate BM25 similarity scores between texts.
    Returns a similarity matrix.
    """
    # Tokenize texts for BM25
    tokenized_texts = [text.lower().split() for text in texts]
    bm25 = BM25Okapi(tokenized_texts)
    
    # Calculate BM25 scores for each text against all others
    similarity_matrix = np.zeros((len(texts), len(texts)))
    for i, text in enumerate(tokenized_texts):
        scores = bm25.get_scores(text)
        # Normalize scores to [0, 1] range for comparison
        max_score = scores.max() if scores.max() > 0 else 1.0
        similarity_matrix[i] = scores / max_score
    
    # Make symmetric (average with transpose)
    similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2
    
    return similarity_matrix

def calculate_all_similarities(
    original: str,
    rephrasings: List[str],
    embedding_model=None
) -> Dict:
    """
    Calculate all similarity metrics for a set of prompts.
    
    Returns a dictionary with:
    - original_vs_rephrasings: List of dicts with similarities between original and each rephrasing
    - pairwise_rephrasings: List of dicts with pairwise similarities between rephrasings
    - summary_stats: Summary statistics for each metric
    """
    all_texts = [original] + rephrasings
    
    # Calculate all similarity matrices
    if EMBEDDINGS_AVAILABLE and embedding_model is not None:
        print("  Calculating embedding similarities...")
        embedding_sim = calculate_embedding_similarity(all_texts, embedding_model)
    else:
        print("  Skipping embedding similarities (not available)...")
        embedding_sim = np.eye(len(all_texts))  # Identity matrix as placeholder
    
    print("  Calculating TF-IDF similarities...")
    tfidf_sim, _ = calculate_tfidf_similarity(all_texts)
    
    print("  Calculating BM25 similarities...")
    bm25_sim = calculate_bm25_similarity(all_texts)
    
    print("  Calculating Levenshtein distances...")
    # Levenshtein is computed pairwise
    levenshtein_sim = np.zeros((len(all_texts), len(all_texts)))
    for i in range(len(all_texts)):
        for j in range(len(all_texts)):
            levenshtein_sim[i, j] = normalize_levenshtein(all_texts[i], all_texts[j])
    
    # Extract original vs rephrasings similarities
    original_vs_rephrasings = []
    for idx, rephrasing in enumerate(rephrasings):
        rephrasing_idx = idx + 1  # +1 because original is at index 0
        result_dict = {
            'rephrasing_index': idx,
            'rephrasing': rephrasing,
            'tfidf_cosine_similarity': float(tfidf_sim[0, rephrasing_idx]),
            'bm25_similarity': float(bm25_sim[0, rephrasing_idx]),
            'levenshtein_similarity': float(levenshtein_sim[0, rephrasing_idx])
        }
        if EMBEDDINGS_AVAILABLE and embedding_model is not None:
            result_dict['embedding_cosine_similarity'] = float(embedding_sim[0, rephrasing_idx])
        else:
            result_dict['embedding_cosine_similarity'] = None
        original_vs_rephrasings.append(result_dict)
    
    # Extract pairwise rephrasing similarities
    pairwise_rephrasings = []
    for i, j in combinations(range(len(rephrasings)), 2):
        # +1 because original is at index 0
        idx_i, idx_j = i + 1, j + 1
        result_dict = {
            'rephrasing_1_index': i,
            'rephrasing_2_index': j,
            'rephrasing_1': rephrasings[i],
            'rephrasing_2': rephrasings[j],
            'tfidf_cosine_similarity': float(tfidf_sim[idx_i, idx_j]),
            'bm25_similarity': float(bm25_sim[idx_i, idx_j]),
            'levenshtein_similarity': float(levenshtein_sim[idx_i, idx_j])
        }
        if EMBEDDINGS_AVAILABLE and embedding_model is not None:
            result_dict['embedding_cosine_similarity'] = float(embedding_sim[idx_i, idx_j])
        else:
            result_dict['embedding_cosine_similarity'] = None
        pairwise_rephrasings.append(result_dict)
    
    # Calculate summary statistics
    summary_stats = {}
    metrics = ['tfidf_cosine_similarity', 'bm25_similarity', 'levenshtein_similarity']
    if EMBEDDINGS_AVAILABLE and embedding_model is not None:
        metrics.insert(0, 'embedding_cosine_similarity')
    
    for metric in metrics:
        original_values = [item[metric] for item in original_vs_rephrasings if item[metric] is not None]
        pairwise_values = [item[metric] for item in pairwise_rephrasings if item[metric] is not None]
        
        if len(original_values) == 0 or len(pairwise_values) == 0:
            continue
            
        summary_stats[metric] = {
            'original_vs_rephrasings': {
                'mean': float(np.mean(original_values)),
                'std': float(np.std(original_values)),
                'min': float(np.min(original_values)),
                'max': float(np.max(original_values)),
                'median': float(np.median(original_values))
            },
            'pairwise_rephrasings': {
                'mean': float(np.mean(pairwise_values)),
                'std': float(np.std(pairwise_values)),
                'min': float(np.min(pairwise_values)),
                'max': float(np.max(pairwise_values)),
                'median': float(np.median(pairwise_values))
            }
        }
    
    return {
        'original_vs_rephrasings': original_vs_rephrasings,
        'pairwise_rephrasings': pairwise_rephrasings,
        'summary_stats': summary_stats
    }

def process_all_perturbations(
    perturbations: List[Dict],
    embedding_model_name: str = "all-MiniLM-L6-v2"
) -> pd.DataFrame:
    """
    Process all perturbations and calculate similarity metrics.
    
    Args:
        perturbations: List of perturbation dictionaries
        embedding_model_name: Name of the sentence transformer model to use (optional)
    
    Returns:
        DataFrame with all similarity metrics
    """
    embedding_model = None
    if EMBEDDINGS_AVAILABLE:
        try:
            print(f"Loading embedding model: {embedding_model_name}")
            embedding_model = SentenceTransformer(embedding_model_name)
        except Exception as e:
            print(f"Warning: Could not load embedding model: {e}")
            print("Continuing without embedding similarity...")
            embedding_model = None
    else:
        print("Embeddings not available, skipping embedding similarity calculations...")
    
    all_results = []
    all_original_vs_rephrasings = []
    all_pairwise_rephrasings = []
    all_summary_stats = []
    
    for prompt_idx, prompt_data in enumerate(tqdm(perturbations, desc="Processing prompts")):
        original_main = prompt_data['original_main']
        rephrasings = prompt_data['rephrasings']
        
        print(f"\nProcessing prompt {prompt_idx + 1}/{len(perturbations)}")
        print(f"  Original: {original_main[:100]}...")
        print(f"  Number of rephrasings: {len(rephrasings)}")
        
        # Calculate similarities
        similarities = calculate_all_similarities(
            original_main,
            rephrasings,
            embedding_model
        )
        
        # Add prompt metadata to results
        for item in similarities['original_vs_rephrasings']:
            item['prompt_index'] = prompt_idx
            item['original_main'] = original_main
            all_original_vs_rephrasings.append(item)
        
        for item in similarities['pairwise_rephrasings']:
            item['prompt_index'] = prompt_idx
            item['original_main'] = original_main
            all_pairwise_rephrasings.append(item)
        
        # Add summary stats
        summary_with_metadata = similarities['summary_stats'].copy()
        summary_with_metadata['prompt_index'] = prompt_idx
        summary_with_metadata['original_main'] = original_main
        summary_with_metadata['num_rephrasings'] = len(rephrasings)
        all_summary_stats.append(summary_with_metadata)
    
    # Create DataFrames
    df_original_vs = pd.DataFrame(all_original_vs_rephrasings)
    df_pairwise = pd.DataFrame(all_pairwise_rephrasings)
    
    # Create summary DataFrame
    summary_rows = []
    for stats in all_summary_stats:
        prompt_idx = stats.pop('prompt_index')
        original_main = stats.pop('original_main')
        num_rephrasings = stats.pop('num_rephrasings')
        
        for metric_name, metric_stats in stats.items():
            row = {
                'prompt_index': prompt_idx,
                'original_main': original_main,
                'num_rephrasings': num_rephrasings,
                'metric': metric_name,
                'original_vs_rephrasings_mean': metric_stats['original_vs_rephrasings']['mean'],
                'original_vs_rephrasings_std': metric_stats['original_vs_rephrasings']['std'],
                'original_vs_rephrasings_min': metric_stats['original_vs_rephrasings']['min'],
                'original_vs_rephrasings_max': metric_stats['original_vs_rephrasings']['max'],
                'original_vs_rephrasings_median': metric_stats['original_vs_rephrasings']['median'],
                'pairwise_rephrasings_mean': metric_stats['pairwise_rephrasings']['mean'],
                'pairwise_rephrasings_std': metric_stats['pairwise_rephrasings']['std'],
                'pairwise_rephrasings_min': metric_stats['pairwise_rephrasings']['min'],
                'pairwise_rephrasings_max': metric_stats['pairwise_rephrasings']['max'],
                'pairwise_rephrasings_median': metric_stats['pairwise_rephrasings']['median']
            }
            summary_rows.append(row)
    
    df_summary = pd.DataFrame(summary_rows)
    
    # Print overall summary
    print("\n=== Overall Summary Statistics ===")
    metrics_to_print = ['tfidf_cosine_similarity', 'bm25_similarity', 'levenshtein_similarity']
    if EMBEDDINGS_AVAILABLE and embedding_model is not None:
        metrics_to_print.insert(0, 'embedding_cosine_similarity')
    
    for metric in metrics_to_print:
        metric_df = df_summary[df_summary['metric'] == metric]
        if len(metric_df) == 0:
            continue
        print(f"\n{metric}:")
        print(f"  Original vs Rephrasings:")
        print(f"    Mean: {metric_df['original_vs_rephrasings_mean'].mean():.4f}")
        print(f"    Std: {metric_df['original_vs_rephrasings_mean'].std():.4f}")
        print(f"  Pairwise Rephrasings:")
        print(f"    Mean: {metric_df['pairwise_rephrasings_mean'].mean():.4f}")
        print(f"    Std: {metric_df['pairwise_rephrasings_mean'].std():.4f}")
    
    return df_original_vs, df_pairwise, df_summary

def main():
    """Main function to run similarity analysis."""
    # Load perturbations
    perturbations_file = "data/perturbations.json"
    print(f"Loading perturbations from {perturbations_file}...")
    perturbations = load_perturbations(perturbations_file)
    print(f"Loaded {len(perturbations)} prompts with rephrasings")
    
    # Process all perturbations
    df_original_vs, df_pairwise, df_summary = process_all_perturbations(perturbations)
    
    print("\n=== Analysis Complete ===")
    print(f"Total prompts analyzed: {len(perturbations)}")
    print(f"Total original vs rephrasing comparisons: {len(df_original_vs)}")
    print(f"Total pairwise rephrasing comparisons: {len(df_pairwise)}")
    
    return df_original_vs, df_pairwise, df_summary

if __name__ == "__main__":
    df_original_vs, df_pairwise, df_summary = main()


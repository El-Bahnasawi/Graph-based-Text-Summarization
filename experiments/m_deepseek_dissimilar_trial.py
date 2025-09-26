import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import time

class DiversityPageRank:
    def __init__(self):
        self.last_result = None
    
    def rank(self, sentence_embeddings, damping=0.85, max_iter=100, 
             tol=1e-6, diversity_strength=0.5, top_k=3):
        """
        Diversity-focused PageRank that considers both similarity and diversity
        
        Args:
            sentence_embeddings: Array of sentence embeddings
            damping: PageRank damping factor
            max_iter: Maximum iterations
            tol: Convergence tolerance
            diversity_strength: How much to weight diversity (0-1)
            top_k: Number of sentences to return
        """
        n = len(sentence_embeddings)
        
        # Calculate cosine similarity matrix
        similarity_matrix = cosine_similarity(sentence_embeddings)
        np.fill_diagonal(similarity_matrix, 0)  # Remove self-similarity
        
        # Calculate diversity matrix (inverse of similarity)
        # Add small epsilon to avoid division by zero
        diversity_matrix = 1 / (similarity_matrix + 1e-8)
        
        # Normalize diversity matrix
        diversity_matrix = diversity_matrix / np.max(diversity_matrix)
        
        # Combine similarity and diversity
        # Higher diversity_strength puts more emphasis on diverse sentences
        combined_matrix = (1 - diversity_strength) * similarity_matrix + \
                         diversity_strength * diversity_matrix
        
        # Convert to transition matrix (column stochastic)
        column_sums = combined_matrix.sum(axis=0)
        column_sums[column_sums == 0] = 1  # Avoid division by zero
        transition_matrix = combined_matrix / column_sums
        
        # Initialize PageRank vector
        pagerank = np.ones(n) / n
        
        # Power iteration
        for _ in range(max_iter):
            new_pagerank = (1 - damping) / n + damping * np.dot(transition_matrix, pagerank)
            
            # Check convergence
            if np.linalg.norm(new_pagerank - pagerank, 1) < tol:
                break
                
            pagerank = new_pagerank
        
        # Get top-k sentences
        ranked_indices = np.argsort(pagerank)[::-1][:top_k]
        ranked_sentences = [(i, pagerank[i]) for i in ranked_indices]
        
        return ranked_sentences, pagerank

def evaluate_diversity_approach(dataset_path, sample_size=50, diversity_strengths=None):
    """
    Evaluate the diversity-focused PageRank approach
    """
    if diversity_strengths is None:
        diversity_strengths = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    # Load data
    df = pd.read_csv(dataset_path)
    df = df.head(sample_size)
    
    # Initialize SBERT embedder
    from sent_embd import create_embedder
    embedder = create_embedder("sbert", device="cuda")
    
    # Initialize ROUGE scorer
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    results = []
    
    for diversity_strength in diversity_strengths:
        print(f"Evaluating diversity strength: {diversity_strength}")
        
        config_results = {
            'rouge1': [], 'rouge2': [], 'rougeL': [], 'rouge_l2': [], 'time': []
        }
        
        diversity_pagerank = DiversityPageRank()
        
        for i, row in df.iterrows():
            article = row['article']
            reference = row['highlights']
            
            try:
                start_time = time.time()
                
                # Split into sentences
                from Summarizer import split_into_sentences
                sents = split_into_sentences(article)
                
                if len(sents) < 3:  # Need at least 3 sentences for meaningful comparison
                    continue
                
                # Generate embeddings
                embeddings = embedder.encode(sents)
                
                # Run diversity-focused PageRank
                ranked_sentences, scores = diversity_pagerank.rank(
                    embeddings, 
                    diversity_strength=diversity_strength,
                    top_k=3
                )
                
                # Get summary sentences (in original order)
                summary_indices = sorted([idx for idx, score in ranked_sentences])
                summary_sentences = [sents[idx] for idx in summary_indices]
                generated_summary = ' '.join(summary_sentences)
                
                generation_time = time.time() - start_time
                
                # Calculate ROUGE scores
                rouge_scores = scorer.score(reference, generated_summary)
                rouge_l2 = np.sqrt(
                    rouge_scores['rouge1'].fmeasure**2 + 
                    rouge_scores['rouge2'].fmeasure**2 + 
                    rouge_scores['rougeL'].fmeasure**2
                )
                
                config_results['rouge1'].append(rouge_scores['rouge1'].fmeasure)
                config_results['rouge2'].append(rouge_scores['rouge2'].fmeasure)
                config_results['rougeL'].append(rouge_scores['rougeL'].fmeasure)
                config_results['rouge_l2'].append(rouge_l2)
                config_results['time'].append(generation_time)
                
            except Exception as e:
                print(f"Error processing article {i}: {str(e)}")
                continue
        
        # Calculate averages
        avg_results = {
            'diversity_strength': diversity_strength,
            'avg_rouge1': np.mean(config_results['rouge1']),
            'avg_rouge2': np.mean(config_results['rouge2']),
            'avg_rougeL': np.mean(config_results['rougeL']),
            'avg_rouge_l2': np.mean(config_results['rouge_l2']),
            'avg_time': np.mean(config_results['time']),
            'n_articles': len(config_results['rouge1'])
        }
        
        results.append(avg_results)
        print(f"  ROUGE L2-Norm: {avg_results['avg_rouge_l2']:.4f}")
    
    return pd.DataFrame(results)

def plot_diversity_results(results_df):
    """Plot results of diversity-focused PageRank evaluation"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # ROUGE scores plot
    ax1.plot(results_df['diversity_strength'], results_df['avg_rouge_l2'], 
             'o-', linewidth=2, markersize=8, label='ROUGE L2-Norm')
    ax1.plot(results_df['diversity_strength'], results_df['avg_rouge1'], 
             's--', alpha=0.7, label='ROUGE-1')
    ax1.plot(results_df['diversity_strength'], results_df['avg_rouge2'], 
             '^--', alpha=0.7, label='ROUGE-2')
    ax1.plot(results_df['diversity_strength'], results_df['avg_rougeL'], 
             'd--', alpha=0.7, label='ROUGE-L')
    
    ax1.set_xlabel('Diversity Strength')
    ax1.set_ylabel('ROUGE Score')
    ax1.set_title('Diversity-Focused PageRank Performance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Mark the best diversity strength
    best_idx = results_df['avg_rouge_l2'].idxmax()
    best_strength = results_df.loc[best_idx, 'diversity_strength']
    best_score = results_df.loc[best_idx, 'avg_rouge_l2']
    ax1.axvline(x=best_strength, color='red', linestyle='--', alpha=0.7)
    ax1.text(best_strength + 0.02, best_score - 0.05, 
             f'Best: {best_strength}\nScore: {best_score:.3f}', 
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))
    
    # Processing time plot
    ax2.plot(results_df['diversity_strength'], results_df['avg_time'], 
             'o-', color='orange', linewidth=2, markersize=8)
    ax2.set_xlabel('Diversity Strength')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('Processing Time vs Diversity Strength')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('diversity_pagerank_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    best_idx = results_df['avg_rouge_l2'].idxmax()
    best_config = results_df.loc[best_idx]
    
    print("="*60)
    print("DIVERSITY-FOCUSED PAGERANK RESULTS")
    print("="*60)
    print(f"Best diversity strength: {best_config['diversity_strength']:.1f}")
    print(f"ROUGE-1: {best_config['avg_rouge1']:.4f}")
    print(f"ROUGE-2: {best_config['avg_rouge2']:.4f}")
    print(f"ROUGE-L: {best_config['avg_rougeL']:.4f}")
    print(f"ROUGE L2-Norm: {best_config['avg_rouge_l2']:.4f}")
    print(f"Processing time: {best_config['avg_time']:.2f}s")
    print("="*60)
    
    return best_config

# Example usage
if __name__ == "__main__":
    # Evaluate different diversity strengths
    diversity_results = evaluate_diversity_approach(
        dataset_path="Datasets/cnn_dailymail/validation.csv",
        sample_size=100,
        diversity_strengths=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    )
    
    # Plot results
    best_config = plot_diversity_results(diversity_results)
    
    # Save results
    diversity_results.to_csv('diversity_pagerank_results.csv', index=False)




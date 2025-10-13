# evaluate_summarization.py
import pandas as pd
import numpy as np
from summarizer import summarize_text
from rouge_score import rouge_scorer
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv
import os
import math

class SummarizationEvaluator:
    def __init__(self, dataset_path, sample_size=100, use_gpu=True, embedder_kind="sbert", sim_threshold=0.5, sum_num_sents=3):
        self.dataset_path = dataset_path
        self.sample_size = sample_size
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.use_gpu = use_gpu
        self.embedder_kind = embedder_kind
        self.sim_threshold = sim_threshold
        self.sum_num_sents = sum_num_sents
        self.results = []
        
        # Initialize embedder once based on type
        from sent_embd import create_embedder
        
        if embedder_kind in ["tfidf", "bow"]:
            model_path = f"models/{embedder_kind}_model.pkl"
            self.embedder = create_embedder(embedder_kind, model_path=model_path)
            print(f"Using pre-trained {embedder_kind.upper()} model from {model_path}")
        else:
            device = "cuda" if use_gpu else "cpu"
            self.embedder = create_embedder(embedder_kind, device=device)
            print(f"Using {embedder_kind.upper()} embedder on {device}")
        
    def load_data(self):
        """Load the CNN/DailyMail dataset"""
        print("Loading dataset...")
        df = pd.read_csv(self.dataset_path)
        self.df = df.head(self.sample_size)
        print(f"Loaded {len(self.df)} articles")
        
    def evaluate(self, save_to_csv=True):
        """Evaluate the summarization on all articles"""
        print("Evaluating summaries...")
        
        # Create results file with header if it doesn't exist
        results_file = 'detailed_results.csv'
        write_header = not os.path.exists(results_file)
        
        if save_to_csv:
            f = open(results_file, 'a', newline='', encoding='utf-8')
            writer = csv.DictWriter(f, fieldnames=[
                'article_id', 'reference', 'generated', 'rouge1', 'rouge2', 'rougeL', 'rouge_l2', 'time', 'embedder', 'sim_threshold'
            ])
            
            if write_header:
                writer.writeheader()
        
        for i, row in tqdm(self.df.iterrows(), total=len(self.df)):
            article = row['article']
            reference = row['highlights']
            
            try:
                start_time = time.time()
                
                # FIX: Now unpack 3 values instead of 2
                top_indices, summary_sentences, graph = summarize_text(
                    article, 
                    k=self.sum_num_sents, 
                    sim_threshold=self.sim_threshold, 
                    embedder=self.embedder,
                    enable_visualization=False
                )
                
                generation_time = time.time() - start_time
                generated_summary = ' '.join(summary_sentences)
                scores = self.scorer.score(reference, generated_summary)
                
                rouge_l2 = math.sqrt(
                    scores['rouge1'].fmeasure**2 + 
                    scores['rouge2'].fmeasure**2 + 
                    scores['rougeL'].fmeasure**2
                )
                
                result = {
                    'article_id': i,
                    'reference': reference,
                    'generated': generated_summary,
                    'rouge1': scores['rouge1'].fmeasure,
                    'rouge2': scores['rouge2'].fmeasure,
                    'rougeL': scores['rougeL'].fmeasure,
                    'rouge_l2': rouge_l2,
                    'time': generation_time,
                    'embedder': self.embedder_kind,
                    'sim_threshold': self.sim_threshold
                }
                
                self.results.append(result)
                
                if save_to_csv:
                    writer.writerow(result)
                    f.flush()
                    
            except Exception as e:
                print(f"Error processing article {i}: {str(e)}")
                continue
        
        if save_to_csv:
            f.close()
                
    def calculate_stats(self):
        """Calculate overall statistics"""
        if not self.results:
            return None
            
        rouge1_scores = [r['rouge1'] for r in self.results]
        rouge2_scores = [r['rouge2'] for r in self.results]
        rougeL_scores = [r['rougeL'] for r in self.results]
        rouge_l2_scores = [r['rouge_l2'] for r in self.results]
        times = [r['time'] for r in self.results]
        
        stats = {
            'avg_rouge1': np.mean(rouge1_scores),
            'avg_rouge2': np.mean(rouge2_scores),
            'avg_rougeL': np.mean(rougeL_scores),
            'avg_rouge_l2': np.mean(rouge_l2_scores),
            'std_rouge1': np.std(rouge1_scores),
            'std_rouge2': np.std(rouge2_scores),
            'std_rougeL': np.std(rougeL_scores),
            'std_rouge_l2': np.std(rouge_l2_scores),
            'avg_time': np.mean(times),
            'total_time': np.sum(times),
            'n_articles': len(self.results),
            'embedder': self.embedder_kind,
            'sim_threshold': self.sim_threshold
        }
        
        return stats
        
    def plot_results(self, stats):
        """Plot the evaluation results"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'ROUGE L2-Norm']
        means = [
            stats['avg_rouge1'], 
            stats['avg_rouge2'], 
            stats['avg_rougeL'],
            stats['avg_rouge_l2']
        ]
        stds = [
            stats['std_rouge1'], 
            stats['std_rouge2'], 
            stats['std_rougeL'],
            stats['std_rouge_l2']
        ]
        
        ax1.bar(metrics, means, yerr=stds, capsize=5, alpha=0.7, 
                color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
        ax1.set_ylabel('F1 Score')
        ax1.set_title('ROUGE Scores')
        ax1.set_ylim(0, max(means) * 1.2)
        
        ax2.bar(['Avg Time/Article', 'Total Time'], 
                [stats['avg_time'], stats['total_time']], 
                color=['orange', 'purple'])
        ax2.set_ylabel('Time (seconds)')
        ax2.set_title('Processing Time')
        
        rouge_l2_scores = [r['rouge_l2'] for r in self.results]
        ax3.hist(rouge_l2_scores, bins=20, alpha=0.7, color='teal')
        ax3.set_xlabel('ROUGE L2-Norm Score')
        ax3.set_ylabel('Frequency')
        ax3.set_title('ROUGE L2-Norm Distribution')
        ax3.axvline(stats['avg_rouge_l2'], color='red', linestyle='dashed', linewidth=1)
        ax3.text(stats['avg_rouge_l2']*1.05, ax3.get_ylim()[1]*0.9, 
                f'Mean: {stats["avg_rouge_l2"]:.3f}', color='red')
        
        plt.suptitle(f"Evaluation Results ({self.embedder_kind.upper()} & sim_threshold: {self.sim_threshold})", fontsize=16, y=1.02)
        plt.tight_layout()

        plt.savefig(f'evaluation_results_{self.embedder_kind}.png', dpi=150, bbox_inches='tight')
        plt.show()
        
    def save_results(self, stats):
        """Save results to files"""
        stats_df = pd.DataFrame([stats])
        stats_df.to_csv(f'summary_statistics_{self.embedder_kind}.csv', index=False)
        
        print(f"Results saved to summary_statistics_{self.embedder_kind}.csv")
        
    def run_evaluation(self):
        """Run the complete evaluation pipeline"""
        self.load_data()
        self.evaluate()
        
        if not self.results:
            print("No results to evaluate!")
            return
            
        stats = self.calculate_stats()
        
        print("\n=== EVALUATION RESULTS ===")
        print(f"Embedder: {self.embedder_kind.upper()}")
        print(f"Articles processed: {stats['n_articles']}")
        print(f"Average ROUGE-1: {stats['avg_rouge1']:.4f} (±{stats['std_rouge1']:.4f})")
        print(f"Average ROUGE-2: {stats['avg_rouge2']:.4f} (±{stats['std_rouge2']:.4f})")
        print(f"Average ROUGE-L: {stats['avg_rougeL']:.4f} (±{stats['std_rougeL']:.4f})")
        print(f"Average ROUGE L2-Norm: {stats['avg_rouge_l2']:.4f} (±{stats['std_rouge_l2']:.4f})")
        print(f"Average time per article: {stats['avg_time']:.2f} seconds")
        print(f"Total processing time: {stats['total_time']:.2f} seconds")
        
        self.plot_results(stats)
        self.save_results(stats)
    
    def hyperparameter_tuning(self, thresholds):
        """Run hyperparameter tuning for different similarity thresholds"""
        print(f"Running hyperparameter tuning for {self.embedder_kind}...")
        
        # Load data once
        self.load_data()
        
        # Store results for each threshold
        tuning_results = []
        
        for threshold in tqdm(thresholds):
            self.sim_threshold = threshold
            self.results = []  # Reset results for each threshold
            
            # Evaluate without saving to CSV
            self.evaluate(save_to_csv=False)
            
            # Calculate stats
            stats = self.calculate_stats()
            if stats:
                tuning_results.append({
                    'threshold': threshold,
                    'rouge_l2': stats['avg_rouge_l2'],
                    'rouge1': stats['avg_rouge1'],
                    'rouge2': stats['avg_rouge2'],
                    'rougeL': stats['avg_rougeL']
                })
        
        return tuning_results


def plot_hyperparameter_tuning(results_dict):
    """Plot hyperparameter tuning results for all embedders"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    colors = ['blue', 'red', 'green']
    embedders = list(results_dict.keys())
    
    for idx, embedder in enumerate(embedders):
        results = results_dict[embedder]
        thresholds = [r['threshold'] for r in results]
        rouge_l2_scores = [r['rouge_l2'] for r in results]
        
        axes[idx].plot(thresholds, rouge_l2_scores, 'o-', color=colors[idx], linewidth=2, markersize=8)
        axes[idx].set_xlabel('Similarity Threshold')
        axes[idx].set_ylabel('ROUGE L2-Norm Score')
        axes[idx].set_title(f'{embedder.upper()} Embedder')
        axes[idx].grid(True, alpha=0.3)
        
        # Mark the best threshold
        best_idx = np.argmax(rouge_l2_scores)
        best_threshold = thresholds[best_idx]
        best_score = rouge_l2_scores[best_idx]
        axes[idx].axvline(x=best_threshold, color='orange', linestyle='--', alpha=0.7)
        axes[idx].text(best_threshold+0.02, best_score-0.05, 
                      f'Best: {best_threshold}\nScore: {best_score:.3f}', 
                      bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))
    
    plt.suptitle('Hyperparameter Tuning: Similarity Threshold vs ROUGE L2-Norm', fontsize=16)
    plt.tight_layout()
    plt.savefig('hyperparameter_tuning_results.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Hyperparameter tuning configuration
    thresholds = np.arange(0.1, 1.0, 0.1)  # Test thresholds from 0.1 to 0.9
    embedders_to_test = ["sbert", "tfidf", "bow"]
    
    # Store all tuning results
    all_tuning_results = {}
    
    for embedder_kind in embedders_to_test:
        print(f"\n{'='*50}")
        print(f"Testing {embedder_kind.upper()} embedder")
        print(f"{'='*50}")
        
        # Initialize evaluator
        evaluator = SummarizationEvaluator(
            dataset_path="Datasets/cnn_dailymail/validation.csv",
            sample_size=100,
            use_gpu=True,
            embedder_kind=embedder_kind,
            sim_threshold=0.5,  # Default value, will be overridden in tuning
            sum_num_sents=3
        )
        
        # Run hyperparameter tuning
        tuning_results = evaluator.hyperparameter_tuning(thresholds)
        all_tuning_results[embedder_kind] = tuning_results
        
        # Print best result
        best_result = max(tuning_results, key=lambda x: x['rouge_l2'])
        print(f"Best threshold for {embedder_kind}: {best_result['threshold']} "
              f"(ROUGE L2: {best_result['rouge_l2']:.4f})")
    
    # Plot all results
    plot_hyperparameter_tuning(all_tuning_results)
    
    # Save tuning results to CSV
    tuning_df = pd.DataFrame()
    for embedder, results in all_tuning_results.items():
        for result in results:
            result['embedder'] = embedder
        tuning_df = pd.concat([tuning_df, pd.DataFrame(results)], ignore_index=True)
    
    tuning_df.to_csv('hyperparameter_tuning_results.csv', index=False)
    print("Hyperparameter tuning results saved to hyperparameter_tuning_results.csv")
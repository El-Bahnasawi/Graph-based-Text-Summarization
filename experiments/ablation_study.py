import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from Evaluator import SummarizationEvaluator
from Summarizer import summarize_text
from sent_embd import create_embedder
import time
import os

class AblationStudy:
    def __init__(self, dataset_path, sample_size=100, use_gpu=True):
        self.dataset_path = dataset_path
        self.sample_size = sample_size
        self.use_gpu = use_gpu
        self.results = []
        
    def run_ablation_study(self):
        """Run ablation study with different configurations"""
        print("Starting ablation study...")
        
        # Define the configurations to test
        configurations = [
            # Base configuration (best from hyperparameter tuning)
            {
                'name': 'SBERT (Optimal)',
                'embedder_kind': 'sbert',
                'sim_threshold': 0.3,
                'use_sent_sent': True,
                'use_sent_name': True,
                'use_name_name': True
            },
            # Without sentence-sentence edges
            {
                'name': 'No Sent-Sent Edges',
                'embedder_kind': 'sbert',
                'sim_threshold': 0.3,
                'use_sent_sent': False,
                'use_sent_name': True,
                'use_name_name': True
            },
            # Without sentence-name edges
            {
                'name': 'No Sent-Name Edges',
                'embedder_kind': 'sbert',
                'sim_threshold': 0.3,
                'use_sent_sent': True,
                'use_sent_name': False,
                'use_name_name': True
            },
            # Without name-name edges
            {
                'name': 'No Name-Name Edges',
                'embedder_kind': 'sbert',
                'sim_threshold': 0.3,
                'use_sent_sent': True,
                'use_sent_name': True,
                'use_name_name': False
            },
            # Only sentence-sentence edges
            {
                'name': 'Only Sent-Sent Edges',
                'embedder_kind': 'sbert',
                'sim_threshold': 0.3,
                'use_sent_sent': True,
                'use_sent_name': False,
                'use_name_name': False
            },
            # TFIDF baseline
            {
                'name': 'TFIDF Baseline',
                'embedder_kind': 'tfidf',
                'sim_threshold': 0.1,
                'use_sent_sent': True,
                'use_sent_name': True,
                'use_name_name': True
            },
            # BOW baseline
            {
                'name': 'BOW Baseline',
                'embedder_kind': 'bow',
                'sim_threshold': 0.1,
                'use_sent_sent': True,
                'use_sent_name': True,
                'use_name_name': True
            }
        ]
        
        # Load data once
        print("Loading dataset...")
        df = pd.read_csv(self.dataset_path)
        df = df.head(self.sample_size)
        
        # Initialize embedders
        embedders = {}
        for config in configurations:
            if config['embedder_kind'] not in embedders:
                if config['embedder_kind'] in ["tfidf", "bow"]:
                    model_path = f"models/{config['embedder_kind']}_model.pkl"
                    embedders[config['embedder_kind']] = create_embedder(
                        config['embedder_kind'], model_path=model_path
                    )
                else:
                    device = "cuda" if self.use_gpu else "cpu"
                    embedders[config['embedder_kind']] = create_embedder(
                        config['embedder_kind'], device=device
                    )
        
        # Initialize ROUGE scorer
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Evaluate each configuration
        for config in configurations:
            print(f"\nEvaluating: {config['name']}")
            
            config_results = {
                'rouge1': [],
                'rouge2': [],
                'rougeL': [],
                'rouge_l2': [],
                'time': []
            }
            
            embedder = embedders[config['embedder_kind']]
            
            for i, row in df.iterrows():
                article = row['article']
                reference = row['highlights']
                
                try:
                    start_time = time.time()
                    
                    # Custom summarization with ablation settings
                    top_indices, summary_sentences = self.custom_summarize(
                        article, 
                        embedder=embedder,
                        sim_threshold=config['sim_threshold'],
                        use_sent_sent=config['use_sent_sent'],
                        use_sent_name=config['use_sent_name'],
                        use_name_name=config['use_name_name']
                    )
                    
                    generation_time = time.time() - start_time
                    generated_summary = ' '.join(summary_sentences)
                    
                    # Calculate ROUGE scores
                    scores = scorer.score(reference, generated_summary)
                    rouge_l2 = np.sqrt(
                        scores['rouge1'].fmeasure**2 + 
                        scores['rouge2'].fmeasure**2 + 
                        scores['rougeL'].fmeasure**2
                    )
                    
                    config_results['rouge1'].append(scores['rouge1'].fmeasure)
                    config_results['rouge2'].append(scores['rouge2'].fmeasure)
                    config_results['rougeL'].append(scores['rougeL'].fmeasure)
                    config_results['rouge_l2'].append(rouge_l2)
                    config_results['time'].append(generation_time)
                    
                except Exception as e:
                    print(f"Error processing article {i}: {str(e)}")
                    continue
            
            # Calculate average metrics
            avg_results = {
                'config_name': config['name'],
                'embedder': config['embedder_kind'],
                'sim_threshold': config['sim_threshold'],
                'avg_rouge1': np.mean(config_results['rouge1']),
                'avg_rouge2': np.mean(config_results['rouge2']),
                'avg_rougeL': np.mean(config_results['rougeL']),
                'avg_rouge_l2': np.mean(config_results['rouge_l2']),
                'avg_time': np.mean(config_results['time']),
                'n_articles': len(config_results['rouge1'])
            }
            
            self.results.append(avg_results)
            print(f"  ROUGE L2-Norm: {avg_results['avg_rouge_l2']:.4f}")
            print(f"  Avg Time: {avg_results['avg_time']:.2f}s")
        
        return self.results
    
    def custom_summarize(self, text, embedder, sim_threshold=0.5, 
                        use_sent_sent=True, use_sent_name=True, use_name_name=True):
        """Custom summarization function with configurable graph components"""
        from Summarizer import split_into_sentences, _extract_names_per_sentence
        from Summarizer import build_graph_from_similarity, add_sentence_name_edges
        from reactive_graph import ReactiveGraph
        from Rank import PageRank
        
        # Split text into sentences
        sents = split_into_sentences(text)
        if not sents:
            return [], []
        
        # Generate embeddings
        X = embedder.encode(sents)
        
        # Calculate similarity matrix
        def cosine_similarity_matrix(X):
            Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
            S = np.matmul(Xn, Xn.T)
            np.fill_diagonal(S, 0.0)
            return S
        
        S = cosine_similarity_matrix(X.astype(np.float64))
        
        # Build graph
        g = ReactiveGraph(directed=False)
        
        # Add sentence nodes
        n = len(sents)
        for i in range(n):
            g.add_node(i, text=sents[i], kind="sent", value=1.0 / max(n, 1))
        
        # Add edges based on configuration
        if use_sent_sent:
            # Add sentence-sentence edges
            for i in range(n):
                for j in range(i + 1, n):
                    w = float(S[i, j])
                    if w >= sim_threshold:
                        g.add_edge(i, j, value=round(w, 3))
        
        if use_sent_name or use_name_name:
            # Extract names and add name-related edges
            names_per_sent = _extract_names_per_sentence(sents)
            
            # Modify the add_sentence_name_edges function to respect the flags
            def custom_add_sentence_name_edges(g, sents, names_per_sent, 
                                              include_sent_name=True, include_name_name=True):
                """Custom version that respects the ablation flags"""
                def _name_node_id(name):
                    return f"NAME: {name}"
                
                # Sentence <-> Name edges
                if include_sent_name:
                    for i, names in enumerate(names_per_sent):
                        for nm in names:
                            nm = nm.strip()
                            if not nm:
                                continue
                            nid = _name_node_id(nm)
                            if nid not in g.nx:
                                g.add_node(nid, label=nm, kind="name", value=0.0)
                            
                            if g.nx.has_edge(i, nid):
                                w = float(g.nx.edges[i, nid].get("value", 0.0)) + 1.0
                                g.set_edge_attrs(i, nid, value=round(w, 3))
                            elif g.nx.has_edge(nid, i):
                                w = float(g.nx.edges[nid, i].get("value", 0.0)) + 1.0
                                g.set_edge_attrs(nid, i, value=round(w, 3))
                            else:
                                g.add_edge(i, nid, value=1.0)
                
                # Name <-> Name edges
                if include_name_name:
                    from collections import Counter
                    pair_counts = Counter()
                    for names in names_per_sent:
                        uniq = sorted(set(n.strip() for n in names if n.strip()))
                        for a_idx in range(len(uniq)):
                            for b_idx in range(a_idx + 1, len(uniq)):
                                a, b = uniq[a_idx], uniq[b_idx]
                                pair_counts[(a, b)] += 1
                    
                    for (a, b), c in pair_counts.items():
                        na, nb = _name_node_id(a), _name_node_id(b)
                        if na not in g.nx:
                            g.add_node(na, label=a, kind="name", value=0.0)
                        if nb not in g.nx:
                            g.add_node(nb, label=b, kind="name", value=0.0)
                        if g.nx.has_edge(na, nb):
                            w = float(g.nx.edges[na, nb].get("value", 0.0)) + float(c)
                            g.set_edge_attrs(na, nb, value=round(w, 3))
                        elif g.nx.has_edge(nb, na):
                            w = float(g.nx.edges[nb, na].get("value", 0.0)) + float(c)
                            g.set_edge_attrs(nb, na, value=round(w, 3))
                        else:
                            g.add_edge(na, nb, value=float(c))
            
            custom_add_sentence_name_edges(
                g, sents, names_per_sent, 
                include_sent_name=use_sent_name, 
                include_name_name=use_name_name
            )
        
        # Run PageRank
        pr = PageRank()
        res = pr.rank(
            g,
            damping=0.85,
            max_iter=50,
            tol=1e-6,
            sort_desc=True,
            live_update=False,
            update_attr="value",
            sleep_between_iters=0
        )
        
        # Select top sentences
        sent_scores = {i: s for i, s in res.scores.items() if isinstance(i, int)}
        top_sorted = sorted(sent_scores.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
        top_ids = sorted([i for i, _ in top_sorted])
        top_sentences = [sents[i] for i in top_ids]
        
        return top_ids, top_sentences
    
    def plot_results(self):
        """Plot the ablation study results"""
        if not self.results:
            print("No results to plot!")
            return
            
        df = pd.DataFrame(self.results)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        
        # First subplot: ROUGE scores comparison
        x_pos = np.arange(len(df))
        bar_width = 0.2
        
        ax1.bar(x_pos - bar_width*1.5, df['avg_rouge1'], bar_width, 
                label='ROUGE-1', alpha=0.8, color='#1f77b4')
        ax1.bar(x_pos - bar_width*0.5, df['avg_rouge2'], bar_width, 
                label='ROUGE-2', alpha=0.8, color='#ff7f0e')
        ax1.bar(x_pos + bar_width*0.5, df['avg_rougeL'], bar_width, 
                label='ROUGE-L', alpha=0.8, color='#2ca02c')
        ax1.bar(x_pos + bar_width*1.5, df['avg_rouge_l2'], bar_width, 
                label='ROUGE L2-Norm', alpha=0.8, color='#d62728')
        
        ax1.set_xlabel('Configuration', fontsize=14, fontweight='bold')
        ax1.set_ylabel('ROUGE Score', fontsize=14, fontweight='bold')
        ax1.set_title('Ablation Study: ROUGE Scores by Configuration', fontsize=16, fontweight='bold', pad=20)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([f"{row['config_name']}\n({row['embedder']})" for _, row in df.iterrows()], 
                           rotation=45, ha='right', fontsize=10)
        ax1.legend(loc='upper right', fontsize=11, frameon=True, fancybox=True, shadow=True)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, row in df.iterrows():
            ax1.text(i - bar_width*1.5, row['avg_rouge1'] + 0.005, f'{row["avg_rouge1"]:.3f}', 
                     ha='center', va='bottom', fontsize=8, fontweight='bold')
            ax1.text(i - bar_width*0.5, row['avg_rouge2'] + 0.005, f'{row["avg_rouge2"]:.3f}', 
                     ha='center', va='bottom', fontsize=8, fontweight='bold')
            ax1.text(i + bar_width*0.5, row['avg_rougeL'] + 0.005, f'{row["avg_rougeL"]:.3f}', 
                     ha='center', va='bottom', fontsize=8, fontweight='bold')
            ax1.text(i + bar_width*1.5, row['avg_rouge_l2'] + 0.005, f'{row["avg_rouge_l2"]:.3f}', 
                     ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Second subplot: Processing time comparison
        ax2.bar(x_pos, df['avg_time'], color='#9467bd', alpha=0.8)
        ax2.set_xlabel('Configuration', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Time (seconds)', fontsize=14, fontweight='bold')
        ax2.set_title('Ablation Study: Processing Time by Configuration', fontsize=16, fontweight='bold', pad=20)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f"{row['config_name']}\n({row['embedder']})" for _, row in df.iterrows()], 
                           rotation=45, ha='right', fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, row in df.iterrows():
            ax2.text(i, row['avg_time'] + 0.05, f'{row["avg_time"]:.2f}s', 
                     ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('ablation_study_results.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        # Print summary table
        print("\n" + "="*100)
        print("ABLATION STUDY SUMMARY")
        print("="*100)
        summary_df = df[['config_name', 'embedder', 'sim_threshold', 
                        'avg_rouge1', 'avg_rouge2', 'avg_rougeL', 'avg_rouge_l2', 'avg_time']].copy()
        summary_df.columns = ['Configuration', 'Embedder', 'Threshold', 
                             'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'ROUGE L2-Norm', 'Time (s)']
        summary_df = summary_df.round(4)
        print(summary_df.to_string(index=False))
        print("="*100)
        
        # Save results to CSV
        df.to_csv('ablation_study_results.csv', index=False)
        print("Results saved to ablation_study_results.csv")


if __name__ == "__main__":
    # Run ablation study
    ablation = AblationStudy(
        dataset_path="Datasets/cnn_dailymail/validation.csv",
        sample_size=100,  # Use smaller sample for faster evaluation
        use_gpu=True
    )
    
    results = ablation.run_ablation_study()
    ablation.plot_results()
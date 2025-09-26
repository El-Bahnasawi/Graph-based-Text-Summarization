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

class WeightTuning:
    def __init__(self, dataset_path, sample_size=50, use_gpu=True):
        self.dataset_path = dataset_path
        self.sample_size = sample_size
        self.use_gpu = use_gpu
        self.results = []
        
    def run_weight_tuning(self):
        """Run weight tuning for SBERT with different parameters"""
        print("Starting weight tuning for SBERT...")
        
        # Define parameter ranges
        sim_thresholds = np.arange(0.1, 1.0, 0.1)
        sent_name_weights = np.arange(0.0, 2.1, 0.2)
        name_name_weights = np.arange(0.0, 2.1, 0.2)
        
        # Load data
        print("Loading dataset...")
        df = pd.read_csv(self.dataset_path)
        df = df.head(self.sample_size)
        
        # Initialize SBERT embedder
        device = "cuda" if self.use_gpu else "cpu"
        embedder = create_embedder("sbert", device=device)
        
        # Initialize ROUGE scorer
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Counter for progress
        total_combinations = len(sim_thresholds) * len(sent_name_weights) * len(name_name_weights)
        current = 0
        
        # Evaluate each parameter combination
        for sim_threshold in sim_thresholds:
            for sent_name_weight in sent_name_weights:
                for name_name_weight in name_name_weights:
                    current += 1
                    print(f"Evaluating {current}/{total_combinations}: "
                          f"sim_threshold={sim_threshold:.1f}, "
                          f"sent_name_weight={sent_name_weight:.1f}, "
                          f"name_name_weight={name_name_weight:.1f}")
                    
                    config_results = {
                        'rouge1': [],
                        'rouge2': [],
                        'rougeL': [],
                        'rouge_l2': [],
                        'time': []
                    }
                    
                    for i, row in df.iterrows():
                        article = row['article']
                        reference = row['highlights']
                        
                        try:
                            start_time = time.time()
                            
                            # Custom summarization with weight parameters
                            top_indices, summary_sentences = self.custom_summarize(
                                article, 
                                embedder=embedder,
                                sim_threshold=sim_threshold,
                                sent_name_weight=sent_name_weight,
                                name_name_weight=name_name_weight
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
                        'sim_threshold': sim_threshold,
                        'sent_name_weight': sent_name_weight,
                        'name_name_weight': name_name_weight,
                        'avg_rouge1': np.mean(config_results['rouge1']),
                        'avg_rouge2': np.mean(config_results['rouge2']),
                        'avg_rougeL': np.mean(config_results['rougeL']),
                        'avg_rouge_l2': np.mean(config_results['rouge_l2']),
                        'avg_time': np.mean(config_results['time']),
                        'n_articles': len(config_results['rouge1'])
                    }
                    
                    self.results.append(avg_results)
        
        return self.results
    
    def custom_summarize(self, text, embedder, sim_threshold=0.5, 
                        sent_name_weight=1.0, name_name_weight=1.0):
        """Custom summarization function with adjustable weights"""
        from Summarizer import split_into_sentences, _extract_names_per_sentence
        from Summarizer import build_graph_from_similarity
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
        
        # Add sentence-sentence edges
        for i in range(n):
            for j in range(i + 1, n):
                w = float(S[i, j])
                if w >= sim_threshold:
                    g.add_edge(i, j, value=round(w, 3))
        
        # Extract names and add name-related edges with adjustable weights
        names_per_sent = _extract_names_per_sentence(sents)
        
        # Custom function to add name edges with weights
        def custom_add_sentence_name_edges(g, sents, names_per_sent, 
                                          sent_name_weight=1.0, name_name_weight=1.0):
            """Custom version with adjustable weights"""
            def _name_node_id(name):
                return f"NAME: {name}"
            
            # Sentence <-> Name edges
            for i, names in enumerate(names_per_sent):
                for nm in names:
                    nm = nm.strip()
                    if not nm:
                        continue
                    nid = _name_node_id(nm)
                    if nid not in g.nx:
                        g.add_node(nid, label=nm, kind="name", value=0.0)
                    
                    if g.nx.has_edge(i, nid):
                        w = float(g.nx.edges[i, nid].get("value", 0.0)) + sent_name_weight
                        g.set_edge_attrs(i, nid, value=round(w, 3))
                    elif g.nx.has_edge(nid, i):
                        w = float(g.nx.edges[nid, i].get("value", 0.0)) + sent_name_weight
                        g.set_edge_attrs(nid, i, value=round(w, 3))
                    else:
                        g.add_edge(i, nid, value=float(sent_name_weight))
            
            # Name <-> Name edges
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
                
                # Apply name_name_weight as multiplier
                weight_value = float(c) * name_name_weight
                if g.nx.has_edge(na, nb):
                    w = float(g.nx.edges[na, nb].get("value", 0.0)) + weight_value
                    g.set_edge_attrs(na, nb, value=round(w, 3))
                elif g.nx.has_edge(nb, na):
                    w = float(g.nx.edges[nb, na].get("value", 0.0)) + weight_value
                    g.set_edge_attrs(nb, na, value=round(w, 3))
                else:
                    g.add_edge(na, nb, value=weight_value)
        
        custom_add_sentence_name_edges(
            g, sents, names_per_sent, 
            sent_name_weight=sent_name_weight,
            name_name_weight=name_name_weight
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
    """Plot the weight tuning results"""
    if not self.results:
        print("No results to plot!")
        return
        
    df = pd.DataFrame(self.results)
    
    # Find best configuration
    best_idx = df['avg_rouge_l2'].idxmax()
    best_config = df.loc[best_idx]
    print(f"\nBest configuration:")
    print(f"  sim_threshold: {best_config['sim_threshold']:.1f}")
    print(f"  sent_name_weight: {best_config['sent_name_weight']:.1f}")
    print(f"  name_name_weight: {best_config['name_name_weight']:.1f}")
    print(f"  ROUGE L2-Norm: {best_config['avg_rouge_l2']:.4f}")
    
    # Create heatmaps for different parameter combinations
    self.plot_heatmaps(df)
    
    # Create radar chart for best configuration comparison - pass best_config
    self.plot_radar_chart(df, best_config)
    
    # Save results
    df.to_csv('weight_tuning_results.csv', index=False)
    print("Results saved to weight_tuning_results.csv")

def plot_radar_chart(self, df, best_config):
    """Create radar chart comparing best configurations"""
    # Find best configuration for each parameter type
    best_sim = df.loc[df.groupby('sim_threshold')['avg_rouge_l2'].idxmax()]
    best_sent_name = df.loc[df.groupby('sent_name_weight')['avg_rouge_l2'].idxmax()]
    best_name_name = df.loc[df.groupby('name_name_weight')['avg_rouge_l2'].idxmax()]
    
    # Prepare data for radar chart
    categories = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'ROUGE L2-Norm']
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, polar=True)
    
    # Plot function
    def plot_radar(values, label, color):
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]  # Complete the circle
        angles += angles[:1]  # Complete the circle
        ax.plot(angles, values, linewidth=2, label=label, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)
    
    # Plot best overall
    plot_radar([best_config['avg_rouge1'], best_config['avg_rouge2'], 
               best_config['avg_rougeL'], best_config['avg_rouge_l2']], 
              'Best Overall', 'red')
    
    # Plot best by sim_threshold
    plot_radar([best_sim['avg_rouge1'].mean(), best_sim['avg_rouge2'].mean(), 
               best_sim['avg_rougeL'].mean(), best_sim['avg_rouge_l2'].mean()], 
              'Best by sim_threshold', 'blue')
    
    # Plot best by sent_name_weight
    plot_radar([best_sent_name['avg_rouge1'].mean(), best_sent_name['avg_rouge2'].mean(), 
               best_sent_name['avg_rougeL'].mean(), best_sent_name['avg_rouge_l2'].mean()], 
              'Best by sent_name_weight', 'green')
    
    # Plot best by name_name_weight
    plot_radar([best_name_name['avg_rouge1'].mean(), best_name_name['avg_rouge2'].mean(), 
               best_name_name['avg_rougeL'].mean(), best_name_name['avg_rouge_l2'].mean()], 
              'Best by name_name_weight', 'orange')
    
    # Add labels
    ax.set_thetagrids(np.degrees(np.linspace(0, 2*np.pi, len(categories), endpoint=False)), categories)
    ax.set_title('Radar Chart: Best Configurations Comparison')
    ax.legend(loc='upper right')
    
    plt.savefig('weight_tuning_radar.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    def plot_heatmaps(self, df):
        """Create heatmaps for parameter interactions"""
        # Heatmap 1: sim_threshold vs sent_name_weight (average name_name_weight)
        pivot1 = df.pivot_table(values='avg_rouge_l2', 
                               index='sim_threshold', 
                               columns='sent_name_weight',
                               aggfunc='mean')
        
        # Heatmap 2: sim_threshold vs name_name_weight (average sent_name_weight)
        pivot2 = df.pivot_table(values='avg_rouge_l2', 
                               index='sim_threshold', 
                               columns='name_name_weight',
                               aggfunc='mean')
        
        # Heatmap 3: sent_name_weight vs name_name_weight (average sim_threshold)
        pivot3 = df.pivot_table(values='avg_rouge_l2', 
                               index='sent_name_weight', 
                               columns='name_name_weight',
                               aggfunc='mean')
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        sns.heatmap(pivot1, annot=True, fmt=".3f", cmap="YlOrRd", ax=axes[0])
        axes[0].set_title('ROUGE L2-Norm: sim_threshold vs sent_name_weight')
        axes[0].set_xlabel('sent_name_weight')
        axes[0].set_ylabel('sim_threshold')
        
        sns.heatmap(pivot2, annot=True, fmt=".3f", cmap="YlOrRd", ax=axes[1])
        axes[1].set_title('ROUGE L2-Norm: sim_threshold vs name_name_weight')
        axes[1].set_xlabel('name_name_weight')
        axes[1].set_ylabel('sim_threshold')
        
        sns.heatmap(pivot3, annot=True, fmt=".3f", cmap="YlOrRd", ax=axes[2])
        axes[2].set_title('ROUGE L2-Norm: sent_name_weight vs name_name_weight')
        axes[2].set_xlabel('name_name_weight')
        axes[2].set_ylabel('sent_name_weight')
        
        plt.tight_layout()
        plt.savefig('weight_tuning_heatmaps.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_radar_chart(self, df):
        """Create radar chart comparing best configurations"""
        # Find best configuration for each parameter type
        best_sim = df.loc[df.groupby('sim_threshold')['avg_rouge_l2'].idxmax()]
        best_sent_name = df.loc[df.groupby('sent_name_weight')['avg_rouge_l2'].idxmax()]
        best_name_name = df.loc[df.groupby('name_name_weight')['avg_rouge_l2'].idxmax()]
        
        # Prepare data for radar chart
        categories = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'ROUGE L2-Norm']
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, polar=True)
        
        # Plot function
        def plot_radar(values, label, color):
            angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
            values += values[:1]  # Complete the circle
            angles += angles[:1]  # Complete the circle
            ax.plot(angles, values, linewidth=2, label=label, color=color)
            ax.fill(angles, values, alpha=0.1, color=color)
        
        # Plot best overall
        plot_radar([best_config['avg_rouge1'], best_config['avg_rouge2'], 
                   best_config['avg_rougeL'], best_config['avg_rouge_l2']], 
                  'Best Overall', 'red')
        
        # Plot best by sim_threshold
        plot_radar([best_sim['avg_rouge1'].mean(), best_sim['avg_rouge2'].mean(), 
                   best_sim['avg_rougeL'].mean(), best_sim['avg_rouge_l2'].mean()], 
                  'Best by sim_threshold', 'blue')
        
        # Plot best by sent_name_weight
        plot_radar([best_sent_name['avg_rouge1'].mean(), best_sent_name['avg_rouge2'].mean(), 
                   best_sent_name['avg_rougeL'].mean(), best_sent_name['avg_rouge_l2'].mean()], 
                  'Best by sent_name_weight', 'green')
        
        # Plot best by name_name_weight
        plot_radar([best_name_name['avg_rouge1'].mean(), best_name_name['avg_rouge2'].mean(), 
                   best_name_name['avg_rougeL'].mean(), best_name_name['avg_rouge_l2'].mean()], 
                  'Best by name_name_weight', 'orange')
        
        # Add labels
        ax.set_thetagrids(np.degrees(np.linspace(0, 2*np.pi, len(categories), endpoint=False)), categories)
        ax.set_title('Radar Chart: Best Configurations Comparison')
        ax.legend(loc='upper right')
        
        plt.savefig('weight_tuning_radar.png', dpi=300, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    # Run weight tuning
    tuning = WeightTuning(
        dataset_path="Datasets/cnn_dailymail/validation.csv",
        sample_size=50,  # Use smaller sample for faster tuning
        use_gpu=True
    )
    
    results = tuning.run_weight_tuning()
    tuning.plot_results()
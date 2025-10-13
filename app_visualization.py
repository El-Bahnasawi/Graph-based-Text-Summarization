# app_visualization.py
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from reactive_graph import ReactiveGraph

def create_demo_graph():
    """Create a simple demo graph for the landing page"""
    g = ReactiveGraph(directed=False)
    
    # Add some demo nodes and edges
    nodes = [
        (0, "Sentence 1", 0.8),
        (1, "Sentence 2", 0.6),
        (2, "Sentence 3", 0.9),
        (3, "Entity A", 0.3),
        (4, "Entity B", 0.4)
    ]
    
    for node_id, label, value in nodes:
        kind = "name" if "Entity" in label else "sent"
        g.add_node(node_id, text=label, kind=kind, value=value)
    
    # Add edges
    edges = [(0, 1, 0.7), (1, 2, 0.8), (0, 3, 0.5), (2, 4, 0.6), (3, 4, 0.9)]
    for u, v, w in edges:
        g.add_edge(u, v, value=w)
    
    return g

def create_graph_visualization(graph, sentence_scores=None):
    """
    Create a matplotlib visualization of the graph with final PageRank scores
    
    Args:
        graph: The ReactiveGraph object
        sentence_scores: Optional dict of {sentence_index: final_score} from PageRank
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use spring layout for node positions
    pos = nx.spring_layout(graph.nx, seed=42, k=1.5, iterations=50)
    
    # Separate nodes by type
    sent_nodes = [n for n, data in graph.nx.nodes(data=True) if data.get('kind') == 'sent']
    name_nodes = [n for n, data in graph.nx.nodes(data=True) if data.get('kind') == 'name']
    
    # Separate edges by type
    sent_sent_edges = []
    sent_name_edges = []
    name_name_edges = []
    
    for u, v, data in graph.nx.edges(data=True):
        u_kind = graph.nx.nodes[u].get('kind', 'sent')
        v_kind = graph.nx.nodes[v].get('kind', 'sent')
        
        if u_kind == 'sent' and v_kind == 'sent':
            sent_sent_edges.append((u, v))
        elif u_kind == 'name' and v_kind == 'name':
            name_name_edges.append((u, v))
        else:
            sent_name_edges.append((u, v))
    
    # Node sizes based on FINAL PageRank scores
    sent_sizes = []
    for node in sent_nodes:
        # Use provided sentence_scores if available, otherwise fall back to node attribute
        if sentence_scores and node in sentence_scores:
            score = sentence_scores[node]
        else:
            score = graph.nx.nodes[node].get('value', 0.1)
        sent_sizes.append(300 + 1200 * score)
    
    name_sizes = [400] * len(name_nodes)  # Fixed size for entity nodes
    
    # Draw edges first (so they appear behind nodes)
    if sent_sent_edges:
        nx.draw_networkx_edges(graph.nx, pos, edgelist=sent_sent_edges, 
                              edge_color='#2E86AB', width=2.5, ax=ax, alpha=0.7, label='Sentence-Sentence')
    if sent_name_edges:
        nx.draw_networkx_edges(graph.nx, pos, edgelist=sent_name_edges, 
                              edge_color='#A23B72', width=2, ax=ax, alpha=0.6, label='Sentence-Entity')
    if name_name_edges:
        nx.draw_networkx_edges(graph.nx, pos, edgelist=name_name_edges, 
                              edge_color='#F18F01', width=2, ax=ax, alpha=0.8, label='Entity-Entity')
    
    # Draw nodes
    if sent_nodes:
        nx.draw_networkx_nodes(graph.nx, pos, nodelist=sent_nodes, 
                              node_color='#1f77b4', node_size=sent_sizes, ax=ax, 
                              alpha=0.9, edgecolors='white', linewidths=2)
    if name_nodes:
        nx.draw_networkx_nodes(graph.nx, pos, nodelist=name_nodes, 
                              node_color='#2ca02c', node_size=name_sizes, ax=ax, 
                              alpha=0.9, edgecolors='white', linewidths=2)
    
    # Draw labels with FINAL scores
    labels = {}
    for node, data in graph.nx.nodes(data=True):
        if data.get('kind') == 'name':
            labels[node] = data.get('label', str(node))
        else:
            # For sentences, show the FINAL PageRank score
            if sentence_scores and node in sentence_scores:
                score = sentence_scores[node]
            else:
                score = data.get('value', 0)
            labels[node] = f"{score:.3f}"
    
    nx.draw_networkx_labels(graph.nx, pos, labels, font_size=9, ax=ax, 
                           font_weight='bold', bbox=dict(boxstyle="round,pad=0.3", 
                           facecolor="white", alpha=0.8))
    
    ax.set_title("📊 Knowledge Graph Visualization (Final PageRank Scores)", fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    # Create custom legend
    legend_elements = []
    if sent_nodes:
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', 
                      markersize=10, label='Sentence Nodes', alpha=0.9)
        )
    if name_nodes:
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c', 
                      markersize=10, label='Entity Nodes', alpha=0.9)
        )
    if sent_sent_edges:
        legend_elements.append(
            plt.Line2D([0], [0], color='#2E86AB', lw=3, label='Sentence-Sentence Edges')
        )
    if sent_name_edges:
        legend_elements.append(
            plt.Line2D([0], [0], color='#A23B72', lw=3, label='Sentence-Entity Edges')
        )
    if name_name_edges:
        legend_elements.append(
            plt.Line2D([0], [0], color='#F18F01', lw=3, label='Entity-Entity Edges')
        )
    
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1), 
                  frameon=True, fancybox=True, shadow=True, ncol=1)
    
    return fig

def create_score_comparison_chart(initial_scores, final_scores):
    """Create a chart comparing initial vs final scores"""
    if not initial_scores or not final_scores:
        return None
        
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sentences = list(final_scores.keys())
    initial_values = [initial_scores.get(sent, 0) for sent in sentences]
    final_values = [final_scores[sent] for sent in sentences]
    
    x = np.arange(len(sentences))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, initial_values, width, label='Initial Scores', alpha=0.7, color='skyblue')
    bars2 = ax.bar(x + width/2, final_values, width, label='Final PageRank Scores', alpha=0.7, color='coral')
    
    ax.set_xlabel('Sentence Index')
    ax.set_ylabel('Score')
    ax.set_title('Initial vs Final PageRank Scores')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Sent {i+1}' for i in sentences])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def create_performance_chart(processing_times, rouge_scores):
    """Create performance comparison charts"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Processing time chart
    methods = list(processing_times.keys())
    times = list(processing_times.values())
    bars = ax1.bar(methods, times, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax1.set_ylabel('Processing Time (seconds)')
    ax1.set_title('Processing Time by Embedding Method')
    ax1.set_xticklabels(methods, rotation=45)
    
    # Add value labels on bars
    for bar, time in zip(bars, times):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{time:.2f}s', ha='center', va='bottom')
    
    # ROUGE scores chart
    metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
    for method, scores in rouge_scores.items():
        ax2.plot(metrics, [scores['rouge1'], scores['rouge2'], scores['rougeL']], 
                marker='o', label=method, linewidth=2)
    
    ax2.set_ylabel('F1 Score')
    ax2.set_title('ROUGE Scores by Embedding Method')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_hyperparameter_tuning_plot(tuning_results):
    """Create hyperparameter tuning visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    embedders = list(tuning_results.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, embedder in enumerate(embedders):
        if idx >= len(axes):
            break
            
        results = tuning_results[embedder]
        thresholds = [r['threshold'] for r in results]
        rouge_l2 = [r['rouge_l2'] for r in results]
        rouge1 = [r['rouge1'] for r in results]
        rouge2 = [r['rouge2'] for r in results]
        rougeL = [r['rougeL'] for r in results]
        
        ax = axes[idx]
        ax.plot(thresholds, rouge_l2, 'o-', color=colors[idx], linewidth=3, markersize=6, label='ROUGE L2')
        ax.plot(thresholds, rouge1, 's--', color=colors[idx], alpha=0.7, markersize=4, label='ROUGE-1')
        ax.plot(thresholds, rouge2, '^--', color=colors[idx], alpha=0.7, markersize=4, label='ROUGE-2')
        ax.plot(thresholds, rougeL, 'd--', color=colors[idx], alpha=0.7, markersize=4, label='ROUGE-L')
        
        ax.set_xlabel('Similarity Threshold')
        ax.set_ylabel('Score')
        ax.set_title(f'{embedder.upper()} Embedder')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Mark best threshold
        best_idx = rouge_l2.index(max(rouge_l2))
        best_threshold = thresholds[best_idx]
        best_score = rouge_l2[best_idx]
        ax.axvline(x=best_threshold, color='red', linestyle=':', alpha=0.8)
        ax.text(best_threshold + 0.02, best_score - 0.1, 
               f'Best: {best_threshold}\nScore: {best_score:.3f}', 
               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Hide empty subplots
    for idx in range(len(embedders), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Hyperparameter Tuning: Similarity Threshold vs ROUGE Scores', fontsize=16)
    plt.tight_layout()
    return fig

def create_component_analysis_chart(component_results):
    """Create chart showing impact of different graph components"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    configurations = list(component_results.keys())
    rouge_scores = [component_results[config]['rouge_l2'] for config in configurations]
    
    bars = ax.bar(configurations, rouge_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    ax.set_ylabel('ROUGE L2-Norm Score')
    ax.set_title('Impact of Graph Components on Summary Quality')
    ax.set_xticklabels(configurations, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, score in zip(bars, rouge_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def create_sentence_score_distribution(sentence_scores):
    """Create histogram of sentence scores"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    scores = list(sentence_scores.values())
    ax.hist(scores, bins=20, alpha=0.7, color='teal', edgecolor='black')
    ax.set_xlabel('PageRank Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Final PageRank Scores')
    ax.axvline(np.mean(scores), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(scores):.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load your results
diversity_results = pd.read_csv('diversity_pagerank_results.csv')

# Create enhanced visualization
plt.figure(figsize=(15, 10))

# Main performance plot
plt.subplot(2, 2, 1)
plt.plot(diversity_results['diversity_strength'], diversity_results['avg_rouge_l2'], 
         'o-', linewidth=3, markersize=10, label='ROUGE L2-Norm', color='#2ca02c')
plt.plot(diversity_results['diversity_strength'], diversity_results['avg_rouge1'], 
         's--', alpha=0.8, label='ROUGE-1', color='#1f77b4')
plt.plot(diversity_results['diversity_strength'], diversity_results['avg_rouge2'], 
         '^--', alpha=0.8, label='ROUGE-2', color='#ff7f0e')
plt.plot(diversity_results['diversity_strength'], diversity_results['avg_rougeL'], 
         'd--', alpha=0.8, label='ROUGE-L', color='#d62728')

# Highlight best point
best_idx = diversity_results['avg_rouge_l2'].idxmax()
best_strength = diversity_results.loc[best_idx, 'diversity_strength']
best_score = diversity_results.loc[best_idx, 'avg_rouge_l2']
plt.axvline(x=best_strength, color='red', linestyle='--', alpha=0.7, linewidth=2)
plt.plot(best_strength, best_score, 'o', markersize=12, color='red', markeredgewidth=2, markeredgecolor='white')

plt.xlabel('Diversity Strength', fontsize=12, fontweight='bold')
plt.ylabel('ROUGE Score', fontsize=12, fontweight='bold')
plt.title('Diversity-Focused PageRank: Performance vs Diversity Strength', fontsize=14, fontweight='bold')
plt.legend(loc='lower left')
plt.grid(True, alpha=0.3)
plt.ylim(0.25, 0.45)

# Add annotation for best point
plt.annotate(f'Optimal: Strength={best_strength}\nROUGE L2={best_score:.4f}',
             xy=(best_strength, best_score),
             xytext=(best_strength+0.1, best_score-0.05),
             arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8),
             fontweight='bold')

# Processing time plot
plt.subplot(2, 2, 2)
plt.plot(diversity_results['diversity_strength'], diversity_results['avg_time'], 
         'o-', linewidth=3, markersize=10, color='#9467bd')
plt.xlabel('Diversity Strength', fontsize=12, fontweight='bold')
plt.ylabel('Time (seconds)', fontsize=12, fontweight='bold')
plt.title('Processing Time vs Diversity Strength', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# Improvement over baseline plot
plt.subplot(2, 2, 3)
baseline_score = diversity_results.loc[diversity_results['diversity_strength'] == 0.0, 'avg_rouge_l2'].values[0]
improvement = ((diversity_results['avg_rouge_l2'] - baseline_score) / baseline_score) * 100

plt.plot(diversity_results['diversity_strength'], improvement, 
         'o-', linewidth=3, markersize=10, color='#e377c2')
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.xlabel('Diversity Strength', fontsize=12, fontweight='bold')
plt.ylabel('Improvement Over Baseline (%)', fontsize=12, fontweight='bold')
plt.title('Performance Improvement Over Traditional PageRank', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# Highlight the best improvement
best_improvement = improvement.max()
plt.annotate(f'+{best_improvement:.2f}% improvement',
             xy=(best_strength, best_improvement),
             xytext=(best_strength+0.1, best_improvement-1),
             arrowprops=dict(arrowstyle='->', color='purple', lw=1.5),
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="purple", alpha=0.8),
             fontweight='bold')

# Parameter sensitivity plot
plt.subplot(2, 2, 4)
# Calculate the sensitivity (derivative approximation)
strengths = diversity_results['diversity_strength'].values
scores = diversity_results['avg_rouge_l2'].values
sensitivity = np.diff(scores) / np.diff(strengths)

plt.plot(strengths[:-1], sensitivity, 'o-', linewidth=3, markersize=10, color='#8c564b')
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.xlabel('Diversity Strength', fontsize=12, fontweight='bold')
plt.ylabel('Sensitivity (ΔScore/ΔStrength)', fontsize=12, fontweight='bold')
plt.title('Sensitivity Analysis: How Score Changes with Strength', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('enhanced_diversity_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Print comprehensive summary
print("="*70)
print("COMPREHENSIVE DIVERSITY ANALYSIS SUMMARY")
print("="*70)
print(f"Optimal diversity strength: {best_strength:.1f}")
print(f"Best ROUGE L2-Norm: {best_score:.4f}")
print(f"Improvement over baseline: +{((best_score - baseline_score)/baseline_score)*100:.2f}%")
print(f"Processing time: {diversity_results.loc[best_idx, 'avg_time']:.3f}s")
print(f"Number of articles evaluated: {diversity_results.loc[best_idx, 'n_articles']}")
print("="*70)
print("\nIndividual ROUGE scores at optimal setting:")
print(f"  ROUGE-1: {diversity_results.loc[best_idx, 'avg_rouge1']:.4f}")
print(f"  ROUGE-2: {diversity_results.loc[best_idx, 'avg_rouge2']:.4f}")
print(f"  ROUGE-L: {diversity_results.loc[best_idx, 'avg_rougeL']:.4f}")
print("="*70)
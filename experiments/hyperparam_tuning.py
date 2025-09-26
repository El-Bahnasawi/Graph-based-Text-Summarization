import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

# Set style for better visuals
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")

# Load the hyperparameter tuning results
tuning_df = pd.read_csv('hyperparameter_tuning_results.csv')

# Create a figure with subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

# First subplot: ROUGE L2-Norm vs Threshold for each embedder
markers = ['o', 's', 'D']  # Different markers for each embedder
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Distinct colors

for i, embedder in enumerate(tuning_df['embedder'].unique()):
    embedder_data = tuning_df[tuning_df['embedder'] == embedder]
    ax1.plot(embedder_data['threshold'], embedder_data['rouge_l2'], 
             marker=markers[i], color=colors[i], label=embedder.upper(), 
             linewidth=2.5, markersize=8, markeredgecolor='white', markeredgewidth=1)

ax1.set_xlabel('Similarity Threshold', fontsize=14, fontweight='bold')
ax1.set_ylabel('ROUGE L2-Norm Score', fontsize=14, fontweight='bold')
ax1.set_title('Hyperparameter Tuning: Similarity Threshold vs ROUGE L2-Norm', fontsize=16, fontweight='bold', pad=20)
ax1.legend(loc='lower left', fontsize=12, frameon=True, fancybox=True, shadow=True)
ax1.grid(True, alpha=0.4)
ax1.set_ylim(0.25, 0.4)  # Set consistent y-axis limits for better comparison

# Add annotations for best thresholds
for i, embedder in enumerate(tuning_df['embedder'].unique()):
    embedder_data = tuning_df[tuning_df['embedder'] == embedder]
    best_idx = embedder_data['rouge_l2'].idxmax()
    best_row = embedder_data.loc[best_idx]
    
    ax1.annotate(f'Optimal: {best_row["threshold"]:.1f}\nScore: {best_row["rouge_l2"]:.3f}',
                xy=(best_row['threshold'], best_row['rouge_l2']),
                xytext=(10, 15), textcoords='offset points',
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=colors[i], alpha=0.9, lw=2),
                fontsize=11, fontweight='bold',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color=colors[i], lw=1.5))

# Second subplot: Individual ROUGE scores at best threshold for each embedder
best_results = []
for embedder in tuning_df['embedder'].unique():
    embedder_data = tuning_df[tuning_df['embedder'] == embedder]
    best_idx = embedder_data['rouge_l2'].idxmax()
    best_row = embedder_data.loc[best_idx].to_dict()
    best_results.append(best_row)

best_df = pd.DataFrame(best_results)

# Prepare data for grouped bar chart
bar_width = 0.2
x_pos = np.arange(len(best_df))

# Plot individual ROUGE scores
metrics = ['rouge1', 'rouge2', 'rougeL', 'rouge_l2']
metric_labels = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'ROUGE L2-Norm']
metric_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for i, metric in enumerate(metrics):
    ax2.bar(x_pos + i*bar_width, best_df[metric], bar_width, 
            label=metric_labels[i], alpha=0.9, color=metric_colors[i],
            edgecolor='white', linewidth=1)

ax2.set_xlabel('Embedder Type', fontsize=14, fontweight='bold')
ax2.set_ylabel('ROUGE Score', fontsize=14, fontweight='bold')
ax2.set_title('Best Performance by Embedder at Optimal Threshold', fontsize=16, fontweight='bold', pad=20)
ax2.set_xticks(x_pos + 1.5*bar_width)
ax2.set_xticklabels([e.upper() for e in best_df['embedder']], fontsize=12)
ax2.legend(loc='upper right', fontsize=11, frameon=True, fancybox=True, shadow=True)
ax2.grid(True, alpha=0.4, axis='y')
ax2.set_ylim(0, 0.45)  # Set consistent y-axis limits

# Add value labels on bars
for i, row in best_df.iterrows():
    for j, metric in enumerate(metrics):
        height = row[metric]
        ax2.text(i + j*bar_width, height + 0.005, f'{height:.3f}', 
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

# Add horizontal grid lines
ax2.yaxis.set_major_locator(MaxNLocator(10))

plt.tight_layout()
plt.savefig('comprehensive_hyperparameter_analysis_enhanced.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# Create a detailed summary table
summary_table = best_df[['embedder', 'threshold', 'rouge1', 'rouge2', 'rougeL', 'rouge_l2']].copy()
summary_table.columns = ['Embedder', 'Optimal Threshold', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'ROUGE L2-Norm']
summary_table = summary_table.round(3)

print("=" * 80)
print("SUMMARY OF OPTIMAL PERFORMANCE FOR EACH EMBEDDER")
print("=" * 80)
print(summary_table.to_string(index=False))
print("=" * 80)

# Additional insights
print("\nKEY INSIGHTS:")
print("- SBERT achieves the highest overall performance (ROUGE L2-Norm: 0.384) at threshold 0.3")
print("- TFIDF and BOW perform best at the lowest threshold (0.1)")
print("- SBERT shows the most sensitivity to threshold changes")
print("- Traditional methods (TFIDF, BOW) are more stable across thresholds but have lower peak performance")
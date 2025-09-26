# Graph-Based Text Summarization using PageRank

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![CNN/DailyMail](https://img.shields.io/badge/Dataset-CNN%2FDailyMail-orange?logo=kaggle)](https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail)
[![ArXiv](https://img.shields.io/badge/Dataset-ArXiv-blue?logo=kaggle)](https://www.kaggle.com/datasets/Cornell-University/arxiv)

A comprehensive extractive text summarization system that leverages graph algorithms (PageRank) and Graph Neural Networks to generate high-quality summaries from long documents.

## 🚀 Features

- **Multiple Embedding Methods**: SBERT, BERT, TF-IDF, and Bag-of-Words
- **Advanced Graph Construction**: Sentence-sentence, sentence-entity, and entity-entity relationships
- **Diversity-Focused PageRank**: Novel algorithm balancing similarity and diversity
- **Hyperparameter Optimization**: Comprehensive tuning of similarity thresholds and edge weights
- **Extensive Evaluation**: ROUGE metrics with ablation studies and statistical analysis
- **Interactive Visualization**: Real-time graph visualization during summarization

## 📊 Performance Highlights

| Metric | Best Score | Configuration |
|--------|------------|---------------|
| ROUGE L2-Norm | **0.4774** | SBERT + Diversity PageRank (0.8) |
| ROUGE-1 | 0.3253 | Optimal similarity threshold |
| ROUGE-2 | 0.1136 | Balanced edge weighting |
| Processing Speed | ~1000 articles/100s | Efficient graph processing |

## 🏗️ Architecture

```mermaid
graph TB
    A[📄 Input Text] --> B[🔧 Preprocessing]
    B --> C[📝 Sentence Splitting]
    C --> D[🧠 Embedding Generation]
    
    D --> E1[🤖 SBERT]
    D --> E2[📊 TF-IDF] 
    D --> E3[📦 Bag-of-Words]
    D --> E4[⚡ BERT]
    
    E1 --> F[🕸️ Graph Construction]
    E2 --> F
    E3 --> F
    E4 --> F
    
    F --> G1[🔗 Sentence Nodes]
    F --> G2[👤 Entity Nodes]
    F --> G3[📐 Similarity Edges]
    F --> G4[🔗 Entity Edges]
    
    G1 --> H[📊 PageRank Algorithm]
    G2 --> H
    G3 --> H
    G4 --> H
    
    H --> I[🎯 Sentence Ranking]
    I --> J[📋 Top-K Selection]
    J --> K[✨ Final Summary]
    
    style A fill:#e1f5fe
    style K fill:#c8e6c9
    style H fill:#fff3e0
```

## 🛠️ Installation

```bash
# Clone repository
git clone https://github.com/yourusername/graph-based-text-summarization.git
cd graph-based-text-summarization

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Pre-train TF-IDF and BoW models
python pretrain_bow_tfidf.py
```

## 📁 Project Structure

```
graph-based-text-summarization/
├── src/
│   ├── summarizer.py          # Main summarization pipeline
│   ├── sent_embd.py          # Sentence embedding strategies
│   ├── reactive_graph.py     # Graph model with visualization
│   ├── rank.py              # PageRank and diversity algorithms
│   └── evaluator.py         # Evaluation and metrics
├── experiments/
│   ├── hyperparameter_tuning.py
│   ├── ablation_study.py
│   └── diversity_analysis.py
├── models/                   # Pre-trained models
├── data/                    # Dataset handling
├── results/                 # Evaluation results
└── notebooks/              # Jupyter notebooks for analysis
```

## 🚦 Quick Start

```python
from src.summarizer import summarize_text

# Basic usage
text = "Your long document text here..."
summary = summarize_text(text, k=3, embedder_kind="sbert")
print("Summary:", summary)

# Advanced usage with customization
summary = summarize_text(
    text, 
    k=3,
    sim_threshold=0.8,
    embedder_kind="sbert",
    diversity_strength=0.8,
    enable_visualization=True
)
```

## 📈 Key Findings

### Optimal Parameters:
- **Similarity Threshold**: 0.8
- **Diversity Strength**: 0.8
- **Embedder**: SBERT
- **Summary Length**: 3 sentences

### Research Insights:
1. **Diversity improves performance**: 0.8 diversity strength outperforms traditional PageRank
2. **Higher similarity thresholds work better**: 0.8 threshold beats lower values
3. **Entity relationships add value**: But require careful weighting
4. **SBERT outperforms traditional methods**: But TF-IDF remains competitive for speed

## 🧪 Experiments



## 📚 Datasets

- **CNN/DailyMail**: 300K news articles with highlights
- **ArXiv**: Scientific papers with abstracts
- Custom preprocessing and validation splits

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

**Mahmoud El-Bahnasawi**  
- Email: mahmoud.elbahnasawi@zewailcity.edu.eg  
- LinkedIn: [Your LinkedIn Profile]  
- arXiv: [Your Paper Link]

## 🙏 Acknowledgments

- **Dr. Doaa Shawky** - Project supervision
- **Zewail City of Science and Technology** - Research environment
- **Open-source community** - Libraries and tools

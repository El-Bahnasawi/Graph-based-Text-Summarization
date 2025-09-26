from src.Summarizer import summarize_text

# Basic usage
text = """ or nearly 30 years, ArXiv has served the public and research communities by providing open access to scholarly articles, from the vast branches of physics to the many subdisciplines of computer science to everything in between, including math, statistics, electrical engineering, quantitative biology, and economics. This rich corpus of information offers significant, but sometimes overwhelming depth.
In these times of unique global challenges, efficient extraction of insights from data is essential. To help make the arXiv more accessible, we present a free, open pipeline on Kaggle to the machine-readable arXiv dataset: a repository of 1.7 million articles, with relevant features such as article titles, authors, categories, abstracts, full text PDFs, and more.\
Our hope is to empower new use cases that can lead to the exploration of richer machine learning techniques that combine multi-modal features towards applications like trend analysis, paper recommender engines, category prediction, co-citation networks, knowledge graph construction and semantic search interfaces. """

summary = summarize_text(text, k=3, embedder_kind="sbert")
print("Summary:", summary)

# Advanced usage with customization
summary = summarize_text(
    text, 
    k=3,
    sim_threshold=0.8,
    embedder_kind="sbert",
    enable_visualization=True
)
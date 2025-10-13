from sent_embd import create_embedder
import pandas as pd
import os
import random
from tqdm import tqdm

def train_and_save_models():
    # Load your training corpus
    print("Loading training corpus...")
    df = pd.read_csv("Datasets/cnn_dailymail/train.csv")
    
    # Randomly sample 100,000 articles
    if len(df) > 100000:
        df = df.sample(n=100000, random_state=42)  # Use random_state for reproducibility
        print(f"Sampled 100,000 articles from the corpus")
    else:
        print(f"Using all {len(df)} articles (less than 100,000)")
    
    corpus = df["article"].tolist()
    print(f"Final corpus size: {len(corpus)} articles")
    
    # Create directory for models if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Train and save TF-IDF model
    print("Training TF-IDF model...")
    tfidf_embedder = create_embedder("tfidf", max_features=20000, ngram_range=(1, 2))
    
    # Fit the model with progress indication
    tfidf_embedder.fit(corpus)
    
    # Save the fitted model
    tfidf_model_path = "models/tfidf_model.pkl"
    tfidf_embedder.save(tfidf_model_path)
    print(f"TF-IDF model saved to {tfidf_model_path}")
    
    # Train and save BoW model
    print("Training BoW model...")
    bow_embedder = create_embedder("bow", max_features=20000, ngram_range=(1, 2))
    
    # Fit the model
    bow_embedder.fit(corpus)
    
    # Save the fitted model
    bow_model_path = "models/bow_model.pkl"
    bow_embedder.save(bow_model_path)
    print(f"BoW model saved to {bow_model_path}")
    
    print("All models trained and saved successfully!")

if __name__ == "__main__":
    train_and_save_models()
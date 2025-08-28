import json
import numpy as np
import os
from tqdm import tqdm
from typing import List, Dict, Any
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import seaborn as sns
import random
from collections import defaultdict
import argparse

# --- Evaluation Functions ---
def evaluate_matching(embeddings_data: List[Dict[str, Any]], num_neg_samples: int = 5) -> Dict[str, float]:
    """Evaluate concept matching capability"""
    sim_scores = []
    labels = []
    concept_dict = {c["name"]: c for c in embeddings_data}
    
    # Create list of concept names for negative sampling
    all_names = [c["name"] for c in embeddings_data]
    
    for concept in tqdm(embeddings_data, desc="Matching Evaluation"):
        # Positive sample: informal description and formal definition of the same concept
        emb = np.array(concept["embedding"])
        pos_sim = 1 - cosine(emb, emb)  # Compare with self
        sim_scores.append(pos_sim)
        labels.append(1)
        
        # Negative samples: randomly pair different concepts
        for _ in range(num_neg_samples):
            neg_name = random.choice([n for n in all_names if n != concept["name"]])
            neg_concept = concept_dict[neg_name]
            neg_emb = np.array(neg_concept["embedding"])
            neg_sim = 1 - cosine(emb, neg_emb)
            sim_scores.append(neg_sim)
            labels.append(0)
    
    # Calculate AUC
    auc = roc_auc_score(labels, sim_scores)
    return {"matching_auc": auc}

def evaluate_classification(embeddings_data: List[Dict[str, Any]]) -> Dict[str, float]:
    """Evaluate semantic type classification capability"""
    # Prepare data
    embeddings = [np.array(c["embedding"]) for c in embeddings_data]
    semantic_types = [c["semantic_type"] for c in embeddings_data]
    
    # Process semantic type format (could be string or list)
    processed_types = []
    for st in semantic_types:
        if isinstance(st, str):
            processed_types.append([st])
        elif isinstance(st, list):
            processed_types.append(st)
        else:
            processed_types.append([])  # Empty list indicates no type
    
    # Multi-label binarization
    all_types = set()
    for types in processed_types:
        all_types.update(types)
    
    # If no types or too few types, return default values
    if not all_types or len(embeddings) < 2:
        return {"micro_f1": 0.0, "macro_f1": 0.0}
    
    mlb = MultiLabelBinarizer()
    y_bin = mlb.fit_transform(processed_types)
    
    # If too few samples, use leave-one-out validation
    if len(embeddings) < 5:
        test_size = 0.4
    else:
        test_size = 0.2
    
    # Use PCA for dimensionality reduction
    pca = PCA(n_components=min(50, len(embeddings[0])), random_state=42)
    try:
        embeddings_pca = pca.fit_transform(embeddings)
    except Exception as e:
        print(f"PCA failed, using original embeddings: {str(e)}")
        embeddings_pca = embeddings
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings_pca, y_bin, test_size=test_size, random_state=42
    )
    
    # Use multi-label classifier
    try:
        clf = OneVsRestClassifier(LogisticRegression(max_iter=1000, n_jobs=-1))
        clf.fit(X_train, y_train)
        
        # Evaluate
        y_pred = clf.predict(X_test)
        micro_f1 = f1_score(y_test, y_pred, average="micro")
        macro_f1 = f1_score(y_test, y_pred, average="macro")
    except Exception as e:
        print(f"Classification failed: {str(e)}")
        micro_f1, macro_f1 = 0.0, 0.0
    
    return {"micro_f1": micro_f1, "macro_f1": macro_f1}

def evaluate_clustering(embeddings_data: List[Dict[str, Any]]) -> Dict[str, float]:
    """Evaluate clustering quality in embedding space"""
    # Prepare data
    embeddings = [np.array(c["embedding"]) for c in embeddings_data]
    
    # If insufficient samples, return default value
    if len(embeddings) < 2:
        return {"silhouette_score": 0.0}
    
    # Use PCA for dimensionality reduction
    pca = PCA(n_components=min(50, len(embeddings[0])), random_state=42)
    try:
        embeddings_pca = pca.fit_transform(embeddings)
    except Exception as e:
        print(f"PCA failed, using original embeddings: {str(e)}")
        embeddings_pca = embeddings
    
    # Determine number of clusters (based on semantic types)
    semantic_types = []
    for c in embeddings_data:
        st = c["semantic_type"]
        if isinstance(st, str):
            semantic_types.append(st)
        elif isinstance(st, list) and st:
            semantic_types.append(st[0])  # Use first type as representative
        else:
            semantic_types.append("unknown")
    
    unique_types = len(set(semantic_types))
    
    # Ensure reasonable number of clusters
    n_clusters = min(unique_types, len(embeddings))
    if n_clusters < 2:
        n_clusters = min(5, len(embeddings))  # Use default cluster count
    
    # Clustering
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_pca)
        
        # Evaluate with silhouette score
        sil_score = silhouette_score(embeddings_pca, cluster_labels)
    except Exception as e:
        print(f"Clustering failed: {str(e)}")
        sil_score = 0.0
    
    return {"silhouette_score": sil_score}

# --- Visualization ---
def plot_results(model_results: Dict[str, Dict[str, Dict[str, float]]], output_dir: str = "."):
    """Visualize model performance comparison results"""
    # Prepare data
    metrics_data = defaultdict(list)
    model_names = list(model_results.keys())
    
    for model, results in model_results.items():
        metrics_data["Model"].append(model)
        metrics_data["Matching AUC"].append(results["matching"]["matching_auc"])
        metrics_data["Classification Micro-F1"].append(results["classification"]["micro_f1"])
        metrics_data["Clustering Silhouette"].append(results["clustering"]["silhouette_score"])
    
    # Create plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Matching AUC
    if model_names:
        sns.barplot(x="Model", y="Matching AUC", data=metrics_data, ax=axes[0])
        axes[0].set_title("Concept Matching AUC")
        axes[0].set_ylim(0, 1.05)
        axes[0].tick_params(axis='x', rotation=45)
    
    # Classification Micro-F1
    if model_names:
        sns.barplot(x="Model", y="Classification Micro-F1", data=metrics_data, ax=axes[1])
        axes[1].set_title("Semantic Type Classification (Micro-F1)")
        axes[1].set_ylim(0, 1.05)
        axes[1].tick_params(axis='x', rotation=45)
    
    # Clustering Silhouette
    if model_names:
        sns.barplot(x="Model", y="Clustering Silhouette", data=metrics_data, ax=axes[2])
        axes[2].set_title("Clustering Quality (Silhouette Score)")
        axes[2].set_ylim(-0.1, 1.0)
        axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "model_performance_comparison.png")
    plt.savefig(plot_path)
    plt.show()
    return plot_path

# --- Main Execution Function ---
def analyze_embeddings(embedding_dir: str = "./embedding_data", output_dir: str = "."):
    """Analyze generated embedding results"""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all embedding files
    embedding_files = [f for f in os.listdir(embedding_dir) 
                      if f.startswith("concepts_with_embeddings_") and f.endswith(".json")]
    
    if not embedding_files:
        print(f"No embedding files found in {embedding_dir}")
        return
    
    # Store performance results
    model_results = {}
    
    # Analyze each embedding file
    for file in embedding_files:
        try:
            # Extract model name from filename
            model_name = file.replace("concepts_with_embeddings_", "").replace(".json", "")
            print(f"\n{'='*50}")
            print(f"Analyzing embeddings for model: {model_name}")
            print(f"{'='*50}")
            
            # Load embedding data
            file_path = os.path.join(embedding_dir, file)
            with open(file_path, "r", encoding="utf-8") as f:
                embeddings_data = json.load(f)
            
            print(f"Loaded {len(embeddings_data)} embeddings")
            
            # Evaluate model
            results = {
                "matching": evaluate_matching(embeddings_data),
                "classification": evaluate_classification(embeddings_data),
                "clustering": evaluate_clustering(embeddings_data)
            }
            
            model_results[model_name] = results
            
            # Print current model results
            print(f"\n{model_name} Performance:")
            print(f"- Matching AUC: {results['matching']['matching_auc']:.4f}")
            print(f"- Classification Micro-F1: {results['classification']['micro_f1']:.4f}")
            print(f"- Clustering Silhouette: {results['clustering']['silhouette_score']:.4f}")
        except Exception as e:
            print(f"Error analyzing {file}: {str(e)}")
    
    # Print final comparison results
    if model_results:
        print("\n\n" + "="*60)
        print("Embedding Performance Comparison Summary")
        print("="*60)
        print(f"{'Model':<20} {'Matching AUC':<12} {'Micro-F1':<10} {'Silhouette':<10}")
        for model, res in model_results.items():
            print(f"{model:<20} {res['matching']['matching_auc']:.4f}{'':<8} "
                  f"{res['classification']['micro_f1']:.4f}{'':<8} "
                  f"{res['clustering']['silhouette_score']:.4f}")
        
        # Save results
        results_path = os.path.join(output_dir, "embedding_performance_results.json")
        with open(results_path, "w") as f:
            json.dump(model_results, f, indent=2)
        
        # Visualize results
        plot_path = plot_results(model_results, output_dir)
        
        print(f"\nAnalysis complete. Results saved to:")
        print(f"- Performance results: {results_path}")
        print(f"- Comparison plot: {plot_path}")
    else:
        print("\nNo valid analysis results to display")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze embedding performance")
    parser.add_argument("--embedding_dir", default="./embedding_data", 
                        help="Directory containing embedding JSON files")
    parser.add_argument("--output_dir", default=".", 
                        help="Directory to save analysis results")
    args = parser.parse_args()
    
    analyze_embeddings(args.embedding_dir, args.output_dir)
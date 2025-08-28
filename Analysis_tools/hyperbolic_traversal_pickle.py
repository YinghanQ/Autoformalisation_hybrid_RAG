import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModel

# Configure Matplotlib for proper font rendering
plt.rcParams["font.family"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False  # Properly render negative signs


# ----------------------
# One-shot Sentence Embedding
# ----------------------
class QueryEmbedder:
    def __init__(self, model_name_or_path, device="cpu"):
        """Initialize sentence encoder (consistent with original embedding model)"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.device = device
        self.model.to(device)
        self.model.eval()

    def preprocess(self, text):
        """Preprocess text (consistent with original data processing)"""
        return self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(self.device)

    def embed(self, text):
        """Generate one-shot embedding for the input text"""
        inputs = self.preprocess(text)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Use [CLS] token's hidden state as sentence embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        return cls_embedding


# ----------------------
# Hyperbolic Space Core Operations
# ----------------------
def artanh(x):
    x = np.clip(x, -1 + 1e-10, 1 - 1e-10)
    return 0.5 * np.log((1 + x) / (1 - x))

def p_log_map(u):
    norm_u = np.linalg.norm(u, ord=2)
    if norm_u < 1e-10:
        return u
    return (u / norm_u) * artanh(norm_u)

def p_exp_map(v):
    norm_v = np.linalg.norm(v, ord=2)
    if norm_v < 1e-10:
        return v
    return (v / norm_v) * np.tanh(norm_v)

def hyperbolic_interpolation(u, v, num_steps=10):
    u_tangent = p_log_map(u)
    v_tangent = p_log_map(v)
    steps = np.linspace(0, 1, num_steps)
    interpolated_tangent = [(1 - t) * u_tangent + t * v_tangent for t in steps]
    return [p_exp_map(tangent_vec) for tangent_vec in interpolated_tangent]

def euclidean_interpolation(u, v, num_steps=10):
    return [(1 - t) * u + t * v for t in np.linspace(0, 1, num_steps)]


# ----------------------
# Distance Calculation Functions
# ----------------------
def poincare_distance(u, v):
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    numerator = 2 * np.linalg.norm(u - v)**2
    denominator = (1 - norm_u**2) * (1 - norm_v**2)
    denominator = max(denominator, 1e-10)
    return artanh(np.sqrt(numerator / denominator))

def euclidean_distance(u, v):
    return np.linalg.norm(u - v)


# ----------------------
# Nearest Neighbor Search
# ----------------------
def find_nearest_entities(target_vec, vectors, vocab, top_k=10, distance_type="hyperbolic"):
    distances = []
    for vec in vectors:
        if distance_type == "hyperbolic":
            dist = poincare_distance(target_vec, vec)
        else:
            dist = euclidean_distance(target_vec, vec)
        distances.append(dist)
    sorted_indices = np.argsort(distances)[:top_k]
    return [(vocab[idx], distances[idx]) for idx in sorted_indices]


# ----------------------
# Path Analysis Functions
# ----------------------
def jaccard_similarity(set1, set2):
    """Calculate Jaccard similarity between two sets"""
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0

def compute_path_similarities(path):
    """Calculate Jaccard similarities between consecutive points in a path"""
    return [
        jaccard_similarity(set(path[i - 1]), set(path[i]))
        for i in range(1, len(path))
    ]

def midpoint_generalization(path):
    """Calculate Jaccard similarities between midpoint and endpoints"""
    midpoint = set(path[len(path)//2])
    start = set(path[0])
    end = set(path[-1])
    sim_start = jaccard_similarity(midpoint, start)
    sim_end = jaccard_similarity(midpoint, end)
    return sim_start, sim_end, jaccard_similarity(start, end)

def generalization_score(mid_sim_start, mid_sim_end):
    """Define generalization metric"""
    return 1 - max(mid_sim_start, mid_sim_end)

def analyze_and_visualize_paths(hyperbolic_path, euclidean_path, output_dir="analysis_visualizations"):
    """Perform comprehensive path analysis and generate visualizations"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate similarities for both paths
    euclidean_sim = compute_path_similarities(euclidean_path)
    hyperbolic_sim = compute_path_similarities(hyperbolic_path)
    
    # Midpoint analysis
    euc_mid_sim_start, euc_mid_sim_end, euc_start_end = midpoint_generalization(euclidean_path)
    hyp_mid_sim_start, hyp_mid_sim_end, hyp_start_end = midpoint_generalization(hyperbolic_path)
    
    # Generalization scores
    hyp_gen_score = generalization_score(hyp_mid_sim_start, hyp_mid_sim_end)
    euc_gen_score = generalization_score(euc_mid_sim_start, euc_mid_sim_end)
    
    # Plot similarity curves
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(euclidean_sim)+1), euclidean_sim, label='Euclidean', marker='o', linestyle='-', color='blue')
    plt.plot(range(1, len(hyperbolic_sim)+1), hyperbolic_sim, label='Hyperbolic', marker='s', linestyle='-', color='red')
    plt.xlabel('Step')
    plt.ylabel('Jaccard Similarity')
    plt.title('Local Semantic Continuity Along Paths')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    similarity_plot_path = os.path.join(output_dir, 'path_similarities.png')
    plt.savefig(similarity_plot_path, dpi=300)
    plt.close()
    
    # Print analysis results
    print("\n=== Path Analysis Results ===")
    
    print("\nEuclidean Path:")
    print(f"Average local similarity: {np.mean(euclidean_sim):.3f}")
    print(f"Midpoint vs Start: {euc_mid_sim_start:.3f}")
    print(f"Midpoint vs End:   {euc_mid_sim_end:.3f}")
    print(f"Start vs End:      {euc_start_end:.3f}")
    print(f"Generalization Score: {euc_gen_score:.3f}")
    
    print("\nHyperbolic Path:")
    print(f"Average local similarity: {np.mean(hyperbolic_sim):.3f}")
    print(f"Midpoint vs Start: {hyp_mid_sim_start:.3f}")
    print(f"Midpoint vs End:   {hyp_mid_sim_end:.3f}")
    print(f"Start vs End:      {hyp_start_end:.3f}")
    print(f"Generalization Score: {hyp_gen_score:.3f}")
    
    return {
        "euclidean": {
            "similarities": euclidean_sim,
            "midpoint_similarities": (euc_mid_sim_start, euc_mid_sim_end),
            "start_end_similarity": euc_start_end,
            "generalization_score": euc_gen_score
        },
        "hyperbolic": {
            "similarities": hyperbolic_sim,
            "midpoint_similarities": (hyp_mid_sim_start, hyp_mid_sim_end),
            "start_end_similarity": hyp_start_end,
            "generalization_score": hyp_gen_score
        }
    }


# ----------------------
# Latent Space Traversal
# ----------------------
def traverse_latent_space(word1, word2, embeddings_dict, vocab, vectors, num_steps=10, top_k=10, interpolation_type="hyperbolic"):
    """Perform latent space traversal between two entities"""
    if word1 not in embeddings_dict:
        print(f"Error: Entity '{word1}' not found in embeddings")
        return None
    if word2 not in embeddings_dict:
        print(f"Error: Entity '{word2}' not found in embeddings")
        return None
    
    u = embeddings_dict[word1]
    v = embeddings_dict[word2]
    
    if interpolation_type == "hyperbolic":
        interpolated_points = hyperbolic_interpolation(u, v, num_steps=num_steps)
        interpolation_method = "Hyperbolic (geodesic) interpolation"
        distance_type = "hyperbolic"
    else:
        interpolated_points = euclidean_interpolation(u, v, num_steps=num_steps)
        interpolation_method = "Euclidean (linear) interpolation"
        distance_type = "euclidean"
    
    print(f"=== Latent Space Traversal: {word1} -> {word2} ===")
    print(f"Using {interpolation_method}\n")
    
    results = []
    path_entities = []  # Store entities along the path for later analysis
    
    for i, point in enumerate(interpolated_points):
        t = i / (num_steps - 1)
        nearest = find_nearest_entities(point, vectors, vocab, top_k=top_k, distance_type=distance_type)
        entity_names = [entity for entity, _ in nearest]
        path_entities.append(entity_names)
        
        print(f"Point {i+1} (t={t:.2f}):")
        print(f"  Nearest entities: {entity_names}\n")
        
        results.append({
            "step": i, 
            "t": t, 
            "vector": point, 
            "nearest": nearest
        })
    
    # Midpoint analysis
    midpoint_idx = num_steps // 2
    mid_t = results[midpoint_idx]['t']
    mid_nearest = results[midpoint_idx]['nearest']
    mid_entity_names = [entity for entity, _ in mid_nearest]
    
    print(f"*** MIDPOINT ANALYSIS (t={mid_t:.1f}) ***")
    print(f"Entities near midpoint: {mid_entity_names}")
    
    # Save midpoint embedding
    mid_vector = results[midpoint_idx]["vector"]
    os.makedirs("latent_output", exist_ok=True)
    filename = f"latent_output/{word1}_{word2}_{interpolation_type}_midpoint.npy"
    np.save(filename, mid_vector)
    print(f"Saved midpoint embedding to: {filename}\n")
    
    # Return both results and path entities for analysis
    return {
        "results": results,
        "path_entities": path_entities,
        "interpolation_type": interpolation_type,
        "word_pair": (word1, word2)
    }


# ----------------------
# 3D Visualization
# ----------------------
def visualize_traversal_3d(word1, word2, traversal_results, all_vectors, interpolation_type="hyperbolic"):
    """Generate 3D visualization of the traversal path using PCA"""
    from sklearn.decomposition import PCA
    
    interpolated_vectors = [res["vector"] for res in traversal_results]
    all_vectors_pca = np.vstack([all_vectors, interpolated_vectors])
    
    pca = PCA(n_components=3)
    pca.fit(all_vectors_pca)
    
    background_3d = pca.transform(all_vectors)
    path_3d = pca.transform(interpolated_vectors)
    
    fig = plt.figure(figsize=(16, 14))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(
        background_3d[:, 0], background_3d[:, 1], background_3d[:, 2],
        c='lightgray', alpha=0.2, s=10, label='All entities'
    )
    
    ax.plot(
        path_3d[:, 0], path_3d[:, 1], path_3d[:, 2],
        'r-', linewidth=2, label='Traversal path'
    )
    
    ax.scatter(path_3d[0, 0], path_3d[0, 1], path_3d[0, 2],
               c='green', s=100, marker='*', label=f'Start: {word1}')
    ax.scatter(path_3d[-1, 0], path_3d[-1, 1], path_3d[-1, 2],
               c='red', s=100, marker='*', label=f'End: {word2}')
    mid_idx = len(path_3d) // 2
    ax.scatter(path_3d[mid_idx, 0], path_3d[mid_idx, 1], path_3d[mid_idx, 2],
               c='blue', s=100, marker='*', label='Midpoint')
    
    ax.set_title(f'Traversal from {word1} to {word2} in {interpolation_type} space')
    ax.set_xlabel('PCA Dimension 1')
    ax.set_ylabel('PCA Dimension 2')
    ax.set_zlabel('PCA Dimension 3')
    ax.legend()
    
    os.makedirs("visualizations", exist_ok=True)
    filename = f"visualizations/3d_traversal_{word1}_{word2}_{interpolation_type}.png"
    plt.savefig(filename, dpi=300)
    print(f"Generating 3D visualization...")
    print(f"Saved 3D visualization to: {filename}")
    plt.close(fig)


# ----------------------
# Query Embedding and Analysis
# ----------------------
def embed_and_analyze_query(query, model_path, embeddings_dict, vocab, vectors, top_k=10, distance_type="hyperbolic"):
    """Embed a query sentence and analyze its nearest neighbors in the embedding space"""
    embedder = QueryEmbedder(model_path)
    query_embedding = embedder.embed(query)
    print(f"\n=== One-shot Embedding for Query: '{query}' ===")
    print(f"Embedding shape: {query_embedding.shape}")
    
    # Project to hyperbolic space if needed
    if distance_type == "hyperbolic":
        query_embedding = p_exp_map(query_embedding / np.linalg.norm(query_embedding) * 0.5)
    
    # Find nearest entities
    nearest_entities = find_nearest_entities(
        query_embedding, 
        vectors, 
        vocab, 
        top_k=top_k,
        distance_type=distance_type
    )
    
    # Print results
    print(f"\nTop {top_k} nearest entities:")
    for i, (entity, distance) in enumerate(nearest_entities, 1):
        print(f"  {i}. {entity} (distance: {distance:.4f})")
    
    # Save query embedding
    os.makedirs("query_embeddings", exist_ok=True)
    query_filename = f"query_embeddings/query_{query[:30].replace(' ', '_')}_{distance_type}.npy"
    np.save(query_filename, query_embedding)
    print(f"\nSaved query embedding to: {query_filename}")
    
    return query_embedding


# ----------------------
# Main Function
# ----------------------
def main():
    parser = argparse.ArgumentParser(description='Hyperbolic/Euclidean Embedding Analysis Tool')
    # Basic parameters
    parser.add_argument("--embedding_path", type=str, help="Path to pickle embedding dictionary")
    parser.add_argument("--model", type=str, default="hyperbolic", choices=["hyperbolic", "euclidean"])
    parser.add_argument("--dim", type=int, default=300, help="Embedding dimensionality")
    parser.add_argument("--num_points", type=int, default=10, help="Number of interpolation points")
    parser.add_argument("--top_k", type=int, default=10, help="Number of nearest entities to display")
    parser.add_argument("--visualize", action='store_true', help="Generate 3D visualization")
    parser.add_argument("--analyze_paths", action='store_true', help="Perform path similarity analysis")
    
    # Word pair parameters
    parser.add_argument("--word_pairs", type=str, help='JSON string of word pairs, e.g., "[[\"a\",\"b\"], [\"c\",\"d\"]]"')
    parser.add_argument("--word_pairs_file", type=str, help="Path to JSON file containing word pairs")
    
    # Query embedding parameters
    parser.add_argument("--query", type=str, help="Query sentence to embed (one-shot)")
    parser.add_argument("--encoder_path", type=str, help="Path to pre-trained encoder model for one-shot embedding")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="analysis_results", help="Directory for output files")
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    analysis_vis_dir = os.path.join(args.output_dir, "analysis_visualizations")
    os.makedirs(analysis_vis_dir, exist_ok=True)
    
    # Load existing embeddings if available
    embeddings_dict = None
    vocab = []
    vectors = None
    if args.embedding_path and os.path.exists(args.embedding_path):
        with open(args.embedding_path, 'rb') as f:
            embeddings_dict = pickle.load(f)
        vocab = list(embeddings_dict.keys())
        vectors = np.array([embeddings_dict[word] for word in vocab])
        print(f"Loaded embedding dictionary with {len(embeddings_dict)} entities")
    else:
        if args.word_pairs or args.word_pairs_file or args.analyze_paths:
            print("Error: --embedding_path is required for word pair traversal and path analysis")
            return
        print("Warning: No existing embeddings loaded - only query embedding will be processed")
    
    # Process query embedding if specified
    if args.query:
        if not args.encoder_path:
            print("Error: --encoder_path is required for query embedding")
            return
        if vectors is not None:
            embed_and_analyze_query(
                query=args.query,
                model_path=args.encoder_path,
                embeddings_dict=embeddings_dict,
                vocab=vocab,
                vectors=vectors,
                top_k=args.top_k,
                distance_type=args.model
            )
        else:
            # Generate embedding without neighbor analysis
            embedder = QueryEmbedder(args.encoder_path)
            query_embedding = embedder.embed(args.query)
            query_dir = os.path.join(args.output_dir, "query_embeddings")
            os.makedirs(query_dir, exist_ok=True)
            query_filename = f"{query_dir}/query_{args.query[:30].replace(' ', '_')}.npy"
            np.save(query_filename, query_embedding)
            print(f"Saved query embedding to: {query_filename}")
    
    # Process word pair traversal if specified
    traversal_results = {}
    if args.word_pairs or args.word_pairs_file:
        # Load word pairs
        word_pairs = []
        if args.word_pairs_file and os.path.exists(args.word_pairs_file):
            with open(args.word_pairs_file, 'r') as f:
                word_pairs = json.load(f)
        elif args.word_pairs:
            try:
                word_pairs = json.loads(args.word_pairs)
            except json.JSONDecodeError:
                print("Error: Invalid JSON format for --word_pairs")
                print("Example format: '[[\"word1\",\"word2\"], [\"word3\",\"word4\"]]'")
                return
        
        # Validate word pairs format
        if not isinstance(word_pairs, list) or not all(isinstance(pair, list) and len(pair) == 2 for pair in word_pairs):
            print("Error: Word pairs must be a list of 2-element lists")
            print("Example: [[\"word1\",\"word2\"], [\"word3\",\"word4\"]]")
            return
        
        # Perform traversal for each word pair
        for word1, word2 in word_pairs:
            key = f"{word1}_{word2}"
            traversal_results[key] = {}
            
            # Run hyperbolic traversal if needed
            if args.model == "hyperbolic" or args.analyze_paths:
                hyp_result = traverse_latent_space(
                    word1=word1, word2=word2,
                    embeddings_dict=embeddings_dict,
                    vocab=vocab, vectors=vectors,
                    num_steps=args.num_points,
                    top_k=args.top_k,
                    interpolation_type="hyperbolic"
                )
                traversal_results[key]["hyperbolic"] = hyp_result
                
                if args.visualize and hyp_result:
                    visualize_traversal_3d(
                        word1, word2, 
                        hyp_result["results"], 
                        vectors,
                        interpolation_type="hyperbolic"
                    )
            
            # Run euclidean traversal if needed for comparison
            if args.model == "euclidean" or args.analyze_paths:
                euc_result = traverse_latent_space(
                    word1=word1, word2=word2,
                    embeddings_dict=embeddings_dict,
                    vocab=vocab, vectors=vectors,
                    num_steps=args.num_points,
                    top_k=args.top_k,
                    interpolation_type="euclidean"
                )
                traversal_results[key]["euclidean"] = euc_result
                
                if args.visualize and euc_result:
                    visualize_traversal_3d(
                        word1, word2, 
                        euc_result["results"], 
                        vectors,
                        interpolation_type="euclidean"
                    )
    
    # Perform path analysis if requested
    if args.analyze_paths and traversal_results:
        for pair_key, results in traversal_results.items():
            if "hyperbolic" in results and "euclidean" in results:
                print(f"\n=== Path Comparison Analysis: {pair_key} ===")
                analyze_and_visualize_paths(
                    results["hyperbolic"]["path_entities"],
                    results["euclidean"]["path_entities"],
                    output_dir=analysis_vis_dir
                )
    
    print("\nAll operations completed!")


if __name__ == "__main__":
    main()
    
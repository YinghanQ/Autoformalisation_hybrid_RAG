import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
from load_data import Data
from model import *
from utils import *
import argparse
import os
import matplotlib.cm as cm

class LatentSpaceTraversal:
    def __init__(self, model_path, d, model_name="poincare", dim=40, cuda=False):
        self.d = d
        self.model_name = model_name
        self.dim = dim
        self.device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
        
        # Build index mappings for entities and relations
        self.entity_idxs = {self.d.entities[i]: i for i in range(len(self.d.entities))}
        self.idx_entities = {i: self.d.entities[i] for i in range(len(self.d.entities))}
        self.relation_idxs = {self.d.relations[i]: i for i in range(len(self.d.relations))}
        
        # Load trained model
        self.model = self.load_model(model_path)
        
    def load_model(self, model_path):
        """Load trained model"""
        if self.model_name == "poincare":
            model = MuRP(self.d, self.dim, self.device).to(self.device)
        else:
            model = MuRE(self.d, self.dim).to(self.device)
        
        # Load saved model parameters
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model
    
    def get_entity_embedding(self, entity_name):
        """Get entity embedding vector"""
        if entity_name not in self.entity_idxs:
            raise ValueError(f"Entity '{entity_name}' not found in vocabulary")
        
        entity_idx = self.entity_idxs[entity_name]
        with torch.no_grad():
            if self.model_name == "poincare":
                # For hyperbolic model, use Eh embedding
                embedding = self.model.Eh.weight[entity_idx].clone()
            else:
                # For Euclidean model, use E embedding
                embedding = self.model.E.weight[entity_idx].clone()
        return embedding
    
    def hyperbolic_interpolation(self, start_emb, end_emb, num_points=10):
        """Perform geodesic interpolation in hyperbolic space"""
        interpolated_points = []
        
        # Ensure points are in hyperbolic space (norm < 1)
        eps = 1e-5
        start_norm = torch.norm(start_emb, 2)
        end_norm = torch.norm(end_emb, 2)
        
        if start_norm >= 1:
            start_emb = start_emb / (start_norm + eps)
        if end_norm >= 1:
            end_emb = end_emb / (end_norm + eps)
        
        for i in range(num_points):
            t = i / (num_points - 1)  # Interpolation parameter from 0 to 1
            
            # Geodesic interpolation in hyperbolic space
            # Use exponential and logarithmic maps for interpolation
            start_tangent = p_log_map(start_emb)
            end_tangent = p_log_map(end_emb)
            
            # Linear interpolation in tangent space
            interpolated_tangent = (1 - t) * start_tangent + t * end_tangent
            
            # Map back to hyperbolic space
            interpolated_point = p_exp_map(interpolated_tangent)
            
            # Ensure point is in hyperbolic space
            norm = torch.norm(interpolated_point, 2)
            if norm >= 1:
                interpolated_point = interpolated_point / (norm + eps)
            
            interpolated_points.append(interpolated_point)
        
        return torch.stack(interpolated_points)
    
    def euclidean_interpolation(self, start_emb, end_emb, num_points=10):
        """Perform linear interpolation in Euclidean space"""
        interpolated_points = []
        
        for i in range(num_points):
            t = i / (num_points - 1)  # Interpolation parameter from 0 to 1
            interpolated_point = (1 - t) * start_emb + t * end_emb
            interpolated_points.append(interpolated_point)
        
        return torch.stack(interpolated_points)
    
    def find_nearest_entities(self, target_embedding, top_k=5):
        """Find entities most similar to target embedding"""
        similarities = []
        
        with torch.no_grad():
            if self.model_name == "poincare":
                # Hyperbolic distance calculation
                all_embeddings = self.model.Eh.weight
                for i, emb in enumerate(all_embeddings):
                    # Ensure embedding is in hyperbolic space
                    norm = torch.norm(emb, 2)
                    if norm >= 1:
                        emb = emb / (norm + 1e-5)
                    
                    target_norm = torch.norm(target_embedding, 2)
                    if target_norm >= 1:
                        target_embedding = target_embedding / (target_norm + 1e-5)
                    
                    # Calculate hyperbolic distance
                    diff = p_sum(-emb, target_embedding)
                    dist = 2 * artanh(torch.clamp(torch.norm(diff, 2), 1e-10, 1-1e-5))
                    similarities.append((i, -dist.item()))  # Negative distance as similarity
            else:
                # Euclidean distance calculation
                all_embeddings = self.model.E.weight
                for i, emb in enumerate(all_embeddings):
                    dist = torch.norm(emb - target_embedding, 2)
                    similarities.append((i, -dist.item()))  # Negative distance as similarity
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k most similar entities
        nearest_entities = []
        for i in range(min(top_k, len(similarities))):
            entity_idx, similarity = similarities[i]
            entity_name = self.idx_entities[entity_idx]
            nearest_entities.append((entity_name, similarity))
        
        return nearest_entities
    
    def traverse_latent_space(self, entity1, entity2, num_points=10, top_k=3):
        """Perform latent space traversal between two entities"""
        print(f"\n=== Latent Space Traversal: {entity1} -> {entity2} ===")
        
        # Get embeddings for both entities
        try:
            start_emb = self.get_entity_embedding(entity1)
            end_emb = self.get_entity_embedding(entity2)
        except ValueError as e:
            print(f"Error: {e}")
            return None
        
        # Choose interpolation method based on model type
        if self.model_name == "poincare":
            interpolated_points = self.hyperbolic_interpolation(start_emb, end_emb, num_points)
            print(f"Using hyperbolic (geodesic) interpolation")
        else:
            interpolated_points = self.euclidean_interpolation(start_emb, end_emb, num_points)
            print(f"Using Euclidean (linear) interpolation")
        
        # Analyze each interpolation point
        results = []
        for i, point in enumerate(interpolated_points):
            t = i / (num_points - 1)
            nearest_entities = self.find_nearest_entities(point, top_k)
            
            print(f"\nPoint {i+1} (t={t:.2f}):")
            print(f"  Nearest entities: {[entity for entity, _ in nearest_entities]}")
            
            results.append({
                'position': t,
                'embedding': point,
                'nearest_entities': nearest_entities
            })
        
        # Analyze midpoint (most important result)
        midpoint_idx = num_points // 2
        midpoint_embedding = results[midpoint_idx]['embedding']
        midpoint_entities = results[midpoint_idx]['nearest_entities']
        
        print(f"\n*** MIDPOINT ANALYSIS (t=0.5) ***")
        print(f"Entities near midpoint: {[entity for entity, _ in midpoint_entities]}")
        
        # Create output directory if it doesn't exist
        os.makedirs("latent_output", exist_ok=True)
        
        # Save midpoint embedding vector
        filename = f"latent_output/{entity1}_{entity2}_{self.model_name}_midpoint.pt"
        torch.save(midpoint_embedding, filename)
        print(f"Saved midpoint embedding to: {filename}")
        
        return results
    
    def run_traversal_experiments(self, word_pairs, num_points=10, top_k=10):
        """Run a series of traversal experiments"""
        print(f"Running Latent Space Traversal experiments with {self.model_name} model")
        print(f"Model device: {self.device}")
        print(f"Number of interpolation points: {num_points}")
        print(f"Top-k nearest entities: {top_k}")
        
        all_results = {}
        
        for category, pairs in word_pairs.items():
            print(f"\n{'='*50}")
            print(f"Category: {category}")
            print(f"{'='*50}")
            
            category_results = {}
            for entity1, entity2 in pairs:
                result = self.traverse_latent_space(entity1, entity2, num_points, top_k)
                if result:
                    category_results[f"{entity1}_{entity2}"] = result
            
            all_results[category] = category_results
        
        return all_results
    
    def format_traversal_output(self, word_pairs, num_points=10, top_k=10):
        """Format traversal results output"""
        print(f"\n{'='*80}")
        print("LATENT SPACE TRAVERSAL RESULTS")
        print(f"{'='*80}")
        print(f"Model: {self.model_name.upper()}")
        print(f"{'='*80}")
        
        for category, pairs in word_pairs.items():
            for entity1, entity2 in pairs:
                try:
                    # Get embeddings for both entities
                    start_emb = self.get_entity_embedding(entity1)
                    end_emb = self.get_entity_embedding(entity2)
                    
                    # Choose interpolation method based on model type
                    if self.model_name == "poincare":
                        interpolated_points = self.hyperbolic_interpolation(start_emb, end_emb, num_points)
                    else:
                        interpolated_points = self.euclidean_interpolation(start_emb, end_emb, num_points)
                    
                    # Get nearest neighbors for midpoint
                    midpoint_idx = num_points // 2
                    midpoint_embedding = interpolated_points[midpoint_idx]
                    nearest_entities = self.find_nearest_entities(midpoint_embedding, top_k)
                    
                    # Create output directory and save midpoint embedding
                    os.makedirs("latent_output", exist_ok=True)
                    filename = f"latent_output/{entity1}_{entity2}_{self.model_name}_midpoint.pt"
                    torch.save(midpoint_embedding, filename)
                    
                    # Format output
                    entity_names = [entity for entity, _ in nearest_entities]
                    entity_names_str = ", ".join(entity_names)
                    
                    print(f"{category}\t{entity1} - {entity2}\t{entity_names_str}\t(Midpoint saved to: {filename})")
                    
                except ValueError as e:
                    print(f"{category}\t{entity1} - {entity2}\tError: {e}")
                except Exception as e:
                    print(f"{category}\t{entity1} - {entity2}\tError: {e}")
        
        print(f"{'='*80}")
    
    def compare_euclidean_hyperbolic_results(self, word_pairs, num_points=10, top_k=10):
        """Compare traversal results between Euclidean and hyperbolic spaces"""
        print(f"\n{'='*100}")
        print("EUCLIDEAN vs HYPERBOLIC LATENT SPACE TRAVERSAL COMPARISON")
        print(f"{'='*100}")
        print(f"{'Category':<15} {'Word Pair':<20} {'Euclidean':<40} {'Hyperbolic':<40}")
        print(f"{'-'*100}")
        
        for category, pairs in word_pairs.items():
            for entity1, entity2 in pairs:
                try:
                    # Get embeddings for both entities
                    start_emb = self.get_entity_embedding(entity1)
                    end_emb = self.get_entity_embedding(entity2)
                    
                    # Euclidean interpolation
                    euclidean_points = self.euclidean_interpolation(start_emb, end_emb, num_points)
                    midpoint_idx = num_points // 2
                    euclidean_midpoint = euclidean_points[midpoint_idx]
                    euclidean_nearest = self.find_nearest_entities(euclidean_midpoint, top_k)
                    euclidean_names = [entity for entity, _ in euclidean_nearest]
                    euclidean_str = ", ".join(euclidean_names)
                    
                    # Hyperbolic interpolation
                    hyperbolic_points = self.hyperbolic_interpolation(start_emb, end_emb, num_points)
                    hyperbolic_midpoint = hyperbolic_points[midpoint_idx]
                    hyperbolic_nearest = self.find_nearest_entities(hyperbolic_midpoint, top_k)
                    hyperbolic_names = [entity for entity, _ in hyperbolic_nearest]
                    hyperbolic_str = ", ".join(hyperbolic_names)
                    
                    # Save midpoint embedding vectors
                    os.makedirs("latent_output", exist_ok=True)
                    torch.save(euclidean_midpoint, f"latent_output/{entity1}_{entity2}_euclidean_midpoint.pt")
                    torch.save(hyperbolic_midpoint, f"latent_output/{entity1}_{entity2}_hyperbolic_midpoint.pt")
                    
                    # Format output
                    word_pair = f"{entity1} - {entity2}"
                    print(f"{category:<15} {word_pair:<20} {euclidean_str:<40} {hyperbolic_str:<40}")
                    
                except ValueError as e:
                    word_pair = f"{entity1} - {entity2}"
                    print(f"{category:<15} {word_pair:<20} {'Error: Entity not found':<40} {'Error: Entity not found':<40}")
                except Exception as e:
                    word_pair = f"{entity1} - {entity2}"
                    print(f"{category:<15} {word_pair:<20} {'Error: ' + str(e):<40} {'Error: ' + str(e):<40}")
        
        print(f"{'-'*100}")
        print(f"{'='*100}")
    
    def visualize_traversal_3d(self, entity1, entity2, num_points=10):
        """3D latent space traversal visualization using PCA dimensionality reduction, showing all interpolation points"""
        try:
            start_emb = self.get_entity_embedding(entity1)
            end_emb = self.get_entity_embedding(entity2)
        except ValueError as e:
            print(f"Error: {e}")
            return
        
        # Choose interpolation method based on model type
        if self.model_name == "poincare":
            interpolated_points = self.hyperbolic_interpolation(start_emb, end_emb, num_points)
            title_suffix = " (Hyperbolic Space)"
        else:
            interpolated_points = self.euclidean_interpolation(start_emb, end_emb, num_points)
            title_suffix = " (Euclidean Space)"
        
        # Convert interpolation points to numpy array
        points_np = interpolated_points.detach().cpu().numpy()
        
        # Use PCA for 3D dimensionality reduction
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        
        # Get entire embedding space as background
        all_embeddings = self.model.Eh.weight.detach().cpu().numpy() if self.model_name == "poincare" \
                        else self.model.E.weight.detach().cpu().numpy()
        
        # Fit PCA with both interpolation points and all embeddings
        combined = np.vstack([points_np, all_embeddings])
        pca.fit(combined)
        
        # Transform interpolation points and all embeddings
        points_3d = pca.transform(points_np)
        background_3d = pca.transform(all_embeddings)
        
        # Calculate distance of background points from origin (for color mapping)
        background_distances = np.linalg.norm(background_3d, axis=1)
        
        # Create 3D figure
        fig = plt.figure(figsize=(16, 14))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot background points (all entities) - use blue gradient, darker near origin
        scatter = ax.scatter(
            background_3d[:, 0], background_3d[:, 1], background_3d[:, 2], 
            c=background_distances, 
            cmap='Blues_r',  # Reversed blue colormap: dark blue to light blue
            alpha=0.25, 
            s=50, 
            label='All Entities'
        )
        
        # Add colorbar
        cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label('Distance from Origin', fontsize=12)
        
        # Plot traversal path - use red gradient
        for i in range(len(points_3d) - 1):
            # Use gradient from green to red for path
            t = i / (len(points_3d) - 1)
            color = (t, 0.5, 1 - t)  # RGB: from green(0,1,0) to red(1,0,0)
            ax.plot(
                points_3d[i:i+2, 0], points_3d[i:i+2, 1], points_3d[i:i+2, 2],
                color=color, linewidth=3.0, alpha=0.8
            )
        
        # Generate different colors for each point (using viridis colormap)
        point_colors = cm.viridis(np.linspace(0, 1, len(points_3d)))
        
        # Mark all interpolation points
        for i, point in enumerate(points_3d):
            # Get nearest neighbor entity for this point (take only the most similar one)
            with torch.no_grad():
                embedding_tensor = torch.tensor(points_np[i]).float().to(self.device)
                nearest_entities = self.find_nearest_entities(embedding_tensor, top_k=1)
                if nearest_entities:
                    label = f"Point {i}: {nearest_entities[0][0]}"
                else:
                    label = f"Point {i}"
            
            # Plot point and add label
            ax.scatter(
                point[0], point[1], point[2],
                c=[point_colors[i]],  # Each point uses different color
                s=120,  # Point size
                edgecolor='black',
                depthshade=False,
                alpha=0.9,
                label=label
            )
        
        # Specifically mark start and end points
        ax.scatter(points_3d[0, 0], points_3d[0, 1], points_3d[0, 2], 
                  c='green', s=250, edgecolor='black', depthshade=False, 
                  marker='*', alpha=1.0, label=f'Start: {entity1}')
        ax.scatter(points_3d[-1, 0], points_3d[-1, 1], points_3d[-1, 2], 
                  c='red', s=250, edgecolor='black', depthshade=False, 
                  marker='*', alpha=1.0, label=f'End: {entity2}')
        
        # Set axis labels
        ax.set_xlabel('PCA Component 1', fontsize=14)
        ax.set_ylabel('PCA Component 2', fontsize=14)
        ax.set_zlabel('PCA Component 3', fontsize=14)
        
        plt.title(f'3D Latent Space Traversal: {entity1} → {entity2}{title_suffix}', fontsize=16, pad=20)
        
        # Add legend (place outside chart to the right)
        ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=10, title="Points Legend")
        
        # Add grid
        ax.grid(True, alpha=0.2)
        
        # Adjust viewing angle
        ax.view_init(elev=30, azim=45)
        
        # Save image to file
        os.makedirs("visualizations", exist_ok=True)
        filename = f"visualizations/3d_traversal_{entity1}_{entity2}_{self.model_name}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved 3D visualization to: {filename}")
        
        plt.tight_layout()
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Latent Space Traversal for Knowledge Graph Embeddings')
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model")
    parser.add_argument("--dataset", type=str, default="WN18RR", help="Dataset: FB15k-237 or WN18RR")
    parser.add_argument("--model", type=str, default="poincare", help="Model: poincare or euclidean")
    parser.add_argument("--dim", type=int, default=300, help="Embedding dimensionality")
    parser.add_argument("--cuda", action='store_true', help="Use CUDA if available")
    parser.add_argument("--num_points", type=int, default=10, help="Number of interpolation points")
    parser.add_argument("--top_k", type=int, default=10, help="Number of nearest entities to show")
    parser.add_argument("--visualize", action='store_true', help="Show 3D visualization")
    parser.add_argument("--compare", action='store_true', help="Compare Euclidean vs Hyperbolic results")
    parser.add_argument("--format_output", action='store_true', help="Use formatted table output")
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file '{args.model_path}' not found!")
        return
    
    # Load data
    data_dir = f"data/{args.dataset}/"
    d = Data(data_dir=data_dir)
    
    # Create traverser
    traversal = LatentSpaceTraversal(
        model_path=args.model_path,
        d=d,
        model_name=args.model,
        dim=args.dim,
        cuda=args.cuda
    )
    
    # Define word pairs for traversal experiments
    # Adjust these word pairs according to your dataset
    word_pairs = {
        # "Operations": [
            # ("dotproduct", "matrix_multiplication"),
            # ("matrix_addition", "addmonoidhom"),
            # ("star_operation", "dotproduct")
         #],
        # "Structures": [
        #     ("vector", "matrix"),
        #     ("zero_matrix", "diagonal_matrix"),
        #     ("symmetric_matrix", "block_matrix"),
        # ],
         # "Technology": [
         #  ("trace", "matrix"),
        #  #   ("transpose", "conjugate_transpose"),
        #     ("determinant", "kronecker_product")
         # ],
         "Properties": [
          ("absolute_value", "non-negativity"),
          # ("pairwise", "orthogonal_rows"),
          # ("issymm", "isdiag")
       ]
    }

    # Choose output format based on parameters
    if args.compare:
        # Compare Euclidean and hyperbolic space results
        traversal.compare_euclidean_hyperbolic_results(word_pairs, args.num_points, args.top_k)
    elif args.format_output:
        # Format output results
        traversal.format_traversal_output(word_pairs, args.num_points, args.top_k)
    else:
        # Standard detailed output
        results = traversal.run_traversal_experiments(word_pairs, args.num_points, args.top_k)
    
    # Visualization example (if requested)
    if args.visualize:
        print("\nGenerating 3D visualization...")
        # Check if dictionary is empty
        if word_pairs:
            # Get first category
            first_category = list(word_pairs.keys())[0]
            
            # Check if this category has word pairs
            if word_pairs[first_category]:
                first_pair = word_pairs[first_category][0]
                traversal.visualize_traversal_3d(first_pair[0], first_pair[1], args.num_points)
            else:
                print(f"Category '{first_category}' has no word pairs. Skipping visualization.")
        else:
            print("No word pairs defined. Skipping visualization.")
    
    print("\nTraversal experiments completed!")


if __name__ == "__main__":
    main()
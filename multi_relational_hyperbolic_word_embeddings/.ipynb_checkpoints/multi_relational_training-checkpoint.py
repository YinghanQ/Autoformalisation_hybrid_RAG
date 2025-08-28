import random  
import numpy as np  
import torch  
import os
import pickle 
import argparse
import time 
from collections import defaultdict  
from dynaconf import settings  
# from nltk import download
# download("stopwords")  
print("Test")  
from multi_relational.load_data import Data  # Import data loader module
from multi_relational.model import *  # Import all model classes
from multi_relational.rsgd import *  # Import Riemannian SGD optimizer
from web.embeddings import load_embedding  # Import embedding loader
from web.evaluate import evaluate_similarity  # Import similarity evaluation function
from web.evaluate import evaluate_similarity_hyperbolic  # Import hyperbolic similarity evaluation
from web.datasets.similarity import fetch_SimVerb3500  
from tqdm import tqdm  
from six import iteritems 
print("Test")  
    
class Experiment:
    """
    Main experiment class for training and evaluating multi-relational embedding models
    """

    def __init__(self, learning_rate=50, dim=40, nneg=50, model="poincare",
                 num_iterations=500, batch_size=128, dataset="wiktionary", cuda=False):
        # Initialize experiment parameters with default values
        self.model_type = model  # Type of model: "poincare" or "euclidean"
        self.dataset = dataset  # Dataset name
        self.learning_rate = learning_rate  # Learning rate for optimization
        self.dim = dim  # Dimensionality of embeddings
        self.nneg = nneg  # Number of negative samples per positive sample
        self.num_iterations = num_iterations  # Total training iterations
        self.batch_size = batch_size  # Batch size for training
        self.cuda = cuda  # Whether to use GPU acceleration
        self.best_eval_score = 0.0  # Track the best evaluation score
        # Define evaluation tasks (currently only SimVerb3500 dev set)
        self.tasks = {
            'SimVerb3500-dev': fetch_SimVerb3500(which='dev'), 
            #'MEN-dev': fetch_MEN(which = "dev"),  # Commented out but available
        }

        
    def get_data_idxs(self, data):
        # Convert entity and relation names to their corresponding indices
        data_idxs = [(self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]], \
                      self.entity_idxs[data[i][2]]) for i in range(len(data))]
        return data_idxs
    
    def get_er_vocab(self, data, idxs=[0, 1, 2]):
        # Create a vocabulary mapping (entity, relation) pairs to tail entities
        er_vocab = defaultdict(list)  # Default dictionary that returns empty list for missing keys
        for triple in data:
            # Map (head_entity, relation) to list of tail entities
            er_vocab[(triple[idxs[0]], triple[idxs[1]])].append(triple[idxs[2]])
        return er_vocab

    def evaluate(self, model, embeddings_path, embeddings_type="poincare"):
        # Load pre-trained embeddings from file
        embeddings = load_embedding(embeddings_path, format="dict", normalize=True, lower=True, clean_words=False)

        # Calculate similarity evaluation results
        sp_correlations = []  # List to store Spearman correlation scores
        for name, examples in iteritems(self.tasks):
            if embeddings_type == "poincare":
                # Evaluate using hyperbolic similarity for Poincaré embeddings
                score = evaluate_similarity_hyperbolic(embeddings, examples.X, examples.y)
                print(name, score)  # Print task name and score
                sp_correlations.append(score)  # Add score to list
            else:
                # Evaluate using standard similarity for Euclidean embeddings
                score = evaluate_similarity(embeddings, examples.X, examples.y)
                print(name, score)  # Print task name and score
                sp_correlations.append(score)  # Add score to list
        # Return average score across all tasks
        return np.mean(sp_correlations)


    def train_and_eval(self):
        # Main training and evaluation method
        print("Training the %s multi-relational model..." % self.model_type)
        
        # Create mapping from entity/relation names to indices
        self.entity_idxs = {d.entities[i]:i for i in range(len(d.entities))}
        self.relation_idxs = {d.relations[i]:i for i in range(len(d.relations))}
        
        # Set device (GPU if available and enabled, otherwise CPU)
        device = "cuda" if self.cuda else "cpu"

        # Convert training data to index format
        train_data_idxs = self.get_data_idxs(d.train_data)
        print("Number of training data points: %d" % len(train_data_idxs))

        # Initialize the appropriate model based on type
        if self.model_type == "poincare":
            model = torch.jit.script(MuRP(d, self.dim))  # Poincaré ball model
        else:
            model = torch.jit.script(MuRE(d, self.dim))  # Euclidean model
        
        # Get parameter names for the optimizer
        param_names = [name for name, param in model.named_parameters()]
        # Initialize Riemannian SGD optimizer (suitable for hyperbolic space)
        opt = RiemannianSGD(model.parameters(), lr=self.learning_rate, param_names=param_names)
        
        # Move model to GPU if CUDA is enabled
        if self.cuda:
            model.cuda()
            
        # Convert training data indices to tensor and move to appropriate device
        train_data_idxs_tensor = torch.tensor(train_data_idxs, device=device)
        # Get list of all entity indices for negative sampling
        entity_idxs_lst = list(self.entity_idxs.values())
        # Pre-sample negative examples for efficiency during training
        negsamples_tbl = torch.tensor(
            np.random.choice(entity_idxs_lst, 
                           size=(len(train_data_idxs) // self.batch_size, self.batch_size, self.nneg)),
            device=device
        )

        print("Starting training...")
        # Main training loop
        for it in tqdm(range(1, self.num_iterations+1)):
            model.train()  # Set model to training mode

            # Initialize loss storage
            losses = torch.zeros((len(train_data_idxs) // self.batch_size) + 1, device=device)
            batch_cnt = 0
            # Shuffle training data at the beginning of each epoch
            train_data_idxs = train_data_idxs_tensor[torch.randperm(train_data_idxs_tensor.shape[0])]
            
            # Process data in batches
            for j in tqdm(range(0, len(train_data_idxs), self.batch_size)):
                # Get current batch of data
                data_batch = train_data_idxs[j:j+self.batch_size]
                # Randomly select negative samples from pre-sampled table
                negsamples = negsamples_tbl[torch.randint(0, len(train_data_idxs) // self.batch_size, (1,))].squeeze()
                
                # Prepare input tensors: head entities, relations, and tail entities
                e1_idx = torch.tile(torch.unsqueeze(data_batch[:, 0], 0).T, (1, negsamples.shape[1]+1))
                r_idx = torch.tile(torch.unsqueeze(data_batch[:, 1], 0).T, (1, negsamples.shape[1]+1))
                e2_idx = torch.cat((torch.unsqueeze(data_batch[:, 2], 0).T, negsamples[:data_batch.shape[0]]), dim=1)

                # Create target labels: 1 for positive samples, 0 for negatives
                targets = torch.zeros(e1_idx.shape, device=device)
                targets[:, 0] = 1  # First column is positive sample

                # Reset gradients
                opt.zero_grad()

                # Forward pass: get model predictions
                predictions = model.forward(e1_idx, r_idx, e2_idx)      
                # Calculate loss
                loss = model.loss(predictions, targets)
                # Backward pass: compute gradients
                loss.backward()
                # Update model parameters
                opt.step()
                # Store loss for monitoring
                losses[batch_cnt] = loss.detach()
                batch_cnt += 1
                
            # Print training progress
            print("Iteration:", it)    
            print("Loss:", torch.mean(losses).item())

            # Start evaluation phase
            model.eval()  # Set model to evaluation mode
            with torch.no_grad():  # Disable gradient computation for efficiency
                # Save embeddings for external evaluation
                print("Saving the embeddings for evaluation...")
                embeddings_dict = {}
                # Extract embeddings for all entities
                for entity in tqdm(self.entity_idxs):
                    if self.model_type == "poincare":
                        # Get Poincaré embeddings
                        embeddings_dict[entity] = model.Eh.weight[self.entity_idxs[entity]].detach().cpu().numpy()
                    else:
                        # Get Euclidean embeddings
                        embeddings_dict[entity] = model.E.weight[self.entity_idxs[entity]].detach().cpu().numpy()

                # Set up output paths
                out_emb_path = os.path.join(settings["output_path"], "embeddings")
                out_model_path = os.path.join(settings["output_path"], "models")
                outfile_path = os.path.join(out_emb_path, "model_dict_"+self.model_type+"_" + str(self.dim) + "_" + self.dataset)
                
                # Create output directories if they don't exist
                if (not os.path.exists(out_emb_path)):
                    os.makedirs(out_emb_path)
                if (not os.path.exists(out_model_path)):
                    os.makedirs(out_model_path)

                # Save embeddings to file using pickle
                pickle.dump(embeddings_dict, open(outfile_path, "wb"))
                # Evaluate embeddings on similarity tasks
                score = self.evaluate(model, outfile_path, self.model_type)
                print("Evaluation score:", score)
                
                # Check if this is the best model so far
                if score > self.best_eval_score:
                    # Save the best model's embeddings
                    print("New best model, saving the embeddings...")
                    outfile_path = os.path.join(out_emb_path, "best_model_dict_" + self.model_type + "_" + str(self.dim) + "_" + self.dataset)
                    pickle.dump(embeddings_dict, open(outfile_path, "wb"))
                    self.best_eval_score = score  # Update best score
                    
                    # Save complete model checkpoint
                    torch.save({ 
                        "epoch": it,  # Current iteration
                        "model_state_dict": model.state_dict(),  # Model parameters
                        "optimizer_state_dict": opt.state_dict(),  # Optimizer state
                        "loss": losses,  # Loss values
                    }, os.path.join(out_model_path, "best_model_checkpoint_" + self.model_type + "_" + str(self.dim) + "_" + self.dataset+".pt"))

if __name__ == '__main__':
    # Main execution block
    print("Loading setup----")
    # Set up command-line argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="wiktionary", nargs="?",
                    help="Which dataset to use: FB15k-237 or WN18RR.")
    parser.add_argument("--model", type=str, default="poincare", nargs="?",
                    help="Which model to use: poincare or euclidean.")
    parser.add_argument("--num_iterations", type=int, default=500, nargs="?",
                    help="Number of iterations.")
    parser.add_argument("--batch_size", type=int, default=128, nargs="?",
                    help="Batch size.")
    parser.add_argument("--nneg", type=int, default=50, nargs="?",
                    help="Number of negative samples.")
    parser.add_argument("--lr", type=float, default=50, nargs="?",
                    help="Learning rate.")
    parser.add_argument("--dim", type=int, default=40, nargs="?",
                    help="Embedding dimensionality.")
    parser.add_argument("--cuda", type=bool, default=True, nargs="?",
                    help="Whether to use cuda (GPU) or not (CPU).")

    # Parse command-line arguments
    args = parser.parse_args()
    print("Loading Data----")
    dataset = args.dataset
    data_dir = "data/%s/" % dataset  # Construct data directory path
    
    # Set random seeds for reproducibility
    torch.backends.cudnn.deterministic = True  # Ensure CUDA operations are deterministic
    seed = 40
    np.random.seed(seed)  # Seed numpy random generator
    torch.manual_seed(seed)  # Seed PyTorch random generator
    if torch.cuda.is_available:  # Seed all GPUs if available
        torch.cuda.manual_seed_all(seed)
        
    # Load data
    d = Data(data_dir=data_dir)
    print("Start training")  # Print training start message
    
    # Initialize experiment with parsed arguments
    experiment = Experiment(learning_rate=args.lr, batch_size=args.batch_size, 
                            num_iterations=args.num_iterations, dim=args.dim, 
                            cuda=args.cuda, nneg=args.nneg, model=args.model, dataset=args.dataset)
    # Start training and evaluation
    experiment.train_and_eval()
    
    # Print information about model saving locations
    import os
    print("\n\n===== Model Saving Locations =====")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Output path configuration: {settings.get('output_path', 'Not set')}")
    
    # Calculate expected save paths
    base_path = settings.get("output_path", ".")
    emb_path = os.path.join(base_path, "embeddings")  # Embeddings directory path
    model_path = os.path.join(base_path, "models")  # Models directory path
    
    print(f"Embeddings will be saved to: {emb_path}")
    print(f"Model checkpoints will be saved to: {model_path}")
    print("=======================\n")
# Autoformalisation_RAG Project

This project implements **autoformalisation for interactive theorem provers (ITPs) using large language models (LLMs)**, leveraging hybrid embeddings to enhance retrieval and code generation for Mathlib4.

---

## Project Overview

The workflow integrates natural language understanding with formal structural reasoning to robustly translate informal mathematics into Lean 4 code:

1.  **Core Methodology**
    Concepts and multiple related pieces of information are extracted from Mathlib4 using regex and LLMs, creating an augmented dataset.

2.  **Conversion to Natural Language**
    Extracted concepts are transformed into informal, human-readable descriptions.

3.  **Hybrid Embedding Knowledge Base Construction**
    A knowledge base is built using hybrid embeddings:
    -   **Semantic embeddings** – capture linguistic and contextual information from natural language queries.
    -   **Multi-Relational Hyperbolic embeddings** – encode hierarchical and multi-relational dependencies of mathematical concepts in Mathlib4, clustering related concepts and preserving structure.
        > Implemented in `Autoformalisation_RAG/multi_relational_hyperbolic_embeddings`, adapted from [neuro-symbolic-ai/multi_relational_hyperbolic_word_embeddings](https://github.com/neuro-symbolic-ai/multi_relational_hyperbolic_word_embeddings.git) and [ibalazevic/multirelational-poincare](https://github.com/ibalazevic/multirelational-poincare.git).

4.  **Structural Augmentation in Retrieval**
    Beyond semantic similarity, hyperbolic embeddings provide structural information to improve retrieval relevance.

5.  **Code Generation**
    Retrieved semantic and structural information is integrated during generation, improving import selection and correctness of Lean 4 code.

This approach ensures both natural language understanding and formal structural reasoning are leveraged for robust autoformalisation.

---

## Repository Structure & Key Components

### 1. Informalisation and Mathematical DSRL
Notebooks and data for informalisation and semantic role labeling of mathematical definitions:

-   [`Clean_and_process.ipynb`](Autoformalisation_RAG/Informalisation_and_Mathematical_DSRL/Clean_and_process.ipynb) – Data cleaning and preprocessing.
-   [`Informalisation.ipynb`](Autoformalisation_RAG/Informalisation_and_Mathematical_DSRL/Informalisation.ipynb) – Converts mathematical definitions to informal descriptions.
-   [`Mathematical_Definition_Semantic_Role_Labeling.ipynb`](Autoformalisation_RAG/Informalisation_and_Mathematical_DSRL/Mathematical_Definition_Semantic_Role_Labeling.ipynb) – Semantic role labeling of mathematical definitions.
-   [`informal_data/`](Autoformalisation_RAG/Informalisation_and_Mathematical_DSRL/informal_data) – Stores original informal data.
-   [`informal_data_with_triples/`](Autoformalisation_RAG/Informalisation_and_Mathematical_DSRL/informal_data_with_triples) – Extracted triples from informal descriptions.

### 2. Embedding Evaluation & Construction
Notebooks for constructing and evaluating embeddings:

-   [`Text_Embedding_Mathlib4.ipynb`](Autoformalisation_RAG/Text_Embedding_Mathlib4.ipynb) – Construct text embeddings for Mathlib4.
-   `multi_relational_hyperbolic_embeddings/` – Directory containing code for Multi-Relational Hyperbolic embeddings, adapted from [neuro-symbolic-ai](https://github.com/neuro-symbolic-ai/multi_relational_hyperbolic_word_embeddings.git) and [ibalazevic](https://github.com/ibalazevic/multirelational-poincare.git), to encode hierarchy and multi-relational dependencies for robust retrieval.

### 3. RAG-Based Autoformalisation
Core RAG-based autoformalisation pipeline:

-   [`RAG_Based_Autoformalisation.ipynb`](Autoformalisation_RAG/RAG_Based_Autoformalisation.ipynb) – Integrates retrieval with code generation for Lean 4.

### 4. Retrieval and Baseline Experiments
Notebooks for module retrieval and embedding performance comparison:

-   [`Baseline_GPT_Retrival.ipynb`](Autoformalisation_RAG/Baseline_GPT_Retrival.ipynb) – GPT-based module retrieval baseline.
-   [`semantic_vs_hybrid_evaluation.ipynb`](Autoformalisation_RAG/semantic_vs_hybrid_evaluation.ipynb) – Compare semantic vs hybrid embedding retrieval performance.
-   [`Comparing_Embedding_Model_Performance.ipynb`](Autoformalisation_RAG/Comparing_Embedding_Model_Performance.ipynb) – Evaluate different embedding models on semantic similarity benchmarks.

---

## Installation & Dependencies
1.  **Data from Mathlib4**: This project requires the Mathlib4 source for data extraction.
    ```bash
    git clone https://github.com/leanprover-community/mathlib4.git
    ```
    Update the relevant paths in the notebooks to point to your local `mathlib4` directory.
2.  **OpenAI API Key**: Set your `OPENAI_API_KEY` environment variable for LLM access.
3.  **(Optional) Pre-trained Models & Data and some relative evalustion data**: Pre-trained embedding models and processed datasets can be downloaded from:
    [Google Drive Folder](https://drive.google.com/drive/folders/1rXL0cA9kc9PY6WEKEKhqVC5DJc_RKt8Z?usp=drive_link)
    > To evaluate the performance of Retrieval-Augmented Generation (RAG), the following pre-trained models and datasets can be downloaded from the above link and placed in the specified directories:

    - **Pre-trained Embedding Models and Outputs**  
    Path: `./Mathlib4_embeddings/outputs_cleaned`  
    Purpose: Utilized for semantic retrieval and embedding similarity evaluation.

    - **Merged Dataset with Embeddings and Knowledge Graph Triples**  
    Path: `./Informalisation_and_Mathematical_DSRL/merged_with_embeddings_and_triples.json`  
    Purpose: Used for RAG-based retrieval and downstream informalisation experiments.

---

## Usage / How to Run

1.  **Data Preparation**:
    Run the notebooks in the `Informalisation_and_Mathematical_DSRL/` directory in order:
    -   `Clean_and_process.ipynb`
    -   `Informalisation.ipynb`
    -   `Mathematical_Definition_Semantic_Role_Labeling.ipynb`

2.  **Embedding Construction**:
    -   **Semantic Embeddings**: Run `Text_Embedding_Mathlib4.ipynb` to generate semantic embeddings using sentence-transformers models.
    -   **Multi-Relational Hyperbolic Embeddings**: Build the multi-relational hyperbolic embedding knowledge base using the code in the `multi_relational_hyperbolic_embeddings/` directory. This component is adapted from the [Multi-Relational Hyperbolic Word Embeddings](https://github.com/neuro-symbolic-ai/multi_relational_hyperbolic_word_embeddings.git) repository, which implements hyperbolic embeddings for encoding hierarchical relationships and multi-relational dependencies between mathematical concepts.

    **Training Hyperbolic Embeddings**:
    
    Navigate to the `multi_relational_hyperbolic_embeddings/` folder and run the training command:

    ```bash
    cd multi_relational_hyperbolic_embeddings/
    CUDA_VISIBLE_DEVICES=0 python ./multi_relational_training.py \
        --model poincare \
        --dataset my_dataset_cleaned \
        --num_iterations 300 \
        --nneg 50 \
        --batch_size 128 \
        --lr 50 \
        --dim 300
    ```

    **Parameter Explanation**:
    - `--model poincare`: Use the Poincaré ball model for hyperbolic embeddings
    - `--dataset my_dataset_cleaned`: Use the preprocessed dataset file
    - `--num_iterations 300`: Train for 300 iterations
    - `--nneg 50`: Number of negative samples for contrastive learning
    - `--batch_size 128`: Training batch size
    - `--lr 50`: Learning rate (typically higher in hyperbolic space)
    - `--dim 300`: Embedding dimensionality

    **Note**: Ensure your dataset is properly formatted as triplets (head, relation, tail) in the required format before training.

3.  **Evaluation**:
    -   Use `Comparing_Embedding_Model_Performance.ipynb` to benchmark embedding models.
    -   Use `semantic_vs_hybrid_evaluation.ipynb` to compare retrieval performance.

4.  **Autoformalisation**:
    -   Run `RAG_Based_Autoformalisation.ipynb` for the complete module retrieval and Lean 4 code generation pipeline.

---

## References

1.  **Multi-relational Poincaré Graph Embeddings**  
    Ivana Balažević, Carl Allen, and Timothy M. Hospedales.  
    *Neural Information Processing Systems (NeurIPS)*, 2019.  
    [[Paper]](https://arxiv.org/pdf/1905.09791.pdf)  
    [[Code]](https://github.com/ibalazevic/multirelational-poincare)

2.  **Multi-Relational Hyperbolic Word Embeddings from Natural Language Definitions**  
    Prateek Vijayvergiya, Francesca Toni, and Edoardo D. Ponti.  
    *ArXiv*, 2023.  
    [[Paper]](https://arxiv.org/abs/2305.07303)  
    [[Code]](https://github.com/neuro-symbolic-ai/multi_relational_hyperbolic_word_embeddings)

3.  **RAG (Retrieval-Augmented Generation)**  
    Lewis, Patrick, et al.  
    "Retrieval-augmented generation for knowledge-intensive nlp tasks."  
    *Advances in Neural Information Processing Systems* 33 (2020): 9459-9474.

4.  **Chain-of-Thought Prompting**  
    Wei, Jason, et al.  
    "Chain-of-thought prompting elicits reasoning in large language models."  
    *Advances in Neural Information Processing Systems* 35 (2022): 24824-24837.

5.  **Mathlib4**  
    The Lean 4 Mathematical Library.  
    [[Link]](https://github.com/leanprover-community/mathlib4)

6.  **Sentence-T5**  
    Ni, Jianmo, et al.  
    "Sentence-t5: Scalable sentence encoders from pre-trained text-to-text models."  
    *Findings of the Association for Computational Linguistics: ACL 2022*. 2022.


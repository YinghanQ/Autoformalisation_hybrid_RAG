# -*- coding: utf-8 -*-

"""
Functions for fetching similarity datasets (modified for local file access)
"""

import os
import numpy as np
import pandas as pd
from sklearn.utils import Bunch

# Set base directory for dataset storage (configurable via environment variable)
DATA_BASE_DIR = os.environ.get('WORD_SIMILARITY_DATA', './similarity_data')

def _get_local_path(filename, subdirectory=None):
    """Construct local file path"""
    if subdirectory:
        return os.path.join(DATA_BASE_DIR, subdirectory, filename)
    return os.path.join(DATA_BASE_DIR, filename)


def fetch_MEN(which="all", form="natural"):
    """Read MEN dataset from local file"""
    filenames = {
        "dev": 'datasets/MEN/MEN_dataset_lemma_form.dev',
        "test": 'datasets/MEN/MEN_dataset_lemma_form.test',
        "all": 'datasets/MEN/MEN_dataset_lemma_form_full'
    }
    filepath = _get_local_path(filenames[which])
    
    # Try different separators
    try:
        data = pd.read_csv(filepath, header=None, sep=" ", names=['word1', 'word2', 'score'])
    except:
        try:
            data = pd.read_csv(filepath, header=None, sep="\t", names=['word1', 'word2', 'score'])
        except:
            data = pd.read_csv(filepath, header=None, delim_whitespace=True, 
                              names=['word1', 'word2', 'score'])
    
    # Handle lemmatized forms
    if form == "natural":
        # Remove POS tags (e.g., "-n")
        data['word1'] = data['word1'].apply(lambda x: x.split('-')[0] if isinstance(x, str) else x)
        data['word2'] = data['word2'].apply(lambda x: x.split('-')[0] if isinstance(x, str) else x)
    elif form != "lem":
        raise ValueError("Unrecognized form argument. Use 'natural' or 'lem'")
    
    # Ensure all scores are numeric
    data['score'] = pd.to_numeric(data['score'], errors='coerce')
    
    # Print sample for verification
    print(f"Loaded MEN data sample (form={form}):")
    print(f"  word1: '{data.iloc[0]['word1']}', word2: '{data.iloc[0]['word2']}', score: {data.iloc[0]['score']}")
    
    return Bunch(
        X=data[['word1', 'word2']].values.astype("object"), 
        y=data['score'].values.astype(float) / 5.0
    )


def fetch_WS353(which="all"):
    """Read WS353 dataset from local file"""
    filenames = {
        "all": 'datasets/ws-353/wordsim353-english.txt',
        "relatedness": 'datasets/ws-353/wordsim353-english-rel.txt',
        "similarity": 'datasets/ws-353/wordsim353-english-sim.txt',
        "set1": 'datasets/wordsim353/set1.csv',
        "set2": 'datasets/wordsim353/set2.csv'
    }
    filepath = _get_local_path(filenames[which])
    data = pd.read_csv(filepath, header=0, sep="\t")
    
    X = data.values[:, 0:2]
    y = data.values[:, 2].astype(float)

    if data.values.shape[1] > 3:
        sd = np.std(data.values[:, 2:15].astype(float), axis=1).flatten()
        return Bunch(X=X.astype("object"), y=y, sd=sd)
    return Bunch(X=X.astype("object"), y=y)

def fetch_RG65():
    """
    Fetch Rubenstein and Goodenough dataset for testing attributional and
    relatedness similarity

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'X': matrix of 2 words per column,
        'y': vector with scores,
        'sd': vector of std of scores if available (for set1 and set2)

    References
    ----------
    Rubenstein, Goodenough, "Contextual correlates of synonymy", 1965

    Notes
    -----
    Scores were scaled by factor 10/4
    """
    filepath = _get_local_path('datasets/rg65.csv')
    
    # Correctly read semicolon-separated data
    data = pd.read_csv(filepath, header=None, sep=';', names=['word1', 'word2', 'score'])
    
    # Check data format
    print(f"Loaded RG65 data sample:\n{data.head(3)}")
    print(f"Score range: {data['score'].min()} to {data['score'].max()}")
    
    return Bunch(X=data[['word1', 'word2']].values.astype("object"),
                 y=data['score'].values.astype(float) * 10.0 / 4.0)


def fetch_SimLex999(which='all'):
    """Read SimLex999 dataset from local file"""
    filepath = _get_local_path('datasets/SimLex-999/SimLex-999.txt')
    data = pd.read_csv(filepath, sep="\t")
    
    X = data[['word1', 'word2']].values
    y = data['SimLex999'].values
    sd = data['SD(SimLex)'].values
    conc = data[['conc(w1)', 'conc(w2)', 'concQ']].values
    POS = data[['POS']].values
    assoc = data[['Assoc(USF)', 'SimAssoc333']].values
    
    if which == 'all':
        idx = np.arange(len(X))
    elif which == '333':
        idx = np.where(assoc[:, 1] == 1)[0]
    else:
        raise ValueError("Unrecognized subset")
    
    return Bunch(X=X[idx].astype("object"), y=y[idx], sd=sd[idx], 
                 conc=conc[idx], POS=POS[idx], assoc=assoc[idx])


def fetch_SimVerb3500(which='all'):
    filenames = {
        "all": 'datasets/simverb/data/SimVerb-3500.txt',
        "dev": 'datasets/simverb/data/SimVerb-500-dev.txt',
        "test": 'datasets/simverb/data/SimVerb-3000-test.txt'
    }
    filepath = _get_local_path(filenames[which], subdirectory='')
    
    # Use tab separator, specify column names
    data = pd.read_csv(filepath, sep="\t", header=None, 
                      names=['word1', 'word2', 'pos', 'score', 'relation'])
    
    print(f"Loaded SimVerb3500 data sample:\n{data.head(3)}")
    print(f"Total verb pairs: {len(data)}")
    
    return Bunch(
        X=data[['word1', 'word2']].values.astype("object"), 
        y=data['score'].values.astype(float),
        pos=data['pos'].values,  # Add part-of-speech information
        relation=data['relation'].values  # Add relation type
    )


def fetch_SCWS():
    """Read SCWS dataset (with context) from local file"""
    filepath = _get_local_path('datasets/ehuang_sim_wcontext/SCWS/ratings.txt', subdirectory='')
    
    try:
        # Manually read and process file
        data = []
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                # Skip empty lines
                if not line.strip():
                    continue
                
                # Split fields - use tab separator
                parts = line.strip().split('\t')
                
                # Verify minimum field count (at least 15 fields)
                if len(parts) < 15:
                    continue
                
                try:
                    # Extract fixed position fields
                    # According to actual data format:
                    # Field 0: line number
                    # Field 1: word1
                    # Field 2: word1 POS
                    # Field 3: word2
                    # Field 4: word2 POS
                    # Field 5: word1 context
                    # Field 6: word2 context
                    # Field 7: average score
                    # Fields 8-18: 10 ratings
                    word1 = parts[1]
                    word2 = parts[3]
                    
                    # Average score is in the 8th field (index 7)
                    avg_score = float(parts[7])
                    
                    # Scores are in fields 9-19 (index 8-18)
                    scores = list(map(float, parts[8:19]))
                    
                    # Extract context
                    word1_context = parts[5]
                    word2_context = parts[6]
                    
                    # Try to extract sentence (if available)
                    sentence = ""
                    if len(parts) > 19:
                        sentence = " ".join(parts[19:])
                    
                    # Add to dataset
                    data.append({
                        'word1': word1,
                        'word2': word2,
                        'word1_context': word1_context,
                        'word2_context': word2_context,
                        'sentence': sentence,
                        'avg_score': avg_score,
                        'scores': scores
                    })
                    
                except (IndexError, ValueError) as e:
                    # Print problematic line for debugging
                    print(f"Error parsing line: {line.strip()}")
                    print(f"Error message: {str(e)}")
                    continue
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        if len(df) > 0:
            print(f"Successfully loaded SCWS dataset, total {len(df)} word pairs")
            print(f"Data sample:\n{df.iloc[0]}")
        else:
            print("SCWS dataset loaded but no valid data found")
        
        # Extract key information
        X = df[['word1', 'word2']].values.astype("object")
        mean_scores = df['avg_score'].values.astype(float)
        all_scores = np.array(df['scores'].tolist())
        sd_scores = np.std(all_scores, axis=1)
        
        # Also return context information
        contexts = df[['word1_context', 'word2_context', 'sentence']].values
        
        return Bunch(
            X=X, 
            y=mean_scores,
            sd=sd_scores,
            contexts=contexts,
            all_scores=all_scores
        )
    
    except Exception as e:
        import traceback
        print(f"Failed to load SCWS dataset: {str(e)}")
        traceback.print_exc()
        # Return empty dataset to prevent subsequent crashes
        return Bunch(
            X=np.array([], dtype="object").reshape(0, 2),
            y=np.array([], dtype=float),
            sd=np.array([], dtype=float),
            contexts=np.array([], dtype="object").reshape(0, 3),
            all_scores=np.array([], dtype=float).reshape(0, 11)
        )
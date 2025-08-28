# -*- coding: utf-8 -*-
"""
 Evaluation functions
"""
"""
 Classes and function for answering analogy questions
"""
"""
Base class for embedding.

NOTE: This file was adapted from the polyglot package
"""

import logging
from collections import OrderedDict

import numpy as np
import sys

from six import text_type
from six import PY2
from six import iteritems
from six import string_types
from .utils import _open
from .vocabulary import Vocabulary, CountedVocabulary, OrderedVocabulary
from six.moves import cPickle as pickle
from six.moves import range
from functools import partial
from .utils import standardize_string, to_utf8

from sklearn.metrics import pairwise_distances

logger = logging.getLogger(__name__)


class Embedding(object):
    """ Mapping a vocabulary to a d-dimensional points."""

    def __init__(self, vocabulary, vectors):
        self.vocabulary = vocabulary
        self.vectors = np.asarray(vectors)
        if len(self.vocabulary) != self.vectors.shape[0]:
            raise ValueError("Vocabulary has {} items but we have {} "
                             "vectors."
                             .format(len(vocabulary), self.vectors.shape[0]))

        if len(self.vocabulary.words) != len(set(self.vocabulary.words)):
            logger.warning("Vocabulary has duplicates.")

    def __getitem__(self, k):
        return self.vectors[self.vocabulary[k]]

    def __setitem__(self, k, v):
        if not v.shape[0] == self.vectors.shape[1]:
            raise RuntimeError("Please pass vector of len {}".format(self.vectors.shape[1]))

        if k not in self.vocabulary:
            self.vocabulary.add(k)
            self.vectors = np.vstack([self.vectors, v.reshape(1, -1)])
        else:
            self.vectors[self.vocabulary[k]] = v

    def __contains__(self, k):
        return k in self.vocabulary

    def __delitem__(self, k):
        """Remove the word and its vector from the embedding.

        Note:
         This operation costs \\theta(n). Be careful putting it in a loop.
        """
        index = self.vocabulary[k]
        del self.vocabulary[k]
        self.vectors = np.delete(self.vectors, index, 0)

    def __len__(self):
        return len(self.vocabulary)

    def __iter__(self):
        for w in self.vocabulary:
            yield w, self[w]

    @property
    def words(self):
        return self.vocabulary.words

    @property
    def shape(self):
        return self.vectors.shape

    def get(self, k, default=None):
        try:
            return self[k]
        except KeyError as e:
            return default

    def standardize_words(self, lower=False, clean_words=False, inplace=False):
        tw = self.transform_words(partial(standardize_string, lower=lower, clean_words=clean_words), inplace=inplace,
                                  lower=lower)

        if clean_words:
            tw = tw.transform_words(partial(lambda w: w.strip(" ")), inplace=inplace, lower=lower)
        return tw

    def transform_words(self, f, inplace=False, lower=False):
        """
        Transform words in vocabulary according to following strategy.
        Prefer shortest and most often occurring words- after transforming by some (lambda f) function.

        This allow eliminate noisy and wrong coded words.

        Strategy is implemented for all types of Vocabulary- they can be polymorphicaly extended.

        Parameters
        ----------
        f: lambda
            Function called on each word- for transformation it.

        inplace: bool, default: False
            Return new Embedding instance or modify existing

        lower: bool, default: False
            If true, will convert all words to lowercase

        Returns
        -------
        e: Embedding
        Instance of Embedding class with this same Vocabulary type as previous.
        """
        id_map = OrderedDict()
        word_count = len(self.vectors)
        # store max word length before f(w)- in corpora
        words_len = {}
        # store max occurrence count of word
        counts = {}
        is_vocab_generic = False

        curr_words = self.vocabulary.words
        curr_vec = self.vectors

        if isinstance(self.vocabulary, CountedVocabulary):
            _, counter_of_words = self.vocabulary.getstate()
        elif isinstance(self.vocabulary, OrderedVocabulary):
            # range in python3 is lazy
            counter_of_words = range(len(self.vocabulary.words) - 1, -1, -1)

        elif isinstance(self.vocabulary, Vocabulary):
            is_vocab_generic = True
            # if corpora contain lowercase version of word i- for case Vocabulary
            lowered_words = {}

            if lower:

                for w, v in zip(self.vocabulary.words, self.vectors):
                    wl = w.lower()
                    if wl == w:
                        lowered_words[wl] = v
                    elif wl != w and wl not in lowered_words:
                        lowered_words[wl] = v

                curr_words = list(lowered_words.keys())
                curr_vec = np.asanyarray(list(lowered_words.values()))

        else:
            raise NotImplementedError(
                'This kind of Vocabulary is not implemented in transform_words strategy and can not be matched')

        for id, w in enumerate(curr_words):

            fw = f(w)
            if len(fw) and fw not in id_map:
                id_map[fw] = id

                if not is_vocab_generic:
                    counts[fw] = counter_of_words[id]
                words_len[fw] = len(w)

                # overwrite
            elif len(fw) and fw in id_map:
                if not is_vocab_generic and counter_of_words[id] > counts[fw]:
                    id_map[fw] = id

                    counts[fw] = counter_of_words[id]
                    words_len[fw] = len(w)
                elif is_vocab_generic and len(w) < words_len[fw]:
                    # for generic Vocabulary
                    id_map[fw] = id

                    words_len[fw] = len(w)
                elif not is_vocab_generic and counter_of_words[id] == counts[fw] and len(w) < words_len[fw]:
                    id_map[fw] = id

                    counts[fw] = counter_of_words[id]
                    words_len[fw] = len(w)

                logger.warning("Overwriting {}".format(fw))

        if isinstance(self.vocabulary, CountedVocabulary):
            words_only = id_map.keys()
            vectors = curr_vec[[id_map[w] for w in words_only]]
            words = {w: counter_of_words[id_map[w]] for w in words_only}

        elif isinstance(self.vocabulary, OrderedVocabulary):
            words = sorted(id_map.keys(), key=lambda x: id_map[x])
            vectors = curr_vec[[id_map[w] for w in words]]

        elif isinstance(self.vocabulary, Vocabulary):
            words = sorted(id_map.keys(), key=lambda x: id_map[x])
            vectors = curr_vec[[id_map[w] for w in words]]

        logger.info("Transformed {} into {} words".format(word_count, len(words)))

        if inplace:
            self.vectors = vectors
            self.vocabulary = self.vocabulary.__class__(words)

            return self
        else:
            return Embedding(vectors=vectors, vocabulary=self.vocabulary.__class__(words))

    def most_frequent(self, k, inplace=False):
        """Only most frequent k words to be included in the embeddings."""

        assert isinstance(self.vocabulary, OrderedVocabulary), \
            "most_frequent can be called only on Embedding with OrderedVocabulary"

        vocabulary = self.vocabulary.most_frequent(k)
        vectors = np.asarray([self[w] for w in vocabulary])
        if inplace:
            self.vocabulary = vocabulary
            self.vectors = vectors
            return self
        return Embedding(vectors=vectors, vocabulary=vocabulary)

    def normalize_words(self, ord=2, inplace=False):
        """Normalize embeddings matrix row-wise.

        Parameters
        ----------
          ord: normalization order. Possible values {1, 2, 'inf', '-inf'}
        """
        if ord == 2:
            ord = None  # numpy uses this flag to indicate l2.
        vectors = self.vectors.T / np.linalg.norm(self.vectors, ord, axis=1)
        if inplace:
            self.vectors = vectors.T
            return self
        return Embedding(vectors=vectors.T, vocabulary=self.vocabulary)

    def nearest_neighbors(self, word, k=1, exclude=[], metric="cosine"):
        """
        Find nearest neighbor of given word

        Parameters
        ----------
          word: string or vector
            Query word or vector.

          k: int, default: 1
            Number of nearest neighbours to return.

          metric: string, default: 'cosine'
            Metric to use.

          exclude: list, default: []
            Words to omit in answer

        Returns
        -------
          n: list
            Nearest neighbors.
        """
        if isinstance(word, string_types):
            assert word in self, "Word not found in the vocabulary"
            v = self[word]
        else:
            v = word

        D = pairwise_distances(self.vectors, v.reshape(1, -1), metric=metric)

        if isinstance(word, string_types):
            D[self.vocabulary.word_id[word]] = D.max()

        for w in exclude:
            D[self.vocabulary.word_id[w]] = D.max()

        return [self.vocabulary.id_word[id] for id in D.argsort(axis=0).flatten()[0:k]]

    @staticmethod
    def from_gensim(model):
        word_count = {}
        vectors = []
        for word, vocab in sorted(iteritems(model.vocab), key=lambda item: -item[1].count):
            word = standardize_string(word)
            if word:
                vectors.append(model.syn0[vocab.index])
                word_count[word] = vocab.count
        vocab = CountedVocabulary(word_count=word_count)
        vectors = np.asarray(vectors)
        return Embedding(vocabulary=vocab, vectors=vectors)

    @staticmethod
    def from_word2vec_vocab(fvocab):
        counts = {}
        with _open(fvocab) as fin:
            for line in fin:

                word, count = standardize_string(line).split()
                if word:
                    counts[word] = int(count)
        return CountedVocabulary(word_count=counts)

    @staticmethod
    def _from_word2vec_binary(fname):
        with _open(fname, 'rb') as fin:
            words = []
            header = fin.readline()
            vocab_size, layer1_size = list(map(int, header.split()))  # throws for invalid file format
            logger.info("Loading #{} words with {} dim".format(vocab_size, layer1_size))
            vectors = np.zeros((vocab_size, layer1_size), dtype=np.float32)
            binary_len = np.dtype("float32").itemsize * layer1_size
            for line_no in range(vocab_size):
                # mixed text and binary: read text first, then binary
                word = []
                while True:
                    ch = fin.read(1)
                    if ch == b' ':
                        break
                    if ch != b'\n':  # ignore newlines in front of words (some binary files have newline, some don't)
                        word.append(ch)

                words.append(b''.join(word).decode("latin-1"))
                vectors[line_no, :] = np.fromstring(fin.read(binary_len), dtype=np.float32)

            if len(words) < vocab_size:
                logger.warning("Omitted {} words".format(vocab_size - len(words)))
            elif len(words) > vocab_size:
                raise RuntimeError("Read too many words, incorrect file")

            return words, vectors

    @staticmethod
    def _from_word2vec_text(fname):
        with _open(fname, 'r') as fin:
            words = []

            header = fin.readline()
            ignored = 0
            vocab_size, layer1_size = list(map(int, header.split()))  # throws for invalid file format
            vectors = np.zeros(shape=(vocab_size, layer1_size), dtype=np.float32)
            for line_no, line in enumerate(fin):
                try:
                    parts = text_type(line, encoding="utf-8").split(' ')
                    w = parts[0]
                    parts = list(map(lambda x: x.strip(), parts[1:]))
                    parts.insert(0, w)

                except TypeError as e:
                    parts = line.split(' ')
                    w = parts[0]
                    parts = list(map(lambda x: x.strip(), parts[1:]))
                    parts.insert(0, w)

                except Exception as e:
                    logger.warning("We ignored line number {} because of errors in parsing"
                                   "\n{}".format(line_no, e))
                    continue

                # We differ from Gensim implementation.
                # Our assumption that a difference of one happens because of having a
                # space in the word.
                if len(parts) == layer1_size + 1:
                    word, vectors[line_no - ignored] = parts[0], list(map(np.float32, parts[1:]))
                elif len(parts) == layer1_size + 2 and parts[-1]:
                    # last element after splitting is not empty- some glove corpora have additional space
                    word, vectors[line_no - ignored] = parts[:2], list(map(np.float32, parts[2:]))
                    word = u" ".join(word)
                elif not parts[-1]:
                    # omit last value - empty string
                    word, vectors[line_no - ignored] = parts[0], list(map(np.float32, parts[1:-1]))
                else:
                    ignored += 1
                    logger.warning("We ignored line number {} because of unrecognized "
                                   "number of columns {}".format(line_no, parts[:-layer1_size]))
                    continue

                words.append(word)

            if ignored:
                vectors = vectors[0:-ignored]

            if len(words) < vocab_size:
                logger.warning("Omitted {} words".format(vocab_size - len(words)))
            elif len(words) > vocab_size:
                raise RuntimeError("Read too many words, incorrect file")

            return words, vectors

    @staticmethod
    def from_glove(fname, vocab_size, dim):
        with _open(fname, 'r') as fin:

            words = []
            words_uniq = set()

            ignored = 0
            vectors = np.zeros(shape=(vocab_size, dim), dtype=np.float32)
            for line_no, line in enumerate(fin):
                try:
                    parts = text_type(line, encoding="utf-8").split(' ')
                    parts[1:] = map(lambda x: np.float32(x.strip()), parts[1:])
                except TypeError as e:

                    parts = line.split(' ')
                    parts[1:] = map(lambda x: np.float32(x.strip()), parts[1:])

                except Exception as e:
                    ignored += 1

                    logger.warning("We ignored line number {} because of errors in parsing"
                                   "\n{}".format(line_no, e))
                    continue

                try:
                    if parts[0] not in words_uniq:
                        word, vectors[line_no - ignored] = parts[0], list(parts[len(parts) - dim:])
                        words.append(word)
                        words_uniq.add(word)
                    else:
                        ignored += 1
                        logger.warning(
                            "We ignored line number {} - following word is duplicated in file:\n{}\n".format(line_no,
                                                                                                             parts[0]))

                except Exception as e:
                    ignored += 1
                    logger.warning("We ignored line number {} because of errors in parsing"
                                   "\n{}".format(line_no, e))

            return Embedding(vocabulary=OrderedVocabulary(words), vectors=vectors[0:len(words)])

    @staticmethod
    def from_dict(d):
        for k in d:  # Standardize
            d[k] = np.array(d[k]).flatten()
        return Embedding(vectors=list(d.values()), vocabulary=Vocabulary(d.keys()))

    @staticmethod
    def to_word2vec(w, fname, binary=False):
        """
        Store the input-hidden weight matrix in the same format used by the original
        C word2vec-tool, for compatibility.

        Parameters
        ----------
        w: Embedding instance

        fname: string
          Destination file
        """
        logger.info("storing %sx%s projection weights into %s" % (w.vectors.shape[0], w.vectors.shape[1], fname))
        with _open(fname, 'wb') as fout:
            fout.write(to_utf8("%s %s\n" % w.vectors.shape))
            # store in sorted order: most frequent words at the top
            for word, vector in zip(w.vocabulary.words, w.vectors):
                if binary:
                    fout.write(to_utf8(word) + b" " + vector.astype("float32").tostring())
                else:
                    fout.write(to_utf8("%s %s\n" % (word, ' '.join("%.15f" % val for val in vector))))

    @staticmethod
    def from_word2vec(fname, fvocab=None, binary=False):
        """
        Load the input-hidden weight matrix from the original C word2vec-tool format.

        `binary` is a boolean indicating whether the data is in binary word2vec format.
        Word counts are read from `fvocab` filename, if set (this is the file generated
        by `-save-vocab` flag of the original C tool).
        """
        vocabulary = None
        if fvocab is not None:
            logger.info("loading word counts from %s" % (fvocab))
            vocabulary = Embedding.from_word2vec_vocab(fvocab)

        logger.info("loading projection weights from %s" % (fname))
        if binary:
            words, vectors = Embedding._from_word2vec_binary(fname)
        else:
            words, vectors = Embedding._from_word2vec_text(fname)

        if not vocabulary:
            vocabulary = OrderedVocabulary(words=words)

        if len(words) != len(set(words)):
            raise RuntimeError("Vocabulary has duplicates")

        e = Embedding(vocabulary=vocabulary, vectors=vectors)

        return e

    @staticmethod
    def load(fname):
        """Load an embedding dump generated by `save`"""

        content = _open(fname).read()
        if PY2:
            state = pickle.loads(content, encoding='latin1')
        else:
            state = pickle.loads(content, encoding='latin1')
        voc, vec = state
        if len(voc) == 2:
            words, counts = voc
            word_count = dict(zip(words, counts))
            vocab = CountedVocabulary(word_count=word_count)
        else:
            vocab = OrderedVocabulary(voc)
        return Embedding(vocabulary=vocab, vectors=vec)

    def save(self, fname):
        """Save a pickled version of the embedding into `fname`."""

        vec = self.vectors
        voc = self.vocabulary.getstate()
        state = (voc, vec)
        with open(fname, 'wb') as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

import logging
from collections import OrderedDict
import six
from six.moves import range
import scipy
import pandas as pd
from itertools import product

logger = logging.getLogger(__name__)
import sklearn
from .datasets.analogy import *
from .utils import batched
# web.embedding import Embedding

class SimpleAnalogySolver(sklearn.base.BaseEstimator):
    """
    Answer analogy questions

    Parameters
    ----------
    w : Embedding instance

    method : {"add", "mul"}
      Method to use when finding analogy answer, see "Improving Distributional Similarity
      with Lessons Learned from Word Embeddings" O. Levy et al. 2014.

    batch_size : int
      Batch size to use while computing accuracy. This is because of extensive memory usage.

    k: int
      If not None will select k top most frequent words from embedding before doing analogy prediction
      (this can offer significant speedups)

    Note
    ----
    It is suggested to normalize and standardize embedding before passing it to SimpleAnalogySolver.
    To speed up code consider installing OpenBLAS and setting OMP_NUM_THREADS.
    """

    def __init__(self, w, method="add", batch_size=300, k=None):
        self.w = w
        self.batch_size = batch_size
        self.method = method
        self.k = k

    def score(self, X, y):
        """
        Calculate accuracy on analogy questions dataset

        Parameters
        ----------
        X : array-like, shape (n_samples, 3)
          Analogy questions.

        y : array-like, shape (n_samples, )
          Analogy answers.

        Returns
        -------
        acc : float
          Accuracy
        """
        return np.mean(y == self.predict(X))

    def predict(self, X):
        """
        Answer analogy questions

        Parameters
        ----------
        X : array-like, shape (n_samples, 3)
          Analogy questions.

        Returns
        -------
        y_pred : array-like, shape (n_samples, )
          Predicted words.
        """
        w = self.w.most_frequent(self.k) if self.k else self.w
        words = self.w.vocabulary.words
        word_id = self.w.vocabulary.word_id
        mean_vector = np.mean(w.vectors, axis=0)
        output = []

        missing_words = 0
        for query in X:
            for query_word in query:
                if query_word not in word_id:
                    missing_words += 1
        if missing_words > 0:
            logger.warning("Missing {} words. Will replace them with mean vector".format(missing_words))

        # Batch due to memory constaints (in dot operation)
        for id_batch, batch in enumerate(batched(range(len(X)), self.batch_size)):
            ids = list(batch)
            X_b = X[ids]
            if id_batch % np.floor(len(X) / (10. * self.batch_size)) == 0:
                logger.info("Processing {}/{} batch".format(int(np.ceil(ids[1] / float(self.batch_size))),
                                                            int(np.ceil(X.shape[0] / float(self.batch_size)))))

            A, B, C = np.vstack(w.get(word, mean_vector) for word in X_b[:, 0]), \
                      np.vstack(w.get(word, mean_vector) for word in X_b[:, 1]), \
                      np.vstack(w.get(word, mean_vector) for word in X_b[:, 2])

            if self.method == "add":
                D = np.dot(w.vectors, (B - A + C).T)
            elif self.method == "mul":
                D_A = np.log((1.0 + np.dot(w.vectors, A.T)) / 2.0 + 1e-5)
                D_B = np.log((1.0 + np.dot(w.vectors, B.T)) / 2.0 + 1e-5)
                D_C = np.log((1.0 + np.dot(w.vectors, C.T)) / 2.0 + 1e-5)
                D = D_B - D_A + D_C
            else:
                raise RuntimeError("Unrecognized method parameter")

            # Remove words that were originally in the query
            for id, row in enumerate(X_b):
                D[[w.vocabulary.word_id[r] for r in row if r in
                   w.vocabulary.word_id], id] = np.finfo(np.float32).min

            output.append([words[id] for id in D.argmax(axis=0)])

        return np.array([item for sublist in output for item in sublist])

import logging
import torch
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from .datasets.similarity import fetch_MEN, fetch_WS353, fetch_SimLex999, fetch_MTurk, fetch_RG65, fetch_RW, fetch_TR9856
from .datasets.categorization import fetch_AP, fetch_battig, fetch_BLESS, fetch_ESSLI_1a, fetch_ESSLI_2b, \
    fetch_ESSLI_2c
# from analogy import *
from six import iteritems
# from web.embedding import Embedding

logger = logging.getLogger(__name__)

def calculate_purity(y_true, y_pred):
    """
    Calculate purity for given true and predicted cluster labels.

    Parameters
    ----------
    y_true: array, shape: (n_samples, 1)
      True cluster labels

    y_pred: array, shape: (n_samples, 1)
      Cluster assingment.

    Returns
    -------
    purity: float
      Calculated purity.
    """
    assert len(y_true) == len(y_pred)
    true_clusters = np.zeros(shape=(len(set(y_true)), len(y_true)))
    pred_clusters = np.zeros_like(true_clusters)
    for id, cl in enumerate(set(y_true)):
        true_clusters[id] = (y_true == cl).astype("int")
    for id, cl in enumerate(set(y_pred)):
        pred_clusters[id] = (y_pred == cl).astype("int")

    M = pred_clusters.dot(true_clusters.T)
    return 1. / len(y_true) * np.sum(np.max(M, axis=1))


def evaluate_categorization(w, X, y, method="all", seed=None):
    """
    Evaluate embeddings on categorization task.

    Parameters
    ----------
    w: Embedding or dict
      Embedding to test.

    X: vector, shape: (n_samples, )
      Vector of words.

    y: vector, shape: (n_samples, )
      Vector of cluster assignments.

    method: string, default: "all"
      What method to use. Possible values are "agglomerative", "kmeans", "all.
      If "agglomerative" is passed, method will fit AgglomerativeClustering (with very crude
      hyperparameter tuning to avoid overfitting).
      If "kmeans" is passed, method will fit KMeans.
      In both cases number of clusters is preset to the correct value.

    seed: int, default: None
      Seed passed to KMeans.

    Returns
    -------
    purity: float
      Purity of the best obtained clustering.

    Notes
    -----
    KMedoids method was excluded as empirically didn't improve over KMeans (for categorization
    tasks available in the package).
    """

    if isinstance(w, dict):
        w = Embedding.from_dict(w)

    assert method in ["all", "kmeans", "agglomerative"], "Uncrecognized method"

    mean_vector = np.mean(w.vectors, axis=0, keepdims=True)
    words = np.vstack(w.get(word, mean_vector) for word in X.flatten())
    ids = np.random.RandomState(seed).choice(range(len(X)), len(X), replace=False)

    # Evaluate clustering on several hyperparameters of AgglomerativeClustering and
    # KMeans
    best_purity = 0

    if method == "all" or method == "agglomerative":
        best_purity = calculate_purity(y[ids], AgglomerativeClustering(n_clusters=len(set(y)),
                                                                       affinity="euclidean",
                                                                       linkage="ward").fit_predict(words[ids]))
        logger.debug("Purity={:.3f} using affinity={} linkage={}".format(best_purity, 'euclidean', 'ward'))
        for affinity in ["cosine", "euclidean"]:
            for linkage in ["average", "complete"]:
                purity = calculate_purity(y[ids], AgglomerativeClustering(n_clusters=len(set(y)),
                                                                          affinity=affinity,
                                                                          linkage=linkage).fit_predict(words[ids]))
                logger.debug("Purity={:.3f} using affinity={} linkage={}".format(purity, affinity, linkage))
                best_purity = max(best_purity, purity)

    if method == "all" or method == "kmeans":
        purity = calculate_purity(y[ids], KMeans(random_state=seed, n_init=10, n_clusters=len(set(y))).
                                  fit_predict(words[ids]))
        logger.debug("Purity={:.3f} using KMeans".format(purity))
        best_purity = max(purity, best_purity)

    return best_purity



def evaluate_on_semeval_2012_2(w):
    """
    Simple method to score embedding using SimpleAnalogySolver

    Parameters
    ----------
    w : Embedding or dict
      Embedding or dict instance.

    Returns
    -------
    result: pandas.DataFrame
      Results with spearman correlation per broad category with special key "all" for summary
      spearman correlation
    """
    if isinstance(w, dict):
        w = Embedding.from_dict(w)

    data = fetch_semeval_2012_2()
    mean_vector = np.mean(w.vectors, axis=0, keepdims=True)
    categories = data.y.keys()
    results = defaultdict(list)
    for c in categories:
        # Get mean of left and right vector
        prototypes = data.X_prot[c]
        prot_left = np.mean(np.vstack(w.get(word, mean_vector) for word in prototypes[:, 0]), axis=0)
        prot_right = np.mean(np.vstack(w.get(word, mean_vector) for word in prototypes[:, 1]), axis=0)

        questions = data.X[c]
        question_left, question_right = np.vstack(w.get(word, mean_vector) for word in questions[:, 0]), \
                                        np.vstack(w.get(word, mean_vector) for word in questions[:, 1])

        scores = np.dot(prot_left - prot_right, (question_left - question_right).T)

        c_name = data.categories_names[c].split("_")[0]
        # NaN happens when there are only 0s, which might happen for very rare words or
        # very insufficient word vocabulary
        cor = scipy.stats.spearmanr(scores, data.y[c]).correlation
        results[c_name].append(0 if np.isnan(cor) else cor)

    final_results = OrderedDict()
    final_results['all'] = sum(sum(v) for v in results.values()) / len(categories)
    for k in results:
        final_results[k] = sum(results[k]) / len(results[k])
    return pd.Series(final_results)


def evaluate_analogy(w, X, y, method="add", k=None, category=None, batch_size=100):
    """
    Simple method to score embedding using SimpleAnalogySolver

    Parameters
    ----------
    w : Embedding or dict
      Embedding or dict instance.

    method : {"add", "mul"}
      Method to use when finding analogy answer, see "Improving Distributional Similarity
      with Lessons Learned from Word Embeddings"

    X : array-like, shape (n_samples, 3)
      Analogy questions.

    y : array-like, shape (n_samples, )
      Analogy answers.

    k : int, default: None
      If not None will select k top most frequent words from embedding

    batch_size : int, default: 100
      Increase to increase memory consumption and decrease running time

    category : list, default: None
      Category of each example, if passed function returns accuracy per category
      in addition to the overall performance.
      Analogy datasets have "category" field that can be supplied here.

    Returns
    -------
    result: dict
      Results, where each key is for given category and special empty key "" stores
      summarized accuracy across categories
    """
    if isinstance(w, dict):
        w = Embedding.from_dict(w)

    assert category is None or len(category) == y.shape[0], "Passed incorrect category list"

    solver = SimpleAnalogySolver(w=w, method=method, batch_size=batch_size, k=k)
    y_pred = solver.predict(X)

    if category is not None:
        results = OrderedDict({"all": np.mean(y_pred == y)})
        count = OrderedDict({"all": len(y_pred)})
        correct = OrderedDict({"all": np.sum(y_pred == y)})
        for cat in set(category):
            results[cat] = np.mean(y_pred[category == cat] == y[category == cat])
            count[cat] = np.sum(category == cat)
            correct[cat] = np.sum(y_pred[category == cat] == y[category == cat])

        return pd.concat([pd.Series(results, name="accuracy"),
                          pd.Series(correct, name="correct"),
                          pd.Series(count, name="count")],
                         axis=1)
    else:
        return np.mean(y_pred == y)


def evaluate_on_WordRep(w, max_pairs=1000, solver_kwargs={}):
    """
    Evaluate on WordRep dataset

    Parameters
    ----------
    w : Embedding or dict
      Embedding or dict instance.

    max_pairs: int, default: 1000
      Each category will be constrained to maximum of max_pairs pairs
      (which results in max_pair * (max_pairs - 1) examples)

    solver_kwargs: dict, default: {}
      Arguments passed to SimpleAnalogySolver. It is suggested to limit number of words
      in the dictionary.

    References
    ----------
    Bin Gao, Jiang Bian, Tie-Yan Liu (2015)
     "WordRep: A Benchmark for Research on Learning Word Representations"
    """
    if isinstance(w, dict):
        w = Embedding.from_dict(w)

    data = fetch_wordrep()
    categories = set(data.category)

    accuracy = {}
    correct = {}
    count = {}
    for cat in categories:
        X_cat = data.X[data.category == cat]
        X_cat = X_cat[0:max_pairs]

        logger.info("Processing {} with {} pairs, {} questions".format(cat, X_cat.shape[0]
                                                                       , X_cat.shape[0] * (X_cat.shape[0] - 1)))

        # For each category construct question-answer pairs
        size = X_cat.shape[0] * (X_cat.shape[0] - 1)
        X = np.zeros(shape=(size, 3), dtype="object")
        y = np.zeros(shape=(size,), dtype="object")
        id = 0
        for left, right in product(X_cat, X_cat):
            if not np.array_equal(left, right):
                X[id, 0:2] = left
                X[id, 2] = right[0]
                y[id] = right[1]
                id += 1

        # Run solver
        solver = SimpleAnalogySolver(w=w, **solver_kwargs)
        y_pred = solver.predict(X)
        correct[cat] = float(np.sum(y_pred == y))
        count[cat] = size
        accuracy[cat] = float(np.sum(y_pred == y)) / size

    # Add summary results
    correct['wikipedia'] = sum(correct[c] for c in categories if c in data.wikipedia_categories)
    correct['all'] = sum(correct[c] for c in categories)
    correct['wordnet'] = sum(correct[c] for c in categories if c in data.wordnet_categories)

    count['wikipedia'] = sum(count[c] for c in categories if c in data.wikipedia_categories)
    count['all'] = sum(count[c] for c in categories)
    count['wordnet'] = sum(count[c] for c in categories if c in data.wordnet_categories)

    accuracy['wikipedia'] = correct['wikipedia'] / count['wikipedia']
    accuracy['all'] = correct['all'] / count['all']
    accuracy['wordnet'] = correct['wordnet'] / count['wordnet']

    return pd.concat([pd.Series(accuracy, name="accuracy"),
                      pd.Series(correct, name="correct"),
                      pd.Series(count, name="count")], axis=1)


def evaluate_similarity(w, X, y):
    """
    Calculate Spearman correlation between cosine similarity of the model
    and human rated similarity of word pairs

    Parameters
    ----------
    w : Embedding or dict
      Embedding or dict instance.

    X: array, shape: (n_samples, 2)
      Word pairs

    y: vector, shape: (n_samples,)
      Human ratings

    Returns
    -------
    cor: float
      Spearman correlation
    """
    if isinstance(w, dict):
        w = Embedding.from_dict(w)

    missing_words = 0
    words = w.vocabulary.word_id
    for query in X:
        for query_word in query:
            if query_word not in words:
                missing_words += 1
    if missing_words > 0:
        logger.warning("Missing {} words. Will replace them with mean vector".format(missing_words))


    mean_vector = np.mean(w.vectors, axis=0, keepdims=True)
    A = np.vstack([w.get(word, mean_vector) for word in X[:, 0]])
    B = np.vstack([w.get(word, mean_vector) for word in X[:, 1]])
    scores = np.array([v1.dot(v2.T)/(np.linalg.norm(v1)*np.linalg.norm(v2)) for v1, v2 in zip(A, B)])
    return scipy.stats.spearmanr(scores, y).correlation


def poincare_distance(u, v):
    """
    From: https://github.com/facebookresearch/poincare-embeddings/blob/master/model.py#L48
    """
    if (type(u) == np.ndarray):
        u = torch.from_numpy(u)

    if (type(v) == np.ndarray):
        v = torch.from_numpy(v)

    boundary = 1 - 1e-5
    squnorm = torch.clamp(torch.sum(u * u, dim=-1), 0, boundary)
    sqvnorm = torch.clamp(torch.sum(v * v, dim=-1), 0, boundary)
    sqdist = torch.sum(torch.pow(u - v, 2), dim=-1)
    x = sqdist / ((1 - squnorm) * (1 - sqvnorm)) * 2 + 1
    # arcosh
    z = torch.sqrt(torch.pow(x, 2) - 1)
    return torch.log(x + z)


def evaluate_similarity_hyperbolic(w, X, y):
    """
    Calculate Spearman correlation between cosine similarity of the model
    and human rated similarity of word pairs

    Parameters
    ----------
    w : Embedding or dict
      Embedding or dict instance.

    X: array, shape: (n_samples, 2)
      Word pairs

    y: vector, shape: (n_samples,)
      Human ratings

    Returns
    -------
    cor: float
      Spearman correlation
    """
    if isinstance(w, dict):
        w = Embedding.from_dict(w)

    missing_words = 0
    words = w.vocabulary.word_id
    for query in X:
        for query_word in query:
            if query_word not in words:
                missing_words += 1
    if missing_words > 0:
        logger.warning("Missing {} words. Will replace them with mean vector".format(missing_words))


    mean_vector = np.mean(w.vectors, axis=0, keepdims=True)
    A = np.vstack([w.get(word, mean_vector) for word in X[:, 0]])
    B = np.vstack([w.get(word, mean_vector) for word in X[:, 1]])
    scores = np.array([poincare_distance(v1,v2) for v1, v2 in zip(A, B)])
    return scipy.stats.spearmanr(-scores, y).correlation


def minkowski_dot(u, v):
    """
    from https://github.com/lateral/minkowski
    `u` and `v` are vectors in Minkowski space.
    """
    rank = u.shape[-1] - 1
    euc_dp = u[:rank].dot(v[:rank])
    return euc_dp - u[rank] * v[rank]


def evaluate_similarity_minkowski(w, X, y):
    """
    Calculate Spearman correlation between cosine similarity of the model
    and human rated similarity of word pairs

    Parameters
    ----------
    w : Embedding or dict
      Embedding or dict instance.

    X: array, shape: (n_samples, 2)
      Word pairs

    y: vector, shape: (n_samples,)
      Human ratings

    Returns
    -------
    cor: float
      Spearman correlation
    """
    if isinstance(w, dict):
        w = Embedding.from_dict(w)

    missing_words = 0
    words = w.vocabulary.word_id
    for query in X:
        for query_word in query:
            if query_word not in words:
                missing_words += 1
    if missing_words > 0:
        logger.warning("Missing {} words. Will replace them with mean vector".format(missing_words))


    mean_vector = np.mean(w.vectors, axis=0, keepdims=True)
    A = np.vstack([w.get(word, mean_vector) for word in X[:, 0]])
    B = np.vstack([w.get(word, mean_vector) for word in X[:, 1]])
    scores = np.array([minkowski_dot(v1,v2) for v1, v2 in zip(A, B)])
    return scipy.stats.spearmanr(scores, y).correlation

def evaluate_on_all(w):
    """
    Evaluate Embedding on all fast-running benchmarks

    Parameters
    ----------
    w: Embedding or dict
      Embedding to evaluate.

    Returns
    -------
    results: pandas.DataFrame
      DataFrame with results, one per column.
    """
    if isinstance(w, dict):
        w = Embedding.from_dict(w)

    # Calculate results on similarity
    logger.info("Calculating similarity benchmarks")
    similarity_tasks = {
        "MEN": fetch_MEN(),
        "WS353": fetch_WS353(),
        "WS353R": fetch_WS353(which="relatedness"),
        "WS353S": fetch_WS353(which="similarity"),
        "SimLex999": fetch_SimLex999(),
        "RW": fetch_RW(),
        "RG65": fetch_RG65(),
        "MTurk": fetch_MTurk(),
    }

    similarity_results = {}

    for name, data in iteritems(similarity_tasks):
        similarity_results[name] = evaluate_similarity(w, data.X, data.y)
        logger.info("Spearman correlation of scores on {} {}".format(name, similarity_results[name]))

    # Calculate results on analogy
    logger.info("Calculating analogy benchmarks")
    analogy_tasks = {
        "Google": fetch_google_analogy(),
        "MSR": fetch_msr_analogy()
    }

    analogy_results = {}

    for name, data in iteritems(analogy_tasks):
        analogy_results[name] = evaluate_analogy(w, data.X, data.y)
        logger.info("Analogy prediction accuracy on {} {}".format(name, analogy_results[name]))

    analogy_results["SemEval2012_2"] = evaluate_on_semeval_2012_2(w)['all']
    logger.info("Analogy prediction accuracy on {} {}".format("SemEval2012", analogy_results["SemEval2012_2"]))

    # Calculate results on categorization
    logger.info("Calculating categorization benchmarks")
    categorization_tasks = {
        "AP": fetch_AP(),
        "BLESS": fetch_BLESS(),
        "Battig": fetch_battig(),
        "ESSLI_2c": fetch_ESSLI_2c(),
        "ESSLI_2b": fetch_ESSLI_2b(),
        "ESSLI_1a": fetch_ESSLI_1a()
    }

    categorization_results = {}

    # Calculate results using helper function
    for name, data in iteritems(categorization_tasks):
        categorization_results[name] = evaluate_categorization(w, data.X, data.y)
        logger.info("Cluster purity on {} {}".format(name, categorization_results[name]))

    # Construct pd table
    cat = pd.DataFrame([categorization_results])
    analogy = pd.DataFrame([analogy_results])
    sim = pd.DataFrame([similarity_results])
    results = cat.join(sim).join(analogy)

    return results

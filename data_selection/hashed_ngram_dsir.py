from typing import List, Optional, Dict, Callable, Union, Iterable
import hashlib
from tqdm import tqdm
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import word_tokenize
from nltk import ngrams as get_ngrams
import numpy as np
from sklearn.mixture import GaussianMixture
from sentence_transformers import SentenceTransformer
from pathlib import Path

from data_selection.base import (
        DSIR,
        default_load_dataset_fn,
        default_parse_example_fn,
        _iterate_virtually_sharded_dataset,
)

from data_selection.utils import parallelize


wpt = WordPunctTokenizer()


def hash_buckets(text: str, num_buckets: int = 10000) -> int:
    return int(hashlib.sha256(text.encode('utf-8')).hexdigest(), 16) % num_buckets


def get_ngram_counts(line: str,
                     n: int = 2,
                     num_buckets: int = 10000,
                     counts: Optional[np.ndarray] = None,
                     tokenizer: Callable = wpt.tokenize) -> np.ndarray:
    '''Return ngram count features given a string.

    Args:
        line: string to get ngram counts from
        n: n in ngrams
        num_buckets: number of buckets to hash ngrams into
        counts: pre-initialized counts array
        tokenizer: tokenization function to use. Defaults to word_tokenize from nltk
    '''
    words = tokenizer(line.lower())

    if counts is None:
        counts = np.zeros(num_buckets, dtype=int)

    for w in words:
        counts[hash_buckets(w, num_buckets=num_buckets)] += 1
    for i in range(2, n + 1):
        for ng in list(get_ngrams(words, i)):
            ng = ' '.join(ng)
            counts[hash_buckets(ng, num_buckets=num_buckets)] += 1
    return counts


class HashedNgramDSIR(DSIR):
    """DSIR with hashed n-gram features."""

    def __init__(self,
                 raw_datasets: List[str],
                 target_datasets: List[str],
                 cache_dir: str,
                 raw_load_dataset_fn: Callable[[str], Iterable[Dict]] = default_load_dataset_fn,
                 raw_parse_example_fn: Callable[[Dict], str] = default_parse_example_fn,
                 target_load_dataset_fn: Callable[[str], Iterable[Dict]] = default_load_dataset_fn,
                 target_parse_example_fn: Callable[[Dict], str] = default_parse_example_fn,
                 num_proc: Optional[int] = None,
                 ngrams: int = 2,
                 num_buckets: int = 10000,
                 tokenizer: str = 'wordpunct',
                 min_example_length: int = 100,
                 target_laplace_smoothing: float = 0.0,
                 separate_targets: bool = False,
                 target_proportions: Optional[List[float]] = None) -> None:
        '''Initialize the HashedNgramDSIR object.

        Args:
            raw_datasets: List of data paths
            target_datasets: List of data paths
            cache_dir: place to store cached log_importance_weights
            load_dataset_fn: Function to load a dataset from a path. Defaults to default_load_dataset_fn.
            parse_example_fn: Function that takes in an example dict and returns a string.
                              Defaults to returning the 'text' field of the example.
            num_proc: number of processes to use for parallelization. Defaults to number of cores.
            ngrams: N in N-grams. 2 means both unigram and bigrams.
            num_buckets: number of buckets to hash ngrams into.
            tokenizer: word_tokenize or wordpunct
            min_example_length: minimum number of tokens in an example to be considered.
            target_laplace_smoothing: Smooth the target hashed ngram distribution. This parameter is a pseudo-count. This could be useful for small target datasets.
            separate_targets: whether to select data separately for each target and then join them
            target_proportions: weighting across multiple targets if separate_targets=True. Set to None to weight by the size of each target dataset
        '''
        super().__init__(
                raw_datasets=raw_datasets,
                target_datasets=target_datasets,
                cache_dir=cache_dir,
                raw_load_dataset_fn=raw_load_dataset_fn,
                raw_parse_example_fn=raw_parse_example_fn,
                target_load_dataset_fn=target_load_dataset_fn,
                target_parse_example_fn=target_parse_example_fn,
                num_proc=num_proc,
                separate_targets=separate_targets,
                target_proportions=target_proportions)
        if tokenizer == 'word_tokenize':
            self.tokenizer = word_tokenize
        elif tokenizer == 'wordpunct':
            self.tokenizer = wpt.tokenize
        else:
            raise ValueError('tokenizer not recognized')
        self.ngrams = ngrams
        self.num_buckets = num_buckets
        self.min_example_length = min_example_length
        self.raw_probs = None
        self.target_probs = None
        self.log_diff = None
        self.target_laplace_smoothing = target_laplace_smoothing
        self.gmm_raw_log_prob = None
        self.gmm_target_log_prob = None

    def featurizer(self, text: str) -> np.ndarray:
        return get_ngram_counts(text, tokenizer=self.tokenizer, num_buckets=self.num_buckets, n=self.ngrams)

    def importance_estimator(self, features: np.ndarray) -> Union[float, np.ndarray]:
        return np.dot(self.log_diff, features)

    def get_perexample_metadata(self, ex: Dict, features: np.ndarray) -> int:
        """Returns the example length."""
        remainder = self.ngrams * (self.ngrams - 1) / 2
        return (features.sum() + remainder) // self.ngrams

    def perexample_metadata_filter(self, concat_metadata: np.ndarray) -> np.array:
        """Filters out short examples."""
        return concat_metadata >= self.min_example_length

    def _fit_bow(self,
                 paths: List[str],
                 num_tokens_to_fit: Optional[int] = None,
                 load_dataset_fn: Callable[[str], Iterable[Dict]] = default_load_dataset_fn,
                 parse_example_fn: Callable[[Dict], str] = default_parse_example_fn) -> np.ndarray:

        sharded_datasets = self._get_virtually_sharded_datasets(paths)

        def job(args: Dict):
            path = args['path']
            num_shards = args['num_shards']
            shard_idx = args['shard_idx']

            counts = np.zeros(self.num_buckets).astype(int)
            dataset = load_dataset_fn(path)
            iterator = _iterate_virtually_sharded_dataset(dataset, num_shards, shard_idx)
            for ex in tqdm(iterator, miniters=10000, maxinterval=1000000):
                if parse_example_fn is not None:
                    text = parse_example_fn(ex)
                else:
                    text = ex
                counts = get_ngram_counts(text,
                                          n=self.ngrams,
                                          num_buckets=self.num_buckets,
                                          counts=counts,
                                          tokenizer=self.tokenizer)

                if num_tokens_to_fit is not None and counts.sum() > num_tokens_to_fit // len(sharded_datasets):
                    break

            return counts

        all_counts = parallelize(job, sharded_datasets, self.num_proc)
        counts = sum(all_counts)

        return counts

    def fit_ng_importance_estimator(self, num_tokens_to_fit: Union[str, int] = 'auto') -> None:
        '''Fit the importance estimator.
        Args:
            num_tokens_to_fit: number of tokens to fit the raw dataset importance estimator on.
                               Set to "all" to fit on all tokens, and "auto" to determine
                               the number of tokens to fit on automatically (100k * num_buckets).
                               Set to an integer to fit on that many tokens.
        '''
        if num_tokens_to_fit == 'auto':
            num_tokens_to_fit = 100000 * self.num_buckets
        elif num_tokens_to_fit == 'all':
            num_tokens_to_fit = None

        self.raw_probs = self._fit_bow(
                self.raw_datasets,
                num_tokens_to_fit=num_tokens_to_fit,
                parse_example_fn=self.raw_parse_example_fn,
                load_dataset_fn=self.raw_load_dataset_fn)
        
        self.raw_probs = self.raw_probs / self.raw_probs.sum()
        print("NG raw probs shape = ", self.raw_probs.shape)
        if self.separate_targets:
            target_probs = []
            target_proportions = []

            for target_dataset in self.target_datasets:
                curr_target_probs = self._fit_bow(
                        [target_dataset],
                        num_tokens_to_fit=num_tokens_to_fit,
                        parse_example_fn=self.target_parse_example_fn,
                        load_dataset_fn=self.target_load_dataset_fn)
                target_proportions.append(curr_target_probs.sum())
                # smoothing
                curr_target_probs = curr_target_probs + self.target_laplace_smoothing
                curr_target_probs = curr_target_probs / curr_target_probs.sum()
                target_probs.append(curr_target_probs)
            target_proportions = np.asarray(target_proportions)
            if self.target_proportions is None:
                self.target_proportions = target_proportions / target_proportions.sum()

            self.target_probs = np.asarray(target_probs)

        else:
            self.target_probs = self._fit_bow(
                    self.target_datasets,
                    num_tokens_to_fit=None,  # fit on all tokens for target
                    parse_example_fn=self.target_parse_example_fn,
                    load_dataset_fn=self.target_load_dataset_fn)
            # smoothing
            self.target_probs = self.target_probs + self.target_laplace_smoothing
            print("NG target probs shape = ", self.target_probs.shape)
            self.target_probs = self.target_probs / self.target_probs.sum()

        self.log_diff = np.log(self.target_probs + 1e-8) - np.log(self.raw_probs + 1e-8)
    
    def fit_gmm_importance_estimator(self, num_components=3, raw_text_emb_path=None, target_text_emb_path=None) -> None:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        gmm = GaussianMixture(n_components=num_components, covariance_type='full', random_state=42)

        print("We are here")
        if raw_text_emb_path is None:
            print("Enter 1")
            sharded_datasets = self._get_virtually_sharded_datasets(self.raw_datasets)
            
            def job(args: Dict):
                path = args["path"]
                num_shards = args["num_shards"]
                shard_idx = args["shard_idx"]
                
                # get raw dataset embeddings
                parsed_text = []
                raw_dataset = self.raw_load_dataset_fn(path)
                iterator = _iterate_virtually_sharded_dataset(raw_dataset, num_shards, shard_idx)
                for ex in tqdm(iterator, miniters=10000, maxinterval=1000000):
                    if self.raw_parse_example_fn is not None:
                        text = self.raw_parse_example_fn(ex)
                    else:
                        text = ex
                    parsed_text.append(text)
                return parsed_text
            
            all_raw_parsed_text = parallelize(job, sharded_datasets, self.num_proc)
            # stack all_text_emb
            all_raw_parsed_text = np.concatenate(all_raw_parsed_text, axis=0)
            # encode using sentence transformer
            raw_text_emb = model.encode(all_raw_parsed_text)
            # save all_raw_text_emb
            np.save(self.cache_dir / "raw_text_emb.npy", raw_text_emb)
            print("Raw text embeddings = ", raw_text_emb)
            # fit GMM to raw dataset embeddings
            gmm_raw = gmm.fit(raw_text_emb)
        else:
            print("Enter 2")
            # load raw text emb
            raw_text_emb = np.load(raw_text_emb_path)
            # fit GMM to raw dataset embeddings
            gmm_raw = gmm.fit(raw_text_emb)
            print("Raw GMM mean = ", gmm_raw.means_)
            print("Raw GMM  covariances = ", gmm_raw.covariances_)
        
        # get target dataset embeddings
        sharded_datasets = self._get_virtually_sharded_datasets(self.target_datasets)
        
        if target_text_emb_path is None:
            def job(args: Dict):
                path = args["path"]
                num_shards = args["num_shards"]
                shard_idx = args["shard_idx"]
                
                # get target dataset embeddings
                parsed_text = []
                target_dataset = self.target_load_dataset_fn(path)
                iterator = _iterate_virtually_sharded_dataset(target_dataset, num_shards, shard_idx)
                for ex in tqdm(iterator, miniters=10000, maxinterval=1000000):
                    if self.target_parse_example_fn is not None:
                        text = self.target_parse_example_fn(ex)
                    else:
                        text = ex
                    parsed_text.append(text)
                return parsed_text
            
            all_target_parsed_text = parallelize(job, sharded_datasets, self.num_proc)
            # stack all_text_emb
            all_target_parsed_text = np.concatenate(all_target_parsed_text, axis=0)
            # encode parsed text into embeddings
            target_text_emb = model.encode(all_target_parsed_text)
            # save target text emb
            np.save(self.cache_dir / "target_text_emb.npy", target_text_emb)
            print("Target text embeddings = ", target_text_emb)
            # fit GMM to target dataset embeddings
            gmm_target = gmm.fit(target_text_emb)
        else:
            # load target_text_emb
            target_text_emb = np.load(target_text_emb_path)
            # fit GMM to target text embeddings
            gmm_target = gmm.fit(target_text_emb)
            print("Target GMM mean = ", gmm_target.means_)
            print("Target GMM  covariances = ", gmm_target.covariances_)
            
        # get raw dataset log probability under respective gmm
        gmm_raw_log_prob = gmm_raw.score_samples(raw_text_emb)
        gmm_target_log_prob = gmm_target.score_samples(raw_text_emb)
        
        print("GMM raw log prob = ", gmm_raw_log_prob)
        print("GMM target log prob = ", gmm_target_log_prob)
        self.gmm_log_importance_weights = np.asarray(gmm_target_log_prob) + 1e-8 - (np.asarray(gmm_raw_log_prob) + 1e-8) #(N,)
        
        
    def compute_hybrid_importance_weight(self, ng_importance_weights=None, alpha=0.5) -> None:
        '''
        Estimate hybrid importance weight for samples in raw datasets.
        '''
        
        if ng_importance_weights is None:
            sharded_datasets = self._get_virtually_sharded_datasets(self.raw_datasets)
            def job(args: Dict):
                path = args['path']
                num_shards = args['num_shards']
                shard_idx = args['shard_idx']
                overall_idx = args['overall_idx']

                log_importance_weights = []
                perexample_metadata = []

                dataset = self.raw_load_dataset_fn(path)
                iterator = _iterate_virtually_sharded_dataset(dataset, num_shards, shard_idx)
                for ex in tqdm(iterator, miniters=10000, maxinterval=1000000):
                    if self.raw_parse_example_fn is not None:
                        text = self.raw_parse_example_fn(ex)
                    else:
                        text = ex
                    features = self.featurizer(text)
                    log_importance_weights.append(self.importance_estimator(features))
                    if perexample_metadata is not None:
                        try:
                            perexample_metadata.append(self.get_perexample_metadata(ex, features))
                        except NotImplementedError:
                            perexample_metadata = None

                ng_log_importance_weights = np.asarray(log_importance_weights)
                print("ng log importance weights shape = ", ng_log_importance_weights.shape)
                return ng_log_importance_weights

            ng_log_importance_weights = parallelize(job, sharded_datasets, self.num_proc)
            print("Length of all ng log importance weights = ", len(ng_log_importance_weights))
            ng_log_importance_weights = np.concatenate(ng_log_importance_weights, axis=0)
            print("All log importance weights shape = ", ng_log_importance_weights.shape)
            print("All gmm log importance weights shape = ", self.gmm_log_importance_weights.shape)
            print("NG log importance weights", ng_log_importance_weights)
            print("GMM log importance weights", self.gmm_log_importance_weights)
            self.hybrid_log_importance_weights = alpha * (ng_log_importance_weights + 1e-8) + (1 - alpha) * (self.gmm_log_importance_weights + 1e-8)
            print("Hybrid log importance weights = ", self.hybrid_log_importance_weights)
            save_path = self.cache_dir / "hybrid_importance_weights.npy"
            np.save(save_path, self.hybrid_log_importance_weights)
        # else:
            

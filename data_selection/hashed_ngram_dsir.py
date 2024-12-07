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
import os

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

def fit_gmm_step(gmm, data_chunk, n_init=1, n_components=3, covariance_type="diag", random_state=42, max_iter=10):
    gmm_new = GaussianMixture(n_components=n_components,
                        covariance_type=covariance_type,
                        n_init=n_init,
                        max_iter=max_iter,
                        random_state=random_state,
                        means_init=gmm.means_,
                        weights_init=gmm.weights_,
                        precisions_init=gmm.precisions_)
    gmm_new.fit(data_chunk)
    return gmm_new

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
    
    def fit_gmm_importance_estimator(self, 
                                     chunk_size=10, 
                                     raw_max_samples: Union[str, int] = "all", 
                                     target_max_samples: Union[str, int] = "all", 
                                     raw_text_emb_path=None, 
                                     target_text_emb_path=None) -> None:
        '''
        Fit GMM models on raw and target dataset iteratively on chunks of text embeddings. 
        Compute NN importance weights.
        chunk_size: number of text embeddings to be loaded at each iteration
        raw_max_samples: total number of raw text embeddings to be used to fit GMM, only used when `raw_text_emb_path` are provided
                        possible values = integer or "all"
                        when raw_max_samples == "all", all the samples in raw_text_emb_path are used.
        target_max_samples: total number of target text embeddings to be used to fit GMM, only used when `target_text_emb_path` are provided
                        possible values = integer or "all"
                        when raw_max_samples == "all", all the samples in raw_text_emb_path are used
        raw_text_emb_path: dirname to raw text embeddings
        target_text_emb_path: dirname to target text embeddings
        '''
        model = SentenceTransformer("all-MiniLM-L6-v2")
        
        gmm_raw_kwargs = {"n_components":1000, "covariance_type": "diag", "random_state": 42}
        gmm_target_kwargs = {"n_components":50, "covariance_type": "diag", "random_state": 42}
        gmm_raw = GaussianMixture(**gmm_raw_kwargs)
        gmm_target = GaussianMixture(**gmm_target_kwargs)
        
        # fit gmm on raw dataset
        # save raw dataset text embeddings in chunks if not provided
        if raw_text_emb_path is None:
            sharded_datasets = self._get_virtually_sharded_datasets(self.raw_datasets)
            raw_text_emb_path = self.cache_dir / "raw text embeddings"
            if not os.path.isdir(raw_text_emb_path):
                print("Creat folder for saving raw text embedding at 'cache dir/raw text embeddings'")
                os.mkdir(raw_text_emb_path)
            def job(args: Dict):
                path = args["path"]
                num_shards = args["num_shards"]
                shard_idx = args["shard_idx"]
                
                # get raw dataset embeddings
                parsed_text = []
                raw_dataset = self.raw_load_dataset_fn(path)
                iterator = _iterate_virtually_sharded_dataset(raw_dataset, num_shards, shard_idx)
                chunk = 0
                for ex in tqdm(iterator, miniters=10000, maxinterval=1000000):
                    if self.raw_parse_example_fn is not None:
                        text = self.raw_parse_example_fn(ex)
                    else:
                        text = ex
                    parsed_text.append(text)
                    if len(parsed_text) == chunk_size:
                        raw_text_emb = model.encode(parsed_text)
                        np.save(raw_text_emb_path / f"{chunk}.npy", raw_text_emb)
                        # clear current chunk
                        parsed_text.clear()
                        chunk += 1
                if parsed_text:
                    raw_text_emb = model.encode(parsed_text)
                    np.save(raw_text_emb_path / f"{chunk}.npy", raw_text_emb)
                    chunk += 1
                return
            parallelize(job, sharded_datasets, self.num_proc)
        
        # fit gmm on raw dataset iteratively 
        if raw_max_samples == "all":
            # if raw_max_samples is "all", use all the chunks found in the raw_text_emb_path
            n_chunks = len(os.listdir(raw_text_emb_path))
        else:
            n_chunks = raw_max_samples // chunk_size # max samples controls how many chunks to be load
        chunk = 0
        # print("chunk = ", chunk)
        raw_text_emb_chunk = np.load(raw_text_emb_path / f"{chunk}.npy")
        target_size = 1000
        repeat_times = target_size // raw_text_emb_chunk.shape[0] + 1
        raw_text_emb_chunk = np.tile(raw_text_emb_chunk, (repeat_times, 1))[:target_size, :]
        # print(f"raw text embedding chunk {chunk} = ", raw_text_emb_chunk)
        gmm_raw = gmm_raw.fit(raw_text_emb_chunk)
        # print("raw GMM means = ", gmm_raw.means_)
        for chunk in range(1, n_chunks):
            # print("chunk = ", chunk)
            # load raw text emb
            raw_text_emb_chunk = np.load(raw_text_emb_path / f"{chunk}.npy")
            raw_text_emb_chunk = np.tile(raw_text_emb_chunk, (repeat_times, 1))[:target_size, :]
            # print(f"raw text embedding chunk {chunk} = ", raw_text_emb_chunk)
            # fit GMM to raw dataset embeddings
            gmm_raw = fit_gmm_step(gmm_raw, raw_text_emb_chunk, **gmm_raw_kwargs)
            # print("raw GMM means = ", gmm_raw.means_)
            # clear the current chunk before moving to the next
            del raw_text_emb_chunk
        
        # fit gmm on target dataset
        # save chunks of target dataset embeddings if not provided
        if target_text_emb_path is None:
            # get target dataset embeddings
            sharded_datasets = self._get_virtually_sharded_datasets(self.target_datasets)
            target_text_emb_path = self.cache_dir / "target text embeddings"
            if not os.path.isdir(target_text_emb_path):
                print("Creat folder for saving target text embedding at 'cache dir/target text embeddings'")
                os.mkdir(target_text_emb_path)
            def job(args: Dict):
                path = args["path"]
                num_shards = args["num_shards"]
                shard_idx = args["shard_idx"]
                
                # get target dataset embeddings
                parsed_text = []
                target_dataset = self.target_load_dataset_fn(path)
                chunk = 0
                iterator = _iterate_virtually_sharded_dataset(target_dataset, num_shards, shard_idx)
                for ex in tqdm(iterator, miniters=10000, maxinterval=1000000):
                    if self.target_parse_example_fn is not None:
                        text = self.target_parse_example_fn(ex)
                    else:
                        text = ex
                    parsed_text.append(text)
                    if len(parsed_text) == chunk_size:
                        target_text_emb = model.encode(parsed_text)
                        np.save(target_text_emb_path / f"{chunk}.npy", target_text_emb)
                        parsed_text.clear()
                        chunk += 1
                # if there are parsed text left, save embeddings of the left text
                if parsed_text:
                    target_text_emb = model.encode(parsed_text)
                    np.save(target_text_emb_path / f"{chunk}.npy", target_text_emb)
                    chunk += 1
                return
            parallelize(job, sharded_datasets, self.num_proc)

        # fit gmm iteratively
        if target_max_samples == "all":
            # if target_max_samples is "all", use all the chunks found in the target_text_emb_path
            n_chunks = len(os.listdir(target_text_emb_path))
        else:
            n_chunks = target_max_samples // chunk_size # max samples controls how many chunks to be loaded
        chunk = 0
        target_text_emb_chunk = np.load(os.path.join(target_text_emb_path, f"{chunk}.npy"))
        target_size = 50
        repeat_times = target_size // target_text_emb_chunk.shape[0] + 1
        target_text_emb_chunk = np.tile(target_text_emb_chunk, (repeat_times, 1))[:target_size, :]
        gmm_target = gmm_target.fit(target_text_emb_chunk)
        # print("Target GMM means = ", gmm_target.means_)
        for chunk in range(1, n_chunks):
            # print("Chunk = ", chunk)
            # load target text emb
            target_text_emb_chunk = np.load(os.path.join(target_text_emb_path, f"{chunk}.npy"))
            target_text_emb_chunk = np.tile(target_text_emb_chunk, (repeat_times, 1))[:target_size, :]
            # print(f"target text embedding chunk {chunk}: ", target_text_emb_chunk)
            # fit GMM to target dataset embeddings
            gmm_target = fit_gmm_step(gmm_target, target_text_emb_chunk, **gmm_target_kwargs)
            # print("Target GMM means = ", gmm_target.means_)
            # clear the current chunk before moving to the next
            del target_text_emb_chunk
        
        # get raw dataset log probability under respective gmm
        gmm_raw_log_prob_l = []
        gmm_target_log_prob_l = []
        if raw_max_samples == "all":
            # if raw_max_samples is "all", use all the chunks found in the raw_text_emb_path
            n_chunks = len(os.listdir(raw_text_emb_path))
        else:
            n_chunks = raw_max_samples // chunk_size # max samples controls how many chunks to be load
        for chunk in range(n_chunks):
            # load raw text emb
            raw_text_emb_chunk = np.load(os.path.join(raw_text_emb_path, f"{chunk}.npy"))
            # get log likelihood of the chunk
            gmm_raw_log_prob_chunk = gmm_raw.score_samples(raw_text_emb_chunk)
            gmm_target_log_prob_chunk = gmm_target.score_samples(raw_text_emb_chunk)
            # add to the list of log likelihood
            gmm_raw_log_prob_l.append(gmm_raw_log_prob_chunk)
            gmm_target_log_prob_l.append(gmm_target_log_prob_chunk)
            # clear the current chunk before moving to the next
            del raw_text_emb_chunk, gmm_raw_log_prob_chunk, gmm_target_log_prob_chunk
        gmm_raw_log_prob = np.concatenate(gmm_raw_log_prob_l, axis=0)
        gmm_target_log_prob = np.concatenate(gmm_target_log_prob_l, axis=0)
        
        print("GMM raw log prob = ", gmm_raw_log_prob)
        print("GMM target log prob = ", gmm_target_log_prob)
        self.gmm_log_importance_weights = np.asarray(gmm_target_log_prob) + 1e-8 - (np.asarray(gmm_raw_log_prob) + 1e-8) #(N,)
        print("GMM log importance weights = ", self.gmm_log_importance_weights)
        
        
    def compute_hybrid_importance_weight(self, ng_importance_weights_path=None, alpha=0.5, save_path=None) -> None:
        '''
        Estimate hybrid importance weight for samples in raw datasets.
        ng_importance_weights_path: path to pre-calculated NG importance weights
        alpha: proportion of NG importance weights
        '''
        
        if ng_importance_weights_path is None:
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
                            perexample_metadata.append(self.mple_metget_perexaadata(ex, features))
                        except NotImplementedError:
                            perexample_metadata = None

                ng_log_importance_weights = np.asarray(log_importance_weights)
                return ng_log_importance_weights

            ng_log_importance_weights = parallelize(job, sharded_datasets, self.num_proc)
            ng_log_importance_weights = np.concatenate(ng_log_importance_weights, axis=0)
            print("NG log importance weights", ng_log_importance_weights)
        else:
            ng_log_importance_weights = np.load(ng_importance_weights_path)
        
        self.hybrid_log_importance_weights = alpha * (ng_log_importance_weights + 1e-8) + (1 - alpha) * (self.gmm_log_importance_weights + 1e-8)
        print("Hybrid log importance weights = ", self.hybrid_log_importance_weights)
        print("Hybrid importance weights = ", np.exp(self.hybrid_log_importance_weights))
        save_path = save_path if save_path else self.cache_dir / "hybrid_importance_weights.npy"
        np.save(save_path, self.hybrid_log_importance_weights)
            
            
            

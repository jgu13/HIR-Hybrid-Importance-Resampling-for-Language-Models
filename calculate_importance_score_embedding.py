from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
from pathlib import Path
import json


def default_load_dataset_fn(path: str):
    """Load jsonl dataset from path

    Args:
        path (str): path to dataset file
    """
    with open(path, 'r') as f:
        for line in f:
            if len(line) > 0:
                yield json.loads(line)
                

def default_parse_example_fn(ex) -> str:
    """Default parse function from example dict to string

    Args:
        ex (Dict): example dict
    """
    return ex['text']
                
                
class calculate_important_score_embedding():
    def __init__(self, 
                 raw_datapath,
                 cache_dir, 
                 raw_load_dataset_fn=default_load_dataset_fn,
                 raw_parse_example_fn=default_parse_example_fn,
                 batchsize=1,
                 parallel_token=True):
        # Load the model and enable multi-GPU support
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
        self.parallel = parallel_token
        self.raw_datasets = raw_datapath
        self.cache_dir = Path(cache_dir)
        self.raw_load_dataset_fn = raw_load_dataset_fn
        self.raw_parse_example_fn = raw_parse_example_fn
        if self.parallel:
            self.model.parallel_tokenization = True  # Enables multi-GPU tokenization if available

        # Assume `raw_dataset` is already loaded
        self.embeddings = []
        self.batch_size = batchsize  # Adjust based on memory and dataset size
        
    def compute(self):
        print(f"loading the dataset: {self.raw_datasets}")
        total_length = 5899215 #0
        # with open(self.raw_datasets, 'r') as f:
        #     for line in f:
        #         if len(line) > 0:
        #             total_length+=1
        raw_dataset = self.raw_load_dataset_fn(self.raw_datasets)
        print(f"finish loading dataset, there are {total_length} lines")
        print('start to compute...')
        for i in tqdm(range(0, total_length, self.batch_size), desc="Encoding"):
            try:
                batch = [next(raw_dataset) for _ in range(self.batch_size)]
            except:
                break
            batch_text = [self.raw_parse_example_fn(ex) if self.raw_parse_example_fn else ex for ex in batch]
            batch_emb = self.model.encode(
                batch_text, 
                batch_size=self.batch_size, 
                show_progress_bar=False,
                # device="cuda:0"  # Multi-GPU encoding works automatically
            )
            np.save(self.cache_dir / f"raw_text_emb{i}.npy", batch_emb)
            self.embeddings.append(batch_emb)

        # Save concatenated embeddings
        self.embeddings = np.concatenate(self.embeddings, axis=0)
        np.save(self.cache_dir / "raw_text_emb.npy", self.embeddings)
        print('finish compute the embedding')
        
if __name__ == "__main__":
    # calculate = calculate_important_score_embedding('/home/mcb/users/jgu13/projects/HIR-Hybrid-Importance-Resampling-for-Language-Models/Pile/Pile.jsonl',
    #                                                 '/home/mcb/users/jgu13/projects/HIR-Hybrid-Importance-Resampling-for-Language-Models/dsir_cache/raw_text_embedding')
    calculate = calculate_important_score_embedding('/home/mcb/users/jgu13/projects/HIR-Hybrid-Importance-Resampling-for-Language-Models/Pile/Target_set.jsonl',
                                                '/home/mcb/users/jgu13/projects/HIR-Hybrid-Importance-Resampling-for-Language-Models/dsir_cache/target_set_embedding')
    calculate.compute()
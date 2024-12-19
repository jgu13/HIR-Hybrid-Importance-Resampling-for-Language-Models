import subprocess
import numpy as np
import os


import argparse
parser = argparse.ArgumentParser(description='merge quality stats')
parser.add_argument('--pile_path', type=str, help='path to pile')
args = parser.parse_args()


subsets = ['00']

filenames_qualityscores = [f"{args.pile_path}/chunked/{subset}_128/{subset}_128.json_qualityscores.npz" for subset in subsets]


# merge the numpy
qualityscores = [np.load(fn) for fn in filenames_qualityscores]
print("quality scores loaded")
keys = qualityscores[0].keys()
concatted = {k: np.concatenate([qs[k] for qs in qualityscores], axis=0) for k in keys}

print("saving")

np.savez(f"{args.pile_path}/chunked/combined_all.json_qualityscores.npz",
         **concatted)

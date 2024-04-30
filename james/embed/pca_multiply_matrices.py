
import os
import time
import argparse
import sys

import pandas as pd
import torch
import tqdm
import numpy as np
from sklearn.decomposition import PCA
from torch.quantization import quantize_dynamic
from torch.nn import Embedding, Linear

from sentence_transformers import SentenceTransformer, models

parser = argparse.ArgumentParser()
parser.add_argument("--input_embeddings_file", type=str, help="should end with .npy")
parser.add_argument("--output_embeddings_file", type=str, help="should end with .npy")
parser.add_argument("--pca_matrix_file", type=str)

args = parser.parse_args()
print(vars(args), flush=True)

embeddings = np.load(args.input_embeddings_file)
print(f"{embeddings.shape=}", flush=True)

pca_matrix = np.load(args.pca_matrix_file)
print(f"{pca_matrix.shape=}", flush=True)

print(f"Applying PCA Matrix from: {args.pca_matrix_file}", flush=True)
reduced_embeddings = embeddings@pca_matrix.T
print(f"{reduced_embeddings.shape=}", flush=True)

print(f"Saving reduced embeddings to: {args.output_embeddings_file}", flush=True)
os.makedirs(os.path.dirname(args.output_embeddings_file), exist_ok=True)
np.save(args.output_embeddings_file, reduced_embeddings)
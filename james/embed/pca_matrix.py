
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
parser.add_argument("--input_file", type=str, help="should end with .npy")
parser.add_argument("--output_embeddings_file", type=str, help="should end with .npy")
parser.add_argument("--output_matrix_file", type=str, help="should end with .npy")
parser.add_argument("--model_name", type=str, default="all-MiniLM-L6-v2")
parser.add_argument("--embed_dim", type=int, default=384)

args = parser.parse_args()
print(vars(args), flush=True)

model = SentenceTransformer(args.model_name).cuda()
print(f"Model ({args.model_name}) Initialized", flush=True)

def pca_matrix(model, embeddings, pca_dim, save_matrix_path=None):
  # Compute PCA on the train embeddings matrix
  pca = PCA(n_components=pca_dim)

  reduced_embeddings = pca.fit_transform(embeddings)
  pca_comp = np.asarray(pca.components_)

  if save_matrix_path:
    print(f"Saving pca matrix to: {args.output_matrix_file}", flush=True)
    os.makedirs(os.path.dirname(args.output_matrix_file), exist_ok=True)
    np.save(args.output_matrix_file, pca_comp)

  print(f"Saving reduced embeddings to: {args.output_embeddings_file}", flush=True)
  os.makedirs(os.path.dirname(args.output_embeddings_file), exist_ok=True)
  np.save(args.output_embeddings_file, reduced_embeddings)

pca_matrix(model, np.load(args.input_file), args.embed_dim, args.output_matrix_file)
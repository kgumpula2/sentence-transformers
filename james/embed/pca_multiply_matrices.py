
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

def multiply_matrices(embeddings, pca_matrix, output_embeddings_file):
  print(f"Applying PCA Matrix from: {args.pca_matrix_file}", flush=True)
  reduced_embeddings = embeddings@pca_matrix.T

  print(f"Saving reduced embeddings to: {output_embeddings_file}", flush=True)
  os.makedirs(os.path.dirname(output_embeddings_file), exist_ok=True)
  np.save(output_embeddings_file, reduced_embeddings)

multiply_matrices(np.load(args.input_embeddings_file), np.load(args.pca_matrix_file), args.output_embeddings_file)
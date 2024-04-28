
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
parser.add_argument("--tsv", required=True, type=str)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--output_file", type=str, help="should end with .npy")
parser.add_argument("--model_name", type=str, default="all-MiniLM-L6-v2")
parser.add_argument("--embedding_dimension", type=int, default=)

args = parser.parse_args()
print(vars(args), flush=True)

# query_dev = pd.read_csv("./data/queries/queries.dev.tsv",
#                         header=None, sep="\t", index_col=0, names=["query"])
# collection = pd.read_csv("./data/collection/collection.tsv",
#                          header=None, sep="\t", index_col=0, names=["passage"])
dataframe = pd.read_csv(args.tsv, header=None, sep="\t",
                        index_col=0, names=["text"])
print(f"{len(dataframe)=}", flush=True)
model = SentenceTransformer(args.model_name).cuda()
print(f"Model ({args.model_name}) Initialized", flush=True)


def do_embedding(model, series, batch_size=128):
  embeddings = []
  n = len(series)
  start = time.time()
  for i in tqdm.tqdm(range((n + (batch_size - 1)) // batch_size)):
    a, b = batch_size * i, min(n, batch_size * (i + 1))
    embedding_i = model.encode(series[a:b].tolist())
    embeddings.append(embedding_i)
  end = time.time()
  print(f"Time Taken: {(end-start):0.4f} s")
  return np.vstack(embeddings), end - start

def do_pca_embeddings(embeddings, pca_dim):
  pca = PCA(n_components=pca_dim)
  return pca.fit_transform(embeddings)

def pca_model(model, fit_embeddings, pca_dim, save_model_path=None):
  # Compute PCA on the train embeddings matrix
  pca = PCA(n_components=pca_dim)
  pca.fit(fit_embeddings)
  pca_comp = np.asarray(pca.components_)

  # We add a dense layer to the model, so that it will produce directly embeddings with the new size
  dense = models.Dense(
      in_features=model.get_sentence_embedding_dimension(),
      out_features=pca_dim,
      bias=False,
      activation_function=torch.nn.Identity(),
  )
  dense.linear.weight = torch.nn.Parameter(torch.tensor(pca_comp))
  model.add_module("dense", dense)

  if save_model_path:
    model.save(save_model_path)

  return model

def quantize_model(model, quantization_method="base"):
  if quantization_method == "base":
    q_model = quantize_dynamic(model, {Linear}, dtype=torch.qint8)
  else:
    raise ValueError("Invalid quantization method")
  return q_model

text_embeddings, time_taken = do_embedding(
  model, dataframe["text"], batch_size=args.batch_size)
print(f"{time_taken=}", flush=True)
print(f"{text_embeddings.shape=}", flush=True)

print(f"Saving to: {args.output_file}", flush=True)
os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
np.save(args.output_file, text_embeddings)

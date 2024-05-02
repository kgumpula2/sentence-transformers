import os
import numpy as np
import faiss
import time
import argparse
import sys
from bao.post_quant.embedding_quantization import post_quantize_embeddings

parser = argparse.ArgumentParser()
parser.add_argument("--residing_folder", type=str, required=True)
parser.add_argument("--collection_file", type=str, required=True)
parser.add_argument("--query_file", type=str, required=True)
parser.add_argument("--output_file", type=str, required=True)
parser.add_argument("--output_file_scores", type=str, required=True)
parser.add_argument("--nlist", type=int, default=10000)
parser.add_argument("--nprobe", type=int, default=200)
parser.add_argument("--d", type=int, default=384, help="redundant")
parser.add_argument("--k", type=int, default=1000)
parser.add_argument("--truncation_d", type=int, default=-
                    1, help="dimension used to truncate")
parser.add_argument("--pca_d", type=int, default=-
                    1, help="dimension used to pca")
parser.add_argument("--pca_load_from", type=str, default=None,
                    help="load pretrained pca model")
parser.add_argument("--normalize", action="store_true",
                    help="whether to normalize")
parser.add_argument("--post_quant", type=str, default=None)
args = parser.parse_args()
print(vars(args), flush=True)

# residing_folder = "./data/results/baseline"
# collection_file = "collection_embeddings.npy"
# query_file = "query_embeddings.npy"
# output_file = "knn.npy"
# k = 1000 # knn
# nlist = 10000 # typically 4 * sqrt(n)
# nprobe = 200 # typically arbitrary
# d = 384

residing_folder, collection_file, query_file = args.residing_folder, args.collection_file, args.query_file
output_file, nlist, nprobe, d = args.output_file, args.nlist, args.nprobe, args.d
output_file_scores = args.output_file_scores
truncation_d = args.truncation_d
pca_d, pca_load_from = args.pca_d, args.pca_load_from
normalize = args.normalize
post_quant_precision = args.post_quant
k = args.k

corpus_embeddings = np.load(os.path.join(residing_folder, collection_file))
query_embeddings = np.load(os.path.join(residing_folder, query_file))
assert corpus_embeddings.shape[1] == query_embeddings.shape[1]
d = corpus_embeddings.shape[1]
print(f"Loaded: {corpus_embeddings.shape=}", flush=True)
print(f"Loaded: {query_embeddings.shape=}", flush=True)

if truncation_d != -1:
  print(f"Applying Truncation (d={truncation_d})", flush=True)
  corpus_embeddings = corpus_embeddings[:, :truncation_d]
  query_embeddings = query_embeddings[:, :truncation_d]
if pca_d != -1:
  print(f"Applying PCA (d={pca_d})", flush=True)
  if pca_load_from is not None:
    print(f"Load from: {pca_load_from}", flush=True)
    PCA = faiss.read_VectorTransform(pca_load_from)
    assert PCA.d_in == d
    assert PCA.d_out == pca_d
  else:
    print(f"Training: {d} -> {pca_d}", flush=True)
    PCA = faiss.PCAMatrix(d, pca_d)
    a = time.time()
    PCA.train(corpus_embeddings)
    b = time.time()
    print(f"Time to train PCA: {(b-a):0.4f} s", flush=True)
    output_basename = os.path.dirname(output_file)
    write_file = os.path.join(output_basename, f"pca_{pca_d}.pca")
    os.makedirs(output_basename, exist_ok=True)
    print(f"Saving PCA file to: {write_file}", flush=True)
    faiss.write_VectorTransform(PCA, write_file)
  corpus_embeddings = PCA.apply(corpus_embeddings)
  query_embeddings = PCA.apply(query_embeddings)
if normalize:
  print(f"Applying Normalization", flush=True)
  corpus_embeddings = corpus_embeddings / \
      np.linalg.norm(corpus_embeddings, ord=2, axis=1, keepdims=True)
  query_embeddings = query_embeddings / \
      np.linalg.norm(query_embeddings, ord=2, axis=1, keepdims=True)
if post_quant_precision is not None:
  # TODO: extend for mixed precision for queries | corpus
  corpus_embeddings = post_quantize_embeddings(
    corpus_embeddings, post_quant_precision)
  query_embeddings = post_quantize_embeddings(
    query_embeddings, post_quant_precision)
  print(f"after quantization {corpus_embeddings.shape=}", flush=True)
  print(f"after quantization {query_embeddings.shape=}", flush=True)

# modify this if you do dimensionality reduction
print(f"Post-Processing: {corpus_embeddings.shape=}", flush=True)
print(f"Post-Processing: {query_embeddings.shape=}", flush=True)

assert corpus_embeddings.shape[-1] == query_embeddings.shape[-1]
faiss_d = corpus_embeddings.shape[-1]
quantizer = faiss.IndexFlatIP(faiss_d)
cpu_index = faiss.IndexIVFFlat(quantizer, faiss_d, nlist)
a = time.time()
cpu_index.train(corpus_embeddings)
cpu_index.add(corpus_embeddings)
b = time.time()
print(f"Time to train: {(b-a):0.2f} s", flush=True)

cpu_index.nprobe = nprobe
a = time.time()
D, I = cpu_index.search(query_embeddings, k)
b = time.time()
print(f"Time to search: {(b - a):0.2f} s", flush=True)
print(f"Saving to: {output_file}, {output_file_scores}", flush=True)
os.makedirs(os.path.dirname(output_file), exist_ok=True)
os.makedirs(os.path.dirname(output_file_scores), exist_ok=True)
np.save(output_file, I)
np.save(output_file_scores, D)

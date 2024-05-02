"""
This script runs the evaluation of an SBERT msmarco model on the
MS MARCO dev dataset and reports different performances metrices for cossine similarity & dot-product.

Usage:
python eval_msmarco.py model_name [max_corpus_size_in_thousands]
"""
import sys
from sentence_transformers import LoggingHandler, SentenceTransformer, util, evaluation
# import InformationRetrievalEvaluator
import logging
import os
import tarfile
import torch
from torch.quantization import quantize, quantize_dynamic
from torch.nn import Embedding, Linear
from quanto import Calibration, freeze, qint2, qint4, qint8, quantize
import argparse

    
def named_module_tensors(module, recurse=False):
    for named_parameter in module.named_parameters(recurse=recurse):
      name, val = named_parameter
      flag = True
      if hasattr(val,"_data") or hasattr(val,"_scale"):
        if hasattr(val,"_data"):
          yield name + "._data", val._data
        if hasattr(val,"_scale"):
          yield name + "._scale", val._scale
      else:
        yield named_parameter

    for named_buffer in module.named_buffers(recurse=recurse):
      yield named_buffer

def dtype_byte_size(dtype):
    """
    Returns the size (in bytes) occupied by one parameter of type `dtype`.
    """
    import re
    if dtype == torch.bool:
        return 1 / 8
    bit_search = re.search(r"[^\d](\d+)$", str(dtype))
    if bit_search is None:
        raise ValueError(f"`dtype` is not a valid dtype: {dtype}.")
    bit_size = int(bit_search.groups()[0])
    return bit_size // 8

def compute_module_sizes(model):
    """
    Compute the size of each submodule of a given model.
    """
    from collections import defaultdict
    module_sizes = defaultdict(int)
    for name, tensor in named_module_tensors(model, recurse=True):
      size = tensor.numel() * dtype_byte_size(tensor.dtype)
      name_parts = name.split(".")
      for idx in range(len(name_parts) + 1):
        module_sizes[".".join(name_parts[:idx])] += size

    return module_sizes

def return_quant(arg_int):
    if arg_int == 2:
        return qint2
    elif arg_int == 4:
        return qint4
    elif arg_int == 8:
        return qint8
    elif arg_int == None:
        return None
    else:
        raise ValueError("Invalid quantization bit")

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)
#### /print debug information to stdout

# write arg parser for system arguments used in code not using sys.argv
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2")
parser.add_argument("--corpus_max_size", type=int, default=0)
parser.add_argument("--use_quantization", action="store_true")
parser.add_argument("--weight_quant", type=int, default=2)
parser.add_argument("--activation_quant", type=int, default=None)
args = parser.parse_args()

# You can limit the approx. max size of the corpus. Pass 100 as second parameter and the corpus has a size of approx 100k docs
corpus_max_size = args.corpus_max_size * 1000  

####  Load model
model = SentenceTransformer(args.model)

if args.use_quantization:
    module_sizes = compute_module_sizes(model)
    print(f"The original model size is {module_sizes[''] * 1e-9} GB")
    quantize(model, weights=return_quant(args.weight_quant), activations=args.activation_quant)
    freeze(model)
    module_sizes = compute_module_sizes(model)
    print(f"The quantized model size is {module_sizes[''] * 1e-9} GB")
    # model = torch.compile(model)

### Data files
data_folder = "msmarco-data"
os.makedirs(data_folder, exist_ok=True)

collection_filepath = os.path.join(data_folder, "collection.tsv")
dev_queries_file = os.path.join(data_folder, "queries.dev.small.tsv")
qrels_filepath = os.path.join(data_folder, "qrels.dev.tsv")

### Download files if needed
if not os.path.exists(collection_filepath) or not os.path.exists(dev_queries_file):
    tar_filepath = os.path.join(data_folder, "collectionandqueries.tar.gz")
    if not os.path.exists(tar_filepath):
        logging.info("Download: " + tar_filepath)
        util.http_get(
            "https://msmarco.z22.web.core.windows.net/msmarcoranking/collectionandqueries.tar.gz", tar_filepath
        )

    with tarfile.open(tar_filepath, "r:gz") as tar:
        tar.extractall(path=data_folder)


if not os.path.exists(qrels_filepath):
    util.http_get("https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.dev.tsv", qrels_filepath)

### Load data

corpus = {}  # Our corpus pid => passage
dev_queries = {}  # Our dev queries. qid => query
dev_rel_docs = {}  # Mapping qid => set with relevant pids
needed_pids = set()  # Passage IDs we need
needed_qids = set()  # Query IDs we need

# Load the 6980 dev queries
with open(dev_queries_file, encoding="utf8") as fIn:
    for line in fIn:
        qid, query = line.strip().split("\t")
        dev_queries[qid] = query.strip()


# Load which passages are relevant for which queries
with open(qrels_filepath) as fIn:
    for line in fIn:
        qid, _, pid, _ = line.strip().split("\t")

        if qid not in dev_queries:
            continue

        if qid not in dev_rel_docs:
            dev_rel_docs[qid] = set()
        dev_rel_docs[qid].add(pid)

        needed_pids.add(pid)
        needed_qids.add(qid)


# Read passages
with open(collection_filepath, encoding="utf8") as fIn:
    for line in fIn:
        pid, passage = line.strip().split("\t")
        passage = passage

        if pid in needed_pids or corpus_max_size <= 0 or len(corpus) <= corpus_max_size:
            corpus[pid] = passage.strip()


## Run evaluator
logging.info("Queries: {}".format(len(dev_queries)))
logging.info("Corpus: {}".format(len(corpus)))

ir_evaluator = evaluation.InformationRetrievalEvaluator(
    dev_queries,
    corpus,
    dev_rel_docs,
    show_progress_bar=True,
    corpus_chunk_size=100000,
    precision_recall_at_k=[10, 100],
    mrr_at_k=[10, 100, 1000],
    name="msmarco dev",
)

ir_evaluator(model)

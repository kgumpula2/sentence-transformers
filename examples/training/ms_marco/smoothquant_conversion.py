import torch
from torchao.quantization.smoothquant import swap_linear_with_smooth_fq_linear, smooth_fq_linear_to_inference
from sentence_transformers import LoggingHandler, SentenceTransformer, util, evaluation
import os
import tarfile


# Fuse the int8*int8 -> int32 matmul and subsequent mul op avoiding materialization of the int32 intermediary tensor
torch._inductor.config.force_fuse_int_mm_with_mul = True

# plug in your model
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
# model = SentenceTransformer('models/og_model/', device=device)
# convert linear modules to smoothquant
# linear module in calibration mode
swap_linear_with_smooth_fq_linear(model)


### Data files
data_folder = "msmarco-data"
os.makedirs(data_folder, exist_ok=True)

collection_filepath = os.path.join(data_folder, "collection.tsv")
dev_queries_file = os.path.join(data_folder, "queries.dev.small.tsv")
qrels_filepath = os.path.join(data_folder, "qrels.dev.tsv")

# You can limit the approx. max size of the corpus. Pass 100 as second parameter and the corpus has a size of approx 100k docs
corpus_max_size = 1 * 1000  

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


batch_size = 64
num_batches = 50

# sentences = dev_queries
sentences = dev_queries

# Calibrate the model on num_batches from queries
sentences = list(sentences.values())
model.train()
start = 0
end = start + batch_size
while end <= batch_size * num_batches:
    print(start, end)
    inputs = sentences[start:end]
    encoded_input = model.tokenizer(inputs, padding=True, truncation=True, return_tensors='pt').to(device)
    model(encoded_input)
    start = end
    end += batch_size
    del inputs, encoded_input
    torch.cuda.empty_cache()

# set it to inference mode
smooth_fq_linear_to_inference(model)

# compile the model to improve performance
model = torch.compile(model, mode='max-autotune')
model.save('models/sheesh_corpus_large')
print("DONE WITH SMOOTHQUANT")

source ~/miniconda3/etc/profile.d/conda.sh && 
conda activate $ENV_NAME &&
cd $SENTENCE_TRANSFORMERS && 
PYTHONPATH=. python james/embed/embed.py \
  --output_file data/results/matryoshka/collection_embeddings.npy \
  --batch_size 256 \
  --tsv data/collection/collection.tsv \
  --model_name tomaarsen/mpnet-base-nli-matryoshka


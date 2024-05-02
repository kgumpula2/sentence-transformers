source ~/miniconda3/etc/profile.d/conda.sh && 
conda activate $ENV_NAME &&
cd $SENTENCE_TRANSFORMERS && 
PYTHONPATH=. python james/retrieve/knn.py \
  --residing_folder data/results/mpnet \
  --collection_file collection_embeddings.npy \
  --query_file trels_embeddings.npy \
  --output_file data/results/mpnet/knn.npy \
  --output_file_scores data/results/mpnet/knn_D.npy \
  --nlist 10000 \
  --nprobe 200 \
  --normalize \
  --d 384
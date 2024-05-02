source ~/miniconda3/etc/profile.d/conda.sh && 
conda activate $ENV_NAME &&
cd $SENTENCE_TRANSFORMERS && 
PYTHONPATH=. python james/retrieve/knn.py \
  --residing_folder data/results/reduced \
  --collection_file collection_embeddings_48.npy \
  --query_file query_embeddings_48.npy \
  --output_file data/results/reduced/knn_48.npy \
  --output_file_scores data/results/reduced/knn_48_D.npy \
  --nlist 10000 \
  --nprobe 200 \
  --normalize \
  --d 48
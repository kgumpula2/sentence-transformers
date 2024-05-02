source ~/miniconda3/etc/profile.d/conda.sh && 
conda activate $ENV_NAME &&
cd $SENTENCE_TRANSFORMERS && 
PYTHONPATH=. python james/retrieve/knn.py \
  --residing_folder data/results/reduced \
  --collection_file collection_embeddings_96.npy \
  --query_file query_embeddings_96.npy \
  --output_file data/results/reduced/knn_96.npy \
  --output_file_scores data/results/reduced/knn_96_D.npy \
  --nlist 10000 \
  --nprobe 200 \
  --normalize \
  --d 96
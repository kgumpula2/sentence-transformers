source ~/miniconda3/etc/profile.d/conda.sh && 
conda activate $ENV_NAME &&
cd $SENTENCE_TRANSFORMERS && 
echo "=== 48 ===" &&
PYTHONPATH=. python james/retrieve/knn.py \
  --residing_folder data/results/matryoshka \
  --collection_file collection_embeddings.npy \
  --query_file trels_embeddings.npy \
  --output_file data/results/matryoshka/knn_48.npy \
  --output_file_scores data/results/matryoshka/knn_48_D.npy \
  --nlist 10000 \
  --nprobe 200 \
  --normalize \
  --d 384 \
  --truncation_d 48 &&
echo "=== 96 ===" &&
PYTHONPATH=. python james/retrieve/knn.py \
  --residing_folder data/results/matryoshka \
  --collection_file collection_embeddings.npy \
  --query_file trels_embeddings.npy \
  --output_file data/results/matryoshka/knn_96.npy \
  --output_file_scores data/results/matryoshka/knn_96_D.npy \
  --nlist 10000 \
  --nprobe 200 \
  --normalize \
  --d 384 \
  --truncation_d 96 &&
echo "=== 192 ===" &&
PYTHONPATH=. python james/retrieve/knn.py \
  --residing_folder data/results/matryoshka \
  --collection_file collection_embeddings.npy \
  --query_file trels_embeddings.npy \
  --output_file data/results/matryoshka/knn_192.npy \
  --output_file_scores data/results/matryoshka/knn_192_D.npy \
  --nlist 10000 \
  --nprobe 200 \
  --normalize \
  --d 384 \
  --truncation_d 192
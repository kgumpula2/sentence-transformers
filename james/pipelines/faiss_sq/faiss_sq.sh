source ~/miniconda3/etc/profile.d/conda.sh && 
conda activate $ENV_NAME &&
cd $SENTENCE_TRANSFORMERS && 
echo "=== SQ8 ===" &&
PYTHONPATH=. python james/retrieve/knn.py \
  --residing_folder data/results/baseline \
  --collection_file collection_embeddings.npy \
  --query_file trels_embeddings.npy \
  --output_file data/results/faiss_sq/knn_sq8.npy \
  --output_file_scores data/results/faiss_sq/knn_sq8_D.npy \
  --nlist 10000 \
  --nprobe 200 \
  --normalize \
  --d 384 \
  --normalize \
  --faiss_index_string IVF10000,SQ8 &&
echo "=== SQ6 ===" &&
PYTHONPATH=. python james/retrieve/knn.py \
  --residing_folder data/results/baseline \
  --collection_file collection_embeddings.npy \
  --query_file trels_embeddings.npy \
  --output_file data/results/faiss_sq/knn_sq6.npy \
  --output_file_scores data/results/faiss_sq/knn_sq6_D.npy \
  --nlist 10000 \
  --nprobe 200 \
  --normalize \
  --d 384 \
  --normalize \
  --faiss_index_string IVF10000,SQ6 &&
echo "=== SQ4 ===" &&
PYTHONPATH=. python james/retrieve/knn.py \
  --residing_folder data/results/baseline \
  --collection_file collection_embeddings.npy \
  --query_file trels_embeddings.npy \
  --output_file data/results/faiss_sq/knn_sq4.npy \
  --output_file_scores data/results/faiss_sq/knn_sq4_D.npy \
  --nlist 10000 \
  --nprobe 200 \
  --normalize \
  --d 384 \
  --normalize \
  --faiss_index_string IVF10000,SQ4 &&
echo "=== SQfp16 ===" &&
PYTHONPATH=. python james/retrieve/knn.py \
  --residing_folder data/results/baseline \
  --collection_file collection_embeddings.npy \
  --query_file trels_embeddings.npy \
  --output_file data/results/faiss_sq/knn_sqfp16.npy \
  --output_file_scores data/results/faiss_sq/knn_sqfp16_D.npy \
  --nlist 10000 \
  --nprobe 200 \
  --normalize \
  --d 384 \
  --normalize \
  --faiss_index_string IVF10000,SQfp16
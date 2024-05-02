source ~/miniconda3/etc/profile.d/conda.sh && 
conda activate $ENV_NAME &&
cd $SENTENCE_TRANSFORMERS && 
PYTHONPATH=. python james/embed/pca_matrix.py \
  --output_embeddings_file data/results/reduced/collection_embeddings_96.npy \
  --output_matrix_file data/results/reduced/pca_96.pkl \
  --embed_dim 96 \
  --input_file data/results/baseline/collection_embeddings.npy &&
PYTHONPATH=. python james/embed/pca_multiply_matrices.py \
  --input_embeddings_file data/results/baseline/query_embeddings.npy \
  --output_embeddings_file data/results/reduced/query_embeddings_96.npy \
  --pca_matrix_file data/results/reduced/pca_96.pkl
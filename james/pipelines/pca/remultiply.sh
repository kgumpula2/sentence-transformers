source ~/miniconda3/etc/profile.d/conda.sh && 
conda activate $ENV_NAME &&
cd $SENTENCE_TRANSFORMERS && 
PYTHONPATH=. python james/embed/pca_multiply_matrices.py \
  --input_embeddings_file data/results/baseline/trels_embeddings.npy \
  --output_embeddings_file data/results/reduced/query_embeddings_48.npy \
  --pca_matrix_file data/results/reduced/pca_48.pkl &&
PYTHONPATH=. python james/embed/pca_multiply_matrices.py \
  --input_embeddings_file data/results/baseline/trels_embeddings.npy \
  --output_embeddings_file data/results/reduced/query_embeddings_96.npy \
  --pca_matrix_file data/results/reduced/pca_96.pkl &&
PYTHONPATH=. python james/embed/pca_multiply_matrices.py \
  --input_embeddings_file data/results/baseline/trels_embeddings.npy \
  --output_embeddings_file data/results/reduced/query_embeddings_192.npy \
  --pca_matrix_file data/results/reduced/pca_192.pkl
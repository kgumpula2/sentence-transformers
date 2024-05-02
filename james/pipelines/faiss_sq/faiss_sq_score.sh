source ~/miniconda3/etc/profile.d/conda.sh && 
conda activate $ENV_NAME &&
cd $SENTENCE_TRANSFORMERS && 
echo "=== SQ8 ===" &&
PYTHONPATH=. python james/msmarco/score.py \
  --residing_folder data/results/faiss_sq \
  --knn_index_file knn_sq8.npy \
  --output_file inference_sq8.tsv && 
echo "=== SQ6 ===" &&
PYTHONPATH=. python james/msmarco/score.py \
  --residing_folder data/results/faiss_sq \
  --knn_index_file knn_sq6.npy \
  --output_file inference_sq6.tsv && 
echo "=== SQ4 ===" &&
PYTHONPATH=. python james/msmarco/score.py \
  --residing_folder data/results/faiss_sq \
  --knn_index_file knn_sq4.npy \
  --output_file inference_sq4.tsv
source ~/miniconda3/etc/profile.d/conda.sh && 
conda activate $ENV_NAME &&
cd $SENTENCE_TRANSFORMERS && 
echo "=== 48 === " &&
PYTHONPATH=. python james/msmarco/score.py \
  --residing_folder data/results/mpnet \
  --knn_index_file knn_48.npy \
  --output_file inference_48.tsv && 
echo "=== 96 === " &&
PYTHONPATH=. python james/msmarco/score.py \
  --residing_folder data/results/mpnet \
  --knn_index_file knn_96.npy \
  --output_file inference_96.tsv && 
echo "=== 192 === " &&
PYTHONPATH=. python james/msmarco/score.py \
  --residing_folder data/results/mpnet \
  --knn_index_file knn_192.npy \
  --output_file inference_192.tsv &&
echo "=== 384 === " &&
PYTHONPATH=. python james/msmarco/score.py \
  --residing_folder data/results/mpnet \
  --knn_index_file knn.npy \
  --output_file inference.tsv


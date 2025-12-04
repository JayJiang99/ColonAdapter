CUDA_VISIBLE_DEVICES=0 python evaluate_depth_colonaf.py \
  --data_path ./dataset/SyntheticColon_I \
  --load_weights_folder ./models/weights_39 \
  --eval_split synthetic_colon \
  --eval_mono


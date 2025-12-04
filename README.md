## [ColonAdapter: Geometry Estimation Through Foundation Model Adaptation for Colonoscopy](https://arxiv.org/abs/2511.22250)


This repository provides the **official PyTorch implementation** of the paper  
[*ColonAdapter: Geometry Estimation Through Foundation Model Adaptation for Colonoscopy*](https://arxiv.org/abs/2511.22250).


## TODO / Roadmap

- [x] **Evaluation code**: update and test depth evaluation (`eval.sh`, `evaluate_depth_colonaf.py`).
- [x] **Inference code**: update and test folder-based inference (`infer.sh`, `infer_depth_folder.py`).
- [ ] **Training code**: clean up and release training pipeline (`train.sh`, `trainer_end_to_end_3r.py`, related options).
- [ ] **Pretrained weights**: upload ColonAdapter model checkpoints and document how to download/use them.


The main entry points are:
- **Training**: `train.sh` (stage-end-to-end training)
- **Evaluation**: `eval.sh` (quantitative depth evaluation with GT)
- **Inference**: `infer.sh` (qualitative depth inference on arbitrary images)


## 1. Environment and Dependencies

You have two options to set up the environment.

- **Option A – Use this repository’s Python requirements**  
  - Create a fresh virtualenv or conda environment (Python ≥ 3.8 recommended).
  - Install dependencies:

```bash
pip install -r requirements.txt
```

- **Option B – Reuse a DUSt3R / MonST3R environment**  
  - If you already have a working `dust3r` or `monst3r` environment (from the official repos), you can use it directly:
    - Ensure it has compatible `torch` / `torchvision` and CUDA versions.
    - From this repo root, install any missing extras:

```bash
pip install -r requirements.txt
```

In both cases, a CUDA-capable GPU is strongly recommended for training and evaluation.


## 2. Data Layout

The scripts expect the datasets to be organized similarly to the original Monodepth2 / AF-SfMLearner structure (e.g. EndoVis, C3VD, SyntheticColon).  
The exact `--data_path` you pass in `train.sh` / `eval.sh` should point to the preprocessed dataset root (e.g. C3VD reorganized and undistorted, or SyntheticColon).

Ground-truth depth maps for evaluation should already be exported into the `splits/` structure (e.g. `splits/c3vd_undist_crop_brown/gt_depths.npz`), as used by `evaluate_depth_colonaf.py`.


## 3. Training - TODO:

Please check `options.py` to update model weight and other params.
The recommended way to launch end-to-end training is via `train.sh`:

```bash
bash train.sh
```

Current `train.sh` content:

```bash
CUDA_VISIBLE_DEVICES=3 python train_end_to_end.py \
  --data_path DATA_DIR \
  --log_dir LOG_DIR
```

- **`--data_path`**: root directory of your training dataset (e.g. C3VD or SyntheticColon).
- **`--log_dir`**: where TensorBoard logs, checkpoints, and models are written.

You can edit `train.sh` to:
- Change `CUDA_VISIBLE_DEVICES` to your preferred GPU id(s).
- Swap `--data_path` and `--log_dir` for your own datasets and experiment folders.
- Add extra flags defined in `options.py` (e.g. `--batch_size`, `--num_epochs`, etc.).


## 4. Evaluation (with Ground-Truth Depth)

For quantitative depth evaluation against ground-truth depth maps, use `eval.sh`, which calls `evaluate_depth_colonaf.py`:

```bash
bash eval.sh
```

Current `eval.sh` content:

```bash
CUDA_VISIBLE_DEVICES=3 python evaluate_depth_colonaf.py \
  --data_path DATA_DIR \
  --load_weights_folder WEIGHT_DIR \
  --eval_mono
```

- **`--data_path`**: dataset root with the same structure used during training.
- **`--load_weights_folder`**: path to a checkpoint folder containing `depth_model.pth`.
- **`--eval_mono` / `--eval_stereo`**: select mono or stereo evaluation mode (exactly one must be set).

`evaluate_depth_colonaf.py`:
- Loads your DUSt3R-based depth model from `depth_model.pth`.
- Uses the `splits/.../test_files.txt` and `gt_depths.npz` to run evaluation.
- Prints standard metrics: **Abs Rel, Sq Rel, RMSE, RMSE log, δ<1.25, δ<1.25², δ<1.25³**.

You can adjust the evaluation split and other options using flags in `options.py` (e.g. `--eval_split`, `--min_depth`, `--max_depth`, LoRA settings).


## 5. Inference on a Folder of Images

To run depth inference on arbitrary images (no GT required), use `infer.sh`, which calls `infer_depth_folder.py`:

```bash
bash infer.sh
```

Current `infer.sh` content:

```bash
python infer_depth_folder.py \
  --image_dir IMAGE_FOLDER_DIR \
  --save_dir SAVE_DIR \
  --load_weights_folder WEIGHT_DIR \
  --height 224 \
  --width 224 \
  --eval_mono
```

Key arguments:
- **`--image_dir`**: directory containing input images (`.png`, `.jpg`, etc.).  
  The script sorts the images and forms **consecutive pairs** `(img[i], img[i+1])`.
- **`--save_dir`**: directory where predictions are written.
- **`--load_weights_folder`**: DUSt3R-based checkpoint folder with `depth_model.pth`.
- **`--height`, `--width`**: input resolution; must match what the model was trained with.

For each first image in a pair, `infer_depth_folder.py`:
- Runs the DUSt3R-based model using the same loading configuration as `evaluate_depth_colonaf.py`.
- Extracts the predicted 3D points and uses the z-coordinate as depth.
- Converts depth into a disparity-like map with `disp_to_depth`.
- Saves:
  - `<name>_depth.npy`: raw depth map.
  - `<name>_disp.npy`: disparity map.
  - `<name>_depth.png`: depth visualization (colored with `COLORMAP_INFERNO`).

You can change `--image_dir`, `--save_dir`, and `--load_weights_folder` in `infer.sh` to run on your own images and model weights.


## 6. Configuration Options

Most hyperparameters and paths are defined in `options.py` via `MonodepthOptions`, including:
- **Training**: `--batch_size`, `--learning_rate`, `--num_epochs`, `--scales`, etc.
- **Depth range**: `--min_depth`, `--max_depth`.
- **LoRA / DUSt3R model**: `--lora_rank`, `--lora_alpha`, `--lora_dropout`, `--pretrained_path`.
- **Evaluation**: `--eval_split`, `--eval_mono`, `--eval_stereo`, `--pred_depth_scale_factor`, `--post_process`.

All three main scripts (`train_end_to_end.py`, `evaluate_depth_colonaf.py`, `infer_depth_folder.py`) use this options system, so any CLI changes you make there will be shared across training, evaluation, and inference.


## 7. Troubleshooting

- **CUDA / GPU visibility**:  
  - If you see `RuntimeError: CUDA error` or the model runs on CPU only, check `CUDA_VISIBLE_DEVICES` and your installed CUDA/PyTorch versions.
- **Missing `depth_model.pth`**:  
  - Verify that `--load_weights_folder` contains a valid `depth_model.pth` file (produced by training or downloaded).
- **Dataset path errors**:  
  - Ensure `--data_path` matches the directory structure expected by the dataset loaders in `datasets/`.


## 8. Acknowledgements

This repository builds upon and is inspired by the following excellent open-source projects:

- [AF-SfMLearner](https://github.com/ShuweiShao/AF-SfMLearner/tree/main)
- [DUSt3R](https://github.com/naver/dust3r/tree/main)
- [MonST3R](https://github.com/Junyi42/monst3r)


## [ColonAdapter: Geometry Estimation Through Foundation Model Adaptation for Colonoscopy](https://arxiv.org/abs/2511.22250)


This repository provides the **official PyTorch implementation** of the paper  
[*ColonAdapter: Geometry Estimation Through Foundation Model Adaptation for Colonoscopy*](https://arxiv.org/abs/2511.22250).


## TODO / Roadmap

- [x] **Evaluation code**: update and test depth evaluation (`eval.sh`, `evaluate_depth_colonaf.py`).
- [x] **Inference code**: update and test folder-based inference (`infer.sh`, `infer_depth_folder.py`).
- [ ] **Training code**: clean up and release training pipeline (`train.sh`, `trainer_end_to_end_3r.py`, related options).
- [x] **Pretrained weights**: upload ColonAdapter model checkpoints and document how to download/use them.


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

Ground-truth depth maps for evaluation should already be exported into the `splits/` structure (e.g. `splits/synthetic_colon/gt_depths.npz`), as used by `evaluate_depth_colonaf.py`.


## 3. Training

### Quick Start

```bash
bash train.sh
```

### Dataset Preparation

The dataset structure follows the AF-SfMLearner convention:

```
DATA_ROOT/
  scene_1/
    keyframe_1/
      image_02/
        data/
          0000000001.png
          0000000002.png
          ...
    keyframe_2/
      ...
  scene_2/
    ...
```

You need to create the `splits/` directory with train/val/test split files:

```
splits/
  your_dataset/
    train_files.txt   # one scene path per line, e.g., "scene_1"
    val_files.txt
    test_files.txt
    gt_depths.npz     # optional, for evaluation
    gt_poses.npz      # optional, for pose evaluation
```

**Generating splits from video**: If you have video files (e.g., RGB.mp4), convert them to image sequences using ffmpeg, then extract keyframes and organize them into the directory structure above. Refer to the [AF-SfMLearner dataset preprocessing guide](https://github.com/ShuweiShao/AF-SfMLearner/tree/main/dataset) for detailed instructions on converting Endovis, SCARED, C3VD, or other endoscopic datasets.

**Ground-truth depth for evaluation**: Export depth maps into `gt_depths.npz` (shape: `(N, H, W)`) and poses into `gt_poses.npz` (shape: `(N, 4, 4)`). See `evaluate_depth_colonaf.py` for the expected format.

### Training Command

The `train.sh` script runs end-to-end training:

```bash
CUDA_VISIBLE_DEVICES=0 python train_end_to_end.py \
  --data_path /path/to/DATA_ROOT \
  --log_dir /path/to/LOG_DIR \
  --num_epochs 40 \
  --learning_rate 1e-4 \
  --scheduler_step_size 20 \
  --lora_rank 16 \
  --lora_alpha 1.0 \
  --lora_dropout 0.1 \
  --pretrained_path /path/to/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
```

Key arguments:
- **`--data_path`**: root directory of your training dataset.
- **`--log_dir`**: where TensorBoard logs, checkpoints, and models are written.
- **`--pretrained_path`**: path to [DUSt3R pretrained weights](https://download.europe.naverlabs.com/ComputerVision/DUSt3R/).
- **`--num_epochs`**: number of training epochs (default: 40).
- **`--learning_rate`**: base learning rate (default: 1e-4).
- **`--lora_rank`**, **`--lora_alpha`**, **`--lora_dropout`**: LoRA fine-tuning parameters. **Note**: `lora_alpha=1.0` is critical — other values have been shown to cause training failure.

You can follow stage-one training of [AF-SfMLearner](https://github.com/ShuweiShao/AF-SfMLearner/tree/main) repository for training the affiliation module. After training is complete, specify the directory path containing the trained model weights using the load_weights_folder argument.


## 4. Evaluation (with Ground-Truth Depth)

Download [model weight]( https://1drv.ms/f/c/bee3ae4296a6dabd/IgDH-5hbzl7SRICurcGewxX7AYd5gRVzo35ElUlpz_2ez8Y ) to WEIGHT_DIR. 

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


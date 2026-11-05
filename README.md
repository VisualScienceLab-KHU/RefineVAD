# RefineVAD: Semantic-Guided Feature Recalibration for Weakly Supervised Video Anomaly Detection [AAAI 2026]

**Official implementation of the AAAI 2026 paper, _RefineVAD: Semantic-Guided Feature Recalibration for Weakly Supervised Video Anomaly Detection_.**

**Junhee Lee\***, **ChaeBeen Bang\***, **MyoungChul Kim\***, **MyeongAh Cho†**

*\* Equal contribution. † Corresponding author.*

Kyung Hee University

[![arXiv](https://img.shields.io/badge/arXiv-2511.13204-b31b1b.svg?style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2511.13204)
[![Project Page](https://img.shields.io/badge/Project-Page-2563eb.svg?style=flat-square&logo=googlechrome&logoColor=white)](https://visualsciencelab-khu.github.io/RefineVAD_project/)
[![Dataset](https://img.shields.io/badge/Dataset-Hugging%20Face-FFD21E.svg?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/datasets/JunheeLee/RefineVAD_Dataset)

## Overview

RefineVAD addresses **weakly supervised video anomaly detection (WSVAD)** using video-level annotations. It combines **Motion-aware Temporal Attention and Recalibration (MoTAR)** to model temporal changes with **Category-Oriented Refinement (CORE)** to incorporate anomaly-category semantics into segment-level feature refinement.

![Overview of the RefineVAD framework](assets/main_fig.png)

*Figure: Overview of RefineVAD, including the MoTAR and CORE modules.*

This repository contains the unified implementation for **UCF-Crime** and **XD-Violence**. Select the appropriate JSON configuration to train or evaluate either dataset using the same entry points. The dataset-specific training and inference behavior of the reference code is preserved.

## 1. Environment

The reference results were **successfully reproduced by the authors on an NVIDIA RTX A5000** with the matching pre-extracted features, ground truth, and dataset-specific settings. The implementation below retains this reproduction configuration. Results may vary with different data preprocessing, device, or software environments.

Create a Conda environment with Python 3.10:

```bash
conda create -n refinevad python=3.10 -y
conda activate refinevad
python -m pip install --upgrade pip
```

Install the PyTorch 2.4.1 CUDA 12.1 build and the remaining pinned packages:

```bash
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip check
```

CUDA 12.1 requires a compatible NVIDIA driver. To check your environment:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

The pinned dependencies are listed in [`requirements.txt`](requirements.txt), including `natsort==8.4.0`. Use an appropriately compatible PyTorch build if your system cannot run CUDA 12.1.

## 2. Data preparation

Download **`UCF_Crime.tar.gz`** and **`XD_Violence.tar.gz`** from the [RefineVAD dataset release](https://huggingface.co/datasets/JunheeLee/RefineVAD_Dataset/tree/main). Extract both archives to a directory **outside the Git repository**:

```bash
DOWNLOAD_DIR="/path/to/downloads"
DATA_ROOT="/path/to/VAD_data"
mkdir -p "$DATA_ROOT"
tar -xzf "$DOWNLOAD_DIR/UCF_Crime.tar.gz" -C "$DATA_ROOT"
tar -xzf "$DOWNLOAD_DIR/XD_Violence.tar.gz" -C "$DATA_ROOT"
```

The data layout expected by the released configurations is:

```text
VAD_data/
├── UCF_Crime/
│   ├── visual/ViT14L/<category>/*.npy
│   ├── textual/ViT14L/<category>/*.npy
│   └── list/
│       ├── train_list.txt
│       ├── test_list.txt
│       └── test_gt.npy
└── XD_Violence/
    ├── visual/ViT14L/*.npy
    ├── textual/ViT14L/*.npy
    └── list/
        ├── train_list.csv
        ├── test_list.csv
        └── TestGT_padding_xd.npy
```

Open the JSON file for each dataset and set **`data.base_root`** to that dataset's own folder:

| Dataset | Config file | Example `data.base_root` |
| --- | --- | --- |
| UCF-Crime | [`configs/ucf_crime.json`](configs/ucf_crime.json) | `/path/to/VAD_data/UCF_Crime` |
| XD-Violence | [`configs/xd_violence.json`](configs/xd_violence.json) | `/path/to/VAD_data/XD_Violence` |

The distributed configurations intentionally leave `data.base_root` empty. All relative paths under the dataset root retain their original values; **do not change them unless your dataset structure differs**. Before publishing modifications, clear any machine-specific absolute paths from the tracked JSON files.

**Important for XD-Violence:** Use the `TestGT_padding_xd.npy` file corresponding to the released `ViT14L` features and the test CSV order. The evaluator does not add, drop, or trim segments to force GT alignment. XD training/test CSVs require `path` and `label` columns; category labels such as `B2-0-0` are supported.

## 3. Training

Run commands from the repository root. **Saving checkpoints and creating visualizations are disabled by default.** All hyperparameters and the dataset selection are specified in the JSON configuration.

Train on UCF-Crime:

```bash
bash train.sh --config configs/ucf_crime.json --device cuda
```

Train on XD-Violence:

```bash
bash train.sh --config configs/xd_violence.json --device cuda
```

To **save the best checkpoint**, add `--save-model` and a result folder:

```bash
bash train.sh --config configs/ucf_crime.json --device cuda \
  --save-model --result-folder ucf_run

bash train.sh --config configs/xd_violence.json --device cuda \
  --save-model --result-folder xd_run
```

To also export the best model's per-video prediction plots, add `--visualize`:

```bash
bash train.sh --config configs/xd_violence.json --device cuda \
  --save-model --visualize --result-folder xd_run
```

`--visualize` can also be enabled without `--save-model`. Checkpoints are written only when `--save-model` is provided. Result directories are created under `results/` (UCF-Crime) or `results_xd/` (XD-Violence), and contain `best_model.pth`, `best_metrics.json`, `history.json`, and `resolved_config.json` when saving is enabled. Visualization images are written under `visualizations/` when requested. Without either option, only console metrics are produced.

| Argument | Default | Description |
| --- | --- | --- |
| `--config` | `configs/ucf_crime.json` | Dataset, model, and training settings |
| `--device` | `auto` | `cuda`, `cuda:0`, `cpu`, or `auto` |
| `--save-model` | Off | Save the best checkpoint |
| `--visualize` | Off | Save per-video prediction plots after training |
| `--result-folder` | `default` | Folder name under the dataset's result directory |
| `--output-dir` | Unset | Override the output directory |

## 4. Evaluation only

`test.py` and `test.sh` **only load a checkpoint and evaluate the test split**. They do not create training datasets or update model weights.

Evaluate UCF-Crime:

```bash
bash test.sh --config configs/ucf_crime.json --device cuda \
  --checkpoint results/ucf_run/best_model.pth
```

Evaluate XD-Violence:

```bash
bash test.sh --config configs/xd_violence.json --device cuda \
  --checkpoint results_xd/xd_run/best_model.pth
```

To export evaluation metrics and plots:

```bash
bash test.sh --config configs/xd_violence.json --device cuda \
  --checkpoint results_xd/xd_run/best_model.pth \
  --visualize --output-dir results_xd/xd_run/evaluation
```

Without `--visualize` or `--output-dir`, evaluation prints the metrics without creating files. `--output-dir` alone exports `test_metrics.json`. Model state-dictionary keys remain compatible with the reference `CAM_GATSM_NL` checkpoint format when the model configuration and input dimensions match.

## 5. Reproduction settings

The default configurations in [`configs/ucf_crime.json`](configs/ucf_crime.json) and [`configs/xd_violence.json`](configs/xd_violence.json) contain the hyperparameter settings used to obtain the results reported in the AAAI 2026 paper. For reproduction, use the corresponding configuration without changing its hyperparameters, and set only `data.base_root` to your dataset location.

## 6. Repository structure

```text
RefineVAD/
├── assets/
│   └── main_fig.png
├── configs/
│   ├── ucf_crime.json
│   └── xd_violence.json
├── model/
│   ├── __init__.py
│   ├── refinevad.py
│   └── temporal.py
├── data.py
├── evaluate.py
├── losses.py
├── train.py
├── test.py
├── utils.py
├── train.sh
├── test.sh
├── requirements.txt
├── README.md
└── .gitignore
```

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{lee2026refinevad,
  title={RefineVAD: Semantic-Guided Feature Recalibration for Weakly Supervised Video Anomaly Detection},
  author={Lee, Junhee and Bang, ChaeBeen and Kim, MyoungChul and Cho, MyeongAh},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={7},
  pages={5899--5907},
  year={2026},
  doi={10.1609/aaai.v40i7.37512}
}
```

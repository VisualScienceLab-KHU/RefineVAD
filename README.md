# RefineVAD

**RefineVAD: Semantic-Guided Feature Recalibration for Weakly Supervised Video Anomaly Detection**

A unified implementation for UCF-Crime and XD-Violence using pre-extracted visual and textual features. Select a JSON configuration to switch datasets; training and evaluation use shared entry points with dataset-specific behavior.

## Environment

The authors confirmed exact reproduction of their reference runs on an **NVIDIA RTX A5000** with the matching features, ground truth, and dataset-specific settings. This release preserves those reference training and evaluation behaviors. Reproduction on other hardware or software versions is not a bitwise-equality guarantee.

Use Python 3.10 and the pinned dependencies:

```bash
conda create -n refinevad python=3.10 -y
conda activate refinevad
python -m pip install --upgrade pip
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip check
```

The CUDA 12.1 wheel requires a compatible NVIDIA driver. An already installed `refinevad` environment can be reused by installing `requirements.txt`. The dependency versions are PyTorch 2.4.1, NumPy 1.26.4, SciPy 1.13.1, scikit-learn 1.5.2, pandas 2.2.3, Matplotlib 3.9.4, and natsort 8.4.0.

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

CPU-only installation for smoke tests:

```bash
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

## Data preparation

Download `UCF_Crime.tar.gz` and `XD_Violence.tar.gz` from the [RefineVAD dataset repository](https://huggingface.co/datasets/JunheeLee/RefineVAD_Dataset/tree/main). Extract the archives outside this source repository. For example, after downloading them into a directory of your choice:

```bash
DOWNLOAD_DIR="/path/to/downloads"
DATA_ROOT="/path/to/VAD_data"
mkdir -p "$DATA_ROOT"
tar -xzf "$DOWNLOAD_DIR/UCF_Crime.tar.gz" -C "$DATA_ROOT"
tar -xzf "$DOWNLOAD_DIR/XD_Violence.tar.gz" -C "$DATA_ROOT"
```

The configurations retain the relative paths used by the reference code:

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

Set **only `data.base_root`** in each JSON file to the corresponding dataset directory, not its parent:

| Configuration | `data.base_root` example |
| --- | --- |
| `configs/ucf_crime.json` | `/path/to/VAD_data/UCF_Crime` |
| `configs/xd_violence.json` | `/path/to/VAD_data/XD_Violence` |

`data.base_root` is intentionally empty in the distributed files. Absolute paths are recommended. A relative root is resolved from the repository root. All remaining data paths are resolved beneath that root; no environment variable is required.

Both datasets use `ViT14L` for visual **and** textual features, as selected by the reference shell scripts. Training features must be nonempty two-dimensional `[segments, channels]` NumPy arrays. XD CSV files must contain `path` and `label` columns. Paths inside those CSV files are reduced to their basenames. `A` denotes a normal video; anomaly codes are `B1`, `B2`, `B4`, `B5`, `B6`, and `G`. Labels such as `B2-0-0` and `B1-B2-0` are supported.

The XD ground-truth filename is intentionally **`list/TestGT_padding_xd.npy`**, not `list/test_gt.npy`. Use the ground truth belonging to these exact feature sequences and test-video order. The code does not rename files, add labels, or remove first/last segments to make lengths match. Do not substitute a different GT merely because its filename is similar.

Validate the configured data before training:

```bash
python scripts/check_data.py --config configs/ucf_crime.json
python scripts/check_data.py --config configs/xd_violence.json
```

For an evaluation-only dataset installation:

```bash
python scripts/check_data.py --config configs/xd_violence.json --test-only
```

## Training

Run from the repository root. Both datasets use the same script:

```bash
bash train.sh --config configs/ucf_crime.json --device cuda
bash train.sh --config configs/xd_violence.json --device cuda
```

The default configuration is UCF-Crime. **Checkpoint saving and visualization are both disabled by default.** Per-epoch losses and validation metrics are printed directly to the terminal. Hyperparameters live in the JSON files rather than shell variables.

Save the best checkpoint:

```bash
bash train.sh --config configs/ucf_crime.json --device cuda \
    --save-model --result-folder ucf_reference

bash train.sh --config configs/xd_violence.json --device cuda \
    --save-model --result-folder xd_reference
```

Save the checkpoint and plot the best model's frame-level predictions:

```bash
bash train.sh --config configs/xd_violence.json --device cuda \
    --save-model --visualize --result-folder xd_reference
```

`--visualize` can also be used without `--save-model`: the best weights are kept in memory for plotting, but no checkpoint is written. Visualization runs only after training, not after every epoch.

| Runtime argument | Default | Purpose |
| --- | --- | --- |
| `--config` | `configs/ucf_crime.json` | Dataset and hyperparameter configuration |
| `--device` | `auto` | `cuda`, `cuda:0`, `cpu`, or automatic selection |
| `--save-model` | Off | Save the best model state dictionary |
| `--visualize` | Off | Plot the best model after training |
| `--result-folder` | `default` | Name beneath `results/` or `results_xd/` |
| `--output-dir` | Unset | Explicit output directory, overriding the result folder |

Underscore aliases `--save_model` and `--result_folder`, and `--plot` for `--visualize`, are also accepted. There is no training `--test_only` option. `PYTHON=/path/to/python bash train.sh ...` selects an explicit interpreter.

When saving is enabled, outputs include:

```text
results/ucf_reference/          or results_xd/xd_reference/
├── best_model.pth
├── best_metrics.json
├── history.json
├── resolved_config.json
└── visualizations/            only with --visualize
```

With neither saving nor visualization enabled, training writes no result files. Use a new result folder for each experiment to avoid overwriting a previous run. Validation uses the supplied benchmark test split after every epoch, matching the reference model-selection protocol.

## Evaluation only

`test.py` is separate from training and never constructs a training dataset or optimizer. Supply a checkpoint explicitly:

```bash
bash test.sh --config configs/ucf_crime.json --device cuda \
    --checkpoint results/ucf_reference/best_model.pth

bash test.sh --config configs/xd_violence.json --device cuda \
    --checkpoint results_xd/xd_reference/best_model.pth
```

To save metrics and visualizations:

```bash
bash test.sh --config configs/xd_violence.json --device cuda \
    --checkpoint results_xd/xd_reference/best_model.pth \
    --visualize --output-dir results_xd/xd_reference/evaluation
```

Without `--visualize` or `--output-dir`, evaluation only prints metrics. `--output-dir` alone writes `test_metrics.json`. Visualization normalizes scores only for display; AUC/AP use the original, unnormalized frame scores. The model state-dictionary keys are unchanged from the reference `CAM_GATSM_NL`, so its plain state-dictionary checkpoints can be loaded with a matching configuration and feature dimension.

## Reference configurations

| Setting | UCF-Crime | XD-Violence |
| --- | ---: | ---: |
| Batch size | 64 | 64 |
| Learning rate | 0.00069 | 0.0032 |
| Maximum epochs | 30 | 30 |
| Random seed | 42 | 42 |
| Weight decay | 0.0001 | 0.0001 |
| Sampled segments | 32 | 32 |
| Projection dimension | 467 | 461 |
| Category embedding dimension | 80 | 176 |
| Top-k | 7 | 7 |
| Fold division | 4 | 3 |
| Smoothness coefficient | 0.0005 | 0.0005 |
| Sparsity coefficient | 0.009 | 0.005 |
| Category loss coefficient | 0.22 | 0.21 |
| GMM coefficient | 0.35 | 0.1 |
| Consistency coefficient | 0.0021 | 0.0001 |
| Category positive weight | 10 | 5 |
| Checkpoint selection | AUC | AP |
| Early-stopping patience | 5 | 5 |

The XD GMM and consistency coefficients are retained for the reported auxiliary loss; that loss is **not added to XD's optimization objective**, matching the reference trainer.

### Dataset-specific behavior

| Component | UCF-Crime | XD-Violence |
| --- | --- | --- |
| Training batches | Independently shuffled normal/anomaly loaders, combined with `zip` | Filename-sorted normal/anomaly pairs, followed by one shuffled loader |
| Category supervision | Single-label anomaly cross-entropy | Multi-label anomaly BCE |
| Total loss | MIL + weighted category + GMM | MIL + weighted category |
| LR schedule | OneCycleLR | Linear warmup from `1e-6`, then CosineAnnealingWarmRestarts |
| Scheduler step budget | Anomaly-loader length × maximum epochs | Paired-loader length × maximum epochs |
| Test ordering | `natsorted` UCF names | Original CSV row order |
| Reported metrics | AUC, AP, anomaly-only AUC | AUC, AP |

Both paths resample training videos using uniform temporal interval means. Both evaluate all test segments using raw model scores, repeat each segment score 16 times, and apply Gaussian smoothing with sigma 2 separately to each video. There is no sigmoid before evaluation metrics and no segment trimming.

### Implementation behavior retained for reproduction

The reference trainers call `model.train()` once before the epoch loop. Validation switches the model to `eval()` and does not restore training mode afterward. Consequently, subsequent optimizer updates run with evaluation-mode Dropout/BatchNorm and without the training-only category-GT embedding addition. This behavior is deliberately retained, not silently corrected.

The temporal Transformer is created during the first training forward pass, **after** optimizer construction and the initial Linear-layer initialization pass. It is therefore absent from that optimizer's parameter groups, although it is present in saved model state dictionaries. The hard integer shift selection also does not propagate gradients to its shift predictor. These reference implementation properties remain unchanged. Evaluation constructs the temporal module before loading a checkpoint.

The initial DataLoader probe used to infer the input dimension is preserved because it advances the random-number state before model initialization. For UCF, the normal and anomaly loader lengths may differ; the epoch stops at the shorter loader while the original scheduler budget uses the anomaly loader. Do not alter these behaviors when comparing against the reference runs.

## Validation utilities

Run the synthetic integration test without downloading a dataset:

```bash
python tests/smoke_test.py
```

This tests both datasets, training, checkpoint reload, visualization, and evaluation after the training list has been removed. It uses reduced synthetic dimensions and is not a benchmark-performance test.

To compare against an unpacked copy of the original reference sources:

```bash
python tests/compare_original.py --original-dir /path/to/original/source
```

This optional test runs the original and refactored trainers on the same small CPU fixtures. It compares the full model-state SHA-256 after every optimizer step, optimizer parameter membership, learning rates, epoch metrics, and the best checkpoint. Worker count and pin-memory settings are controlled for the CPU test; production JSON files retain six workers and pinned memory. The original source files are not required for normal training or evaluation.

## Repository layout

```text
RefineVAD/
├── configs/
│   ├── ucf_crime.json
│   └── xd_violence.json
├── model/
│   ├── __init__.py
│   ├── refinevad.py
│   └── temporal.py
├── scripts/check_data.py
├── tests/
│   ├── smoke_test.py
│   └── compare_original.py
├── data.py
├── evaluate.py
├── losses.py
├── train.py
├── test.py
├── utils.py
├── train.sh
├── test.sh
├── requirements.txt
└── README.md
```

Only the reference `_NL` architecture is retained and is exported as `RefineVAD`. Dataset-specific Python config files, alternative model variants, cluster-specific job directives, and ad-hoc single-video CSV exports are not part of this release.

import copy
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn


def load_config(path):
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as stream:
        cfg = json.load(stream)
    if cfg.get("dataset") not in {"ucf_crime", "xd_violence"}:
        raise ValueError("dataset must be ucf_crime or xd_violence.")
    for section in ("data", "model", "training", "loss", "scheduler", "evaluation"):
        if not isinstance(cfg.get(section), dict):
            raise ValueError(f"Missing configuration section: {section}")
    if cfg["model"].get("name") != "RefineVAD":
        raise ValueError("Only the RefineVAD model is supported.")
    base = cfg["data"].get("base_root", "").strip()
    if not base:
        raise ValueError(f"Set data.base_root in {path} to the dataset directory.")
    base = Path(base).expanduser()
    if not base.is_absolute():
        base = Path(__file__).resolve().parent / base
    base = base.resolve()
    if not base.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {base}")
    expected_scheduler = {
        "ucf_crime": "one_cycle", "xd_violence": "linear_warmup_cosine_restarts"
    }[cfg["dataset"]]
    if cfg["scheduler"]["name"] != expected_scheduler:
        raise ValueError(f"{cfg['dataset']} requires scheduler {expected_scheduler}.")
    expected_metric = "auc" if cfg["dataset"] == "ucf_crime" else "ap"
    if cfg["training"]["primary_metric"] != expected_metric:
        raise ValueError(f"{cfg['dataset']} selects checkpoints by {expected_metric}.")
    for section, keys in {
        "data": ("num_segments_per_video", "num_categories"),
        "model": ("proj_embedding", "d_emb", "topk", "fold_div"),
        "training": ("batch_size", "max_epoch", "patience"),
        "evaluation": ("frames_per_segment",),
    }.items():
        for key in keys:
            value = cfg[section][key]
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{section}.{key} must be a positive integer.")
    if cfg["model"]["topk"] > cfg["data"]["num_segments_per_video"]:
        raise ValueError("model.topk cannot exceed data.num_segments_per_video.")
    if cfg["model"]["d_emb"] % 4:
        raise ValueError("model.d_emb must be divisible by four attention heads.")
    if cfg["training"]["batch_size"] < 2:
        raise ValueError("training.batch_size must be at least 2 for category BatchNorm.")
    if cfg["training"]["num_workers"] < 0 or cfg["training"]["lr"] <= 0:
        raise ValueError("num_workers must be nonnegative and lr must be positive.")
    values = {}
    for section in ("data", "model", "training", "loss"):
        values.update(cfg[section])
    values["dataset"] = cfg["dataset"]
    values["base_path"] = str(base)
    for key in ("visual_root", "text_root", "train_list", "test_list", "gt_path"):
        relative = Path(cfg["data"][key])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"data.{key} must be relative to data.base_root.")
        values[key] = str(base / relative)
    values["test_list_path"] = values["test_list"]
    values["gt_feature_path"] = values["gt_path"]
    cfg = copy.deepcopy(cfg)
    cfg["data"]["base_root"] = str(base)
    return cfg, SimpleNamespace(**values)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Allocate a GPU or use --device cpu.")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Supported devices are cpu, cuda, cuda:N and auto.")
    return device


def initialize_weights(model):
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            if layer.weight is not None and layer.weight.dim() >= 2:
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)


def load_weights(model, path, device):
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("Expected a model state_dict checkpoint.")
    model.load_state_dict(state, strict=True)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")

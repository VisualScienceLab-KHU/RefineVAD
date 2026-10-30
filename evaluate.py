import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from natsort import natsorted
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass(frozen=True)
class TestEntry:
    name: str
    visual_path: str
    text_path: str
    category: str


@dataclass
class Evaluation:
    metrics: dict
    entries: list
    frame_scores: list
    ground_truth: np.ndarray


def load_test_entries(args):
    visual_root = os.path.join(args.visual_root, args.visual_backbone)
    text_root = os.path.join(args.text_root, args.text_backbone)
    entries = []
    if args.dataset == "ucf_crime":
        videos = np.atleast_1d(np.loadtxt(args.test_list_path, dtype=str))
        if videos.ndim != 1:
            raise ValueError("The UCF test list must contain one video name per line.")
        for video in natsorted(videos):
            match = re.split(r"(\d)", video, maxsplit=1)
            category = match[0] if match else video
            if category == "Normal_Videos_":
                category = "Testing_Normal_Videos_Anomaly"
            filename = f"{video.replace('x264', '').strip()}16frames.npy"
            entries.append(TestEntry(str(video), os.path.join(visual_root, category, filename),
                                     os.path.join(text_root, category, filename), category))
    else:
        frame = pd.read_csv(args.test_list_path)
        if not {"path", "label"}.issubset(frame.columns):
            raise ValueError("The XD test CSV requires path and label columns.")
        for path, label in zip(frame["path"], frame["label"]):
            filename = os.path.basename(path)
            entries.append(TestEntry(filename, os.path.join(visual_root, filename),
                                     os.path.join(text_root, filename), str(label)))
    if not entries:
        raise ValueError("The test list is empty.")
    return entries


def infer_input_dimension(args):
    entry = load_test_entries(args)[0]
    visual = np.load(entry.visual_path, mmap_mode="r", allow_pickle=False)
    textual = np.load(entry.text_path, mmap_mode="r", allow_pickle=False)
    return int(visual.shape[-1] + textual.shape[-1])


def predict_frames(model, entries, device, frames_per_segment, sigma):
    predictions = []
    for entry in entries:
        visual_feature = np.load(entry.visual_path, allow_pickle=False)
        text_feature = np.load(entry.text_path, allow_pickle=False)
        visual_tensor = torch.tensor(visual_feature, dtype=torch.float32).to(device)
        text_tensor = torch.tensor(text_feature, dtype=torch.float32).to(device)
        if visual_tensor.dim() == 2:
            visual_tensor = visual_tensor.unsqueeze(0)
        if text_tensor.dim() == 2:
            text_tensor = text_tensor.unsqueeze(0)
        fused = torch.cat((visual_tensor, text_tensor), dim=2)
        with torch.no_grad():
            scores, _, _, _ = model(fused)
            scores = scores.cpu().numpy()
        frames = np.repeat(scores, frames_per_segment, axis=1).reshape(-1)
        frames = gaussian_filter1d(frames, sigma=sigma)
        predictions.append(frames)
    return predictions


def evaluate(model, cfg, args, device):
    model.eval()
    entries = load_test_entries(args)
    frame_scores = predict_frames(model, entries, device,
                                  cfg["evaluation"]["frames_per_segment"],
                                  cfg["evaluation"]["smoothing_sigma"])
    predictions = np.concatenate(frame_scores)
    gt = np.load(args.gt_feature_path, allow_pickle=False)
    if gt.ndim != 1 or len(gt) != len(predictions):
        raise ValueError(f"GT shape {gt.shape} does not match {len(predictions)} predicted frames. "
                         f"Check {args.gt_feature_path} and the feature/list combination. "
                         "No frames or segments are automatically removed.")
    metrics = {"auc": float(roc_auc_score(gt, predictions)),
               "ap": float(average_precision_score(gt, predictions))}
    if args.dataset == "ucf_crime":
        anomaly_gt = []
        anomaly_pred = []
        start = 0
        for entry, scores in zip(entries, frame_scores):
            end = start + len(scores)
            if entry.category != "Testing_Normal_Videos_Anomaly":
                anomaly_gt.extend(gt[start:end])
                anomaly_pred.extend(predictions[start:end])
            start = end
        metrics["ano_auc"] = float(roc_auc_score(anomaly_gt, anomaly_pred)) if anomaly_pred else 0.0
    return Evaluation(metrics, entries, frame_scores, gt)


def save_visualizations(result, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory) / "visualizations"
    directory.mkdir(parents=True, exist_ok=True)
    predictions = np.concatenate(result.frame_scores)
    low, high = predictions.min(), predictions.max()
    normalized = (predictions - low) / (high - low) if high > low else np.zeros_like(predictions)
    start = 0
    for index, (entry, scores) in enumerate(zip(result.entries, result.frame_scores)):
        end = start + len(scores)
        labels = result.ground_truth[start:end]
        curve = np.clip(normalized[start:end], 0, 1)
        figure, axis = plt.subplots(figsize=(12, 4))
        axis.plot(np.arange(len(curve)), curve, linewidth=1.5, label="Anomaly score")
        boundaries = np.diff(np.r_[False, labels.astype(bool), False].astype(np.int8))
        first = True
        for begin, finish in zip(np.flatnonzero(boundaries == 1), np.flatnonzero(boundaries == -1)):
            axis.axvspan(begin, finish, alpha=0.2, label="Ground truth" if first else None)
            first = False
        axis.set(xlabel="Frame", ylabel="Normalized score", ylim=(0, 1), title=entry.name)
        axis.legend(loc="upper right")
        figure.tight_layout()
        filename = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(entry.name).stem)
        figure.savefig(directory / f"{index:04d}_{filename}.png", dpi=180)
        plt.close(figure)
        start = end
    print(f"Visualizations: {directory}")

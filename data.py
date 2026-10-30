import csv
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


XD_CATEGORY_ORDER = ("B1", "B2", "B4", "B5", "B6", "G")


def resample_segments(feature, target_segments):
    num_segments, feature_dim = feature.shape
    if num_segments == 0:
        raise ValueError("Cannot resample an empty feature sequence.")
    new_feat = np.zeros((target_segments, feature_dim), dtype=np.float32)
    split_idx = np.linspace(0, num_segments, target_segments + 1, dtype=int)
    for i in range(target_segments):
        start, end = split_idx[i], split_idx[i + 1]
        if start != end:
            new_feat[i] = np.mean(feature[start:end], axis=0)
        else:
            new_feat[i] = feature[min(start, num_segments - 1)]
    return new_feat


def load_training_features(visual_path, text_path, target_segments):
    visual_feature = np.load(visual_path, allow_pickle=False)
    text_feature = np.load(text_path, allow_pickle=False)
    visual_feature = resample_segments(visual_feature, target_segments)
    text_feature = resample_segments(text_feature, target_segments)
    visual_tensor = torch.tensor(visual_feature, dtype=torch.float32)
    text_tensor = torch.tensor(text_feature, dtype=torch.float32)
    return torch.cat((visual_tensor, text_tensor), dim=1)


class UCFDataset(Dataset):
    def __init__(self, args, is_anomaly):
        self.visual_root = os.path.join(args.visual_root, args.visual_backbone)
        self.text_root = os.path.join(args.text_root, args.text_backbone)
        self.num_segments_per_video = args.num_segments_per_video
        self.data_list, self.categories = self._load_feature_list(args, is_anomaly)
        self.category_to_idx = {cat: idx for idx, cat in enumerate(sorted(self.categories))}
        self.num_categories = len(self.category_to_idx)
        if self.num_categories != args.num_categories:
            raise ValueError(f"Expected {args.num_categories} UCF categories, found {self.num_categories}.")

    def _load_feature_list(self, args, is_anomaly):
        data_list = []
        categories = set()
        with open(args.train_list, encoding="utf-8") as stream:
            for line in stream:
                video_name = line.strip()
                category = os.path.dirname(video_name)
                is_anomaly_video = category != "Training_Normal_Videos_Anomaly"
                if is_anomaly_video:
                    categories.add(category)
                if is_anomaly != is_anomaly_video:
                    continue
                video_base = os.path.basename(video_name).replace("x264.mp4", "")
                visual_dir = os.path.join(self.visual_root, category)
                visual_files = [f for f in os.listdir(visual_dir)
                                if f.startswith(video_base) and f.endswith(".npy")]
                if not visual_files:
                    continue
                text_dir = os.path.join(self.text_root, category)
                text_files = [f for f in os.listdir(text_dir)
                              if f.startswith(video_base) and f.endswith(".npy")]
                if not text_files:
                    continue
                data_list.append((os.path.join(visual_dir, visual_files[0]),
                                  os.path.join(text_dir, text_files[0]),
                                  is_anomaly_video, category))
        return data_list, categories

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        visual_path, text_path, label, category = self.data_list[idx]
        feature = load_training_features(visual_path, text_path, self.num_segments_per_video)
        if category == "Training_Normal_Videos_Anomaly":
            cat_onehot = torch.zeros(self.num_categories, dtype=torch.float32)
        else:
            cat_idx = self.category_to_idx[category]
            cat_onehot = F.one_hot(torch.tensor(cat_idx), num_classes=self.num_categories).float()
        return feature, torch.tensor([label], dtype=torch.float32), cat_onehot


class XDDataset(Dataset):
    def __init__(self, args):
        self.visual_root = os.path.join(args.visual_root, args.visual_backbone)
        self.text_root = os.path.join(args.text_root, args.text_backbone)
        self.num_segments_per_video = args.num_segments_per_video
        self.anomaly_list = self._load_feature_list(args, True)
        self.normal_list = self._load_feature_list(args, False)
        self.normal_list = sorted(self.normal_list, key=lambda x: os.path.basename(x[0]))
        self.anomaly_list = sorted(self.anomaly_list, key=lambda x: os.path.basename(x[0]))
        self.category_to_idx = {cat: idx for idx, cat in enumerate(XD_CATEGORY_ORDER)}
        self.num_categories = len(self.category_to_idx)
        if args.num_categories != self.num_categories:
            raise ValueError("XD-Violence requires six anomaly categories.")

    def _load_feature_list(self, args, is_anomaly):
        data_list = []
        with open(args.train_list, newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if not {"path", "label"}.issubset(reader.fieldnames or []):
                raise ValueError("XD training CSV requires path and label columns.")
            for row in reader:
                filename = os.path.basename(row["path"])
                label = row["label"]
                is_anomaly_video = label != "A"
                if is_anomaly != is_anomaly_video:
                    continue
                visual_path = os.path.join(self.visual_root, filename)
                text_path = os.path.join(self.text_root, filename)
                if not os.path.isfile(visual_path) or not os.path.isfile(text_path):
                    continue
                data_list.append((visual_path, text_path, is_anomaly_video,
                                  label if is_anomaly_video else "A"))
        return data_list

    def __len__(self):
        return min(len(self.normal_list), len(self.anomaly_list))

    def __getitem__(self, idx):
        return (self._load_and_process(*self.normal_list[idx]),
                self._load_and_process(*self.anomaly_list[idx]))

    def _load_and_process(self, visual_path, text_path, is_anomaly, category):
        feature = load_training_features(visual_path, text_path, self.num_segments_per_video)
        if category == "A":
            anomaly_flag = torch.tensor([0.0], dtype=torch.float32)
            cat_vector = torch.zeros(self.num_categories, dtype=torch.float32)
        else:
            anomaly_flag = torch.tensor([1.0], dtype=torch.float32)
            cat_vector = torch.zeros(self.num_categories, dtype=torch.float32)
            for token in category.split("-"):
                if token != "0":
                    idx = self.category_to_idx.get(token)
                    if idx is not None:
                        cat_vector[idx] = 1.0
        return feature, anomaly_flag, cat_vector


@dataclass
class TrainingLoaders:
    dataset_name: str
    anomaly: object = None
    normal: object = None
    paired: object = None

    @property
    def steps_per_epoch(self):
        return len(self.paired) if self.paired is not None else min(len(self.normal), len(self.anomaly))

    @property
    def scheduler_steps_per_epoch(self):
        return len(self.paired) if self.paired is not None else len(self.anomaly)

    def probe_input_dimension(self):
        sample = next(iter(self.paired if self.paired is not None else self.anomaly))
        return int((sample[1][0] if self.paired is not None else sample[0]).shape[-1])

    def __iter__(self):
        if self.paired is not None:
            return iter(self.paired)
        return iter(zip(self.normal, self.anomaly))


def make_training_loaders(args):
    options = dict(num_workers=args.num_workers, pin_memory=args.pin_memory,
                   batch_size=args.batch_size, shuffle=True, drop_last=True)
    if args.dataset == "ucf_crime":
        anomaly = UCFDataset(args, is_anomaly=True)
        normal = UCFDataset(args, is_anomaly=False)
        if not len(anomaly) or not len(normal):
            raise ValueError("Both normal and anomaly training features are required.")
        loaders = TrainingLoaders(args.dataset, DataLoader(anomaly, **options),
                                  DataLoader(normal, **options))
        print(f"Training videos: normal={len(normal)}, anomaly={len(anomaly)}")
    else:
        dataset = XDDataset(args)
        if not len(dataset):
            raise ValueError("Both normal and anomaly XD training features are required.")
        loaders = TrainingLoaders(args.dataset, paired=DataLoader(dataset, **options))
        print(f"Training videos: normal={len(dataset.normal_list)}, "
              f"anomaly={len(dataset.anomaly_list)}, paired={len(dataset)}")
    if loaders.steps_per_epoch == 0:
        raise ValueError("The dataset is smaller than batch_size with drop_last=True.")
    return loaders

import argparse
import copy
import json
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts, LinearLR, OneCycleLR, SequentialLR,
)

from data import make_training_loaders
from evaluate import evaluate, save_visualizations
from losses import compute_losses
from model import RefineVAD
from utils import initialize_weights, load_config, seed_everything, select_device, write_json


def build_scheduler(optimizer, cfg, steps_per_epoch):
    schedule = cfg["scheduler"]
    total_steps = steps_per_epoch * cfg["training"]["max_epoch"]
    if cfg["dataset"] == "ucf_crime":
        return OneCycleLR(
            optimizer, max_lr=cfg["training"]["lr"], total_steps=total_steps,
            pct_start=schedule["pct_start"], anneal_strategy=schedule["anneal_strategy"],
            div_factor=schedule["div_factor"], final_div_factor=schedule["final_div_factor"],
        )
    warmup_steps = int(schedule["warmup_fraction"] * total_steps)
    if not 0 < warmup_steps < total_steps:
        raise ValueError("XD warmup requires 0 < int(warmup_fraction * total_steps) < total_steps.")
    return SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=schedule["initial_lr"] / cfg["training"]["lr"],
                     total_iters=warmup_steps),
            CosineAnnealingWarmRestarts(optimizer, T_0=total_steps - warmup_steps,
                                       T_mult=schedule["t_mult"], eta_min=schedule["eta_min"]),
        ],
        milestones=[warmup_steps],
    )


def train(cfg, args, device, output_dir, save_model=False, visualize=False):
    seed_everything(args.random_seed)
    loaders = make_training_loaders(args)
    input_dim = loaders.probe_input_dimension()
    model = RefineVAD(args, input_dim, train=True).to(device)
    initialize_weights(model)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, cfg, loaders.scheduler_steps_per_epoch)
    print(f"Dataset: {args.dataset} | Device: {device} | Input dimension: {input_dim}")
    print(f"Batches/epoch: {loaders.steps_per_epoch} | Epochs: {args.max_epoch} | "
          f"Scheduler batches/epoch: {loaders.scheduler_steps_per_epoch}")
    primary = cfg["training"]["primary_metric"]
    best = 0.0
    best_state = None
    best_metrics = None
    best_epoch = None
    counter = 0
    history = []
    model.train()
    for epoch in range(args.max_epoch):
        training_mode = model.training
        for normal_batch, anomaly_batch in loaders:
            normal_feat, _, normal_cat_gt = normal_batch
            anomaly_feat, _, anomaly_cat_gt = anomaly_batch
            normal_feat = normal_feat.to(device)
            anomaly_feat = anomaly_feat.to(device)
            normal_cat_gt = normal_cat_gt.to(device)
            anomaly_cat_gt = anomaly_cat_gt.to(device)
            optimizer.zero_grad()
            anomaly_output = model(anomaly_feat, anomaly_cat_gt)
            normal_output = model(normal_feat, normal_cat_gt)
            losses = compute_losses(cfg, anomaly_output, normal_output, anomaly_cat_gt, normal_cat_gt)
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch + 1}.")
            losses["total"].backward()
            optimizer.step()
            scheduler.step()
        evaluation = evaluate(model, cfg, args, device)
        row = {"epoch": epoch + 1, "train_mode": training_mode,
               **{f"loss_{name}": float(value.detach()) for name, value in losses.items()},
               "lr": optimizer.param_groups[0]["lr"], **evaluation.metrics}
        history.append(row)
        print(json.dumps(row), flush=True)
        if evaluation.metrics[primary] > best:
            best = evaluation.metrics[primary]
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = dict(evaluation.metrics)
            best_epoch = epoch + 1
            counter = 0
        else:
            counter += 1
            if counter >= args.patience:
                print(f"Early stopping after {counter} epochs without improvement.")
                break
    summary = {"dataset": args.dataset, "primary_metric": primary,
               "best_value": best, "best_epoch": best_epoch, "metrics": best_metrics,
               "epochs_completed": len(history), "input_dim": input_dim}
    print(f"Best {primary}: {best:.6f}")
    if save_model or visualize:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "resolved_config.json", cfg)
        write_json(output_dir / "history.json", history)
        write_json(output_dir / "best_metrics.json", summary)
        if best_state is None:
            raise RuntimeError("No best checkpoint was selected; inspect the validation metrics.")
        if save_model:
            torch.save(best_state, output_dir / "best_model.pth")
            print(f"Checkpoint: {output_dir / 'best_model.pth'}")
        if visualize:
            model.load_state_dict(best_state, strict=True)
            save_visualizations(evaluate(model, cfg, args, device), output_dir)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Train RefineVAD using a dataset JSON configuration.")
    parser.add_argument("--config", default="configs/ucf_crime.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--result-folder", "--result_folder", default="default")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--save-model", "--save_model", action="store_true")
    parser.add_argument("--visualize", "--plot", action="store_true")
    cli = parser.parse_args()
    cfg, args = load_config(cli.config)
    root = "results" if args.dataset == "ucf_crime" else "results_xd"
    output = Path(cli.output_dir) if cli.output_dir else Path(root) / cli.result_folder
    if not cli.output_dir and (Path(cli.result_folder).is_absolute() or ".." in Path(cli.result_folder).parts):
        parser.error("Use --output-dir for an absolute output path.")
    train(cfg, args, select_device(cli.device), output, cli.save_model, cli.visualize)


if __name__ == "__main__":
    main()

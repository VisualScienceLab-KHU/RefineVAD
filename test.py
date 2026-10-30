import argparse
import json
from pathlib import Path

from evaluate import evaluate, infer_input_dimension, save_visualizations
from model import RefineVAD
from utils import load_config, load_weights, seed_everything, select_device, write_json


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved RefineVAD checkpoint without training data.")
    parser.add_argument("--config", default="configs/ucf_crime.json")
    parser.add_argument("--checkpoint", "--ckpt", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--visualize", "--plot", action="store_true")
    parser.add_argument("--output-dir", default=None)
    cli = parser.parse_args()
    cfg, args = load_config(cli.config)
    device = select_device(cli.device)
    seed_everything(args.random_seed)
    input_dim = infer_input_dimension(args)
    model = RefineVAD(args, input_dim, train=False).to(device)
    load_weights(model, cli.checkpoint, device)
    result = evaluate(model, cfg, args, device)
    print(f"Dataset: {args.dataset} | Device: {device} | Input dimension: {input_dim}")
    print(json.dumps(result.metrics, indent=2))
    if cli.output_dir or cli.visualize:
        output = Path(cli.output_dir) if cli.output_dir else Path(cli.checkpoint).resolve().parent / "test"
        write_json(output / "test_metrics.json", result.metrics)
        if cli.visualize:
            save_visualizations(result, output)


if __name__ == "__main__":
    main()

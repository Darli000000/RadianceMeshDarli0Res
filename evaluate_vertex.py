"""Re-use existing dataset/metric/rendering functions for the vertex ablation."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from utils.vertex_checkpoint import load_ablation_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset_path", type=Path, help="Override saved dataset location only.")
    parser.add_argument("--output_path", required=True, type=Path,
                        help="Use a fresh directory; this must differ from the checkpoint directory.")
    parser.add_argument("--render_train", action="store_true")
    options = parser.parse_args()
    directory = options.checkpoint if options.checkpoint.is_dir() else options.checkpoint.parent
    config_path = directory / "config.json"
    if not config_path.exists() and directory.name == "debug_pre_densify":
        config_path = directory.parent / "config.json"
    with config_path.open() as stream:
        config = json.load(stream)
    if options.output_path.resolve() == directory.resolve():
        parser.error("Choose a separate evaluation output directory.")
    if not config.get("eval", False):
        parser.error("Checkpoint was trained without --eval; no held-out split was reserved.")
    if not torch.cuda.is_available():
        parser.error("The existing renderer requires an NVIDIA CUDA GPU.")
    from data.loader import load_dataset
    from utils.test_util import evaluate_and_save

    model = load_ablation_checkpoint(options.checkpoint)
    model.eval()
    train, test, scene = load_dataset(
        options.dataset_path or Path(config["dataset_path"]), config["image_folder"],
        data_device="cpu", eval=config["eval"], resolution=config["resolution"])
    saved_transform = config_path.parent / "transform.txt"
    if saved_transform.exists() and not np.allclose(
            np.loadtxt(saved_transform), scene.transform, atol=1e-5, rtol=1e-5):
        raise ValueError("Dataset camera transform differs from training; use the same scene/split.")
    if not test:
        raise ValueError("No held-out cameras found.")
    options.output_path.mkdir(parents=True, exist_ok=True)
    splits = [("train", train), ("test", test)] if options.render_train else [("test", test)]
    with torch.no_grad():
        results = evaluate_and_save(model, splits, options.output_path,
                                    config["tile_size"], min_t=config["min_t"])
    with (options.output_path / "results.json").open("w") as stream:
        json.dump(results, stream, indent=2, default=lambda v: float(v))


if __name__ == "__main__":
    main()

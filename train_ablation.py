"""Launch the existing train.py with an explicit paired-ablation protocol."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model", required=True, choices=("ingp", "vertex_v2"))
    parser.add_argument("--protocol", choices=("matched_constant", "ingp_original"), default="matched_constant")
    parser.add_argument("--config", type=Path, help="Saved training JSON used as defaults, before CLI overrides.")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--probe-iterations", default="0,2000,4500,4510,5000,18000")
    parser.add_argument("--save-probe-checkpoints", action="store_true",
                        help="Save pre-event MODEL weights (not resumable optimizer state); may use many GB.")
    options, training_options = parser.parse_known_args()
    if options.protocol == "ingp_original" and options.model != "ingp":
        parser.error("ingp_original is a reference run, not a matched vertex protocol.")
    try:
        probes = [int(i) for i in options.probe_iterations.split(",") if i.strip()]
        if any(i < 0 for i in probes) or options.seed < 0:
            raise ValueError()
    except ValueError:
        parser.error("seed and probe iterations must be nonnegative integers.")
    settings = dict(model=options.model, protocol=options.protocol, seed=options.seed,
                    config=str(options.config.resolve()) if options.config else None,
                    probes=probes, save_checkpoints=options.save_probe_checkpoints)
    old_env = os.environ.get("RM_ABLATION_SETTINGS")
    old_argv = sys.argv
    os.environ["RM_ABLATION_SETTINGS"] = json.dumps(settings)
    sys.argv = ["train.py", *training_options]
    try:
        runpy.run_path(str(Path(__file__).with_name("train.py")), run_name="__main__")
    finally:
        sys.argv = old_argv
        if old_env is None:
            os.environ.pop("RM_ABLATION_SETTINGS", None)
        else:
            os.environ["RM_ABLATION_SETTINGS"] = old_env


if __name__ == "__main__":
    main()

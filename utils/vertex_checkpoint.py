"""Load pre-freeze vertex or standard post-freeze checkpoints for evaluation."""
import json
from pathlib import Path

import torch

from models.vertex_color import Model


def load_ablation_checkpoint(path, device="cuda"):
    path = Path(path)
    checkpoint = path / "ckpt.pth" if path.is_dir() else path
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    if "raw_features" in state:
        return Model.load_ckpt(checkpoint, device)
    if "interior_vertices" not in state or "density" not in state:
        raise ValueError("Expected a vertex-color or frozen checkpoint, not an iNGP checkpoint.")
    # Use the existing frozen class; do not route through its loader, which
    # expects config.base_min_t although current train.py saves config.min_t.
    from models.frozen import FrozenTetModel
    with (checkpoint.parent / "config.json").open() as stream:
        config = json.load(stream)
    max_sh_deg = round((state["sh"].shape[1] + 1)**0.5) - 1
    empty = state.get("empty_indices", state["indices"][:0])
    model = FrozenTetModel(
        int_vertices=state["interior_vertices"], ext_vertices=state["ext_vertices"],
        indices=state["indices"], empty_indices=empty, density=state["density"],
        rgb=state["rgb"], gradient=state["gradient"], sh=state["sh"],
        center=state["center"], scene_scaling=state["scene_scaling"],
        max_sh_deg=max_sh_deg)
    model.load_state_dict({k: v for k, v in state.items() if k != "empty_indices"})
    # The existing constructor assigns indices to empty_indices; restore the
    # checkpoint's actual value without changing the freeze implementation.
    model.empty_indices = empty
    model.min_t = float(config["min_t"])
    return model

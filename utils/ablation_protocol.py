"""Optional shared controls; no changes to selection, losses, or freeze algorithms.

matched_constant explicitly overrides the four head-initialization ranges AND
lambda_weight_decay=0 in BOTH models. An original-initialization iNGP reference
is a separate protocol, never mislabeled as this paired experiment.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch

from utils.args import Args


def settings_from_env():
    raw = os.environ.get("RM_ABLATION_SETTINGS")
    return json.loads(raw) if raw else None


def select_model():
    settings = settings_from_env()
    if settings and settings["model"] == "ingp":
        from models.ingp_color import Model, TetOptimizer
    elif settings and settings["model"] == "vertex_v2":
        from models.vertex_color_v2 import Model, TetOptimizer
    else:
        # Preserve the user's current plain train.py default.
        from models.vertex_color import Model, TetOptimizer
    return Model, TetOptimizer


def domain_seed(seed, iteration, domain):
    value = f"radiance-pair-v2:{seed}:{iteration}:{domain}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little") % (2**63 - 1)


def seed_torch(seed):
    torch.manual_seed(seed)


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


@contextmanager
def isolated_observation(model):
    """Diagnostics must not consume training RNGs or change train/eval mode."""
    py_state, np_state = random.getstate(), np.random.get_state()
    modes = [(module, module.training) for module in model.modules()]
    device = torch.device(model.device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices), torch.no_grad():
        try:
            model.eval()
            yield
        finally:
            for module, mode in modes:
                module.training = mode
            random.setstate(py_state)
            np.random.set_state(np_state)


def parse_training_args(defaults, argv=None):
    settings = settings_from_env()
    if settings is None:
        return Args.from_namespace(defaults.get_parser().parse_args(argv)), Session(None)
    values = dict(defaults.as_dict())
    if settings.get("config"):
        supplied = json.loads(Path(settings["config"]).read_text())
        unknown = set(supplied) - set(values)
        if unknown:
            raise ValueError(f"Unknown saved configuration fields: {sorted(unknown)}")
        for key, value in supplied.items():
            values[key] = Path(value) if isinstance(values[key], Path) else value
    parser = argparse.ArgumentParser(allow_abbrev=False)
    for key, value in values.items():
        if isinstance(value, bool):
            parser.add_argument(f"--{key}", default=value, action=argparse.BooleanOptionalAction)
        else:
            parser.add_argument(f"--{key}", default=value, type=Path if isinstance(value, Path) else type(value))
    args = Args.from_namespace(parser.parse_args(argv))
    if args.ckpt:
        parser.error("Paired runs must start fresh; the original checkpoint does not save full optimizer/RNG state.")
    if args.output_path.name == "CHOOSE_A_NEW_ABLATION_DIRECTORY":
        parser.error("Pass an explicit NEW --output_path for this experiment.")
    if args.output_path.exists() and any(args.output_path.iterdir()):
        parser.error("Choose a NEW output_path; refusing to overwrite an existing experiment.")
    session = Session(settings)
    session.requested_config = dict(args.as_dict())
    if settings["protocol"] == "matched_constant":
        for key in ("g_init", "s_init", "d_init", "c_init", "lambda_weight_decay"):
            setattr(args, key, 0.0)
        if args.density_offset != -4 or args.k_samples != 1 or args.percent_alpha != 0:
            parser.error("matched_constant currently requires density_offset=-4, k_samples=1, percent_alpha=0.")
    session.configure_runtime()
    return args, session


class Session:
    def __init__(self, settings):
        self.settings = settings
        self.requested_config = {}
        self.args = None
        self.probes = set(settings["probes"]) if settings else set()

    def configure_runtime(self):
        seed = self.settings["seed"]
        random.seed(seed)
        np.random.seed(seed % 2**32)
        seed_torch(seed)
        torch.set_float32_matmul_precision("high")
        # Fixed seed != bitwise deterministic Slang/gDel kernels. Do not promise it.

    def after_init(self, model, args):
        if not self.settings:
            return
        self.args = args
        self.configure_runtime()
        changed = {k: {"requested": self.requested_config[k], "effective": v}
                   for k, v in args.as_dict().items() if self.requested_config.get(k) != v}
        root = Path(__file__).resolve().parents[1]
        files = ("train.py", "models/ingp_color.py", "models/vertex_color_v2.py", "models/radiance_core.py",
                 "models/frozen.py", "utils/densification.py", "utils/ablation_protocol.py",
                 "models/vertex_color.py", "utils/vertex_model_util.py", "utils/optim.py")
        manifest = dict(format="radiance_ablation_v2", source_base="79251afc5b9030e999284f5ff7a75f87b530ed3f",
                        **self.settings, overrides=changed, effective_config=args.as_dict(),
                        torch_version=torch.__version__, cuda_version=torch.version.cuda,
                        precision=torch.get_float32_matmul_precision(), geometry_seed=2,
                        internal_vertices_sha256=tensor_hash(model.contracted_vertices),
                        exterior_vertices_sha256=tensor_hash(model.ext_vertices),
                        center_sha256=tensor_hash(model.center), scale=float(model.scene_scaling),
                        source_sha256={name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files},
                        vertex_initializer="zero raw = constant activated outputs",
                        vertex_aggregation="mean RAW corners, then common activation",
                        vertex_transfer="source tetrahedron RAW mean; not field-preserving",
                        vertex_lr_mapping="network_lr schedule; encoding_lr has no vertex equivalent",
                        adam_append="zero new moments; retain old rows and tensor-wide step",
                        rng_control="seeded camera order; per-iteration render and densification domains",
                        determinism="not guaranteed for custom CUDA kernels")
        (args.output_path / "protocol.json").write_text(json.dumps(manifest, indent=2, default=str))
        print(f"[ablation] model={self.settings['model']} protocol={self.settings['protocol']} seed={self.settings['seed']}")
        print(f"[ablation] Explicit paired overrides: {changed}")

    def seed_domain(self, iteration, domain):
        if self.settings:
            seed_torch(domain_seed(self.settings["seed"], iteration, domain))

    def log_sample(self, iteration, camera_index):
        if self.settings:
            with (self.args.output_path / "camera_order.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(iteration=iteration, camera_index=int(camera_index))) + "\n")

    def observe(self, model, camera, render, args, directory):
        import imageio.v2 as imageio
        directory.mkdir(parents=True, exist_ok=True)
        with isolated_observation(model):
            image = render(camera, model, scene_scaling=model.scene_scaling, **args.as_dict())["render"].detach().float().cpu()
            target = camera.original_image.detach().float().cpu()
            mse = (image - target).square().mean().item()
            density = model.calc_tet_density().detach().float().cpu()
            finite = density[torch.isfinite(density)]
            summary = dict(vertices=len(model), tetrahedra=len(model.indices),
                           # The unchanged frozen constructor aliases empty_indices
                           # to active indices; do not report that as a real count.
                           empty_tetrahedra=None if model.frozen else len(model.empty_indices), frozen=model.frozen,
                           probe_psnr=-10 * math.log10(max(mse, 1e-12)),
                           density_nonfinite=int(len(density) - len(finite)))
            summary["last_triangulation"] = getattr(model, "last_triangulation_summary", None)
            if not torch.isfinite(image).all() or summary["density_nonfinite"]:
                raise RuntimeError("Non-finite probe image or density; inspect before continuing training.")
            if len(finite):
                # numpy avoids torch.quantile's input-size limit on large meshes.
                summary["density_quantiles_0_10_50_90_100"] = np.quantile(finite.numpy(), [0, .1, .5, .9, 1]).tolist()
                summary["density_fraction_gt_threshold"] = float((finite > args.density_threshold).float().mean())
            (directory / "stats.json").write_text(json.dumps(summary, indent=2))
            imageio.imwrite(directory / "render.png", (image.permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8))
            if self.settings.get("save_checkpoints") and directory.name == "before":
                state = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in model.state_dict().items()}
                state["empty_indices"] = model.empty_indices.detach().cpu()
                torch.save(state, directory / "ckpt.pth")
                (directory / "config.json").write_text(json.dumps(args.as_dict(), default=str, indent=2))
            return summary, image

    @contextmanager
    def event(self, name, iteration, get_model, camera, render, args):
        if iteration not in self.probes:
            yield
            return
        directory = args.output_path / "diagnostics" / f"{iteration:08d}_{name}"
        before, image_before = self.observe(get_model(), camera, render, args, directory / "before")
        try:
            yield
        except Exception as error:
            failure = dict(iteration=iteration, event=name, error=repr(error),
                           last_triangulation=getattr(get_model(), "last_triangulation_summary", None))
            (directory / "failure.json").write_text(json.dumps(failure, indent=2))
            raise
        after, image_after = self.observe(get_model(), camera, render, args, directory / "after")
        delta = dict(iteration=iteration, event=name, before=before, after=after,
                     image_mean_absolute_change=float((image_after - image_before).abs().mean()),
                     image_max_absolute_change=float((image_after - image_before).abs().max()),
                     psnr_change=after["probe_psnr"] - before["probe_psnr"],
                     tetrahedra_change=after["tetrahedra"] - before["tetrahedra"])
        (directory / "event.json").write_text(json.dumps(delta, indent=2))
        print(f"[ablation probe] {iteration} {name}: T {before['tetrahedra']} -> {after['tetrahedra']}, "
              f"fixed-view PSNR {before['probe_psnr']:.2f} -> {after['probe_psnr']:.2f}")

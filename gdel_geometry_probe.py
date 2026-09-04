"""Four disposable gdel3d geometry controls; never edits model/training code.

Usage (from the server repository, with its installed Python environment):
    python gdel_geometry_probe.py gdel_probe_yln53iu1

The input may also be the original probe ZIP. This is a diagnostic, not a fix
or a proof of triangulation correctness. Each case runs in a fresh process.
"""
import argparse
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import zipfile


def read_input(source):
    import numpy as np

    if source.is_dir():
        points = np.load(source / "points.npy", allow_pickle=False)
        metadata = json.loads((source / "input.json").read_text())
    else:
        with zipfile.ZipFile(source) as archive:
            names = [n for n in archive.namelist()
                     if n.endswith("/points.npy") and not n.startswith("__MACOSX/")]
            if len(names) != 1:
                raise ValueError("ZIP must contain exactly one probe points.npy.")
            name = names[0]
            points = np.load(io.BytesIO(archive.read(name)), allow_pickle=False)
            metadata = json.loads(archive.read(name[:-len("points.npy")] + "input.json"))
    points = np.ascontiguousarray(points, dtype=np.float64)
    count = int(metadata["internal_vertices"])
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Expected finite Nx3 points.")
    if not 0 < count < len(points) or len(points) - count != int(metadata["exterior_vertices"]):
        raise ValueError("Interior/exterior counts do not match the input.")
    return points, count


def make_cases(points, internal_count):
    import numpy as np

    shell = points[internal_count:]
    # Recover sphere center from the saved positions, not from camera assumptions.
    matrix = np.column_stack([2 * shell, np.ones(len(shell))])
    solution, _, rank, _ = np.linalg.lstsq(matrix, np.square(shell).sum(axis=1), rcond=None)
    if rank != 4:
        raise ValueError("Sphere fit is rank deficient.")
    center = solution[:3]
    radii = np.linalg.norm(shell - center, axis=1)
    mean_radius = float(radii.mean())
    if mean_radius <= 0 or np.std(radii) / mean_radius > 1e-4:
        raise ValueError("Input exterior does not resemble the expected sphere.")
    noise = np.random.default_rng(20260903).uniform(-1e-4, 1e-4, len(shell))
    cases = {"original": points.copy()}
    for name, expand, jitter in [("expand_1pct", True, False),
                                 ("radial_jitter", False, True),
                                 ("expand_and_jitter", True, True)]:
        values = points.copy()
        factors = np.ones(len(shell))
        if expand:
            factors *= 1.01
        if jitter:
            factors *= 1 + noise
        values[internal_count:] = center + (shell - center) * factors[:, None]
        cases[name] = values
    metadata = dict(internal_vertices=internal_count, exterior_vertices=len(shell),
                    fitted_center=center.tolist(), mean_radius=mean_radius,
                    original_radius_std=float(radii.std()),
                    jitter_relative_amplitude=1e-4, jitter_seed=20260903,
                    expansion_factor=1.01,
                    note="Diagnostic geometry only; training code is unchanged.")
    return cases, metadata


def worker(path):
    import numpy as np
    import torch
    import gdel3d

    points = np.load(path, allow_pickle=False)
    print("NumPy:", np.__version__, "Torch:", torch.__version__,
          "CUDA:", torch.version.cuda, "gdel3d:", gdel3d.__file__, flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("Run this worker on the NVIDIA server.")
    torch.cuda.init()
    print("[1] Del(N).compute; input=", str(path), "shape=", points.shape, flush=True)
    indices, previous = gdel3d.Del(len(points)).compute(points)
    print("[2] compute returned; output shape=", getattr(indices, "shape", None), flush=True)
    torch.cuda.synchronize()
    print("[3] CUDA synchronization completed", flush=True)
    del previous
    del indices
    print("[4] Released results; completed, not a correctness assertion", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, nargs="?")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--prepare-only", action="store_true",
                        help="Generate diagnostic point sets without running GPU workers.")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return 0
    if args.source is None or args.timeout <= 0:
        parser.error("Provide a probe directory/ZIP and a positive timeout.")
    import numpy as np

    points, count = read_input(args.source)
    cases, metadata = make_cases(points, count)
    output = Path(tempfile.mkdtemp(prefix="gdel_geometry_", dir=Path.cwd()))
    metadata["source"] = str(args.source.resolve())
    (output / "geometry.json").write_text(json.dumps(metadata, indent=2))
    print("Diagnostic directory:", output, flush=True)
    for name, values in cases.items():
        np.save(output / (name + ".npy"), values, allow_pickle=False)
    if args.prepare_only:
        print("Prepared inputs only; no GPU validation was performed.")
        return 0
    environment = os.environ.copy()
    environment.update(PYTHONUNBUFFERED="1", CUDA_LAUNCH_BLOCKING="1")
    summary = {}
    for name in cases:
        logfile = output / (name + ".log")
        print("Running", name, "...", flush=True)
        command = [sys.executable, "-X", "faulthandler", str(Path(__file__).resolve()),
                   "--worker", str(output / (name + ".npy"))]
        with logfile.open("w") as stream:
            try:
                result = subprocess.run(command, env=environment, stdout=stream,
                                        stderr=subprocess.STDOUT, timeout=args.timeout)
                code = result.returncode
                status = "completed" if code == 0 else "failed"
                if code < 0:
                    status = signal.Signals(-code).name
            except subprocess.TimeoutExpired:
                code, status = None, "timeout"
        summary[name] = dict(returncode=code, status=status, log=str(logfile))
        (output / "summary.json").write_text(json.dumps(summary, indent=2))
        print(name + ": " + status, flush=True)
    print("Send summary.json and all four .log files from:", output, flush=True)
    print("Do not apply these point changes to formal training as an unverified fix.", flush=True)
    return 0 if all(item["returncode"] == 0 for item in summary.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

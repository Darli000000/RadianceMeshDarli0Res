"""Diagnose gdel3d without editing the repository or installing dependencies.

Run from the Radiance Mesh repository root:
    .venv/bin/python -X faulthandler gdel_probe.py

Captures the first GPU test's actual vertices before triangulation. Replays
those vertices (Tensor / NumPy) and a same-size random cloud in separate
processes. A successful replay is NOT a triangulation correctness test.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile


class CapturedInput(Exception):
    pass


def capture(directory):
    import numpy as np
    import torch
    from unittest.mock import patch

    if not torch.cuda.is_available():
        raise RuntimeError("Run this diagnostic on the NVIDIA server.")
    sys.path.insert(0, str(Path.cwd()))
    from models.radiance_core import RadianceMeshCore
    from tests.test_vertex_color_v2_cuda import VertexV2CudaTests

    def intercept(model, *args, **kwargs):
        # Same conversion used by the failing update_triangulation call.
        points = model.vertices.detach().cpu().double().numpy()
        np.save(directory / "points.npy", points, allow_pickle=False)
        details = dict(
            shape=list(points.shape), dtype=str(points.dtype),
            finite=bool(np.isfinite(points).all()),
            unique_points=int(len(np.unique(points, axis=0))),
            internal_vertices=int(model.num_int_verts),
            exterior_vertices=int(len(model.ext_vertices)),
        )
        (directory / "input.json").write_text(json.dumps(details, indent=2))
        print("Captured exact pre-triangulation input:", details, flush=True)
        raise CapturedInput

    # Patch exists only in this disposable process. No source file is edited.
    method = "test_actual_pcd_geometry_and_initial_render_match"
    case = VertexV2CudaTests(method)
    with patch.object(RadianceMeshCore, "update_triangulation", intercept):
        try:
            getattr(case, method)()
        except CapturedInput:
            print("Input captured; deliberately stopped before gdel3d.", flush=True)
            return
    raise RuntimeError("Expected triangulation interception was not reached.")


def replay(directory, mode):
    from importlib.metadata import version
    import numpy as np
    import torch
    import gdel3d

    print("Python:", sys.version, flush=True)
    print("NumPy:", np.__version__, "Torch:", torch.__version__,
          "Torch CUDA:", torch.version.cuda, flush=True)
    print("gdel3d:", version("gdel3d"), gdel3d.__file__, flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("Run this diagnostic on the NVIDIA server.")
    torch.cuda.init()
    points = np.load(directory / "points.npy", allow_pickle=False)
    if mode == "random_numpy":
        # Diagnostic control only: never used for actual model training.
        rng = np.random.default_rng(27)
        points = rng.uniform(points.min(axis=0), points.max(axis=0), points.shape)
        np.save(directory / "random_points.npy", points, allow_pickle=False)
    points = np.ascontiguousarray(points, dtype=np.float64)
    if not np.isfinite(points).all():
        raise ValueError("Input contains NaN or infinity.")
    argument = torch.from_numpy(points) if mode == "exact_tensor" else points
    print("Mode:", mode, "shape:", points.shape, "input:", type(argument), flush=True)
    print("[1] Calling Del(N).compute(input)", flush=True)
    indices, previous = gdel3d.Del(len(points)).compute(argument)
    print("[2] compute returned:", type(indices), getattr(indices, "shape", None), flush=True)
    torch.cuda.synchronize()
    print("[3] CUDA synchronize completed", flush=True)
    del previous
    print("[4] Python result object released", flush=True)
    del indices
    print("[5] Replay completed (not a correctness assertion)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=int, default=300,
                        help="Maximum seconds per child process (default: 300).")
    parser.add_argument("--worker", choices=["capture", "exact_tensor", "exact_numpy", "random_numpy"])
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args()
    if args.worker:
        if args.directory is None:
            parser.error("--worker requires --directory")
        if args.worker == "capture":
            capture(args.directory)
        else:
            replay(args.directory, args.worker)
        return 0
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if not Path("tests/test_vertex_color_v2_cuda.py").is_file():
        parser.error("Run from the Radiance Mesh repository root.")
    directory = Path(tempfile.mkdtemp(prefix="gdel_probe_", dir=Path.cwd()))
    print("Diagnostic files:", directory, flush=True)
    environment = os.environ.copy()
    environment.update(PYTHONUNBUFFERED="1", CUDA_LAUNCH_BLOCKING="1")
    summary = {}
    for mode in ("capture", "exact_tensor", "exact_numpy", "random_numpy"):
        log = directory / (mode + ".log")
        print("Running", mode, "...", flush=True)
        command = [sys.executable, "-X", "faulthandler", str(Path(__file__).resolve()),
                   "--worker", mode, "--directory", str(directory)]
        with log.open("w") as stream:
            try:
                result = subprocess.run(command, env=environment, stdout=stream,
                                        stderr=subprocess.STDOUT, timeout=args.timeout)
                code = result.returncode
                status = "completed" if code == 0 else "failed"
                if code < 0:
                    status = signal.Signals(-code).name
            except subprocess.TimeoutExpired:
                code, status = None, "timeout"
        summary[mode] = dict(returncode=code, status=status, log=str(log))
        (directory / "summary.json").write_text(json.dumps(summary, indent=2))
        print(mode + ": " + status + "; log=" + str(log), flush=True)
        if mode == "capture" and code != 0:
            print("Capture failed; stopped. Send capture.log and summary.json.", flush=True)
            return 1
    print("Send summary.json, input.json and the three replay .log files.", flush=True)
    print("A failed child is a diagnostic result, not permission to skip the CUDA test.", flush=True)
    return 0 if all(v["returncode"] == 0 for v in summary.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

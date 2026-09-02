"""CPU-testable utilities for the explicit vertex-property ablation."""
import warnings
import numpy as np
import torch
from scipy.spatial import Delaunay, QhullError, cKDTree


@torch.no_grad()
def interpolate_new_vertices(vertices, raw_features, points):
    """Interpolate raw parameters on a PRE-insertion CPU Delaunay locator.

    This locator does not replace the rendering mesh. Its simplices may differ
    from gDel3D near degeneracies. Outside points inherit the nearest vertex.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("New vertex positions must have shape (N, 3).")
    if not torch.isfinite(points).all():
        raise ValueError("Densification produced NaN/Inf positions; no vertices were added.")
    if len(points) == 0:
        return raw_features.new_empty((0, raw_features.shape[1])), 0
    xyz = vertices.detach().cpu().double().numpy()
    query = points.detach().cpu().double().numpy()
    nearest = cKDTree(xyz).query(query)[1]
    result = raw_features[torch.as_tensor(nearest, device=raw_features.device)].clone()
    valid = np.zeros(len(query), dtype=bool)
    try:
        locator = Delaunay(xyz)
        simplex = locator.find_simplex(query, tol=1e-9)
        valid = simplex >= 0
        ids = np.flatnonzero(valid)
        transforms = locator.transform[simplex[valid]]
        first = np.einsum("nij,nj->ni", transforms[:, :3], query[valid] - transforms[:, 3])
        weights = np.concatenate((first, 1 - first.sum(axis=1, keepdims=True)), axis=1)
        good = np.isfinite(weights).all(axis=1) & (weights.min(axis=1) >= -1e-7)
        valid[ids[~good]] = False
        weights = np.maximum(weights[good], 0)
        weights /= weights.sum(axis=1, keepdims=True)
        corners = torch.as_tensor(locator.simplices[simplex[ids[good]]],
                                  device=raw_features.device, dtype=torch.long)
        w = torch.as_tensor(weights, device=raw_features.device, dtype=raw_features.dtype)
        result[torch.as_tensor(ids[good], device=result.device)] = (
            raw_features[corners] * w[..., None]).sum(dim=1)
    except QhullError:
        warnings.warn("Degenerate vertex transfer locator; using nearest-vertex attributes.",
                      RuntimeWarning, stacklevel=2)
    return result, int((~valid).sum())


# Same equations as utils.train_util at ccf615d. Kept separate because importing
# train_util eagerly loads Slang/CUDA, including when only testing LR schedules.
def get_expon_lr_func(lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0,
                      max_steps=1000000):
    lr_init, lr_final = max(lr_init, 1e-20), max(lr_final, 1e-20)

    def helper(step):
        if max_steps == 0:
            return lr_init
        if step < 0:
            return 0.0
        delay = (lr_delay_mult + (1 - lr_delay_mult) * np.sin(
            0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1))) if lr_delay_steps > 0 else 1.0
        t = np.clip(step / max_steps, 0, 1)
        return delay * np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
    return helper


class SpikingLR:
    def __init__(self, duration, max_steps, base_function, peak_start,
                 peak_interval, peak_end, peak_lr_init, peak_lr_final):
        self.duration, self.max_steps = duration, max_steps
        self.base_function = base_function
        self.peak_start, self.peak_interval, self.peak_end = peak_start, peak_interval, peak_end
        self.peak_lr_init, self.peak_lr_final = peak_lr_init, peak_lr_final

    def __call__(self, iteration):
        base = self.base_function(iteration)
        if iteration < self.peak_start:
            return base
        last_peak = (iteration - self.peak_end if iteration > self.peak_end
                     else (iteration - self.peak_start) % self.peak_interval)
        peak_ind = iteration - last_peak
        height = (peak_ind / self.max_steps * (self.peak_lr_final - self.peak_lr_init)
                  + self.peak_lr_init - self.base_function(peak_ind))
        t = np.clip(last_peak / self.duration, 0, 1)
        return base + np.exp(np.log(max(height, 1e-20)) * (1 - t) + np.log(1e-6) * t)

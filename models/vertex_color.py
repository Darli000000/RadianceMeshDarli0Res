"""Explicit vertex properties, compatible with dev/ccf615d's training loop.

No hash grid, MLP, pretrained iNGP, or legacy sh_slang is used. Each vertex owns
[log_density, rgb_residual(3), raw_gradient(3), SH_rest(K*3)]. Four-corner
averaging produces per-tetrahedron raw parameters; the existing activation
and renderer then apply. See docs/vertex_color_ablation.md for experiment
choices, LR mapping, freezing, and limitations.
"""
from pathlib import Path

import numpy as np
import torch
from torch import nn
from scipy.spatial import Delaunay

from models.base_model import BaseModel
from utils.model_util import activate_output, offset_normalize, pre_calc_cell_values
from utils.optim import CustomAdam
from utils.safe_math import safe_exp
from utils.topo_utils import fibonacci_spiral_on_sphere, tet_volumes
from utils.vertex_model_util import (
    get_expon_lr_func, SpikingLR, interpolate_new_vertices,
)


class Model(BaseModel):
    def __init__(self, vertices, ext_vertices, center, scene_scaling,
                 max_sh_deg=3, current_sh_deg=None, density_offset=-4.0,
                 ablate_gradient=False, ablate_circumsphere=True,
                 chunk_size=65536, min_t=0.4, indices=None, **kwargs):
        super().__init__()
        # Degree zero is not supported by the unchanged renderer: it takes
        # mean(SH**2), which is NaN for an empty SH table. current degree may be 0.
        if not 1 <= max_sh_deg <= 4:
            raise ValueError("max_sh_deg must be 1..4; current_sh_deg may be 0.")
        self.max_sh_deg = int(max_sh_deg)
        self.current_sh_deg = self.max_sh_deg if current_sh_deg is None else int(current_sh_deg)
        if not 0 <= self.current_sh_deg <= self.max_sh_deg:
            raise ValueError("current_sh_deg must be between 0 and max_sh_deg.")
        self.chunk_size = int(chunk_size)
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        self.density_offset = float(density_offset)
        self.ablate_gradient = bool(ablate_gradient)
        self.ablate_circumsphere = bool(ablate_circumsphere)
        self.base_min_t = float(min_t)
        vertices = torch.as_tensor(vertices, dtype=torch.float32)
        self.contracted_vertices = nn.Parameter(vertices.detach().clone())
        self.register_buffer("ext_vertices", torch.as_tensor(
            ext_vertices, device=vertices.device, dtype=torch.float32).detach().clone())
        self.register_buffer("center", torch.as_tensor(
            center, device=vertices.device, dtype=torch.float32).reshape(1, 3).clone())
        self.register_buffer("scene_scaling", torch.as_tensor(
            scene_scaling, device=vertices.device, dtype=torch.float32).reshape(()).clone())
        self.sh_dim = (self.max_sh_deg + 1)**2 - 1
        self.raw_dim = 7 + 3 * self.sh_dim
        # Boundary positions are fixed, but their properties remain learnable.
        # Separate tables avoid shifting their Adam states when internal points grow.
        self.raw_features = nn.Parameter(vertices.new_zeros((len(vertices), self.raw_dim)))
        self.ext_raw_features = nn.Parameter(vertices.new_zeros((len(self.ext_vertices), self.raw_dim)))
        self.register_buffer("indices", torch.empty((0, 4), dtype=torch.int32, device=self.device))
        self.register_buffer("empty_indices", torch.empty_like(self.indices))
        self.mask_values, self.frozen, self.linear = True, False, False
        self.feature_dim, self.alpha = 7, 0.0
        if indices is None:
            self.update_triangulation()
        else:
            self.indices = torch.as_tensor(indices, device=self.device, dtype=torch.int32).clone()
            self._validate_indices(self.indices)

    @property
    def device(self):
        return self.contracted_vertices.device

    @property
    def vertices(self):
        # 'contracted_vertices' retains the baseline checkpoint/API name.
        # These coordinates are world/PCA coordinates, NOT NGP-contracted ones.
        return torch.cat((self.contracted_vertices, self.ext_vertices), dim=0)

    @property
    def vertex_features(self):
        return torch.cat((self.raw_features, self.ext_raw_features), dim=0)

    @property
    def min_t(self):
        # train.py passes args.min_t directly to render (no scene-scale factor).
        return self.base_min_t

    @classmethod
    def init_from_pcd(cls, point_cloud, cameras, device, max_sh_deg=3,
                      voxel_size=0.0, **kwargs):
        """Same geometry recipe as ingp_color.init_from_pcd; no network created."""
        torch.manual_seed(2)
        centers = torch.stack([c.camera_center.reshape(3) for c in cameras]).to(device)
        center = centers.mean(dim=0)
        scaling = torch.linalg.norm(centers - center, dim=1, ord=torch.inf).max()
        print(f"Scene scaling: {scaling}. Center: {center}")
        vertices = torch.as_tensor(np.asarray(point_cloud.points)).float().cpu()
        if voxel_size > 0:
            import open3d as o3d
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(vertices.numpy())
            vertices = torch.as_tensor(np.asarray(
                cloud.voxel_down_sample(voxel_size=voxel_size).points)).float()
        vertices = vertices + torch.randn(*vertices.shape) * 1e-3
        radius = torch.linalg.norm(vertices - center.cpu(), dim=1).max().item()
        exterior = fibonacci_spiral_on_sphere(1000, radius, device="cpu") + center.cpu()
        print("[vertex_color] Direct vertex properties; raw four-corner mean; no iNGP.")
        return cls(vertices.to(device), exterior, center, scaling,
                   max_sh_deg=max_sh_deg, **kwargs)

    def _validate_indices(self, indices):
        if indices.ndim != 2 or indices.shape[1] != 4:
            raise ValueError("Tetrahedron indices must have shape (T, 4).")
        if len(indices) and ((indices < 0).any() or (indices >= len(self)).any()):
            raise ValueError("Tetrahedron index is outside the current vertex table.")

    def compute_batch_features(self, vertices, indices, start, end, circumcenters=None):
        # Average RAW parameters before activation (not average RGB after SH).
        ids = indices[start:end].long()
        raw = self.vertex_features[ids].mean(dim=1)
        density = safe_exp(raw[:, :1] + self.density_offset)
        rgb = raw[:, 1:4] + 0.5
        grd = raw[:, 4:7]
        grd = (grd / torch.sqrt(1 + (grd * grd).sum(dim=-1, keepdim=True))).unsqueeze(1)
        if self.ablate_gradient:
            grd = torch.zeros_like(grd)
        sh = raw[:, 7:].reshape(-1, self.sh_dim, 3).half()
        # No spatial NGP query remains. The returned anchor still follows the
        # baseline convention; get_cell_values below uses the baseline centroid.
        if self.ablate_circumsphere:
            cc = vertices[ids].mean(dim=1)
        elif circumcenters is not None:
            cc = circumcenters[start:end]
        else:
            cc = pre_calc_cell_values(vertices, ids)
        return cc, density, rgb, grd, sh

    def get_cell_values(self, camera, mask=None, all_circumcenters=None, radii=None):
        indices = self.indices if mask is None else self.indices[mask]
        if not len(indices):
            return (self.raw_features.new_empty((0, self.sh_dim, 3)),
                    self.raw_features.new_empty((0, self.feature_dim)))
        vertices = self.vertices
        shs, values = [], []
        for start in range(0, len(indices), self.chunk_size):
            end = min(start + self.chunk_size, len(indices))
            _, density, rgb, grd, sh = self.compute_batch_features(
                vertices, indices, start, end)
            ids = indices[start:end]
            # Deliberately match current ingp_color.get_cell_values, not the
            # legacy vertex renderer and not a new barycentric color rasterizer.
            centroids = vertices[ids.long()].mean(dim=1)
            values.append(activate_output(
                camera.camera_center.to(self.device), density, rgb, grd, sh,
                ids, centroids, vertices.detach(), self.current_sh_deg, self.max_sh_deg))
            shs.append(sh.float())
        return torch.cat(shs), torch.cat(values)

    def compute_features(self, offset=False):
        if not len(self.indices):
            raise RuntimeError("There are no tetrahedra to export or freeze.")
        fields = [[] for _ in range(5)]
        vertices = self.vertices
        for start in range(0, len(self.indices), self.chunk_size):
            end = min(start + self.chunk_size, len(self.indices))
            cc, density, rgb, grd, sh = self.compute_batch_features(vertices, self.indices, start, end)
            if offset:
                # BaseModel's PLY contract explicitly requires the color at v0.
                # Match our pre-freeze rendering anchor (centroid).
                tets = vertices[self.indices[start:end].long()]
                cc = tets.mean(dim=1)
                rgb, grd = offset_normalize(rgb, grd, cc, tets)
            for dest, value in zip(fields, (cc, density, rgb, grd, sh)):
                dest.append(value)
        return tuple(torch.cat(items) for items in fields)

    def calc_tet_density(self):
        result = []
        features = self.vertex_features
        for start in range(0, len(self.indices), self.chunk_size):
            ids = self.indices[start:start + self.chunk_size].long()
            result.append(safe_exp(features[:, 0][ids].mean(dim=1) + self.density_offset))
        return torch.cat(result) if result else features.new_empty((0,))

    def get_circumcenters(self):
        return pre_calc_cell_values(self.vertices, self.indices)

    def sh_up(self):
        self.current_sh_deg = min(self.max_sh_deg, self.current_sh_deg + 1)

    @torch.no_grad()
    def update_triangulation(self, high_precision=False, density_threshold=0.0, alpha_threshold=0.0):
        vertices = self.vertices
        if not torch.isfinite(vertices).all():
            raise ValueError("Non-finite geometry before triangulation.")
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        if high_precision or self.device.type != "cuda":
            simplices = Delaunay(vertices.detach().cpu().double().numpy()).simplices.copy()
        else:
            from gdel3d import Del
            simplices, previous = Del(len(vertices)).compute(vertices.detach().cpu().double())
            if isinstance(simplices, torch.Tensor):
                simplices = simplices.detach().cpu().numpy()
            del previous
        simplices = np.asarray(simplices)
        valid = ((simplices >= 0) & (simplices < len(vertices))).all(axis=1)
        indices = torch.as_tensor(simplices[valid], device=self.device, dtype=torch.int32)
        reverse = tet_volumes(vertices[indices.long()]) < 0
        indices[reverse] = indices[reverse][:, [1, 0, 2, 3]]
        self.indices = indices
        if density_threshold > 0 or alpha_threshold > 0:
            density = self.calc_tet_density()
            alpha = self.calc_tet_alpha(mode="min", density=density)
            keep = (density > density_threshold) | (alpha > alpha_threshold)
            self.empty_indices, self.indices = indices[~keep], indices[keep]
        else:
            self.empty_indices = torch.empty_like(indices[:0])
        if not len(self.indices):
            raise RuntimeError("All tetrahedra were culled. Reduce density/alpha thresholds.")
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def get_extra_state(self):
        # Saved by both train.py's final checkpoint and debug_pre_densify.
        return dict(format="vertex_color_v1", max_sh_deg=self.max_sh_deg,
                    current_sh_deg=self.current_sh_deg, density_offset=self.density_offset,
                    ablate_gradient=self.ablate_gradient, ablate_circumsphere=self.ablate_circumsphere,
                    min_t=self.base_min_t, chunk_size=self.chunk_size)

    def set_extra_state(self, state):
        if state.get("format") != "vertex_color_v1" or state["max_sh_deg"] != self.max_sh_deg:
            raise ValueError("Incompatible vertex-color checkpoint.")
        self.current_sh_deg = state["current_sh_deg"]
        self.density_offset = state["density_offset"]
        self.ablate_gradient = state["ablate_gradient"]
        self.ablate_circumsphere = state["ablate_circumsphere"]
        self.base_min_t = state["min_t"]
        self.chunk_size = state["chunk_size"]

    @classmethod
    def load_ckpt(cls, path, device):
        path = Path(path)
        state = torch.load(path / "ckpt.pth" if path.is_dir() else path,
                           map_location=device, weights_only=True)
        if "raw_features" not in state:
            raise ValueError("Not a pre-freeze vertex checkpoint; use load_ablation_checkpoint.")
        settings = dict(state["_extra_state"])
        settings.pop("format")
        model = cls(state["contracted_vertices"], state["ext_vertices"], state["center"],
                    state["scene_scaling"], indices=state["indices"], **settings)
        model.empty_indices = state["empty_indices"].clone()
        model.load_state_dict(state)
        return model


class TetOptimizer:
    def __init__(self, model, network_lr=1e-3, final_network_lr=1e-4,
                 vertices_lr=1e-4, final_vertices_lr=1e-6,
                 vertices_lr_delay_multi=1e-8, lr_delay=0, vert_lr_delay=0,
                 freeze_start=18000, spike_duration=500, densify_interval=500,
                 densify_end=16000, midpoint=2000, **kwargs):
        self.model = model
        if min(freeze_start, spike_duration, densify_interval) <= 0:
            raise ValueError("freeze_start, spike_duration and densify_interval must be positive.")
        self.optim = CustomAdam([dict(params=[model.raw_features], name="raw_features",
                                      lr=network_lr)], betas=(0.9, 0.999), eps=1e-15)
        self.ext_optim = CustomAdam([dict(params=[model.ext_raw_features], name="ext_raw_features",
                                          lr=network_lr)], betas=(0.9, 0.999), eps=1e-15)
        self.vert_lr_multi = float(model.scene_scaling)
        self.vertices_lr = self.vert_lr_multi * vertices_lr
        self.final_vertices_lr = self.vert_lr_multi * final_vertices_lr
        self.vertex_optim = CustomAdam([dict(params=[model.contracted_vertices],
            name="contracted_vertices", lr=self.vertices_lr)])
        base_field = get_expon_lr_func(network_lr, final_network_lr, lr_delay, 1e-8, freeze_start)
        self.net_scheduler_args = SpikingLR(spike_duration, freeze_start, base_field,
            midpoint, densify_interval, densify_end, network_lr, network_lr)
        # train.py prints "Encoding LR" unconditionally. This is only an alias;
        # it plots the actual explicit-property LR, not a nonexistent encoder.
        self.encoder_scheduler_args = self.net_scheduler_args
        base_vertex = get_expon_lr_func(self.vertices_lr, self.final_vertices_lr,
            vert_lr_delay, vertices_lr_delay_multi, freeze_start)
        self.vertex_scheduler_args = SpikingLR(spike_duration, freeze_start, base_vertex,
            midpoint, densify_interval, densify_end, self.vertices_lr, self.vertices_lr)
        self.iteration = 0
        self.last_transfer_fallback_count = 0

    @property
    def sh_optim(self):
        # SH is part of the packed vertex parameters and must not be stepped twice.
        return None

    def main_step(self):
        self.optim.step()
        self.ext_optim.step()

    def main_zero_grad(self):
        self.optim.zero_grad(set_to_none=True)
        self.ext_optim.zero_grad(set_to_none=True)

    def update_learning_rate(self, iteration):
        self.iteration = iteration
        for optimizer in (self.optim, self.ext_optim):
            for group in optimizer.param_groups:
                group["lr"] = self.net_scheduler_args(iteration)
        self.vertices_lr = self.vertex_scheduler_args(iteration)
        self.vertex_optim.param_groups[0]["lr"] = self.vertices_lr

    def regularizer(self, render_pkg, lambda_weight_decay=0.0, **kwargs):
        # The baseline's lambda_weight_decay is specifically hash-embedding L2.
        # Do not silently repurpose it as vertex-property L2. train.py still
        # applies its existing SH and image/distortion regularizers.
        return 0.0

    def update_triangulation(self, **kwargs):
        self.model.update_triangulation(**kwargs)

    @torch.no_grad()
    def add_points(self, new_verts, raw_verts=False):
        if raw_verts:
            raise ValueError("Pass world/PCA coordinates; there are no contracted NGP coordinates.")
        new_verts = torch.as_tensor(new_verts, device=self.model.device, dtype=torch.float32)
        new_features, count = interpolate_new_vertices(
            self.model.vertices, self.model.vertex_features, new_verts)
        self.last_transfer_fallback_count = count
        if not len(new_verts):
            return
        self.model.raw_features = self.optim.cat_tensors_to_optimizer(
            dict(raw_features=new_features))["raw_features"]
        self.model.contracted_vertices = self.vertex_optim.cat_tensors_to_optimizer(
            dict(contracted_vertices=new_verts))["contracted_vertices"]
        # Re-triangulation also accounts for shifted exterior vertex indices.
        self.model.update_triangulation()
        print(f"[vertex_color] Added {len(new_verts)} vertices; nearest-vertex fallback: {count}.")

    @torch.no_grad()
    def split(self, split_point, **kwargs):
        self.add_points(split_point)

    @torch.no_grad()
    def remove_points(self, keep_mask):
        keep = torch.as_tensor(keep_mask, device=self.model.device, dtype=torch.bool)
        if keep.ndim != 1 or len(keep) not in (self.model.num_int_verts, len(self.model)):
            raise ValueError("keep_mask must match internal vertices or all vertices.")
        if len(keep) == len(self.model) and not keep[self.model.num_int_verts:].all():
            raise ValueError("The enclosing exterior geometry must be retained.")
        keep = keep[:self.model.num_int_verts]
        self.model.raw_features = self.optim.prune_optimizer(keep)["raw_features"]
        self.model.contracted_vertices = self.vertex_optim.prune_optimizer(keep)["contracted_vertices"]
        self.model.update_triangulation()

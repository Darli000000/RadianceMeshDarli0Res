"""Paired constant-output initialization; raw-corner-mean vertex ablation.

No network is constructed. Geometry, activation, rendering adapters, culling,
and export are shared with iNGP through RadianceMeshCore. New vertex attributes
are the RAW mean of the actual source tetrahedron supplied by densification.
This is NOT a claim of field-preserving subdivision or the author's old ablation.
"""
from pathlib import Path

import torch
from torch import nn

from models.radiance_core import RadianceMeshCore, activate_raw_properties
from models.vertex_color import TetOptimizer as ExplicitOptimizer
from utils.model_util import pre_calc_cell_values


class Model(RadianceMeshCore):
    def __init__(self, vertices, ext_vertices, center, scene_scaling,
                 max_sh_deg=3, current_sh_deg=None, density_offset=-4.,
                 ablate_circumsphere=True, ablate_gradient=False,
                 chunk_size=308576, min_t=.4, indices=None,
                 g_init=0., s_init=0., d_init=0., c_init=0.,
                 k_samples=1, percent_alpha=0., **kwargs):
        super().__init__()
        if any(float(v) != 0 for v in (g_init, s_init, d_init, c_init)):
            raise ValueError("v2 uses matched_constant output initialization. Run train_ablation.py; "
                             "network weight ranges are NOT vertex feature ranges.")
        if k_samples != 1 or percent_alpha != 0:
            raise ValueError("v2 protocol requires k_samples=1 and percent_alpha=0.")
        if not 1 <= max_sh_deg <= 4:
            raise ValueError("The unchanged renderer requires max_sh_deg in 1..4.")
        current_sh_deg = max_sh_deg if current_sh_deg is None else current_sh_deg
        if not 0 <= current_sh_deg <= max_sh_deg or chunk_size <= 0:
            raise ValueError("Invalid current_sh_deg or chunk_size.")
        vertices = torch.as_tensor(vertices, dtype=torch.float32)
        self.device = vertices.device
        self.contracted_vertices = nn.Parameter(vertices.detach().clone())
        self.register_buffer("ext_vertices", torch.as_tensor(
            ext_vertices, dtype=torch.float32, device=self.device).detach().clone())
        self.register_buffer("center", torch.as_tensor(center, device=self.device).reshape(1, 3).float().clone())
        self.register_buffer("scene_scaling", torch.as_tensor(scene_scaling, device=self.device).float().reshape(()))
        self.max_sh_deg, self.current_sh_deg = int(max_sh_deg), int(current_sh_deg)
        self.chunk_size, self.base_min_t = int(chunk_size), float(min_t)
        self.density_offset = float(density_offset)
        self.ablate_circumsphere, self.ablate_gradient = bool(ablate_circumsphere), bool(ablate_gradient)
        self.feature_dim, self.alpha = 7, 0.
        self.mask_values, self.frozen, self.linear = True, False, False
        self.sh_dim = (max_sh_deg + 1)**2 - 1
        self.raw_dim = 7 + 3 * self.sh_dim
        self.raw_features = nn.Parameter(vertices.new_zeros((len(vertices), self.raw_dim)))
        self.ext_raw_features = nn.Parameter(vertices.new_zeros((len(self.ext_vertices), self.raw_dim)))
        self.register_buffer("indices", torch.empty((0, 4), dtype=torch.int32, device=self.device))
        self.register_buffer("empty_indices", torch.empty_like(self.indices))
        if indices is None:
            self.update_triangulation()
        else:
            self.indices = torch.as_tensor(indices, device=self.device, dtype=torch.int32).clone()
            if self.indices.ndim != 2 or self.indices.shape[1] != 4:
                raise ValueError("indices must have shape (T, 4).")
            if len(self.indices) and ((self.indices < 0).any() or (self.indices >= len(self)).any()):
                raise ValueError("Invalid tetrahedron vertex index.")

    @property
    def min_t(self):
        return self.base_min_t

    @property
    def vertex_features(self):
        return torch.cat([self.raw_features, self.ext_raw_features])

    def compute_batch_features(self, vertices, indices, start, end, circumcenters=None):
        ids = indices[start:end].long()
        raw = self.vertex_features[ids].mean(dim=1)
        density, rgb, gradient, sh = activate_raw_properties(
            raw[:, :1], raw[:, 1:4], raw[:, 4:7],
            raw[:, 7:].reshape(-1, self.sh_dim, 3), self.density_offset)
        if self.ablate_circumsphere:
            anchor = vertices[ids].mean(dim=1)
        elif circumcenters is not None:
            anchor = circumcenters[start:end]
        else:
            anchor = pre_calc_cell_values(vertices, ids)
        # ablate_gradient belongs to the shared rendering adapter, just as in iNGP.
        return anchor, density, rgb, gradient, sh

    def get_extra_state(self):
        return dict(format="vertex_color_v2", initializer="matched_constant",
                    aggregation="raw_corner_mean", new_vertex_rule="source_tet_raw_mean",
                    max_sh_deg=self.max_sh_deg, current_sh_deg=self.current_sh_deg,
                    density_offset=self.density_offset, ablate_gradient=self.ablate_gradient,
                    ablate_circumsphere=self.ablate_circumsphere,
                    min_t=self.base_min_t, chunk_size=self.chunk_size)

    def set_extra_state(self, state):
        for key, value in (("format", "vertex_color_v2"), ("initializer", "matched_constant"),
                           ("aggregation", "raw_corner_mean"), ("new_vertex_rule", "source_tet_raw_mean")):
            if state.get(key) != value:
                raise ValueError(f"Incompatible v2 checkpoint: {key}.")
        if state["max_sh_deg"] != self.max_sh_deg:
            raise ValueError("SH degree does not match checkpoint.")
        for key in ("current_sh_deg", "density_offset", "ablate_gradient", "ablate_circumsphere", "chunk_size"):
            setattr(self, key, state[key])
        self.base_min_t = state["min_t"]

    @classmethod
    def load_ckpt(cls, path, device):
        path = Path(path)
        state = torch.load(path / "ckpt.pth" if path.is_dir() else path,
                           map_location=device, weights_only=True)
        options = dict(state["_extra_state"])
        if options.get("format") != "vertex_color_v2":
            raise ValueError("Expected a v2 pre-freeze checkpoint.")
        for name in ("format", "initializer", "aggregation", "new_vertex_rule"):
            options.pop(name)
        model = cls(state["contracted_vertices"], state["ext_vertices"], state["center"],
                    state["scene_scaling"], indices=state["indices"], **options)
        model.empty_indices = state["empty_indices"].clone()
        model.load_state_dict(state)
        return model


class TetOptimizer(ExplicitOptimizer):
    """Keep Adam/geometry schedules; replace only the v1 attribute transfer.

Old rows retain their moments and the tensor-wide Adam step. New rows have
zero moments and inherit that step (CustomAdam's original append contract).
This is not equivalent to giving each new row a fresh Adam step counter.
"""
    @torch.no_grad()
    def add_points(self, new_verts, raw_verts=False, parent_indices=None):
        if raw_verts:
            raise ValueError("Expected world/PCA coordinates.")
        new_verts = torch.as_tensor(new_verts, device=self.model.device, dtype=torch.float32)
        if new_verts.ndim != 2 or new_verts.shape[1] != 3 or not torch.isfinite(new_verts).all():
            raise ValueError("New vertex positions must be finite (N, 3).")
        if not len(new_verts):
            return
        if parent_indices is None:
            raise ValueError("v2 needs actual parent_indices. Apply the supplied densification patch.")
        parents = torch.as_tensor(parent_indices, device=self.model.device, dtype=torch.long)
        if parents.shape != (len(new_verts), 4) or (parents < 0).any() or (parents >= len(self.model)).any():
            raise ValueError("parent_indices must identify four OLD vertices per new point.")
        # Resolve before appending: exterior indices move when internal rows grow.
        inherited = self.model.vertex_features[parents].mean(dim=1)
        if not torch.isfinite(inherited).all():
            raise ValueError("Non-finite inherited properties; no vertices appended.")
        self.model.raw_features = self.optim.cat_tensors_to_optimizer(
            dict(raw_features=inherited))["raw_features"]
        self.model.contracted_vertices = self.vertex_optim.cat_tensors_to_optimizer(
            dict(contracted_vertices=new_verts))["contracted_vertices"]
        self.model.update_triangulation()
        print(f"[vertex_color_v2] Added {len(new_verts)} vertices; transfer=source_tet_raw_mean.")

    @torch.no_grad()
    def split(self, split_point, parent_indices=None, **kwargs):
        self.add_points(split_point, parent_indices=parent_indices)

"""Shared geometry/render/export contract for iNGP and vertex_color_v2.

Moved from ingp_color at 79251afc. In particular, retain its centroid rendering
anchor and legacy PLY offset convention; this ablation does not repair either
model's export/freeze conventions. CPU triangulation exists for unit tests.
"""
import numpy as np
import torch
from torch import nn

from models.base_model import BaseModel
from utils.model_util import activate_output, offset_normalize, pre_calc_cell_values
from utils.safe_math import safe_exp
from utils.topo_utils import fibonacci_spiral_on_sphere, tet_volumes


def activate_raw_properties(sigma, rgb, gradient, sh, density_offset):
    """sigma: log-density residual; RGB: residual about .5; gradient: unbounded."""
    density = safe_exp(sigma + density_offset)
    rgb = rgb.reshape(-1, 3) + 0.5
    gradient = gradient.reshape(-1, 1, 3)
    gradient = gradient / (gradient.square().sum(-1, keepdim=True) + 1).sqrt()
    return density, rgb, gradient, sh.half()


class RadianceMeshCore(BaseModel):
    @property
    def vertices(self):
        return torch.cat([self.contracted_vertices, self.ext_vertices])

    @property
    def num_int_verts(self):
        return self.contracted_vertices.shape[0]

    def get_circumcenters(self):
        return pre_calc_cell_values(self.vertices, self.indices)

    @classmethod
    def init_from_pcd(cls, point_cloud, cameras, device, max_sh_deg,
                      voxel_size=0.0, **kwargs):
        # Preserve the published implementation's geometry seed and recipe.
        # The paired launcher resets training RNGs AFTER model construction.
        import open3d as o3d
        torch.manual_seed(2)
        centers = torch.stack([c.camera_center.reshape(3) for c in cameras]).to(device)
        center = centers.mean(dim=0)
        scaling = torch.linalg.norm(centers - center.reshape(1, 3), dim=1, ord=torch.inf).max()
        print(f"Scene scaling: {scaling}. Center: {center}")
        vertices = torch.as_tensor(point_cloud.points).float().cpu()
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(vertices.numpy())
        if voxel_size > 0:
            cloud = cloud.voxel_down_sample(voxel_size=voxel_size)
        vertices = torch.as_tensor(np.asarray(cloud.points)).float()
        vertices = vertices + torch.randn(*vertices.shape) * 1e-3
        radius = torch.linalg.norm(vertices - center.cpu().reshape(1, 3), dim=1).max().item()
        ext_vertices = fibonacci_spiral_on_sphere(1000, radius, device="cpu") + center.cpu().reshape(1, 3)
        return cls(vertices.to(device), ext_vertices, center, scaling,
                   max_sh_deg=max_sh_deg, **kwargs)

    def get_cell_values(self, camera, mask=None, all_circumcenters=None, radii=None):
        indices = self.indices[mask] if mask is not None else self.indices
        vertices = self.vertices
        sh_dim = (self.max_sh_deg + 1)**2 - 1
        features = torch.empty((len(indices), self.feature_dim), device=self.device)
        shs = torch.empty((len(indices), sh_dim, 3), device=self.device)
        for start in range(0, len(indices), self.chunk_size):
            end = min(start + self.chunk_size, len(indices))
            _, density, rgb, grd, sh = self.compute_batch_features(
                vertices, indices, start, end, circumcenters=all_circumcenters)
            if self.ablate_gradient:
                grd = torch.zeros_like(grd)
            centroids = vertices[indices[start:end]].mean(dim=1)
            shs[start:end] = sh.reshape(-1, sh_dim, 3)
            features[start:end] = activate_output(
                camera.camera_center.to(self.device), density, rgb, grd,
                sh.reshape(-1, sh_dim, 3), indices[start:end], centroids,
                vertices.detach(), self.current_sh_deg, self.max_sh_deg)
        return shs, features

    def calc_tet_density(self):
        values = []
        vertices = self.vertices
        for start in range(0, len(self.indices), self.chunk_size):
            end = min(start + self.chunk_size, len(self.indices))
            _, density, _, _, _ = self.compute_batch_features(vertices, self.indices, start, end)
            values.append(density.reshape(-1))
        return torch.cat(values) if values else vertices.new_empty((0,))

    def compute_features(self, offset=False):
        vertices, indices = self.vertices, self.indices
        fields = [[] for _ in range(5)]
        for start in range(0, len(indices), self.chunk_size):
            end = min(start + self.chunk_size, len(indices))
            cc, density, rgb, grd, sh = self.compute_batch_features(vertices, indices, start, end)
            if offset:
                # Intentionally retain ingp_color's existing behavior: it returns
                # rgb, not the computed base_color_v0_raw. Do not fix only v2.
                _, grd = offset_normalize(rgb, grd, cc, vertices[indices[start:end]])
            for dest, value in zip(fields, (cc, density, rgb, grd, sh)):
                dest.append(value)
        if not len(indices):
            raise RuntimeError("No active tetrahedra to export or freeze.")
        return tuple(torch.cat(values) for values in fields)

    def sh_up(self):
        self.current_sh_deg = min(self.max_sh_deg, self.current_sh_deg + 1)

    @torch.no_grad()
    def update_triangulation(self, high_precision=False, density_threshold=0.0, alpha_threshold=0.0):
        vertices = self.vertices
        if not torch.isfinite(vertices).all():
            raise ValueError("Non-finite vertex positions before triangulation.")
        if vertices.is_cuda:
            torch.cuda.empty_cache()
        if high_precision or not vertices.is_cuda:
            from scipy.spatial import Delaunay
            indices_np = Delaunay(vertices.detach().cpu().numpy()).simplices.astype(np.int32)
        else:
            from gdel3d import Del
            indices_np, previous = Del(len(vertices)).compute(vertices.detach().cpu().double())
            if isinstance(indices_np, torch.Tensor):
                indices_np = indices_np.detach().cpu().numpy()
            else:
                indices_np = np.asarray(indices_np)
            del previous
        valid = ((indices_np >= 0) & (indices_np < len(vertices))).all(axis=1)
        indices = torch.as_tensor(indices_np[valid], device=vertices.device, dtype=torch.int32)
        reverse = tet_volumes(vertices[indices]) < 0
        indices[reverse] = indices[reverse][:, [1, 0, 2, 3]]
        self.indices = indices
        summary = dict(total=len(indices), density_threshold=float(density_threshold),
                       alpha_threshold=float(alpha_threshold))
        if density_threshold > 0 or alpha_threshold > 0:
            density = self.calc_tet_density()
            alpha = self.calc_tet_alpha(mode="min", density=density)
            self.mask = (density > density_threshold) | (alpha > alpha_threshold)
            self.empty_indices, self.indices = indices[~self.mask], indices[self.mask]
            summary.update(density_pass=int((density > density_threshold).sum()),
                           alpha_pass=int((alpha > alpha_threshold).sum()))
        else:
            self.empty_indices = indices[:0].clone()
        summary.update(kept=len(self.indices), culled=len(self.empty_indices))
        self.last_triangulation_summary = summary
        if not len(self.indices):
            raise RuntimeError("All tetrahedra were culled; inspect density/alpha diagnostics.")
        if vertices.is_cuda:
            torch.cuda.empty_cache()

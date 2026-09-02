"""CPU unit tests. Run: python -m unittest tests.test_vertex_color_model -v

Frozen-source tests execute the unchanged class/function definitions with pure
dependencies, omitting eager Slang imports. They are NOT CUDA renderer tests.
"""
import ast
import gc
import tempfile
import unittest
import json
import sys
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn
import tinyplypy
from plyfile import PlyData

from data.camera import Camera
from models.base_model import BaseModel
from models.vertex_color import Model, TetOptimizer
from utils import model_util, optim
from utils.safe_math import safe_exp, safe_log
from utils.topo_utils import tet_volumes
from utils.vertex_model_util import interpolate_new_vertices, get_expon_lr_func, SpikingLR
from utils.vertex_checkpoint import load_ablation_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def make_model(device="cpu", **kwargs):
    vertices = torch.tensor([[0., 0., 0.], [1., 0., 0.],
                             [0., 1., 0.], [0., 0., 1.]], device=device)
    exterior = torch.tensor([[-2., -2., -2.], [3., 0., 0.],
                             [0., 3., 0.], [0., 0., 3.]], device=device)
    indices = torch.tensor([[0, 1, 2, 3], [0, 5, 6, 7]], dtype=torch.int32, device=device)
    return Model(vertices, exterior, torch.zeros(3, device=device), 2.0,
                 indices=indices, **kwargs)


def isolated_frozen_namespace():
    # Execute the actual repository definitions, not a rewritten frozen model.
    path = ROOT / "models/frozen.py"
    tree = ast.parse(path.read_text())
    definitions = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    namespace = dict(vars(model_util))
    namespace.update(torch=torch, nn=nn, np=np, gc=gc, tinyplypy=tinyplypy,
                     BaseModel=BaseModel, Camera=Camera, Optional=Optional, Tuple=Tuple,
                     Path=Path, optim=optim, safe_exp=safe_exp, safe_log=safe_log,
                     get_expon_lr_func=get_expon_lr_func, SpikingLR=SpikingLR)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class VertexModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.model = make_model(chunk_size=1)
        self.camera = SimpleNamespace(camera_center=torch.tensor([2., 2., -3.]))

    def test_no_neural_parameters_and_shapes(self):
        self.assertEqual(set(dict(self.model.named_parameters())),
                         {"contracted_vertices", "raw_features", "ext_raw_features"})
        self.assertFalse(hasattr(self.model, "backbone"))
        self.assertEqual(self.model.raw_features.shape, (4, 52))
        sh, features = self.model.get_cell_values(self.camera)
        self.assertEqual(sh.shape, (2, 15, 3))
        self.assertEqual(features.shape, (2, 7))
        self.assertTrue(torch.isfinite(features).all())
        torch.testing.assert_close(features[:, 0], torch.full((2,), np.exp(-4), dtype=torch.float32))

    def test_initial_geometry_recipe_without_voxel_downsampling(self):
        cloud = SimpleNamespace(points=np.array([[0., 0., 0.], [1., 0., 0.],
                                                  [0., 1., 0.], [0., 0., 1.]]))
        cameras = [SimpleNamespace(camera_center=torch.tensor(p))
                   for p in ([2., 0., 0.], [0., 2., 0.], [0., 0., 2.])]
        model = Model.init_from_pcd(cloud, cameras, "cpu", max_sh_deg=3,
                                   current_sh_deg=1, voxel_size=0, dataset_path="ignored")
        torch.manual_seed(2)
        expected = torch.tensor(cloud.points).float() + torch.randn(4, 3) * 1e-3
        torch.testing.assert_close(model.contracted_vertices, expected)
        torch.testing.assert_close(model.center, torch.full((1, 3), 2/3))
        self.assertEqual(model.ext_vertices.shape, (1000, 3))
        radius = (expected - model.center).norm(dim=1).max()
        torch.testing.assert_close((model.ext_vertices - model.center).norm(dim=1),
                                   radius.expand(1000), atol=1e-6, rtol=1e-6)
        self.assertEqual(model.current_sh_deg, 1)

    def test_raw_average_activation_and_gradient_bound(self):
        with torch.no_grad():
            self.model.raw_features.copy_(torch.randn_like(self.model.raw_features))
        raw = self.model.raw_features.mean(dim=0, keepdim=True)
        cc, density, rgb, grd, sh = self.model.compute_batch_features(
            self.model.vertices, self.model.indices, 0, 1)
        torch.testing.assert_close(density, safe_exp(raw[:, :1] - 4))
        torch.testing.assert_close(rgb, raw[:, 1:4] + .5)
        expected = raw[:, 4:7] / (1 + raw[:, 4:7].square().sum(-1, keepdim=True)).sqrt()
        torch.testing.assert_close(grd[:, 0], expected)
        self.assertTrue((grd.norm(dim=-1) < 1).all())
        torch.testing.assert_close(sh, raw[:, 7:].reshape(1, 15, 3).half())

    def test_property_and_geometry_gradients(self):
        with torch.no_grad():
            self.model.raw_features[:, 4:7] = .1
            self.model.ext_raw_features[:, 4:7] = .1
        sh, features = self.model.get_cell_values(self.camera)
        (features.sum() + sh.sum()).backward()
        for param in (self.model.raw_features, self.model.ext_raw_features,
                      self.model.contracted_vertices):
            self.assertIsNotNone(param.grad)
            self.assertTrue(torch.isfinite(param.grad).all())
            self.assertGreater(param.grad.abs().sum().item(), 0)
        for lo, hi in ((0, 1), (1, 4), (4, 7), (7, 52)):
            self.assertGreater(self.model.raw_features.grad[:, lo:hi].abs().sum().item(), 0)

    def test_mask_chunk_and_empty(self):
        with torch.no_grad():
            self.model.raw_features.normal_(std=.1)
        full = self.model.get_cell_values(self.camera)
        mask = torch.tensor([False, True])
        selected = self.model.get_cell_values(self.camera, mask, self.model.get_circumcenters()[mask])
        for actual, expected in zip(selected, full):
            torch.testing.assert_close(actual, expected[mask])
        self.model.chunk_size = 100
        for actual, expected in zip(self.model.get_cell_values(self.camera), full):
            torch.testing.assert_close(actual, expected)
        empty = self.model.get_cell_values(self.camera, torch.zeros(2, dtype=torch.bool))
        self.assertEqual(empty[1].shape, (0, 7))

    def test_sh_degree_and_gradient_ablation(self):
        model = make_model(current_sh_deg=0, ablate_gradient=True)
        self.assertEqual(model.current_sh_deg, 0)
        model.sh_up()
        self.assertEqual(model.current_sh_deg, 1)
        with torch.no_grad():
            model.raw_features[:, 4:7] = 10
        self.assertEqual(model.compute_batch_features(model.vertices, model.indices, 0, 2)[3].abs().sum(), 0)
        # Baking reads compute_batch_features, so gradient ablation survives baking.
        with self.assertRaises(ValueError):
            make_model(max_sh_deg=0)

    def test_transfer_affine_field_and_outside(self):
        vertices = self.model.contracted_vertices.detach()
        raw = torch.cat((vertices, 2 * vertices + 3), dim=1)
        query = torch.tensor([[.1, .2, .3], [.25, .25, .25], [10., 0., 0.]])
        actual, count = interpolate_new_vertices(vertices, raw, query)
        torch.testing.assert_close(actual[:2], torch.cat((query[:2], 2 * query[:2] + 3), dim=1))
        torch.testing.assert_close(actual[2], raw[1])
        self.assertEqual(count, 1)

    def test_transfer_empty_and_reject_nonfinite(self):
        values, count = interpolate_new_vertices(self.model.vertices, self.model.vertex_features,
                                                  torch.empty(0, 3))
        self.assertEqual(values.shape, (0, 52))
        self.assertEqual(count, 0)
        with self.assertRaisesRegex(ValueError, "NaN/Inf"):
            interpolate_new_vertices(self.model.vertices, self.model.vertex_features,
                                     torch.tensor([[float("nan"), 0., 0.]]))

    def test_add_before_first_optimizer_step(self):
        optimizer = TetOptimizer(self.model)
        exterior = self.model.ext_raw_features
        optimizer.split(torch.tensor([[.2, .2, .2]]))
        self.assertEqual(self.model.raw_features.shape[0], 5)
        self.assertEqual(self.model.num_int_verts, 5)
        self.assertIs(exterior, self.model.ext_raw_features)
        self.model._validate_indices(self.model.indices)

    def test_add_preserves_adam_moments_and_exterior_alignment(self):
        optimizer = TetOptimizer(self.model)
        (self.model.raw_features.sum() + self.model.ext_raw_features.sum()
         + self.model.contracted_vertices.sum()).backward()
        optimizer.main_step()
        optimizer.vertex_optim.step()
        optimizer.main_zero_grad()
        optimizer.vertex_optim.zero_grad()
        old_features = self.model.raw_features.detach().clone()
        exterior = self.model.ext_raw_features.detach().clone()
        moments = optimizer.optim.get_state_by_name("raw_features")["exp_avg"].clone()
        old_state_step = optimizer.optim.get_state_by_name("raw_features")["step"].clone()
        optimizer.split(torch.tensor([[.2, .2, .2]]))
        state = optimizer.optim.get_state_by_name("raw_features")
        torch.testing.assert_close(state["exp_avg"][:4], moments)
        self.assertEqual(state["exp_avg"][4:].abs().sum(), 0)
        self.assertEqual(state["exp_avg_sq"][4:].abs().sum(), 0)
        torch.testing.assert_close(state["step"], old_state_step)
        torch.testing.assert_close(self.model.raw_features[:4], old_features)
        torch.testing.assert_close(self.model.vertex_features[5:], exterior)
        (self.model.raw_features.sum() + self.model.ext_raw_features.sum()).backward()
        optimizer.main_step()  # resized optimizer must remain usable

    def test_prune_internal_attributes_and_moments(self):
        optimizer = TetOptimizer(self.model)
        self.model.raw_features.sum().backward()
        optimizer.main_step()
        keep = torch.tensor([True, False, True, True])
        saved = self.model.raw_features.detach().clone()
        moments = optimizer.optim.get_state_by_name("raw_features")["exp_avg"].clone()
        optimizer.remove_points(keep)
        torch.testing.assert_close(self.model.raw_features, saved[keep])
        torch.testing.assert_close(optimizer.optim.get_state_by_name("raw_features")["exp_avg"], moments[keep])
        self.model._validate_indices(self.model.indices)

    def test_bad_split_does_not_mutate(self):
        optimizer = TetOptimizer(self.model)
        with self.assertRaises(ValueError):
            optimizer.split(torch.tensor([[float("inf"), 0., 0.]]))
        self.assertEqual(self.model.num_int_verts, 4)

    def test_triangulation_preserves_vertex_fields(self):
        with torch.no_grad():
            self.model.raw_features.normal_()
        saved = self.model.raw_features.clone()
        self.model.update_triangulation(high_precision=True)
        torch.testing.assert_close(self.model.raw_features, saved)
        self.assertEqual(self.model.indices.dtype, torch.int32)
        self.assertTrue((tet_volumes(self.model.vertices[self.model.indices.long()]) >= 0).all())
        torch.testing.assert_close(self.model.calc_tet_density(),
            self.model.compute_features()[1].reshape(-1))
        self.assertTrue(torch.isfinite(self.model.calc_tet_alpha()).all())

    def test_checkpoint_round_trip_including_sh_stage(self):
        self.model.current_sh_deg = 1
        with torch.no_grad():
            self.model.raw_features.normal_(std=.1)
        expected = self.model.get_cell_values(self.camera)
        with tempfile.TemporaryDirectory() as directory:
            torch.save(self.model.state_dict(), Path(directory) / "ckpt.pth")
            restored = Model.load_ckpt(directory, "cpu")
        self.assertEqual(restored.current_sh_deg, 1)
        self.assertEqual(restored.min_t, self.model.min_t)
        for actual, value in zip(restored.get_cell_values(self.camera), expected):
            torch.testing.assert_close(actual, value)

    def test_checkpoint_after_insertion(self):
        TetOptimizer(self.model).split(torch.tensor([[.2, .2, .2]]))
        with tempfile.TemporaryDirectory() as directory:
            torch.save(self.model.state_dict(), Path(directory) / "ckpt.pth")
            restored = load_ablation_checkpoint(directory, "cpu")
        self.assertEqual(restored.num_int_verts, 5)
        torch.testing.assert_close(restored.vertex_features, self.model.vertex_features)
        torch.testing.assert_close(restored.indices, self.model.indices)

    def test_frozen_checkpoint_loader_against_original_class(self):
        namespace = isolated_frozen_namespace()
        frozen = namespace["bake_from_model"](self.model, torch.ones(2, dtype=torch.bool))
        with tempfile.TemporaryDirectory() as directory:
            state = frozen.state_dict()
            state["empty_indices"] = frozen.indices[:0]
            torch.save(state, Path(directory) / "ckpt.pth")
            (Path(directory) / "config.json").write_text(json.dumps({"min_t": .37}))
            module = SimpleNamespace(FrozenTetModel=namespace["FrozenTetModel"])
            with patch.dict(sys.modules, {"models.frozen": module}):
                restored = load_ablation_checkpoint(directory, "cpu")
        self.assertTrue(restored.frozen)
        self.assertEqual(restored.min_t, .37)
        self.assertEqual(len(restored.empty_indices), 0)
        torch.testing.assert_close(restored.get_cell_values(self.camera)[1],
                                   frozen.get_cell_values(self.camera)[1])

    def test_threshold_culling_uses_current_vertex_densities(self):
        with torch.no_grad():
            self.model.raw_features[:, 0] = 3.0
            self.model.ext_raw_features[:, 0] = -10.0
        self.model.update_triangulation()
        all_indices = self.model.indices.clone()
        density = self.model.calc_tet_density()
        alpha = self.model.calc_tet_alpha(density=density)
        keep = (density > .05) | (alpha > .05)
        self.model.update_triangulation(density_threshold=.05, alpha_threshold=.05)
        torch.testing.assert_close(self.model.indices, all_indices[keep])
        torch.testing.assert_close(self.model.empty_indices, all_indices[~keep])

    def test_pruning_exterior_is_rejected(self):
        keep = torch.ones(len(self.model), dtype=torch.bool)
        keep[-1] = False
        with self.assertRaisesRegex(ValueError, "exterior"):
            TetOptimizer(self.model).remove_points(keep)

    def test_ply_export_uses_existing_schema_and_v0_color(self):
        with torch.no_grad():
            self.model.raw_features[:, 4:7] = .2
        expected = self.model.compute_features(offset=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.ply"
            self.model.save2ply(path)
            ply = PlyData.read(path)
            self.assertEqual(len(ply["vertex"]), len(self.model))
            self.assertEqual(len(ply["tetrahedron"]), len(self.model.indices))
            rgb_r = np.asarray(ply["tetrahedron"]["sh_0_r"]) * .28209479177387814 + .5
            np.testing.assert_allclose(rgb_r, expected[2][:, 0].detach().numpy(), atol=1e-6)

    def test_schedule_equations_match_original_source(self):
        path = ROOT / "utils/train_util.py"
        tree = ast.parse(path.read_text())
        nodes = [n for n in tree.body if getattr(n, "name", "") in {"get_expon_lr_func", "SpikingLR"}]
        ns = {"np": np}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
        baseline = ns["get_expon_lr_func"](.001, .0001, 100, .001, 18000)
        ours = get_expon_lr_func(.001, .0001, 100, .001, 18000)
        baseline_spike = ns["SpikingLR"](500, 18000, baseline, 2000, 500, 16000, .001, .001)
        ours_spike = SpikingLR(500, 18000, ours, 2000, 500, 16000, .001, .001)
        for step in (-1, 0, 10, 100, 1999, 2000, 2001, 2500, 15999, 16001, 18000, 30000):
            self.assertAlmostEqual(baseline_spike(step), ours_spike(step), places=14)
        optimizer = TetOptimizer(self.model)
        optimizer.update_learning_rate(2500)
        self.assertEqual(optimizer.optim.param_groups[0]["lr"], optimizer.encoder_scheduler_args(2500))
        self.assertEqual(optimizer.regularizer({}, lambda_weight_decay=1), 0)
        self.assertIsNone(optimizer.sh_optim)

    def test_original_frozen_source_receives_exact_activated_fields(self):
        namespace = isolated_frozen_namespace()
        with torch.no_grad():
            self.model.raw_features.normal_(std=.1)
        expected = tuple(x.detach().clone() for x in self.model.compute_features()[1:])
        frozen = namespace["bake_from_model"](self.model, torch.ones(2, dtype=torch.bool), chunk_size=1)
        for value, wanted in zip((frozen.density, frozen.rgb, frozen.gradient, frozen.sh), expected):
            torch.testing.assert_close(value, wanted)
        self.assertTrue(frozen.frozen)
        self.assertFalse(frozen.vertices.requires_grad)
        sh, features = frozen.get_cell_values(self.camera)
        (features.sum() + sh.float().sum()).backward()
        self.assertTrue(torch.isfinite(frozen.rgb.grad).all())

    def test_original_freeze_optimizer_continues_training(self):
        namespace = isolated_frozen_namespace()
        args = SimpleNamespace(as_dict=lambda: dict(freeze_start=10, iterations=20))
        before = self.model.get_cell_values(self.camera)[1].detach()
        frozen, optimizer = namespace["freeze_model"](
            self.model, torch.ones(2, dtype=torch.bool), args, chunk_size=1)
        # Zero gradient field: differing baseline/frozen anchors cannot change RGB.
        torch.testing.assert_close(frozen.get_cell_values(self.camera)[1], before)
        geometry = frozen.vertices.detach().clone()
        sh, features = frozen.get_cell_values(self.camera)
        (features.sum() + sh.float().square().mean()).backward()
        optimizer.main_step()
        optimizer.main_zero_grad()
        optimizer.sh_optim.step()
        optimizer.sh_optim.zero_grad()
        optimizer.update_learning_rate(11)
        torch.testing.assert_close(frozen.vertices, geometry)
        self.assertIsNone(optimizer.vertex_optim)


if __name__ == "__main__":
    unittest.main()

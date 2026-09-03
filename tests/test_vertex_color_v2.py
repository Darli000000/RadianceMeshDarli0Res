"""CPU semantics/regression tests, not a substitute for the real CUDA tests."""
import ast
import copy
import json
import io
import math
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from models.vertex_color_v2 import Model, TetOptimizer
from models.radiance_core import RadianceMeshCore, activate_raw_properties
from utils import hashgrid, model_util, optim
from utils.args import Args
from utils.ablation_protocol import (Session, domain_seed, isolated_observation, parse_training_args)
from utils.safe_math import safe_exp, safe_div, safe_sqrt
from utils.vertex_checkpoint import load_ablation_checkpoint

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "tests/fixtures/ingp_color_79251afc.py"


def source_definitions(path, names, namespace):
    """Execute exact selected definitions, excluding eager CUDA/Slang imports."""
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def head_class(path):
    namespace = dict(torch=torch, nn=nn, np=np, hashgrid=hashgrid,
                     init_linear=model_util.init_linear, safe_exp=safe_exp,
                     safe_div=safe_div, safe_sqrt=safe_sqrt,
                     activate_raw_properties=activate_raw_properties)
    return source_definitions(path, {"approx_erf", "iNGPDW"}, namespace)["iNGPDW"]


def old_method(name):
    cls = next(n for n in ast.parse(REFERENCE.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "Model")
    node = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name))
    node.decorator_list = []
    namespace = dict(vars(model_util), torch=torch, np=np, Camera=object)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(REFERENCE), "exec"), namespace)
    return namespace[name]


def make_model(device="cpu", **kwargs):
    v = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], device=device)
    ext = torch.tensor([[-2., -2., -2.], [3., 0., 0.], [0., 3., 0.], [0., 0., 3.]], device=device)
    ids = torch.tensor([[0, 1, 2, 3], [0, 5, 6, 7]], device=device, dtype=torch.int32)
    return Model(v, ext, torch.zeros(3, device=device), 2., indices=ids, **kwargs)


class VertexV2Tests(unittest.TestCase):
    def test_shared_methods_are_inherited(self):
        for name in ("init_from_pcd", "get_cell_values", "compute_features", "calc_tet_density", "update_triangulation"):
            self.assertNotIn(name, Model.__dict__)
        cls = next(n for n in ast.parse((ROOT / "models/ingp_color.py").read_text()).body
                   if isinstance(n, ast.ClassDef) and n.name == "Model")
        self.assertEqual(cls.bases[0].id, "RadianceMeshCore")
        self.assertFalse({"init_from_pcd", "get_cell_values", "update_triangulation"} &
                         {n.name for n in cls.body if isinstance(n, ast.FunctionDef)})

    def test_initializer_and_rejected_options(self):
        model = make_model()
        self.assertFalse(hasattr(model, "backbone"))
        self.assertEqual(set(dict(model.named_parameters())), {"contracted_vertices", "raw_features", "ext_raw_features"})
        cc, d, rgb, g, sh = model.compute_features()
        torch.testing.assert_close(d, torch.full((2, 1), math.exp(-4)))
        torch.testing.assert_close(rgb, torch.full((2, 3), .5))
        self.assertEqual(int(torch.count_nonzero(g)), 0)
        self.assertEqual(int(torch.count_nonzero(sh)), 0)
        for options in ({"c_init": .8}, {"k_samples": 2}, {"percent_alpha": .02}):
            with self.assertRaises(ValueError):
                make_model(**options)

    def test_real_ungpu_network_constant_outputs_match(self):
        # The real hashgrid/head implementation on CPU, not a fake zero network.
        network = head_class(ROOT / "models/ingp_color.py")(
            sh_dim=45, L=2, hashmap_dim=2, hidden_dim=8, base_resolution=2,
            log2_hashmap_size=5, g_init=0, s_init=0, d_init=0, c_init=0)
        actual = network(torch.rand(2, 3), torch.ones(2) * .1)
        expected = make_model().compute_features()[1:]
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a.reshape_as(b), b)
        # Only output layers are zero: the whole network was NOT zeroed.
        self.assertGreater(network.color_net[0].weight.abs().sum().item(), 0)

    def test_activation_refactor_preserves_random_network_and_gradients(self):
        kwargs = dict(sh_dim=45, L=2, hashmap_dim=2, hidden_dim=8,
                      base_resolution=2, log2_hashmap_size=5)
        torch.manual_seed(7)
        old = head_class(REFERENCE)(**kwargs)
        torch.manual_seed(7)
        new = head_class(ROOT / "models/ingp_color.py")(**kwargs)
        x, radius = torch.rand(5, 3), torch.rand(5) * .1
        before, after = old(x, radius), new(x, radius)
        for a, b in zip(before, after):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        sum(v.float().sum() for v in before).backward()
        sum(v.float().sum() for v in after).backward()
        for a, b in zip(old.parameters(), new.parameters()):
            if a.grad is None:
                self.assertIsNone(b.grad)
            else:
                torch.testing.assert_close(a.grad, b.grad)

    def test_legacy_render_density_and_export_regression(self):
        model = make_model(chunk_size=1)
        with torch.no_grad():
            model.raw_features.normal_(0, .1)
            model.ext_raw_features.normal_(0, .1)
        camera = SimpleNamespace(camera_center=torch.tensor([2., 2., -3.]))
        for disabled in (False, True):
            model.ablate_gradient = disabled
            for a, b in zip(old_method("get_cell_values")(model, camera), model.get_cell_values(camera)):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        for offset in (False, True):
            for a, b in zip(old_method("compute_features")(model, offset), model.compute_features(offset)):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(old_method("calc_tet_density")(model), model.calc_tet_density())

    def test_all_property_channels_receive_gradients(self):
        model = make_model()
        with torch.no_grad():
            model.raw_features[:, 4:7] = .1
        sh, values = model.get_cell_values(SimpleNamespace(camera_center=torch.tensor([2., 2., -3.])))
        (sh.sum() + values.sum()).backward()
        for lo, hi in ((0, 1), (1, 4), (4, 7), (7, 52)):
            self.assertGreater(model.raw_features.grad[:, lo:hi].abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(model.contracted_vertices.grad).all())

    def test_raw_mean_is_not_mean_activated_density(self):
        model = make_model()
        with torch.no_grad():
            model.raw_features[:, 0] = torch.tensor([0., 0., 0., 4.])
        d = model.compute_features()[1][0]
        torch.testing.assert_close(d, torch.tensor([math.exp(-3)]))
        self.assertFalse(torch.isclose(d, torch.exp(model.raw_features[:, 0] - 4).mean()).all())

    def test_parent_transfer_and_adam_state_preservation(self):
        model = make_model()
        optimizer = TetOptimizer(model)
        with torch.no_grad():
            model.raw_features.normal_(0, .2)
            model.ext_raw_features.normal_(0, .2)
        (model.raw_features.square().sum() + model.ext_raw_features.square().sum() + model.contracted_vertices.square().sum()).backward()
        optimizer.main_step()
        optimizer.vertex_optim.step()
        optimizer.main_zero_grad()
        optimizer.vertex_optim.zero_grad()
        parents = model.indices[[1]].clone()  # includes exterior vertices
        expected = model.vertex_features[parents.long()].mean(dim=1).detach().clone()
        old_ext, old_rows = model.ext_raw_features.detach().clone(), model.raw_features.detach().clone()
        state = optimizer.optim.get_state_by_name("raw_features")
        moments, step = state["exp_avg"].clone(), state["step"].clone()
        optimizer.split(torch.tensor([[.2, .2, .2]]), parent_indices=parents)
        torch.testing.assert_close(model.raw_features[:-1], old_rows)
        torch.testing.assert_close(model.raw_features[-1:], expected)
        torch.testing.assert_close(model.ext_raw_features, old_ext)
        state = optimizer.optim.get_state_by_name("raw_features")
        torch.testing.assert_close(state["exp_avg"][:-1], moments)
        self.assertEqual(float(state["exp_avg"][-1].abs().sum()), 0)
        torch.testing.assert_close(state["step"], step)
        self.assertEqual(len(model.ext_vertices), 4)

    def test_missing_parent_fails_before_mutation(self):
        model = make_model()
        optimizer = TetOptimizer(model)
        before = model.vertices.detach().clone()
        with self.assertRaises(ValueError):
            optimizer.split(torch.tensor([[.1, .1, .1]]))
        torch.testing.assert_close(model.vertices, before)

    def test_transfer_is_explicitly_not_subdivision_invariant(self):
        old_corners = torch.tensor([0., 0., 0., 4.])
        inherited = old_corners.mean()
        child = torch.cat([old_corners[:3], inherited.reshape(1)]).mean()
        self.assertEqual(float(inherited), 1.)
        self.assertEqual(float(child), .25)

    def test_threshold_is_same_or_rule_and_vertices_not_deleted(self):
        model = make_model()
        model.update_triangulation()
        before = model.vertices.detach().clone()
        # Spatially heterogeneous density to ensure both keep/drop outcomes.
        with torch.no_grad():
            model.raw_features[:, 0] = 5
            model.ext_raw_features[:, 0] = -5
        model.update_triangulation(density_threshold=.1, alpha_threshold=.1)
        summary = model.last_triangulation_summary
        self.assertGreater(summary["culled"], 0)
        self.assertGreater(summary["kept"], 0)
        kept = model.indices.clone()
        model.indices = torch.cat([kept, model.empty_indices])
        d = model.calc_tet_density()
        a = model.calc_tet_alpha(density=d)
        expected = (d > .1) | (a > .1)
        self.assertTrue(expected[:len(kept)].all())
        self.assertFalse(expected[len(kept):].any())
        torch.testing.assert_close(model.vertices, before)

    def test_lr_equations_match_baseline_all_30k_steps(self):
        ns = source_definitions(ROOT / "utils/train_util.py", {"get_expon_lr_func", "SpikingLR"}, dict(np=np, math=math))
        ns.update(torch=torch, optim=optim, Model=object)
        baseline_type = source_definitions(REFERENCE, {"TetOptimizer"}, ns)["TetOptimizer"]
        baseline_model = SimpleNamespace(scene_scaling=torch.tensor(2.), contracted_vertices=nn.Parameter(torch.zeros(4, 3)),
            backbone=SimpleNamespace(**{name: nn.Linear(2, 2) for name in ("encoding", "density_net", "color_net", "gradient_net", "sh_net")}))
        config = dict(network_lr=.001, final_network_lr=.0001, vertices_lr=.0001, final_vertices_lr=.000001,
                      lr_delay=0, vert_lr_delay=0, vertices_lr_delay_multi=1e-8, freeze_start=18000,
                      spike_duration=500, densify_interval=500, densify_end=16000)
        old, new = baseline_type(baseline_model, **config), TetOptimizer(make_model(), **config)
        for name in ("net_scheduler_args", "vertex_scheduler_args"):
            a = np.array([getattr(old, name)(i) for i in range(30000)])
            b = np.array([getattr(new, name)(i) for i in range(30000)])
            np.testing.assert_allclose(a, b, rtol=1e-14, atol=0)

    def test_v2_checkpoint_roundtrip_and_old_version_rejection(self):
        model = make_model()
        with torch.no_grad():
            model.raw_features.normal_()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ckpt.pth"
            torch.save(model.state_dict(), path)
            restored = load_ablation_checkpoint(path, "cpu")
            self.assertIsInstance(restored, Model)
            for a, b in zip(model.compute_features(), restored.compute_features()):
                torch.testing.assert_close(a, b)
            state = model.get_extra_state()
            state["format"] = "vertex_color_v1"
            with self.assertRaises(ValueError):
                model.set_extra_state(state)


class ProtocolTests(unittest.TestCase):
    def settings(self, **kwargs):
        return dict(model="vertex_v2", protocol="matched_constant", seed=2, config=None,
                    probes=[0], save_checkpoints=False, **kwargs)

    def test_observer_preserves_all_rngs_and_modes(self):
        model = make_model()
        random.seed(5)
        np.random.seed(5)
        torch.manual_seed(5)
        py, np_state, state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        with isolated_observation(model):
            self.assertFalse(model.training)
            random.random(); np.random.rand(); torch.rand(123)
        self.assertTrue(model.training)
        self.assertEqual(py, random.getstate())
        np.testing.assert_array_equal(np_state[1], np.random.get_state()[1])
        torch.testing.assert_close(state, torch.get_rng_state(), rtol=0, atol=0)

    def test_domain_reseeding_is_independent_of_model_rng_consumption(self):
        session = Session(self.settings())
        session.seed_domain(500, "train_render")
        a = torch.rand(7)
        torch.rand(10000)
        session.seed_domain(500, "train_render")
        torch.testing.assert_close(a, torch.rand(7), rtol=0, atol=0)
        self.assertNotEqual(domain_seed(2, 500, "train_render"), domain_seed(2, 500, "densification"))

    def test_effective_overrides_boolean_flags_and_fresh_output(self):
        with tempfile.TemporaryDirectory() as directory:
            args = Args()
            for k, v in dict(ckpt="", output_path=Path(directory) / "new", eval=True,
                             density_offset=-4., k_samples=1, percent_alpha=0.,
                             g_init=1., s_init=.0001, d_init=.1, c_init=.8,
                             lambda_weight_decay=1.).items():
                setattr(args, k, v)
            with patch.dict(os.environ, RM_ABLATION_SETTINGS=json.dumps(self.settings())):
                effective, _ = parse_training_args(args, ["--eval"])
                self.assertTrue(effective.eval)
                self.assertEqual(effective.c_init, 0.)
                self.assertEqual(effective.lambda_weight_decay, 0.)
                effective, _ = parse_training_args(args, ["--no-eval"])
                self.assertFalse(effective.eval)
                args.output_path.mkdir()
                (args.output_path / "existing.txt").write_text("keep")
                with patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit):
                    parse_training_args(args, [])

    def test_probe_measures_change_without_training(self):
        model = make_model()
        camera = SimpleNamespace(original_image=torch.zeros(3, 2, 2))
        def render(camera, model, **kwargs):
            value = model.raw_features.mean()
            return {"render": value.expand(3, 2, 2)}
        with tempfile.TemporaryDirectory() as directory:
            args = Args()
            args.output_path, args.density_threshold = Path(directory), .1
            session = Session(self.settings())
            with session.event("test", 0, lambda: model, camera, render, args):
                with torch.no_grad():
                    model.raw_features.fill_(.1)
            result = json.loads((Path(directory) / "diagnostics/00000000_test/event.json").read_text())
            self.assertAlmostEqual(result["image_mean_absolute_change"], .1, places=5)
            self.assertIsNone(model.raw_features.grad)
            self.assertTrue(model.training)

    def test_full_config_manifest_and_pair_comparison(self):
        from scripts.compare_ablation_v2 import compare
        tree = ast.parse((ROOT / "train.py").read_text())
        cutoff = next(n.lineno for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
                      and getattr(n.value.func, "id", None) == "parse_training_args")
        assignments = [n for n in tree.body if n.lineno < cutoff and isinstance(n, ast.Assign) and len(n.targets) == 1 and (
            isinstance(n.targets[0], ast.Attribute) and isinstance(n.targets[0].value, ast.Name)
            and n.targets[0].value.id == "args")]
        defaults = Args()
        exec(compile(ast.Module(body=assignments, type_ignores=[]), "defaults", "exec"), dict(args=defaults, Path=Path))
        with tempfile.TemporaryDirectory() as directory, patch("sys.stdout", new=io.StringIO()):
            paths = []
            for name in ("ingp", "vertex_v2"):
                settings = self.settings()
                settings["model"], settings["config"] = name, str(ROOT / "configs/bonsai_ablation_v2.json")
                output = Path(directory) / name
                with patch.dict(os.environ, RM_ABLATION_SETTINGS=json.dumps(settings)):
                    effective, session = parse_training_args(defaults, ["--output_path", str(output)])
                self.assertTrue(effective.eval)
                self.assertEqual(effective.image_folder, "images_4")
                output.mkdir()
                session.after_init(make_model(), effective)
                session.log_sample(0, 7)
                paths.append(output)
            self.assertEqual(compare(*paths), [])


if __name__ == "__main__":
    unittest.main()

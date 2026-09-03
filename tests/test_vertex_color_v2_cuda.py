"""Real CUDA paired render -> backward -> densify -> cull -> freeze -> reload.

Requires the user's already-installed repository dependencies. Uses tile_size=4.
Nonzero-gradient freeze changes are measured, not assumed to be zero: the shared
legacy freeze anchor convention is intentionally not repaired in this ablation.
"""
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from data.camera import Camera
from models.vertex_color_v2 import Model, TetOptimizer
from utils.args import Args
from utils.topo_utils import fibonacci_spiral_on_sphere
from utils.vertex_checkpoint import load_ablation_checkpoint


@unittest.skipUnless(torch.cuda.is_available(), "Requires an NVIDIA CUDA GPU")
class VertexV2CudaTests(unittest.TestCase):
    def config(self):
        return dict(max_sh_deg=3, current_sh_deg=3, density_offset=-4.,
                    g_init=0., s_init=0., d_init=0., c_init=0., k_samples=1,
                    percent_alpha=0., L=2, hashmap_dim=2, hidden_dim=16,
                    log2_hashmap_size=8, base_resolution=2, per_level_scale=2,
                    ablate_circumsphere=True, ablate_gradient=False)

    def test_actual_pcd_geometry_and_initial_render_match(self):
        from models.ingp_color import Model as InGP
        from utils.train_util import render
        points = np.random.default_rng(3).normal(size=(32, 3)).astype(np.float32) * .2
        cloud = SimpleNamespace(points=points)
        cameras = [SimpleNamespace(camera_center=torch.tensor(p, device="cuda"))
                   for p in ([2., 0., 0.], [0., 2., 0.], [0., 0., 2.])]
        baseline = InGP.init_from_pcd(cloud, cameras, "cuda", voxel_size=.01, **self.config())
        vertex = Model.init_from_pcd(cloud, cameras, "cuda", voxel_size=.01, **self.config())
        for key in ("contracted_vertices", "ext_vertices", "center", "scene_scaling"):
            torch.testing.assert_close(getattr(baseline, key), getattr(vertex, key), rtol=0, atol=0)
        self.assertEqual(len(vertex.ext_vertices), 1000)
        # Use identical connectivity to isolate the property backend from gDel ordering.
        vertex.indices = baseline.indices.clone()
        camera = Camera(0, np.eye(3), np.array([0., 0., 3.]), 1.2, 1.2,
                        torch.zeros(3, 64, 64), image_name="synthetic", data_device="cpu")
        baseline.eval(); vertex.eval()
        with torch.no_grad():
            a = render(camera, baseline, tile_size=4, min_t=.1)["render"]
            b = render(camera, vertex, tile_size=4, min_t=.1)["render"]
        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-5)
        # Verify actual compiled/checkpointed iNGP learns from this zero-output init.
        baseline.train()
        render(camera, baseline, tile_size=4, min_t=.1)["render"].square().mean().backward()
        grad = baseline.backbone.color_net[-1].bias.grad
        self.assertIsNotNone(grad)
        self.assertGreater(grad.abs().sum().item(), 0)
        with tempfile.TemporaryDirectory() as directory:
            state = baseline.state_dict()
            state["indices"], state["empty_indices"] = baseline.indices, baseline.empty_indices
            torch.save(state, Path(directory) / "ckpt.pth")
            (Path(directory) / "config.json").write_text(json.dumps(dict(self.config(), min_t=.1, sh_interval=0)))
            restored = load_ablation_checkpoint(directory)
            baseline.eval(); restored.eval()
            with torch.no_grad():
                torch.testing.assert_close(render(camera, baseline, tile_size=4, min_t=.1)["render"],
                                           render(camera, restored, tile_size=4, min_t=.1)["render"])

    def test_real_densification_culling_freezing_and_reload(self):
        from utils.train_util import render
        from utils.densification import collect_render_stats, apply_densification
        from models.frozen import freeze_model
        torch.manual_seed(2)
        vertices = torch.rand((24, 3), device="cuda") * .4
        exterior = fibonacci_spiral_on_sphere(50, 2., device="cuda")
        model = Model(vertices, exterior, torch.zeros(3, device="cuda"), 1., **self.config())
        camera = Camera(0, np.eye(3), np.array([0., 0., 3.]), 1.2, 1.2,
                        torch.zeros(3, 64, 64), image_name="synthetic", data_device="cpu")
        optimizer = TetOptimizer(model, freeze_start=10)
        # Non-constant attributes exercise all channels, not only zero SH/gradients.
        with torch.no_grad():
            model.raw_features.normal_(0, .05)
            model.ext_raw_features.normal_(0, .05)
            model.raw_features[:, 0] += 7
            model.ext_raw_features[:, 0] -= 5
        image = render(camera, model, tile_size=4, min_t=.1)["render"]
        image.square().mean().backward()
        self.assertGreater(model.raw_features.grad.abs().sum().item(), 0)
        optimizer.main_step(); optimizer.main_zero_grad()
        optimizer.vertex_optim.step(); optimizer.vertex_optim.zero_grad()
        args = Args()
        for key, value in dict(tile_size=4, min_tet_count=1, clone_min_contrib=.001,
                               split_min_contrib=.001, within_thresh=1., total_thresh=100.,
                               output_path=None, freeze_start=10, iterations=20, min_t=.1).items():
            setattr(args, key, value)
        model.eval()
        stats = collect_render_stats([camera], model, args, torch.device("cuda"))
        model.train()
        stats.top_ssim.zero_(); stats.top_ssim[0] = 10
        stats.total_var_moments.zero_(); stats.peak_contrib.fill_(1)
        parent = model.indices[[0]].clone()
        expected = model.vertex_features[parent.long()].mean(1).detach().clone()
        center = model.vertices[parent.long()].mean((0, 1)).detach()
        dx, dy = center.new_tensor([.05, 0, 0]), center.new_tensor([0, .05, 0])
        stats.within_var_rays[0, 0] = torch.cat([center - dx, center + dx])
        stats.within_var_rays[0, 1] = torch.cat([center - dy, center + dy])
        old_n = model.num_int_verts
        apply_densification(stats, model, optimizer, args, 1, torch.device("cuda"), camera, None, target_addition=1)
        self.assertEqual(model.num_int_verts, old_n + 1)
        torch.testing.assert_close(model.raw_features[-1:], expected)
        optimizer.update_triangulation(density_threshold=.1, alpha_threshold=.1)
        self.assertGreater(len(model.empty_indices), 0)
        self.assertGreater(len(model.indices), 0)
        self.assertEqual(model.num_int_verts, old_n + 1)
        before = render(camera, model, tile_size=4, min_t=.1)["render"].detach()
        expected_fields = tuple(v.detach().clone() for v in model.compute_features()[1:])
        frozen, frozen_optimizer = freeze_model(model, torch.ones(len(model.indices), device="cuda", dtype=torch.bool), args)
        for actual, wanted in zip((frozen.density, frozen.rgb, frozen.gradient, frozen.sh), expected_fields):
            torch.testing.assert_close(actual, wanted)
        image = render(camera, frozen, tile_size=4, min_t=.1)["render"]
        self.assertTrue(torch.isfinite(image).all())
        print("Shared legacy freeze image MAE:", float((image.detach() - before).abs().mean()))
        image.square().mean().backward()
        frozen_optimizer.main_step(); frozen_optimizer.main_zero_grad()
        frozen_optimizer.sh_optim.step(); frozen_optimizer.sh_optim.zero_grad()
        with tempfile.TemporaryDirectory() as directory:
            state = frozen.state_dict(); state["empty_indices"] = frozen.empty_indices
            torch.save(state, Path(directory) / "ckpt.pth")
            (Path(directory) / "config.json").write_text(json.dumps({"min_t": .1}))
            restored = load_ablation_checkpoint(directory)
            with torch.no_grad():
                torch.testing.assert_close(render(camera, frozen, tile_size=4, min_t=.1)["render"],
                                           render(camera, restored, tile_size=4, min_t=.1)["render"])
        torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()

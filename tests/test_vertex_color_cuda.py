"""Real CUDA smoke test: renderer -> densification -> freeze -> render/load.

Run on the configured NVIDIA server:
CUDA_LAUNCH_BLOCKING=1 uv run --no-sync python -m unittest tests.test_vertex_color_cuda -v
No dataset and no 5,000-iteration warmup required; first Slang compile may take time.
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from data.camera import Camera
from models.vertex_color import Model, TetOptimizer


@unittest.skipUnless(torch.cuda.is_available(), "Requires NVIDIA CUDA and the repository extensions")
class VertexCudaTest(unittest.TestCase):
    def test_render_densify_freeze_and_reload(self):
        from models.frozen import freeze_model
        from utils.args import Args
        from utils.train_util import render
        from utils.densification import collect_render_stats, apply_densification
        from utils.vertex_checkpoint import load_ablation_checkpoint

        torch.manual_seed(2)
        interior = torch.tensor([[0., 0., 0.], [.4, 0., 0.],
                                 [0., .4, 0.], [0., 0., .4]], device="cuda")
        exterior = torch.tensor([[-1., -1., -1.], [2., 0., 0.],
                                 [0., 2., 0.], [0., 0., 2.]], device="cuda")
        model = Model(interior, exterior, torch.zeros(3, device="cuda"), 1.,
                      density_offset=0., max_sh_deg=3)
        camera = Camera(0, np.eye(3), np.array([0., 0., 3.]), 1.2, 1.2,
                        torch.zeros(3, 64, 64), image_name="synthetic",
                        data_device="cpu")
        optimizer = TetOptimizer(model, freeze_start=10)

        package = render(camera, model, tile_size=4, min_t=.1)
        self.assertEqual(package["render"].shape, (3, 64, 64))
        self.assertTrue(torch.isfinite(package["render"]).all())
        package["render"].square().mean().backward()
        self.assertIsNotNone(model.raw_features.grad)
        self.assertTrue(torch.isfinite(model.raw_features.grad).all())
        self.assertGreater(model.raw_features.grad.abs().sum().item(), 0)
        optimizer.main_step()
        optimizer.main_zero_grad()
        optimizer.vertex_optim.step()
        optimizer.vertex_optim.zero_grad()

        args = Args()
        for name, value in dict(tile_size=4, min_tet_count=1, clone_min_contrib=.001,
                                split_min_contrib=.001, within_thresh=1., total_thresh=100.,
                                output_path=None, freeze_start=10, iterations=20).items():
            setattr(args, name, value)
        stats = collect_render_stats([camera], model, args, torch.device("cuda"))
        self.assertTrue(torch.isfinite(stats.tet_moments).all())
        # Force one controlled clone using valid crossing ray segments. This
        # tests the actual apply_densification call, independently of scene quality.
        stats.top_ssim.zero_()
        stats.top_ssim[0] = 10
        stats.total_var_moments.zero_()
        stats.peak_contrib.fill_(1)
        center = model.vertices[model.indices[0].long()].detach().mean(0)
        dx, dy = center.new_tensor([.05, 0., 0.]), center.new_tensor([0., .05, 0.])
        stats.within_var_rays[0, 0] = torch.cat((center - dx, center + dx))
        stats.within_var_rays[0, 1] = torch.cat((center - dy, center + dy))
        previous = model.num_int_verts
        apply_densification(stats, model, optimizer, args, 1, torch.device("cuda"),
                            camera, None, target_addition=1)
        self.assertEqual(model.num_int_verts, previous + 1)
        self.assertEqual(len(model.raw_features), model.num_int_verts)
        render(camera, model, tile_size=4, min_t=.1)["render"].mean().backward()
        optimizer.main_step()
        optimizer.main_zero_grad()
        optimizer.update_triangulation(high_precision=True)

        expected = tuple(v.detach().clone() for v in model.compute_features()[1:])
        frozen, frozen_optimizer = freeze_model(
            model, torch.ones(len(model.indices), dtype=torch.bool, device="cuda"), args)
        for actual, wanted in zip((frozen.density, frozen.rgb, frozen.gradient, frozen.sh), expected):
            torch.testing.assert_close(actual, wanted)
        self.assertFalse(frozen.vertices.requires_grad)
        image = render(camera, frozen, tile_size=4, min_t=.1)["render"]
        self.assertTrue(torch.isfinite(image).all())
        image.square().mean().backward()
        frozen_optimizer.main_step()
        frozen_optimizer.main_zero_grad()
        frozen_optimizer.sh_optim.step()
        frozen_optimizer.sh_optim.zero_grad()
        frozen_optimizer.update_learning_rate(11)
        with tempfile.TemporaryDirectory() as directory:
            state = frozen.state_dict()
            state["empty_indices"] = frozen.empty_indices
            torch.save(state, Path(directory) / "ckpt.pth")
            (Path(directory) / "config.json").write_text(json.dumps({"min_t": .1}))
            frozen.save2ply(Path(directory) / "ckpt.ply")
            restored = load_ablation_checkpoint(directory)
            with torch.no_grad():
                torch.testing.assert_close(
                    render(camera, frozen, tile_size=4, min_t=.1)["render"],
                    render(camera, restored, tile_size=4, min_t=.1)["render"])
        torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()

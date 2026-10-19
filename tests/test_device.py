import unittest
from unittest import mock

import torch

from dexbotic import device as dev


def fake_host(xpu=False, cuda=False):
    return mock.patch.object(dev, "is_available", lambda name: {"xpu": xpu, "cuda": cuda}.get(name, name == "cpu"))


class DetectionTest(unittest.TestCase):
    def test_xpu_beats_cuda(self):
        with fake_host(xpu=True, cuda=True):
            self.assertEqual(dev.default_device_type(), "xpu")
            self.assertEqual(dev.resolve_device(None), torch.device("xpu", 0))

    def test_cuda_when_no_xpu(self):
        with fake_host(cuda=True):
            self.assertEqual(dev.resolve_device(None), torch.device("cuda", 0))

    def test_cpu_fallback(self):
        with fake_host():
            self.assertEqual(dev.resolve_device(None), torch.device("cpu"))

    def test_explicit_device_wins(self):
        with fake_host(xpu=True):
            self.assertEqual(dev.resolve_device("cuda:1"), torch.device("cuda", 1))
            self.assertEqual(dev.resolve_device("cpu"), torch.device("cpu"))
            self.assertEqual(dev.resolve_device("xpu"), torch.device("xpu", 0))


class SeedAndCacheTest(unittest.TestCase):
    def test_manual_seed_all_only_touches_available_backends(self):
        with fake_host(cuda=True), mock.patch("torch.cuda.manual_seed_all") as cuda_seed:
            dev.manual_seed_all(3)
            cuda_seed.assert_called_once_with(3)

    def test_manual_seed_all_without_accelerator_is_a_noop(self):
        with fake_host(), mock.patch("torch.cuda.manual_seed_all") as cuda_seed:
            dev.manual_seed_all(3)
            cuda_seed.assert_not_called()

    def test_empty_cache_follows_device(self):
        with fake_host(xpu=True, cuda=True), mock.patch("torch.cuda.empty_cache") as cuda_empty:
            dev.empty_cache("cpu")
            cuda_empty.assert_not_called()
            dev.empty_cache("cuda:0")
            cuda_empty.assert_called_once()


class InferenceLoadTest(unittest.TestCase):
    def test_load_model_uses_resolved_device(self):
        base_exp = __import__("dexbotic.exp.base_exp", fromlist=["x"])
        model = mock.MagicMock()
        model.to.return_value = model
        with fake_host(xpu=True), \
                mock.patch.object(base_exp.DexboticForCausalLM, "from_pretrained", return_value=model) as load, \
                mock.patch.object(base_exp.AutoTokenizer, "from_pretrained"):
            cfg = base_exp.InferenceConfig(model_name_or_path="x")
            cfg._load_model()
        self.assertEqual(cfg.device, torch.device("xpu", 0))
        self.assertEqual(load.call_args.kwargs["device_map"], {"": "xpu:0"})
        model.to.assert_called_once_with(torch.device("xpu", 0))

    def test_explicit_device_is_kept(self):
        base_exp = __import__("dexbotic.exp.base_exp", fromlist=["x"])
        model = mock.MagicMock()
        model.to.return_value = model
        with mock.patch.object(base_exp.DexboticForCausalLM, "from_pretrained", return_value=model) as load, \
                mock.patch.object(base_exp.AutoTokenizer, "from_pretrained"):
            cfg = base_exp.InferenceConfig(model_name_or_path="x", device="cpu")
            cfg._load_model()
        self.assertEqual(load.call_args.kwargs["device_map"], {"": "cpu"})


@unittest.skipUnless(hasattr(torch, "xpu") and torch.xpu.is_available(), "needs an XPU")
class XpuParityTest(unittest.TestCase):
    def test_small_mlp_matches_cpu(self):
        torch.manual_seed(0)
        net = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.GELU(), torch.nn.Linear(32, 7))
        x = torch.randn(4, 16)

        def run(device):
            module = net.to(device)
            inp = x.to(device)
            with torch.no_grad():
                return module(inp).cpu()

        expected = run("cpu")
        actual = run("xpu:0")
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)

    def test_seed_and_empty_cache_run_on_xpu(self):
        dev.manual_seed_all(1)
        dev.empty_cache("xpu:0")


if __name__ == "__main__":
    unittest.main()

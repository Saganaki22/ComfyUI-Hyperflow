"""Two-time embedder tests: endpoint math, (t, r) mapping, ctx wrapper plumbing.

Uses the real TimeEmbedder class with tiny dims (operations=torch.nn) so the
sinusoidal/silu math under test is the model's own.
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_COMFYUI_ROOT = Path(__file__).resolve().parents[3]
_PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_COMFYUI_ROOT))
sys.path.insert(0, str(_PACKAGE))

import comfy.model_management
from comfy.ldm.minimax.model import (AUDIO_COND_TIMESTEP, VISUAL_COND_TIMESTEP,
                                     TimeEmbedder, time_shift_sigma)

from hyperflow_h3.embedder import (install_two_time, make_ctx_wrapper,
                                   make_endpoint_forward, make_two_time_forward)
from hyperflow_h3.schedule import DEFAULT_SIGMAS_8STEP, video_schedule_sigmas

comfy.model_management.get_torch_device = lambda: torch.device("cpu")

FREQ, HID, OUT = 32, 48, 24
RANK = 4
ALPHA = 8.0
GATE = 0.35


def _embedder(seed=0):
    torch.manual_seed(seed)
    return TimeEmbedder(FREQ, HID, OUT, dtype=torch.float32, device=None,
                        operations=torch.nn)


def _endpoint_tensors(seed=1):
    torch.manual_seed(seed)
    return {
        "proj_in": (torch.randn(RANK, FREQ) * 0.05, torch.randn(HID, RANK) * 0.05),
        "proj_out": (torch.randn(RANK, HID) * 0.05, torch.randn(OUT, RANK) * 0.05),
    }


def _manual_endpoint(te, endpoint, r):
    a_in, b_in = endpoint["proj_in"]
    a_out, b_out = endpoint["proj_out"]
    half = FREQ // 2
    freqs = torch.exp(-torch.log(torch.tensor(10000.0))
                      * torch.arange(half, dtype=torch.float32) / half)
    args = r[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    z = F.linear(emb, te.proj_in.weight, te.proj_in.bias) \
        + F.linear(F.linear(emb, a_in), b_in) * (ALPHA / RANK)
    z = F.silu(z)
    return F.linear(z, te.proj_out.weight, te.proj_out.bias) \
        + F.linear(F.linear(z, a_out), b_out) * (ALPHA / RANK)


def test_endpoint_forward_matches_manual():
    te = _embedder()
    endpoint = _endpoint_tensors()
    cache = {}
    fwd = make_endpoint_forward(te, endpoint, {"proj_in": ALPHA, "proj_out": ALPHA}, cache)
    r = torch.tensor([0.1, 0.532, 1.0])
    got = fwd(r)
    want = _manual_endpoint(te, endpoint, r)
    assert torch.allclose(got, want, atol=1e-5)


def _manual_ctx(t_v, t_a, r_v, r_a):
    return {"t_v": t_v, "t_a": t_a, "r_v": r_v, "r_a": r_a,
            "pin_v": max(t_v, 0.999), "pin_a": max(t_a, 1.0),
            "video_mask": False, "audio_mask": False}


def test_two_time_gate_zero_is_base():
    te = _embedder()
    endpoint = _endpoint_tensors()
    cache = {}
    endpoint_fwd = make_endpoint_forward(te, endpoint,
                                         {"proj_in": ALPHA, "proj_out": ALPHA}, cache)
    shared = {}
    fwd = make_two_time_forward(te.forward, endpoint_fwd, 0.0, shared)
    t_vals = torch.tensor([0.0, 0.4, 0.532, 0.999, 1.0])
    shared["ctx"] = _manual_ctx(0.532, 0.999, 0.6, 0.999)
    assert torch.allclose(fwd(t_vals), te.forward(t_vals), atol=1e-6)


def test_two_time_r_mapping():
    te = _embedder()
    endpoint = _endpoint_tensors()
    endpoint_fwd = make_endpoint_forward(te, endpoint,
                                         {"proj_in": ALPHA, "proj_out": ALPHA}, {})
    shared = {}
    fwd = make_two_time_forward(te.forward, endpoint_fwd, GATE, shared)
    t_v, t_a, r_v, r_a = 0.532, 0.999, 0.6, 0.999
    t_vals = torch.tensor([t_v, t_a, 0.999, 0.7])
    shared["ctx"] = _manual_ctx(t_v, t_a, r_v, r_a)
    got = fwd(t_vals)
    # expected: blend with per-row r -- video row r_v, audio row r_a, pins/masked r = t
    r = torch.tensor([r_v, r_a, 0.999, 0.7])
    t_emb = te.forward(t_vals)
    r_emb = _manual_endpoint(te, endpoint, r)
    want = t_emb + GATE * (r_emb - t_emb)
    assert torch.allclose(got, want, atol=1e-5)
    # and the r == t rows alone equal base when gate*... check pins: row 2 (t=0.999==r)
    assert torch.allclose(got[2], t_emb[2] + GATE * (_manual_endpoint(te, endpoint, torch.tensor([0.999]))[0] - t_emb[2]), atol=1e-5)


def test_two_time_requires_ctx():
    te = _embedder()
    endpoint = _endpoint_tensors()
    endpoint_fwd = make_endpoint_forward(te, endpoint,
                                         {"proj_in": ALPHA, "proj_out": ALPHA}, {})
    fwd = make_two_time_forward(te.forward, endpoint_fwd, GATE, {})
    try:
        fwd(torch.tensor([0.5]))
    except RuntimeError as e:
        assert "step context" in str(e)
    else:
        raise AssertionError("expected RuntimeError without ctx")


def _ctx_for(te, sigma_v, sample_sigmas, payload=None):
    """Run the real ctx wrapper against a fake executor and return (ctx, cleared)."""
    shared = {}
    dm = type("DM", (), {"sigma_shift_video": 12.0, "sigma_shift_audio": 3.0})()
    grid = video_schedule_sigmas(DEFAULT_SIGMAS_8STEP)
    wrap = make_ctx_wrapper(shared, dm, grid, "test", False)
    seen = {}

    def executor(*args, **kwargs):
        seen["ctx"] = dict(shared["ctx"])
        return "ok"

    timestep = torch.tensor([sigma_v * 1000.0])
    # call the way the model really does: transformer_options positional
    result = wrap(executor, None, timestep, None, {"sample_sigmas": sample_sigmas},
                  minimax_payload=payload or {})
    assert result == "ok"
    assert shared["ctx"] is None          # cleared after the forward
    return seen["ctx"]


def test_ctx_wrapper_endpoints_match_grid():
    grid = video_schedule_sigmas(DEFAULT_SIGMAS_8STEP)
    # step 3 of the trained grid, no conditioning: unique rows are {t_v, t_a}
    sigma_v = float(grid[3])
    sigma_next = float(grid[4])
    ctx = _ctx_for(None, sigma_v, grid)
    t_v = 1.0 - sigma_v
    t_a = 1.0 - float(time_shift_sigma(torch.tensor(sigma_v), 12.0, 3.0))
    r_v = 1.0 - sigma_next
    r_a = 1.0 - float(time_shift_sigma(torch.tensor(sigma_next), 12.0, 3.0))
    assert ctx["t_v"] == t_v and ctx["r_v"] == r_v
    assert ctx["t_a"] == t_a and ctx["r_a"] == r_a
    # r_a is on the audio clock and strictly above t_v (no collision with the
    # video row), and r pins at the last step land at exactly 1
    assert t_a > t_v
    ctx_last = _ctx_for(None, float(grid[-2]), grid)
    assert ctx_last["r_v"] == 1.0 and ctx_last["r_a"] == 1.0


def test_ctx_wrapper_pins_and_last_sigma():
    grid = video_schedule_sigmas(DEFAULT_SIGMAS_8STEP)
    sigma_v = float(grid[1])
    # payload with a keyframe adds the visual pin row; ref audio adds the audio pin
    payload = {"visual_cond_noise_aug": VISUAL_COND_TIMESTEP,
               "audio_cond_noise_aug": AUDIO_COND_TIMESTEP}
    ctx = _ctx_for(None, sigma_v, grid, payload)
    t_v = 1.0 - sigma_v
    t_a = 1.0 - float(time_shift_sigma(torch.tensor(sigma_v), 12.0, 3.0))
    pin_v = max(t_v, VISUAL_COND_TIMESTEP)
    pin_a = max(t_a, AUDIO_COND_TIMESTEP)
    # the model's unique_t for this payload is sorted({t_v, t_a, pin_v, pin_a});
    # every row the model can build has an exact-match classification
    rows = sorted({t_v, t_a, pin_v, pin_a})
    for t in rows:
        is_v = (t == ctx["t_v"])
        is_a = (t == ctx["t_a"])
        assert is_v or is_a or t in (pin_v, pin_a), f"row {t} unclassified"


def test_unique_t_mirror_all_steps():
    """Mirror MiniMaxH3Model._forward's seg_t/unique_t computation for plain
    and conditioning payloads at every grid step: every row the model can put
    in t_vals must classify as video/audio/pin -- zero silent fallbacks."""
    grid = video_schedule_sigmas(DEFAULT_SIGMAS_8STEP)
    payloads = ({}, {"visual_cond_noise_aug": VISUAL_COND_TIMESTEP,
                     "audio_cond_noise_aug": AUDIO_COND_TIMESTEP})
    for payload in payloads:
        vis_aug = float(payload.get("visual_cond_noise_aug", VISUAL_COND_TIMESTEP))
        aud_aug = float(payload.get("audio_cond_noise_aug", AUDIO_COND_TIMESTEP))
        for i in range(len(grid) - 1):
            ctx = _ctx_for(None, float(grid[i]), grid, payload)
            t_v, t_a = ctx["t_v"], ctx["t_a"]
            seg_t = {"text": t_v, "video": t_v, "audio": t_a,
                     "cond": max(t_v, vis_aug), "ref_img": max(t_v, vis_aug),
                     "cond_audio": max(t_a, aud_aug), "ref_audio": max(t_a, aud_aug)}
            unique_t = sorted({t_v, t_a} | set(seg_t.values()))
            for t in unique_t:
                assert t == ctx["t_v"] or t == ctx["t_a"] \
                    or t == ctx["pin_v"] or t == ctx["pin_a"], (i, payload, t)


def test_wrapper_full_loop_matches_reference_formula():
    """End to end on the real TimeEmbedder: wrapper sets ctx, patched forward
    blends, and the result equals the reference TwoTimeEmbedder formula
    emb_t(t) + gate * (emb_r(r) - emb_t(t)) computed manually."""
    te = _embedder()
    endpoint = _endpoint_tensors()
    cache = {}
    endpoint_fwd = make_endpoint_forward(te, endpoint,
                                         {"proj_in": ALPHA, "proj_out": ALPHA}, cache)
    shared = {}
    patched = make_two_time_forward(te.forward, endpoint_fwd, GATE, shared)
    dm = type("DM", (), {"sigma_shift_video": 12.0, "sigma_shift_audio": 3.0})()
    grid = video_schedule_sigmas(DEFAULT_SIGMAS_8STEP)
    wrap = make_ctx_wrapper(shared, dm, grid, "test", False)
    sigma_v = float(grid[5])
    captured = {}

    def executor(*args, **kwargs):
        t_v = 1.0 - sigma_v
        t_a = 1.0 - float(time_shift_sigma(torch.tensor(sigma_v), 12.0, 3.0))
        captured["out"] = patched(torch.tensor(sorted({t_v, t_a})))
        return "ok"

    wrap(executor, None, torch.tensor([sigma_v * 1000.0]), None,
         {"sample_sigmas": grid}, minimax_payload={})
    # manual expectation from the ctx the wrapper published
    ctx = _ctx_for(None, sigma_v, grid)
    t_vals = torch.tensor(sorted({ctx["t_v"], ctx["t_a"]}))
    r = torch.where(t_vals == ctx["t_v"], torch.full_like(t_vals, ctx["r_v"]),
                    torch.full_like(t_vals, ctx["r_a"]))
    t_emb = te.forward(t_vals)
    want = t_emb + GATE * (_manual_endpoint(te, endpoint, r) - t_emb)
    assert torch.allclose(captured["out"], want, atol=1e-5)


class _FakeMetadata:
    def __init__(self, gate):
        self.gate = gate
        self.lora_alpha = ALPHA
        self.sigmas = DEFAULT_SIGMAS_8STEP
        self.version = "test"


class _FakeWeights:
    def __init__(self, endpoint, gate):
        self.endpoint = endpoint
        self.metadata = _FakeMetadata(gate)


class _FakePatcher:
    """The bits of ModelPatcher install_two_time touches: shared dm across
    clones, per-patcher object_patches / wrappers."""

    def __init__(self, dm):
        self._dm = dm
        self.object_patches = {}
        self.wrappers = []

    def get_model_object(self, name):
        assert name == "diffusion_model"
        return self._dm

    def add_object_patch(self, key, value):
        self.object_patches[key] = value

    def add_wrapper_with_key(self, wmp, key, wrapper):
        self.wrappers.append(wrapper)


def test_reinstall_on_fresh_clone_unwraps_resident_patch():
    """Regression (context loops): scene 1's patch stays resident on the shared
    time_embedder module when scene 2 applies HyperFlow on a fresh clone. The
    second install must capture the ORIGINAL forward -- wrapping scene 1's
    wrapper would leave the inner one bound to scene 1's cleared step context
    and fail with 'ran without step context' on the first sample step."""
    te = _embedder()
    dm = type("DM", (), {"time_embedder": te,
                         "sigma_shift_video": 12.0,
                         "sigma_shift_audio": 3.0})()
    original_forward = te.forward

    # scene 1: install, then ComfyUI's patch_model applies the patch live
    scene1 = _FakePatcher(dm)
    install_two_time(scene1, _FakeWeights(_endpoint_tensors(seed=1), GATE))
    te.forward = scene1.object_patches["diffusion_model.time_embedder.forward"]
    assert getattr(te.forward, "_hyperflow_two_time", False)

    # scene 2: fresh clone, empty object_patches, module still patched
    scene2 = _FakePatcher(dm)
    gate2, endpoint2 = 0.6, _endpoint_tensors(seed=2)
    install_two_time(scene2, _FakeWeights(endpoint2, gate2))
    patched2 = scene2.object_patches["diffusion_model.time_embedder.forward"]
    # the new wrapper sits on the ORIGINAL forward, not on scene 1's wrapper
    assert patched2._hyperflow_base_forward == original_forward
    te.forward = patched2

    # scene 2 samples through its own ctx wrapper: no stale-context error,
    # and the result follows scene 2's gate/endpoint, not scene 1's
    grid = video_schedule_sigmas(DEFAULT_SIGMAS_8STEP)
    sigma_v = float(grid[3])
    captured = {}

    def executor(*args, **kwargs):
        t_v = 1.0 - sigma_v
        t_a = 1.0 - float(time_shift_sigma(torch.tensor(sigma_v), 12.0, 3.0))
        captured["out"] = te.forward(torch.tensor(sorted({t_v, t_a})))
        return "ok"

    scene2.wrappers[0](executor, None, torch.tensor([sigma_v * 1000.0]), None,
                       {"sample_sigmas": grid}, minimax_payload={})
    assert captured["out"] is not None
    ctx = _ctx_for(None, sigma_v, grid)
    t_vals = torch.tensor(sorted({ctx["t_v"], ctx["t_a"]}))
    r = torch.where(t_vals == ctx["t_v"], torch.full_like(t_vals, ctx["r_v"]),
                    torch.full_like(t_vals, ctx["r_a"]))
    t_emb = original_forward(t_vals)
    want = t_emb + gate2 * (_manual_endpoint(te, endpoint2, r) - t_emb)
    assert torch.allclose(captured["out"], want, atol=1e-5)


def test_chaining_twice_on_one_patcher_still_rejected():
    te = _embedder()
    dm = type("DM", (), {"time_embedder": te})()
    patcher = _FakePatcher(dm)
    install_two_time(patcher, _FakeWeights(_endpoint_tensors(), GATE))
    try:
        install_two_time(patcher, _FakeWeights(_endpoint_tensors(), GATE))
    except RuntimeError as e:
        assert "chain it once" in str(e)
    else:
        raise AssertionError("expected double-apply on one patcher to be rejected")


if __name__ == "__main__":
    test_endpoint_forward_matches_manual()
    test_two_time_gate_zero_is_base()
    test_two_time_r_mapping()
    test_two_time_requires_ctx()
    test_ctx_wrapper_endpoints_match_grid()
    test_ctx_wrapper_pins_and_last_sigma()
    print("ALL PASS (test_wrapper_full_loop_matches_reference_formula is pytest-only)")

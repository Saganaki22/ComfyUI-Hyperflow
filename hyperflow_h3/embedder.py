"""Two-time (t, r) conditioning for ComfyUI's native MiniMax-H3, ported from
hyperflow_h3/embedder.py + blocks.py (Video-Rebirth/hyperflow, Apache-2.0).

HyperFlow conditions every denoising step on the interval it integrates,
``(t, r)`` with ``r = 1 - sigma_next``, not on the point ``t``. ComfyUI's
MiniMax-H3 already reduces the packed sequence's distinct timesteps to a small
sorted table ``t_vals`` and indexes rows into it, so the port replaces exactly
one call -- ``time_embedder(t_vals)`` -- with::

    t_emb = emb_t(t) + gate * (emb_r(r) - emb_t(t))

where ``emb_t`` is the model's own ``time_embedder`` (its LoRA rides the normal
bypass/merge path) and ``emb_r`` is a re-implementation of the same embedder on
raw weights plus the file's ``endpoint_time_embedder`` LoRA, held as captured
tensors -- the module tree is never touched, so ComfyUI's streaming loader
backup/restore is unaffected (the failure mode the Turbo node documents).

Endpoints per row of ``t_vals`` (the pin rule is the official one):

* generated video rows (and text rows, which follow video): r = 1 - sigma_next_v
* generated audio rows: r = 1 - sigma_next_a (audio clock via the model's own
  time_shift_sigma mapping from the video sigma)
* conditioning rows (fl2va keyframes, ref2va refs, cond audio) and masked rows:
  r = t (pinned)

``sigma_next`` is derived from ``transformer_options["sample_sigmas"]`` -- the
same mechanism the native FinalLayer already uses -- so any core sampler works
and no custom sampler node is needed.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.patcher_extension
from comfy.ldm.minimax.model import (AUDIO_COND_TIMESTEP, VISUAL_COND_TIMESTEP,
                                     time_shift_sigma)
from comfy.patcher_extension import WrappersMP

from hyperflow_h3.schedule import video_schedule_sigmas

_log = logging.getLogger("comfy.hyperflow")


def _lora_linear(x, a, b, alpha):
    """B @ A @ x * (alpha / rank), on x's device/dtype."""
    return F.linear(F.linear(x, a.to(device=x.device, dtype=x.dtype)),
                    b.to(device=x.device, dtype=x.dtype)) * (alpha / a.shape[0])


def make_endpoint_forward(te, endpoint, alphas, cache):
    """The endpoint copy of the time embedder: the model's own TimeEmbedder math
    on RAW weights (no base LoRA -- the endpoint in the reference is a deep copy
    made before injection) plus the endpoint_time_embedder LoRA deltas."""

    a_in, b_in = endpoint["proj_in"]
    a_out, b_out = endpoint["proj_out"]
    freq_dim = te.freq_dim

    def endpoint_forward(r):
        # The reference embedder runs fp32 throughout. Quantized bases may store
        # these weights bf16 (comfy.ops auto-casts keep the BASE path alive, but
        # this raw implementation must cast explicitly); fp32 matches the
        # reference math exactly and the matrices are tiny.
        def raw(param):
            return comfy.model_management.cast_to(param, device=r.device) \
                .to(torch.float32)
        w_in = raw(te.proj_in.weight)
        w_out = raw(te.proj_out.weight)
        bias_in = raw(te.proj_in.bias) if te.proj_in.bias is not None else None
        bias_out = raw(te.proj_out.bias) if te.proj_out.bias is not None else None
        # captured LoRA tensors: cache the per-device cast, they are reused every
        # step and the cast result never changes
        key = r.device
        hit = cache.get(key)
        if hit is None:
            hit = tuple(t.to(device=r.device) for t in (a_in, b_in, a_out, b_out))
            cache[key] = hit
        a_in_d, b_in_d, a_out_d, b_out_d = hit

        half = freq_dim // 2
        freqs = torch.exp(-math.log(10000.0)
                          * torch.arange(half, dtype=torch.float32, device=r.device) / half)
        args = r.to(torch.float32)[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        z = F.linear(emb, w_in, bias_in) + _lora_linear(emb, a_in_d, b_in_d,
                                                        alphas["proj_in"])
        z = F.silu(z)
        return F.linear(z, w_out, bias_out) + _lora_linear(z, a_out_d, b_out_d,
                                                           alphas["proj_out"])

    return endpoint_forward


def make_two_time_forward(base_forward, endpoint_forward, gate, shared):
    """Replacement for ``time_embedder.forward``. Refuses to run without step
    context, exactly like the reference TwoTimeEmbedder. Every row of t_vals
    must classify as video/audio/pin; an unclassified row means an endpoint
    silently fell back to single-time -- that only legitimately happens for
    denoise-masked rows, so without masks it warns once instead of degrading
    quietly."""

    def forward(t_vals):
        ctx = shared.get("ctx")
        if ctx is None:
            raise RuntimeError(
                "ApplyHyperFlow: the two-time embedder ran without step context. "
                "Sample through a core guider/sampler (SamplerCustomAdvanced) so "
                "sample_sigmas reach the model, or remove the ApplyHyperFlow node.")
        t_emb = base_forward(t_vals)
        t = t_vals.to(torch.float32)
        is_v = t == ctx["t_v"]
        is_a = t == ctx["t_a"]
        if not shared.get("warned_fallback"):
            fallback = ~(is_v | is_a) & (t != ctx["pin_v"]) & (t != ctx["pin_a"])
            if bool(fallback.any()) and not (ctx["video_mask"] or ctx["audio_mask"]):
                shared["warned_fallback"] = True
                _log.warning(
                    "[hyperflow] unexpected timestep rows %s with no denoise mask "
                    "-- endpoints fall back to single-time for those rows; the "
                    "sampler schedule may not match the trained grid.",
                    [round(float(x), 6) for x in t[fallback].tolist()])
        r = torch.where(is_v, torch.full_like(t, ctx["r_v"]),
                        torch.where(is_a, torch.full_like(t, ctx["r_a"]), t))
        r_emb = endpoint_forward(r)
        return t_emb + gate * (r_emb - t_emb)

    forward._hyperflow_two_time = True
    # Keep the captured base reachable: the module attribute outlives any single
    # ModelPatcher (every clone shares the inner model), so a later install on a
    # fresh clone while this patch is still resident must unwrap to the true
    # original instead of wrapping this wrapper -- a wrapper-on-wrapper leaves
    # the inner one bound to the stale (cleared) step context.
    forward._hyperflow_base_forward = base_forward
    return forward


def _unwrap_two_time(forward):
    """Reduce ``forward`` to the first non-HyperFlow forward beneath it, however
    many HyperFlow wrappers are stacked (loop scenes sharing one module)."""
    seen = set()
    while getattr(forward, "_hyperflow_two_time", False):
        inner = getattr(forward, "_hyperflow_base_forward", None)
        if inner is None or id(inner) in seen:
            raise RuntimeError(
                "ApplyHyperFlow: time_embedder.forward carries a HyperFlow patch "
                "this install cannot unwrap (stale wrapper from an older version?). "
                "Restart ComfyUI and re-run the workflow.")
        seen.add(id(forward))
        forward = inner
    return forward


def make_ctx_wrapper(shared, dm, grid, label, verbose):
    """DIFFUSION_MODEL wrapper: publish the (t, r) context of this step, run,
    clear. Mirrors the model's own float arithmetic for t_v/t_a so the exact
    equality in the forward patch holds, and derives sigma_next from
    sample_sigmas the way the native FinalLayer does."""

    warned = {"grid": False}

    def wrap(executor, *args, **kwargs):
        # the model calls its _forward positionally: (x, timestep, context,
        # transformer_options, minimax_payload=..., ...); read both ways
        timestep = args[1] if len(args) > 1 else kwargs.get("timestep")
        transformer_options = (args[3] if len(args) > 3
                               else kwargs.get("transformer_options")) or {}
        payload = kwargs.get("minimax_payload") or {}

        sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
        shift_v = float(transformer_options.get("minimax_h3_sigma_shift_video",
                                                dm.sigma_shift_video))
        shift_a = float(transformer_options.get("minimax_h3_sigma_shift_audio",
                                                dm.sigma_shift_audio))
        t_v = float(1.0 - sigma_v)
        t_a = float(1.0 - time_shift_sigma(sigma_v, shift_v, shift_a))

        sample_sigmas = transformer_options.get("sample_sigmas")
        if sample_sigmas is None:
            sigma_next = torch.zeros_like(sigma_v)   # FinalLayer's end-of-schedule
        else:
            if not warned["grid"]:
                warned["grid"] = True
                ss = sample_sigmas.flatten().cpu()
                if ss.numel() != grid.numel() or not torch.allclose(ss, grid, atol=1e-4):
                    _log.warning(
                        "[hyperflow] the sampler sigmas (%d points) are not the "
                        "trained 8-step grid; the adapter was distilled on its "
                        "fixed grid -- output quality may deviate.", ss.numel())
            i = int((sample_sigmas - sigma_v).abs().argmin())
            sigma_next = sample_sigmas[min(i + 1, sample_sigmas.shape[0] - 1)]
        r_v = float(1.0 - sigma_next)
        r_a = float(1.0 - time_shift_sigma(sigma_next, shift_v, shift_a))

        vis_aug = float(payload.get("visual_cond_noise_aug", VISUAL_COND_TIMESTEP))
        aud_aug = float(payload.get("audio_cond_noise_aug", AUDIO_COND_TIMESTEP))
        shared["ctx"] = {
            "t_v": t_v, "t_a": t_a, "r_v": r_v, "r_a": r_a,
            "pin_v": max(t_v, vis_aug), "pin_a": max(t_a, aud_aug),
            "video_mask": kwargs.get("denoise_mask") is not None,
            "audio_mask": kwargs.get("audio_denoise_mask") is not None,
        }
        if verbose and not shared.get("logged"):
            shared["logged"] = True
            _log.info("[hyperflow] step ctx: t_v=%.5f r_v=%.5f t_a=%.5f r_a=%.5f",
                      t_v, r_v, t_a, r_a)
        try:
            return executor(*args, **kwargs)
        finally:
            shared["ctx"] = None

    return wrap


def install_two_time(new_model, weights, gate_override=None, verbose=False):
    """Install the two-time embedder on a cloned patcher: the forward-attribute
    patch on time_embedder plus the DIFFUSION_MODEL context wrapper. Idempotence
    guard: chaining ApplyHyperFlow twice on one patcher is rejected. Applying on
    a FRESH clone while a previous scene's patch is still resident on the shared
    module is legal (context loops): the live module forward is unwrapped back
    to the true original before the new wrapper captures it."""
    dm = new_model.get_model_object("diffusion_model")
    te = getattr(dm, "time_embedder", None)
    if te is None:
        raise RuntimeError(
            "ApplyHyperFlow: this MiniMax-H3 base has no time_embedder (pruned / "
            "curve base). HyperFlow's two-time conditioning cannot run on it; use "
            "the pruned-base build of the weights if you must sample this base.")

    patch_key = "diffusion_model.time_embedder.forward"
    if getattr(new_model.object_patches.get(patch_key), "_hyperflow_two_time", False):
        raise RuntimeError("This MODEL already has HyperFlow applied; chain it once.")
    # The module attribute may still carry a wrapper from a previous scene's
    # clone (shared inner model, dynamic-VRAM loops keep it resident): capture
    # the true original forward, never a previous HyperFlow wrapper.
    base_forward = _unwrap_two_time(te.forward)

    gate = weights.metadata.gate if gate_override is None else float(gate_override)
    shared: dict = {}
    cache: dict = {}
    endpoint_forward = make_endpoint_forward(
        te, weights.endpoint,
        {"proj_in": weights.metadata.lora_alpha, "proj_out": weights.metadata.lora_alpha},
        cache)
    forward = make_two_time_forward(base_forward, endpoint_forward, gate, shared)
    new_model.add_object_patch(patch_key, forward)
    new_model.add_wrapper_with_key(
        WrappersMP.DIFFUSION_MODEL, "hyperflow_two_time",
        make_ctx_wrapper(shared, dm, video_schedule_sigmas(weights.metadata.sigmas),
                         weights.metadata.version, verbose))
    return gate


__all__ = ["install_two_time"]

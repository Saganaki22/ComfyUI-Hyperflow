"""HyperFlow weights discovery, header parsing and tensor loading.

The node loads the **ComfyUI-converted** build of the adapter (single
.safetensors, keys already on ComfyUI module paths, q/k/v pre-fused) -- nothing is
translated or fused at load time. Files resolve from models/hyperflow/
(registered by the package ``__init__`` next to models/loras, the same pattern
the VDN-H3 node uses for models/vdn), from a directory holding one, or from an
absolute path. The ``hyperflow.json`` manifests are bundled with this node pack
in assets/, so users only ever download the weights file itself.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors import safe_open

from hyperflow_h3.keys import DIRECT_RE, KEY_RE, fused_qkv_pair, plan_from_keys

#: Hugging Face repo holding the converted builds, fetched on demand when the user
#: enables "download_if_missing". The node downloads exactly the file the bundled
#: manifest names for the chosen variant -- nothing else.
HF_REPO_ID = "drbaph/Hyperflow-Comfyui"

MANIFEST_NAME = "hyperflow.json"
#: manifests ship in assets/, one per base family; users only ever download the
#: .safetensors itself
_ASSETS = Path(__file__).resolve().parent.parent / "assets"
BUNDLED_MANIFEST_FULL = _ASSETS / MANIFEST_NAME
BUNDLED_MANIFEST_PRUNED = _ASSETS / "hyperflow_pruned.json"
BUNDLED_MANIFESTS = (BUNDLED_MANIFEST_FULL, BUNDLED_MANIFEST_PRUNED)

_REQUIRED_METADATA = ("hyperflow", "hyperflow_version", "hyperflow_gate",
                      "lora_alpha", "base_model")

_LAYOUT_ERROR = (
    "{path} is the ORIGINAL diffusers-layout HyperFlow file. This node loads the "
    "ComfyUI-converted build (single .safetensors, ComfyUI module paths) -- grab "
    "the converted file from the HyperFlow Hugging Face repo and drop it into "
    "models/hyperflow/. The original file stays untouched on disk; convert "
    "it yourself with tools/convert_hyperflow.py if needed.")

#: the standalone builds from the MiniMax-H3-Turbo-Lora-ComfyUI repo wrap the
#: same weights as a GENERIC ComfyUI LoRA: every key carries a
#: `diffusion_model.` prefix and a per-module `.alpha` scalar. Those load with
#: the stock Load LoRA node -- this node deliberately does not (their per-module
#: alpha / fc1 row-swap / dropped endpoint differ from the node build).
_GENERIC_LORA_ERROR = (
    "{name} is a standalone/generic ComfyUI LoRA (keys like "
    "'diffusion_model.<module>.lora_A.weight' and '.alpha'), not the HyperFlow "
    "node build. Two options:\n"
    "  1. Use it with the stock 'Load LoRA' node (move it to models/loras/) -- "
    "backbone only, no two-time conditioning.\n"
    "  2. For this node, download the node build '{node_full}' (or "
    "'{node_pruned}' if your MiniMax-H3 base is pruned/curve) from "
    "https://huggingface.co/drbaph/Hyperflow-Comfyui into models/hyperflow/ -- "
    "or tick download_if_missing and let the node fetch it.")


def _node_build_names() -> tuple[str, str]:
    """The published node-build file names, manifest-driven with a hardcoded
    fallback so error text stays helpful even if assets/ were stripped."""
    names = {}
    for variant, manifest in (("full", BUNDLED_MANIFEST_FULL),
                              ("pruned", BUNDLED_MANIFEST_PRUNED)):
        try:
            names[variant] = _manifest_default(manifest)
        except (ValueError, OSError, json.JSONDecodeError, FileNotFoundError):
            names[variant] = ("custom_node_hyperflow_8step_v1.0_comfyui"
                              + ("_pruned" if variant == "pruned" else "")
                              + ".safetensors")
    return names["full"], names["pruned"]


@dataclass(frozen=True)
class HyperFlowMetadata:
    """The safetensors header of a HyperFlow weights file, parsed and validated.
    The converter preserves the header verbatim, so gate/sigmas/rank are read the
    same way from the original and the converted build."""

    version: str
    gate: float
    lora_alpha: float
    base_model: str
    lora_rank: int | None = None
    sigmas: tuple[float, ...] | None = None
    video_shift: float | None = None
    audio_shift: float | None = None
    base_model_revision: str | None = None
    raw: dict[str, str] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, metadata: dict[str, str] | None) -> "HyperFlowMetadata":
        metadata = dict(metadata or {})
        missing = [key for key in _REQUIRED_METADATA if key not in metadata]
        if missing or metadata.get("hyperflow", "").lower() != "true":
            raise ValueError(
                "Not a HyperFlow weights file: the safetensors header must carry "
                f"{list(_REQUIRED_METADATA)} with `hyperflow == \"true\"`; missing "
                f"{missing or ['hyperflow=true']}.")
        return cls(
            version=metadata["hyperflow_version"],
            gate=float(metadata["hyperflow_gate"]),
            lora_alpha=float(metadata["lora_alpha"]),
            base_model=metadata["base_model"],
            lora_rank=int(metadata["lora_rank"]) if "lora_rank" in metadata else None,
            sigmas=_json_tuple(metadata.get("hyperflow_sigmas"), float),
            video_shift=float(metadata["hyperflow_video_shift"]) if "hyperflow_video_shift" in metadata else None,
            audio_shift=float(metadata["hyperflow_audio_shift"]) if "hyperflow_audio_shift" in metadata else None,
            base_model_revision=metadata.get("base_model_revision"),
            raw=metadata)


def _json_tuple(value: str | None, cast) -> tuple | None:
    if value is None:
        return None
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise ValueError(f"Expected a JSON list in the safetensors header, got {value!r}.")
    return tuple(cast(item) for item in parsed)


def _manifest_default(manifest_path: Path) -> str:
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    default = manifest.get("default") if isinstance(manifest, dict) else None
    if not isinstance(default, str) or not default:
        raise ValueError(f"{manifest_path} has no `default` entry naming the recommended weights file.")
    return default


def _within_registered(path: Path, roots) -> bool:
    """Containment check that tolerates Windows junctions and symlinked
    subfolders: a junctioned subfolder
    resolves outside its registered root, so accept a match on EITHER the
    literal or the resolved paths. A '..' traversal filename fails both and
    stays rejected."""
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    for root in roots:
        if path.is_relative_to(root):
            return True
        try:
            if resolved.is_relative_to(root.resolve()):
                return True
        except OSError:
            pass
    return False

def _manifest_defaults() -> list[str]:
    """The recommended weights file of every bundled manifest (full base first)."""
    defaults = []
    for manifest in BUNDLED_MANIFESTS:
        if manifest.is_file():
            try:
                defaults.append(_manifest_default(manifest))
            except (ValueError, OSError, json.JSONDecodeError):
                pass
    return defaults


def _pick_one(candidates: list[str], where: str) -> str:
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected a weights file named by the bundled manifests "
            f"({', '.join(str(m) for m in BUNDLED_MANIFESTS)}) or exactly one "
            f".safetensors file in {where}, found {candidates}.")
    return candidates[0]


def resolve_weights(path_or_name: str | os.PathLike) -> Path:
    """Resolve a combo name (models/hyperflow), a path to a .safetensors file,
    or a directory with one weights file, to the weights file itself."""
    name = str(path_or_name)
    local = Path(name)
    if not local.is_file() and not local.is_dir():
        import folder_paths  # ComfyUI-only path; the standalone converter passes real paths
        full = folder_paths.get_full_path("hyperflow", name)
        if full is not None:
            local = Path(full)
            roots = [Path(root) for root in folder_paths.get_folder_paths("hyperflow")]
            if not _within_registered(local, roots):
                raise ValueError("Model path leaves its registered model directory.")
    if local.is_file():
        if local.suffix != ".safetensors":
            raise FileNotFoundError(f"{local} is not a .safetensors file.")
        return local
    if local.is_dir():
        for default in _manifest_defaults():
            if (local / default).is_file():
                return local / default
        return local / _pick_one(sorted(p.name for p in local.glob("*.safetensors")), str(local))
    raise FileNotFoundError(
        f"{name!r} is not a HyperFlow weights file under models/hyperflow "
        "and not a path on disk.")


def manifest_filename(variant: str) -> str:
    """The published weights file for a base variant, named by its bundled
    manifest. ``variant`` is 'full' or 'pruned' (validated by the caller's
    combo, never free text)."""
    if variant == "full":
        manifest = BUNDLED_MANIFEST_FULL
    elif variant == "pruned":
        manifest = BUNDLED_MANIFEST_PRUNED
    else:
        raise ValueError(f"variant must be 'full' or 'pruned', got {variant!r}.")
    if not manifest.is_file():
        raise FileNotFoundError(f"bundled manifest missing: {manifest}")
    return _manifest_default(manifest)


def _download(repo_id: str, filename: str, dest_dir: Path) -> Path:
    """Fetch exactly one published artifact into ``dest_dir`` (via the HF cache,
    then moved -- keeps .cache metadata out of the model folder)."""
    from huggingface_hub import hf_hub_download
    cached = Path(hf_hub_download(repo_id=repo_id, filename=filename))
    dest = dest_dir / filename
    if not dest.exists():
        import shutil
        shutil.move(str(cached), str(dest))
    return dest


def ensure_downloaded(variant: str, repo_id: str | None = None) -> Path:
    """Download-on-demand for the node's download_if_missing toggle: user-enabled,
    variant picked from a fixed allowlist, file name taken from the bundled
    manifest -- never a free-text path, and only ever into models/hyperflow."""
    repo_id = repo_id if repo_id is not None else HF_REPO_ID
    if not repo_id:
        raise RuntimeError(
            "download_if_missing is enabled but no Hugging Face repo is configured: "
            "upload the converted weights, then set HF_REPO_ID in "
            "hyperflow_h3/weights.py (see the README).")
    import folder_paths
    filename = manifest_filename(variant)
    folders = folder_paths.get_folder_paths("hyperflow")
    if not folders:
        raise RuntimeError("models/hyperflow is not registered; restart ComfyUI.")
    return _download(repo_id, filename, Path(folders[0]))


def read_metadata(path: str | Path) -> HyperFlowMetadata:
    """Read and validate the HyperFlow header of a safetensors file without
    loading any tensor."""
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return HyperFlowMetadata.from_dict(handle.metadata())


@dataclass
class HyperFlowWeights:
    """Tensors keyed by ComfyUI module path, ready for hyperflow_h3.apply.

    lora:    {comfy_module: (A, B)} -- qkv_proj entries are the pre-fused rank-3r
             pairs (A' = cat, B' = block_diag), with ``alphas[module] = 3 * lora_alpha``
             so the effective scale stays alpha/rank per q/k/v part.
    endpoint: {"proj_in": (A, B), "proj_out": (A, B)} -- captured tensors for the
             two-time forward patch; never registered on the module tree.
    alphas:  per-entry alpha (file lora_alpha, or 3x on fused qkv_proj entries).
    """

    lora: dict
    endpoint: dict
    alphas: dict
    rank: int
    metadata: HyperFlowMetadata
    path: Path


def load_weights(path_or_name: str | os.PathLike) -> HyperFlowWeights:
    """Load one converted HyperFlow file: validate the header, group the keys,
    pull every tensor. The original diffusers layout is rejected here -- loading
    it would mean translating and fusing on every load."""
    path = resolve_weights(path_or_name)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        if any(KEY_RE.match(k) is not None for k in keys):
            raise RuntimeError(_LAYOUT_ERROR.format(path=path))
        if any(k.startswith("diffusion_model.") or k.endswith(".alpha") for k in keys):
            node_full, node_pruned = _node_build_names()
            raise RuntimeError(_GENERIC_LORA_ERROR.format(
                name=path.name, node_full=node_full, node_pruned=node_pruned))
        plan = plan_from_keys(keys)     # converted layout only, validated
        metadata = HyperFlowMetadata.from_dict(handle.metadata())
        tensors = {key: handle.get_tensor(key) for key in keys}

    lora, endpoint, alphas = {}, {}, {}
    ranks = set()
    for target, entry in plan.items():
        a, b = tensors[entry["A"]], tensors[entry["B"]]
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
            raise ValueError(f"{target}: not an A [rank, in] / B [out, rank] pair "
                             f"(shapes {tuple(a.shape)}, {tuple(b.shape)}).")
        alpha = metadata.lora_alpha * (3.0 if target.endswith(".attn.qkv_proj") else 1.0)
        if entry["kind"] == "endpoint":
            endpoint[target.rsplit(".", 1)[1]] = (a, b)
            alphas[target] = alpha
            ranks.add(a.shape[0])
        else:
            lora[target] = (a, b)
            alphas[target] = alpha
            # fused qkv entries carry rank 3r; the rest carry the file rank
            ranks.add(a.shape[0] // 3 if target.endswith(".attn.qkv_proj") else a.shape[0])

    rank = metadata.lora_rank if metadata.lora_rank is not None else None
    derived = ranks.pop() if len(ranks) == 1 else None
    if rank is None:
        rank = derived
    if rank is None:
        raise ValueError("Cannot determine the LoRA rank: header has no lora_rank "
                         "and the entries disagree.")
    if derived is not None and derived != rank:
        raise ValueError(f"Header says lora_rank={rank} but the tensors imply {derived}.")
    return HyperFlowWeights(lora=lora, endpoint=endpoint, alphas=alphas, rank=rank,
                            metadata=metadata, path=path)


__all__ = ["BUNDLED_MANIFEST_FULL", "BUNDLED_MANIFEST_PRUNED", "BUNDLED_MANIFESTS",
           "HF_REPO_ID", "MANIFEST_NAME", "HyperFlowMetadata", "HyperFlowWeights",
           "ensure_downloaded", "load_weights", "manifest_filename",
           "read_metadata", "resolve_weights"]

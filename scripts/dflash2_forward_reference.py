"""Generate a complete tiny DFlash2 forward reference using the public source archive.

CPU-only; requires torch, transformers==5.5.0 and safetensors. The archive is
available beside the pinned public checkpoint; no network access is performed.
"""

import argparse
import ast
import hashlib
import importlib
import json
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import save_file
from transformers import Qwen3Config

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--archive", type=Path, required=True)
parser.add_argument("--output-directory", type=Path, required=True)
args = parser.parse_args()
if (
    hashlib.sha256(args.archive.read_bytes()).hexdigest()
    != "a55429af46cd7d17a06f7026189c3fa64c69698e333ce927cacbb32784cd1a16"
):
    raise ValueError("Use the pinned public DFlash2 reference source")
workspace = tempfile.TemporaryDirectory(prefix="dflash2-reference-")
root = Path(workspace.name)
package = root / "public_dflash"
package.mkdir()
(package / "__init__.py").touch()
compat = root / "specforge"
compat.mkdir()
(compat / "__init__.py").touch()
with tarfile.open(args.archive) as archive:
    for name in [
        "dflash.py",
        "dflash2.py",
        "dflash_kernels.py",
        "flex_attention_backend.py",
        "registry.py",
    ]:
        (package / name).write_bytes(
            archive.extractfile("SpecForge/specforge/modeling/draft/" + name).read()
        )
    (compat / "torch_compat.py").write_bytes(
        archive.extractfile("SpecForge/specforge/torch_compat.py").read()
    )
    mask_source = archive.extractfile(
        "SpecForge/specforge/algorithms/common/dflash_family_model.py"
    ).read()
mask_function = next(
    n
    for n in ast.parse(mask_source).body
    if isinstance(n, ast.FunctionDef) and n.name == "create_dflash_sdpa_mask"
)
namespace = {"torch": torch, "Optional": Optional}
exec(  # noqa: S102 - Execute hash-verified public reference definitions.
    compile(
        ast.Module(body=[mask_function], type_ignores=[]), "public-mask.py", "exec"
    ),
    namespace,
)
sys.path.insert(0, str(root))
D = importlib.import_module("public_dflash.dflash2").DFlash2DraftModel
torch.manual_seed(74)
c = {
    "architectures": ["DFlash2DraftModel"],
    "model_type": "qwen3",
    "hidden_size": 8,
    "intermediate_size": 16,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 4,
    "num_hidden_layers": 2,
    "num_target_layers": 4,
    "vocab_size": 13,
    "rms_norm_eps": 1e-6,
    "attention_bias": False,
    "hidden_act": "silu",
    "is_causal": False,
    "layer_types": ["full_attention", "sliding_attention"],
    "sliding_window": 4,
    "use_sliding_window": True,
    "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
    "dflash_config": {
        "block_size": 4,
        "conv_group_size": 2,
        "conv_kernel_size": 3,
        "mask_token_id": 12,
        "selector_rank": 3,
        "selector_top_k": 4,
        "target_layer_ids": [0, 2],
    },
}
config = Qwen3Config(**c)
config._attn_implementation = "sdpa"
model = D(config).eval()
with torch.no_grad():
    for p in model.parameters():
        p.copy_(torch.randn_like(p) * 0.2)
    x = torch.randn(1, 4, 8)
    context = torch.randn(1, 6, 16)
    positions = torch.arange(10).unsqueeze(0)
    mask = namespace["create_dflash_sdpa_mask"](
        torch.tensor([[6]]),
        torch.tensor([[True]]),
        6,
        4,
        torch.device("cpu"),
        sliding_window=4,
    )
    output = model(
        position_ids=positions,
        noise_embedding=x,
        target_hidden=context,
        attention_mask={"full_attention": None, "sliding_attention": mask},
    )
arrays = {
    **model.state_dict(),
    "fixture.embeddings": x,
    "fixture.context": context,
    "fixture.output": output,
}
root = args.output_directory
root.mkdir(parents=True, exist_ok=True)
c["reference"] = {
    "archive_sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest(),
    "torch": torch.__version__,
    "seed": 74,
}
save_file(
    {k: v.detach().contiguous() for k, v in arrays.items()},
    root / "dflash2-forward.safetensors",
)
(root / "dflash2-forward.json").write_text(json.dumps(c, indent=2) + "\n")
print(output.flatten()[:8])

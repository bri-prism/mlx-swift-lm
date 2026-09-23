"""Generate CPU PyTorch fixtures from the pinned public SpecForge DFlash2 source.

Pass the extracted specforge/modeling/draft/dflash2.py as --source. Only the
unmodified grouped-convolution and selector classes execute; no model downloads
or repository imports occur. The source is MIT licensed, see ThirdPartyLicenses.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch import nn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.read_bytes()
    if (
        hashlib.sha256(source).hexdigest()
        != "ecc7c04d6058bf2e8bc6e6434c5638bc7ac41ddab9f5b7ca8bab6be8e75dc90b"
    ):
        raise ValueError("Use the pinned public DFlash2 reference source")
    classes = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef)
        and node.name in ("DFlashGroupedConv", "CandidateSelector")
    ]
    assert len(classes) == 2
    namespace = {"torch": torch, "nn": nn, "F": F}
    exec(  # noqa: S102 - Execute hash-verified public reference definitions.
        compile(ast.Module(body=classes, type_ignores=[]), str(args.source), "exec"),
        namespace,
    )
    torch.manual_seed(73)
    conv = namespace["DFlashGroupedConv"](8, 4, 3, 2)
    selector = namespace["CandidateSelector"](
        hidden_size=8, vocab_size=13, state_rank=3, top_k=4, initializer_range=0.02
    )
    with torch.no_grad():
        for p in conv.parameters():
            p.copy_(torch.randn_like(p) * 0.2)
        for p in selector.parameters():
            p.copy_(torch.randn_like(p) * 0.2)
        x = torch.randn(2, 8, 8)
        prepared, delta = conv.prepare(x)
        sublayer = torch.randn_like(x)
        finished = conv.finish(sublayer, delta)
        ids = torch.tensor([[[1, 4, 8, 11], [0, 2, 5, 9], [3, 6, 10, 12]]] * 2)
        hidden = torch.randn(2, 3, 8)
        unary = torch.randn(2, 3, 4) * 0.1
        anchor = torch.tensor([2, 7])
        scores = selector.score_candidates(
            candidate_ids=ids[:, 0],
            unary_logits=unary[:, 0],
            hidden_states=hidden[:, 0],
            predecessor_ids=anchor,
        )
        path = selector.greedy_path(
            candidate_ids=ids,
            unary_logits=unary,
            hidden_states=hidden,
            anchor_token_ids=anchor,
        )
    arrays = {
        **{"conv." + k: v for k, v in conv.state_dict().items()},
        **{"selector." + k: v for k, v in selector.state_dict().items()},
        "input": x,
        "prepared": prepared,
        "delta": delta,
        "sublayer": sublayer,
        "finished": finished,
        "ids": ids,
        "hidden": hidden,
        "unary": unary,
        "anchor": anchor,
        "scores": scores,
        "path": path,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().contiguous() for k, v in arrays.items()}, args.output)
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "source_sha256": hashlib.sha256(source).hexdigest(),
                "checkpoint_revision": "4cfb6ad03268fed0f60ca96c1a659c0b1c77e50b",
                "source_archive": "https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-DFlash2/blob/4cfb6ad03268fed0f60ca96c1a659c0b1c77e50b/source/specforge-source.tar.gz",
                "torch": torch.__version__,
                "seed": 73,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

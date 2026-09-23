"""Generate reference tensors for the opt-in Swift packed-checkpoint tests.

Run in the published checkpoint's Python environment with mlx-vlm and Pillow.
The checkpoint must include its published runtime/vision_artifact.py loader.
"""

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output filename prefix")
    args = parser.parse_args()
    sys.path.insert(0, str(args.model.resolve() / "runtime"))
    from vision_artifact import load_vl_model

    model, processor, _ = load_vl_model(args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for color in (None, "red", "blue"):
        question = (
            "What is the capital of France? Answer briefly."
            if color is None
            else "<|vision_start|><|image_pad|><|vision_end|>"
            "What color is the image? Answer with one word."
        )
        prompt = (
            f"<|im_start|>user\n{question}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
        if color is None:
            ids = mx.array([processor.tokenizer.encode(prompt, add_special_tokens=False)])
            pixels = grid = None
        else:
            inputs = processor(
                text=prompt,
                images=[Image.new("RGB", (224, 224), color)],
                return_tensors="np",
            )
            ids = mx.array(inputs["input_ids"])
            pixels = mx.array(inputs["pixel_values"])
            grid = mx.array(inputs["image_grid_thw"])
        cache = make_prompt_cache(model.language_model)
        tensors = {} if pixels is None else {"pixels": pixels}
        tokens = []
        x = ids
        for step in range(12):
            if step == 0 and pixels is not None:
                result = model(x, pixels, cache=cache, image_grid_thw=grid)
            else:
                if step == 0:
                    model.language_model._position_ids = None
                result = model.language_model(x, cache=cache)
            logits = (result.logits if hasattr(result, "logits") else result)[:, -1, :]
            logits = logits.astype(mx.float32)
            mx.eval(logits)
            if not mx.all(mx.isfinite(logits)).item():
                raise ValueError("Non-finite reference logits")
            tensors[f"logits_{step}"] = logits
            token = mx.argmax(logits, axis=-1).item()
            tokens.append(token)
            x = mx.array([[token]])
            if token in (248044, 248046):
                break
        prefix = str(args.output) + (f"-vision-{color}" if color else "")
        mx.save_safetensors(prefix + ".safetensors", tensors)
        record = {
            "input_ids": ids[0].tolist(),
            "output_ids": tokens,
            "text": processor.tokenizer.decode(tokens),
        }
        if grid is not None:
            record["grid"] = grid.tolist()
        Path(prefix + ".json").write_text(json.dumps(record, indent=2) + "\n")
        print(color or "text", record["text"], flush=True)


if __name__ == "__main__":
    main()

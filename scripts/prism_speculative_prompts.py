"""Prepare local-tokenizer inputs for the opt-in Swift speculative smoke test."""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-sliding-window", action="store_true")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prompts = [
        ("arithmetic", "Explain why 17 times 23 equals 391, step by step."),
        (
            "code",
            "Write a Python function that removes duplicates from a list while preserving order.",
        ),
        ("prose", "Describe three differences between a mountain and a volcano."),
    ]
    if args.include_sliding_window:
        prompts.append(
            (
                "sliding_window",
                "The blue notebook is on the wooden desk.\n" * 240
                + "Explain three ways to organize a desk.",
            )
        )
    rows = []
    for name, question in prompts:
        prompt = (
            f"<|im_start|>user\n{question}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
        rows.append(
            {
                "name": name,
                "prompt": prompt,
                "input_ids": tokenizer.encode(prompt, add_special_tokens=False),
            }
        )
    args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()

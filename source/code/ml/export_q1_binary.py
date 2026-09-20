#!/usr/bin/env python3
"""Export the set scheduler's selection and RU paths to a C++-readable binary."""

import argparse
import struct
from pathlib import Path

import numpy as np
import torch


TENSOR_NAMES = [
    "local_encoder.0.weight",
    "local_encoder.0.bias",
    "local_encoder.3.weight",
    "local_encoder.3.bias",
    "context_encoder.0.weight",
    "context_encoder.0.bias",
    "decoder.0.weight",
    "decoder.0.bias",
    "decoder.3.weight",
    "decoder.3.bias",
    "selected_head.weight",
    "selected_head.bias",
    "ru_head.weight",
    "ru_head.bias",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "set-equivariant":
        raise ValueError("Binary export requires a set-equivariant checkpoint")

    state = checkpoint["model_state_dict"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        stream.write(b"Q1SET02\0")
        stream.write(
            struct.pack(
                "<5I",
                2,
                int(checkpoint["feature_count"]),
                int(checkpoint["hidden_size"]),
                int(checkpoint["aggregation_classes"]),
                5,
            )
        )
        for name in TENSOR_NAMES:
            values = state[name].detach().cpu().numpy().astype(np.float32, copy=False)
            stream.write(struct.pack("<Q", values.size))
            stream.write(values.tobytes(order="C"))

    print(f"exported={args.output} bytes={args.output.stat().st_size}")


if __name__ == "__main__":
    main()

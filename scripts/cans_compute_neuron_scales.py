"""Compute the domain's CANS intervention or a Grad-only comparison."""

import argparse
from pathlib import Path

import torch

from cans.data_processing import config, save_tensor, sha256
from cans.scaling_solver import cans_scales, grad_only_scales


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--method", choices=["cans", "grad-only"], default="cans")
    parser.add_argument("--strength", type=float, default=0.5,
                        help="Grad-only strength (default: 0.5); CANS uses its domain config")
    parser.add_argument("--gradient", type=Path, help="Use newly extracted gradients")
    parser.add_argument("--fisher", type=Path, help="Use newly estimated Fisher")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, common, domain = config(args.config)
    gradient_path = args.gradient or root / domain["artifact"]
    artifact = torch.load(gradient_path, map_location="cpu", weights_only=True)
    if artifact["metadata"]["model_config_sha256"] != common["model_config_sha256"]:
        raise ValueError("Grad artifact belongs to another model")
    if artifact["metadata"]["data_sha256"] != sha256(root / domain["data"]):
        raise ValueError("Grad artifact belongs to another steering corpus")
    gradient = artifact["gradient"]
    metadata = {"model": common["model"], "model_config_sha256": common["model_config_sha256"],
                "domain": domain["domain"], "method": args.method,
                "gradient_sha256": sha256(gradient_path), "scope": "global/all-token"}
    if args.method == "cans":
        fisher_path = args.fisher or root / common["fisher_artifact"]
        curvature = torch.load(fisher_path, map_location="cpu", weights_only=True)
        if (curvature["metadata"]["model_config_sha256"] != common["model_config_sha256"]
                or curvature["metadata"]["estimator"] != "whole-response-logprob-gradient-square"):
            raise ValueError("Expected matching whole-response Fisher")
        scales, diagnostics = cans_scales(
            domain["objective_sign"] * gradient, curvature["fisher"], **domain["cans"])
        if diagnostics["capped_fraction"] > 0.15:
            raise ValueError("Saturation exceeds the recorded 15% limit")
        metadata.update(diagnostics, fisher_sha256=sha256(fisher_path))
    else:
        if args.strength not in domain["grad-only"]["strengths"]:
            raise ValueError("Choose one of the Grad-only strengths in the domain config")
        top_k = domain["grad-only"]["top_k"]
        multiplier = 1 + domain["objective_sign"] * args.strength
        scales = grad_only_scales(gradient, top_k, multiplier)
        metadata.update(top_k=top_k, multiplier=multiplier, strength=args.strength)
    save_tensor(args.output, {"scales": scales, "metadata": metadata})
    print(metadata)


if __name__ == "__main__":
    main()

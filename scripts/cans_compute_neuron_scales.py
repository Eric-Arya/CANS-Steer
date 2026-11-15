"""Compute the domain's CANS intervention or a Grad-only comparison."""

import argparse
from pathlib import Path

import torch

from cans.data_processing import config, save_tensor, sha256
from cans.scaling_solver import cans_scales, grad_only_scales


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--method", choices=["cans", "grad-only"], default="cans")
    parser.add_argument("--coefficient", type=float,
                        help="Override configured nonnegative c for a signed Grad-only strength sweep")
    parser.add_argument("--cap", type=float,
                        help="Matching CANS cap; must equal the configured CANS cap when present")
    parser.add_argument("--gradient", type=Path, help="Use newly extracted gradients")
    parser.add_argument("--fisher", type=Path, help="Use newly estimated Fisher")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.method != "grad-only" and (args.coefficient is not None or args.cap is not None):
        parser.error("--coefficient and --cap apply only to grad-only")
    if args.method != "cans" and args.fisher is not None:
        parser.error("Grad-only does not use --fisher")
    if args.output.exists():
        raise FileExistsError(f"Use a fresh output path: {args.output}")
    root, common, domain = config(args.config)
    coefficient = args.coefficient
    if args.method == "grad-only" and coefficient is None:
        coefficient = domain.get("grad-only", {}).get("coefficient")
        if coefficient is None:
            parser.error("grad-only requires a configured coefficient or --coefficient")
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
    elif args.method == "grad-only":
        binding = artifact["metadata"]
        if (binding.get("scope") != "global/all-token"
                or binding.get("model_dtype", "float32") != "float32"
                or binding.get("mean_dtype") != "float32"):
            raise ValueError("Grad-only requires FP32 model/mean gradients with global/all-token extraction")
        has_configured_cap = "cap" in domain.get("cans", {})
        configured_cap = domain.get("cans", {}).get("cap")
        cap = args.cap if args.cap is not None else configured_cap
        if cap is None and not has_configured_cap:
            parser.error("Supply --cap equal to the corresponding CANS comparison cap")
        if has_configured_cap and cap != configured_cap:
            raise ValueError("Grad-only cap must equal the configured CANS cap")
        sign = domain["objective_sign"]
        scales = grad_only_scales(gradient, coefficient, cap, objective_sign=sign)
        metadata.update(coefficient=coefficient, cap=cap, objective_sign=sign,
                        formula="1 + c * g" if cap is None else "1 + clip(c * g, -cap, cap)",
                        objective_gradient="g = objective_sign * target_logprob_gradient",
                        source_gradient_semantics="target log-probability gradient before task objective sign",
                        curvature="identity", coordinate_count=gradient.numel(),
                        coordinate_selection="all signed coordinates; no ranking or filtering",
                        fisher_used=False, damping=0, gradient_normalization=False,
                        post_clip_normalization=False, source_metadata=binding,
                        data_sha256=binding["data_sha256"],
                        capped_fraction=0.0 if cap is None else float((coefficient * gradient.double()).abs().gt(cap).double().mean()))
    save_tensor(args.output, {"scales": scales, "metadata": metadata})
    print(metadata)


if __name__ == "__main__":
    main()

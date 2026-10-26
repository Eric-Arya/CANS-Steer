"""Verify release inputs, restore missing files, and prepare official evaluators."""

import argparse
from importlib.metadata import distribution
import json
from pathlib import Path
import subprocess
from urllib.request import urlopen
import zipfile

from cans.data_processing import read_jsonl, sha256


def normalize(text):
    return ' '.join(text.split())


def verify_baselines(root, manifest):
    from baselines.h_neurons import validate_split
    folder = root / 'samples/baselines'
    result={'model':manifest['model'],'verified_files':len(manifest['files']),'domains':{}}
    for domain,task in [('safety','harmbench'),('toxicity','rtp'),('hallucination','truthfulqa')]:
        rows=read_jsonl(folder/domain/'h_neurons.jsonl');split=json.loads((folder/domain/'h_split.json').read_text())
        validate_split(rows,split)
        for r in rows:
            if not r['span'] or r['response'][r['span_start']:r['span_end']]!=r['span']:raise ValueError('Invalid exact span')
        heldout={normalize(r['prompt']) for r in rows if r['id'] in split['heldout']}
        benchmark={normalize(r.get('prompt',r.get('question',''))) for r in read_jsonl(root/f'samples/benchmarks/{task}.jsonl')}
        count=len(heldout&benchmark)
        result['domains'][domain]={'h_examples':len(rows),'train_examples':len(split['train']),'heldout_examples':len(split['heldout']),
            'heldout_source_prompts':len(heldout),'heldout_final_benchmark_prompt_overlap':count,
            'domain_selection_data':f'samples/baselines/{domain}/caa_selection.jsonl',
            'selection_record':f'samples/baselines/{domain}/provenance/selected_h_neurons.json'}
    return result


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--evaluators", action="store_true",
                        help="Prepare official evaluator source and language data")
    parser.add_argument("--base-url", help="Raw release URL for restoring missing files")
    parser.add_argument("--external", type=Path, default=root / "external",
                        help="Destination for evaluator source and language data")
    parser.add_argument("--output", type=Path, help="Optional JSON report for all release checks")
    args = parser.parse_args(argv)
    manifest = json.loads((root / "samples/manifest.json").read_text())
    baseline_manifest = json.loads((root / "samples/baselines/manifest.json").read_text())
    files = manifest["files"] + [{"path": path, **record} for path, record in baseline_manifest["files"].items()]
    for item in files:
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Manifest path escapes the repository")
        if not path.exists() and not args.verify_only:
            if not args.base_url:
                raise FileNotFoundError(f"Missing {item['path']}; supply --base-url or restore from Git")
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".download")
            with urlopen(args.base_url.rstrip("/") + "/" + item["path"], timeout=60) as response:
                temporary.write_bytes(response.read())
            if sha256(temporary) != item["sha256"]:
                raise ValueError(f"Downloaded checksum mismatch: {item['path']}")
            temporary.replace(path)
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise ValueError(f"Missing or modified release input: {item['path']}")
    baseline_audit = verify_baselines(root, baseline_manifest)
    result = {"model": manifest["model"], "verified_files": len(files),
              "cans_files": len(manifest["files"]), "baseline_files": len(baseline_manifest["files"]),
              "domains": baseline_audit["domains"]}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if args.evaluators:
        external = args.external.resolve()
        external.mkdir(parents=True, exist_ok=True)
        source = json.loads(distribution("lm_eval").read_text("direct_url.json") or "{}")
        if source.get("vcs_info", {}).get("commit_id") != manifest["dependencies"]["lm_eval"]["revision"]:
            raise ValueError("Install the pinned LM Evaluation Harness with uv sync --extra eval --frozen")
        for name, dependency in manifest["dependencies"].items():
            if name == "lm_eval":
                continue  # Installed from its pinned commit by uv sync.
            path = external / name
            if not path.exists():
                subprocess.run(["git", "clone", dependency["url"], str(path)], check=True)
                subprocess.run(["git", "-C", str(path), "checkout", dependency["revision"]], check=True)
            revision = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
            if revision != dependency["revision"]:
                raise ValueError(f"Unexpected evaluator revision in {path}")
        with zipfile.ZipFile(root / "samples/ifeval_scorer.zip") as archive:
            for member in archive.namelist():
                if not (external / member).resolve().is_relative_to(external.resolve()):
                    raise ValueError("Unsafe evaluator archive member")
            archive.extractall(external)
        import nltk
        nltk.download("punkt_tab", download_dir=str(external / "nltk_data"), raise_on_error=True)
        import spacy
        spacy.load("en_core_web_sm")
        print(f"Official evaluators are ready in {external}")


if __name__ == "__main__":
    main()

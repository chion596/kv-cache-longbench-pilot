"""Download the original LongBench dataset archive to a user-owned local file.

We do not redistribute the benchmark data in this repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/raw/longbench-data.zip")
    parser.add_argument("--repo-id", default="zai-org/LongBench")
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cache_path = Path(hf_hub_download(
        repo_id=args.repo_id, repo_type="dataset", filename="data.zip", revision=args.revision
    ))
    shutil.copyfile(cache_path, output)
    info = {"repo_id": args.repo_id, "revision": args.revision,
            "archive": str(output), "sha256": sha256_file(output)}
    with output.with_suffix(".provenance.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()

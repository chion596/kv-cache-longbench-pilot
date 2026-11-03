"""Read selected official ZIP members over HTTP; report context independence.

No model inference or package installation. No source documents are saved.
The cluster archive SHA must be compared with the recorded upstream LFS SHA.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import random
import struct
import urllib.request
import zlib

REPO = "https://huggingface.co/datasets/zai-org/LongBench"


def read_url_json(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="5e628be450b7e67fb7ae6e201bd6d8f7056f7672")
    parser.add_argument("--output", default="results/audits/data_independence.json")
    args = parser.parse_args()
    revision = args.revision
    tree = read_url_json(
        f"https://huggingface.co/api/datasets/zai-org/LongBench/tree/{revision}"
    )
    archive_info = next(item for item in tree if item["path"] == "data.zip")
    url = f"{REPO}/resolve/{revision}/data.zip"

    def get_range(spec):
        request = urllib.request.Request(
            url, headers={"Range": "bytes=" + spec, "Accept-Encoding": "identity"}
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            if response.status != 206:
                raise RuntimeError(f"Expected HTTP 206, got {response.status}")
            return response.read()

    tail = get_range("-65536")
    position = tail.rfind(b"PK\x05\x06")
    if position < 0:
        raise RuntimeError("ZIP end record not found")
    end = struct.unpack("<4s4H2IH", tail[position:position + 22])
    central_size, central_offset = end[5], end[6]
    central = get_range(f"{central_offset}-{central_offset + central_size - 1}")
    entries = {}
    position = 0
    while central[position:position + 4] == b"PK\x01\x02":
        header = struct.unpack("<4s6H3I5H2I", central[position:position + 46])
        name_size, extra_size, comment_size = header[10:13]
        name = central[position + 46:position + 46 + name_size].decode()
        entries[name] = (header[4], header[7], header[8], header[16])
        position += 46 + name_size + extra_size + comment_size

    summaries = []
    for dataset in ("multifieldqa_en", "multifieldqa_zh", "passage_retrieval_en"):
        name = next(
            value for value in entries
            if value.endswith("/" + dataset + ".jsonl") or value == dataset + ".jsonl"
        )
        method, expected_crc, compressed_size, offset = entries[name]
        local_header = get_range(f"{offset}-{offset + 29}")
        values = struct.unpack("<4s5H3I2H", local_header)
        start = offset + 30 + values[-2] + values[-1]
        packed = get_range(f"{start}-{start + compressed_size - 1}")
        if len(packed) != compressed_size:
            raise RuntimeError(f"Incomplete member: {name}")
        if method == 8:
            raw = zlib.decompress(packed, -15)
        elif method == 0:
            raw = packed
        else:
            raise RuntimeError(f"Unsupported compression: {method}")
        if zlib.crc32(raw) & 0xFFFFFFFF != expected_crc:
            raise RuntimeError(f"CRC mismatch: {name}")
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        contexts = [hashlib.sha256(row["context"].encode()).hexdigest() for row in rows]
        pilot = sorted(random.Random(2026).sample(range(len(rows)), 40))
        available = sorted(set(range(len(rows))) - set(pilot))
        heldout = sorted(random.Random(2027).sample(available, 80))
        pilot_contexts = {contexts[index] for index in pilot}
        heldout_contexts = {contexts[index] for index in heldout}
        groups = collections.Counter(contexts)
        summary = {
            "dataset": dataset,
            "member_sha256": hashlib.sha256(raw).hexdigest(),
            "n": len(rows),
            "unique_contexts": len(groups),
            "largest_context_group": max(groups.values()),
            "pilot_n": len(pilot),
            "pilot_unique_contexts": len(pilot_contexts),
            "heldout_n": len(heldout),
            "heldout_unique_contexts": len(heldout_contexts),
            "overlapping_contexts": len(pilot_contexts & heldout_contexts),
            "heldout_questions_with_pilot_context": sum(
                contexts[index] in pilot_contexts for index in heldout
            ),
            "pilot_indices": pilot,
            "heldout_indices": heldout,
        }
        summaries.append(summary)
        print(json.dumps({k: v for k, v in summary.items() if not k.endswith("_indices")}), flush=True)

    output = {
        "source": REPO,
        "revision": revision,
        "archive_sha256_from_upstream_lfs": archive_info["lfs"]["oid"],
        "archive_size_bytes": archive_info["size"],
        "verification": "Member CRC and member SHA-256; archive SHA is upstream LFS metadata.",
        "cluster_archive_verified": False,
        "context_grouping": "SHA-256 of exact UTF-8 context, without the question.",
        "selection": "Same algorithms as pilot n40 seed2026 and disjoint n80 seed2027.",
        "datasets": summaries,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved: {destination}", flush=True)


if __name__ == "__main__":
    main()

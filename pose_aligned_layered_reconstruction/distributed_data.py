"""Full manifest verification split across ranks, with durable progress."""
from __future__ import annotations

import json
import time

from .paired_dataset import canonical_json_sha256, sha256_file
from .prepare_official_experiment import atomic_json


def verify_dataset_shard(dataset, rank, world_size, output):
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("invalid verification rank/world size")
    indices = list(range(rank, len(dataset), world_size))
    manifest_sha = sha256_file(dataset.manifest_path)
    begin = time.time()
    verified_bytes = 0

    def publish(count):
        row = {"phase": "verify_data", "rank": rank, "world_size": world_size,
               "verified_objects": count, "total_objects": len(indices),
               "verified_bytes": verified_bytes, "elapsed_seconds": time.time() - begin,
               "updated_epoch": time.time(), "manifest_sha256": manifest_sha}
        atomic_json(output / f"startup_rank{rank}.json", row)
        print(json.dumps({"data_verification": row}), flush=True)

    publish(0)
    for count, index in enumerate(indices, 1):
        row = dataset.rows[index]
        path = dataset._resolve(row["sample"])
        if not row.get("sha256") or sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"paired sample hash differs: {path}")
        verified_bytes += path.stat().st_size
        if count % 10 == 0 or count == len(indices):
            publish(count)
    return {"rank": rank, "world_size": world_size, "verified_objects": len(indices),
            "manifest_sha256": manifest_sha,
            "verified_uid_hash": canonical_json_sha256([dataset.rows[i]["uid"] for i in indices])}


def validate_verification_receipts(dataset, receipts):
    world = len(receipts)
    manifest_sha = sha256_file(dataset.manifest_path)
    if not world:
        raise ValueError("missing verification receipts")
    for rank, receipt in enumerate(receipts):
        indices = range(rank, len(dataset), world)
        if (receipt["rank"] != rank or receipt["world_size"] != world
                or receipt["manifest_sha256"] != manifest_sha
                or receipt["verified_objects"] != len(indices)
                or receipt["verified_uid_hash"] != canonical_json_sha256([dataset.rows[i]["uid"] for i in indices])):
            raise ValueError("incomplete or mismatched distributed data verification")
    return {"all_sample_hashes_verified_before_training": True,
            "verification_method": "disjoint_manifest_index_shards_full_SHA256_then_all_rank_receipts",
            "verification_receipts": receipts}

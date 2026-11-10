#!/usr/bin/env python3
"""Export one self-contained five-level training example from a materialization."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialization", type=Path, required=True)
    parser.add_argument("--subset", default="5k")
    parser.add_argument("--sample-id")
    parser.add_argument("--keypoint-registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def copy(source: str | Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination.as_posix()


def main() -> None:
    args = parse_args()
    subset_dir = args.materialization / "subsets" / args.subset
    levels = {}
    for level in range(1, 6):
        rows = load_jsonl(subset_dir / f"level_{level}.jsonl")
        if not rows:
            raise RuntimeError(f"No Level-{level} conversations in {subset_dir}")
        if args.sample_id:
            matches = [row for row in rows if row["sample_id"] == args.sample_id]
            if len(matches) != 1:
                raise RuntimeError(f"Expected one Level-{level} row for {args.sample_id}")
            levels[level] = matches[0]
        else:
            levels[level] = rows[0]
    sample_id = levels[1]["sample_id"]
    if any(row["sample_id"] != sample_id for row in levels.values()):
        raise RuntimeError("Selected level rows do not describe the same sample")

    canonical_ref = levels[1]["canonical_annotation"]
    canonical_path = Path(canonical_ref["path"])
    canonical = None
    with canonical_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line_number == canonical_ref["line"]:
                canonical = json.loads(line)
                break
    if canonical is None or canonical["sample_id"] != sample_id:
        raise RuntimeError("Canonical reference does not resolve to the selected sample")

    output = args.output
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    source_paths = canonical["provenance"]["raw_paths"]
    image_suffix = Path(source_paths["image"]).suffix.lower()
    relative = {
        "image": copy(source_paths["image"], output / "images" / f"rgb{image_suffix}"),
        "label_2d": copy(source_paths["label_2d"], output / "raw_labels" / "label_2d.json"),
        "label_3d": copy(source_paths["label_3d"], output / "raw_labels" / "label_3d.json"),
        "occlusion": copy(source_paths["occlusion"], output / "raw_labels" / "occlusion.json"),
    }
    if source_paths.get("depth"):
        relative["depth"] = copy(source_paths["depth"], output / "images" / "depth_u16.png")
    if source_paths.get("depth_metadata"):
        relative["depth_metadata"] = copy(
            source_paths["depth_metadata"], output / "raw_labels" / "depth_metadata.json"
        )
    relative["keypoint_registry"] = copy(
        args.keypoint_registry, output / "schemas" / "excavision_keypoints_v1.json"
    )

    canonical["image"]["path"] = relative["image"]
    canonical["depth"]["path"] = relative.get("depth")
    canonical["depth"]["metadata_path"] = relative.get("depth_metadata")
    canonical["provenance"]["raw_paths"] = {
        **{key: None for key in canonical["provenance"]["raw_paths"]}, **relative
    }
    canonical_destination = output / "canonical" / "annotation.json"
    write_json(canonical_destination, canonical)

    portable_rows = []
    for level, row in levels.items():
        row["images"] = [
            relative["image"] if index == 0 else relative["depth"]
            for index, _ in enumerate(row["images"])
        ]
        row["canonical_annotation"] = "canonical/annotation.json"
        destination = output / "qwen" / f"level_{level}.jsonl"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
        portable_rows.append(row)
    (output / "qwen" / "qwen_sft_all_5_levels.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n"
                for row in portable_rows),
        encoding="utf-8",
    )

    readme = f"""# One-image five-level Qwen training bundle

Sample: `{sample_id}` from the curated `{args.subset}` subset.

The bundle is self-contained and uses relative paths. It contains RGB, aligned
16-bit depth and its encoding metadata, raw 2D/3D/occlusion labels, the unified
canonical annotation, the 48-landmark registry, and one Qwen conversation for
each experiment level. Use `qwen/qwen_sft_all_5_levels.jsonl` to load all five
examples, or an individual `qwen/level_N.jsonl` file for a level-specific run.

The five rows deliberately share one image. They are a format and pipeline
test, not a statistically meaningful training dataset.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    files = sorted(path for path in output.rglob("*") if path.is_file())
    manifest = {
        "bundle_schema": "excavision-one-image-qwen-bundle/1.0.0",
        "sample_id": sample_id,
        "source_subset": args.subset,
        "conversation_count": 5,
        "files": [{"path": path.relative_to(output).as_posix(), "sha256": sha256(path),
                   "bytes": path.stat().st_size} for path in files],
    }
    write_json(output / "bundle_manifest.json", manifest)
    if args.archive:
        archive = output.with_suffix(".tar.gz")
        with tarfile.open(archive, "w:gz") as stream:
            stream.add(output, arcname=output.name)
        print(f"Archive: {archive}")
    print(f"Bundle: {output}")


if __name__ == "__main__":
    main()

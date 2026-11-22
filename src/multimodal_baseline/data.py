"""Validate paired annotations and copy only selected images from an uncompressed tar."""

from collections import Counter
import codecs
import csv
import hashlib
import json
import random
from pathlib import Path, PurePosixPath
import tarfile

from .storage import write_json


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_path(root, value):
    return (root / value).resolve()


def dataset_name(config):
    value = config.get("dataset", "newsclippings")
    if value not in ("newsclippings", "verite"):
        raise ValueError(f"Unknown dataset: {value!r}")
    return value


VERITE_BINARY_MAPPING = {"true": "MATCH", "miscaptioned": "MISMATCH", "out-of-context": "MISMATCH"}


def validate_verite_snapshot(config, manifest):
    """Validate a prepared/saved snapshot without CSV, images or a server."""
    if (manifest.get("dataset") != "VERITE" or manifest.get("variant") != "adapted_binary"
            or manifest.get("binary_label_mapping") != config.get("binary_label_mapping")):
        raise ValueError("VERITE snapshot dataset or adapted binary mapping differs from the configuration.")
    rows = manifest["records"]
    if (len(rows) != config["expected_records"] or len({r["id"] for r in rows}) != len(rows)
            or [r["annotation_index"] for r in rows] != list(range(len(rows)))):
        raise ValueError("VERITE snapshot requires every ordered, unique source ID and annotation index.")
    for row in rows:
        if (row["source_category"] not in VERITE_BINARY_MAPPING
                or row["true_label"] != config["binary_label_mapping"][row["source_category"]]
                or not row["caption"].strip() or not row["image_sha256"]
                or row["caption_sha256"] != hashlib.sha256(row["caption"].encode("utf-8")).hexdigest()):
            raise ValueError(f"VERITE snapshot input/category mismatch at source ID {row['id']!r}.")
    for field, source in (("source_category_counts", "source_category"), ("class_counts", "true_label")):
        categories = VERITE_BINARY_MAPPING if source == "source_category" else ("MATCH", "MISMATCH")
        actual = {key: sum(r[source] == key for r in rows) for key in categories}
        if manifest[field] != actual:
            raise ValueError(f"VERITE snapshot {field} does not match its records.")
        expected = config.get("expected_source_counts" if source == "source_category" else "expected_binary_counts")
        if expected is not None and actual != expected:
            raise ValueError(f"VERITE snapshot {field} differs from configured source counts.")


def verite_records(root, config):
    """Read exactly one supplied CSV; keep source IDs/categories and image bytes."""
    mapping = config.get("binary_label_mapping")
    if mapping != VERITE_BINARY_MAPPING:
        raise ValueError("VERITE adapted binary evaluation requires true -> MATCH; miscaptioned/out-of-context -> MISMATCH.")
    source = local_path(root, config["annotations"])
    image_root = local_path(root, config["images_dir"])
    id_column = config.get("id_column", "")
    records, ids, images = [], set(), {}
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = (id_column, "caption", "image_path", "label")
        if (not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames)
                or not set(required) <= set(reader.fieldnames)):
            raise ValueError(f"VERITE CSV requires unique columns {required!r}; found {reader.fieldnames!r}.")
        for index, row in enumerate(reader):
            if None in row or any(row.get(key) is None or not row[key].strip() for key in required):
                raise ValueError(f"VERITE CSV record {index}: missing/malformed required fields.")
            identifier, category = row[id_column].strip(), row["label"]
            if identifier in ids:
                raise ValueError(f"Duplicate VERITE source ID: {identifier!r}")
            if category not in mapping:
                raise ValueError(f"VERITE ID {identifier}: unknown source label {category!r}; expected {list(mapping)}.")
            relative = Path(row["image_path"].replace("\\", "/"))
            if relative.is_absolute() or relative.drive or ".." in relative.parts:
                raise ValueError(f"VERITE ID {identifier}: image_path must be relative to images-dir.")
            image = (image_root / relative).resolve()
            if not image.is_relative_to(image_root) or not image.is_file():
                raise FileNotFoundError(f"VERITE ID {identifier}: missing image within images-dir: {relative}")
            if image not in images:
                images[image] = {"image_sha256": sha256_file(image), "image_bytes": image.stat().st_size}
            records.append({"annotation_index": index, "id": identifier, "caption": row["caption"],
                            "source_category": category, "true_label": mapping[category],
                            "image_path": image.relative_to(root).as_posix() if image.is_relative_to(root) else str(image),
                            "image_relative_path": relative.as_posix(),
                            "caption_sha256": hashlib.sha256(row["caption"].encode("utf-8")).hexdigest(),
                            **images[image]})
            ids.add(identifier)
        physical_lines = reader.line_num
    if not records:
        raise ValueError("VERITE CSV has no data records.")
    source_counts = {key: sum(r["source_category"] == key for r in records) for key in mapping}
    binary_counts = {key: sum(r["true_label"] == key for r in records) for key in ("MATCH", "MISMATCH")}
    if len(records) != config["expected_records"]:
        raise ValueError(f"VERITE CSV has {len(records)} records, expected {config['expected_records']}; source counts={source_counts}. No rows were dropped.")
    for key, actual in (("expected_source_counts", source_counts), ("expected_binary_counts", binary_counts)):
        if key in config and config[key] != actual:
            raise ValueError(f"VERITE {key} mismatch: actual={actual}, configured={config[key]}. No rows were dropped.")
    pairs = {(r["image_sha256"], r["caption"]) for r in records}
    return {"schema_version": 1, "dataset": "VERITE", "variant": "adapted_binary", "split": config["split"],
            "evaluation_protocol": "adapted binary evaluation", "binary_label_mapping": dict(mapping),
            "annotations_sha256": sha256_file(source), "source_id_column": id_column,
            "source_category_counts": source_counts, "class_counts": binary_counts,
            "unique_images": len(images), "records": records,
            "audit": {"records": len(records), "unique_ids": len(ids), "physical_csv_lines": physical_lines,
                      "duplicate_image_caption_pairs": len(records) - len(pairs)}}


def input_identity(manifest):
    """Inference/label identity; storage paths, tar offsets and timestamps are provenance only."""
    keys = ("annotation_index", "pair_index", "id", "image_id", "caption", "image_sha256", "true_label")
    if manifest.get("dataset") == "VERITE":
        return {"dataset": "VERITE", "variant": manifest["variant"], "split": manifest["split"],
                "binary_label_mapping": manifest["binary_label_mapping"],
                "records": [{key: row[key] for key in ("annotation_index", "id", "caption", "image_sha256", "source_category", "true_label")}
                            for row in manifest["records"]]}
    return {"split": manifest.get("split"), "annotations_sha256": manifest.get("annotations_sha256"),
            "records": [{key: row.get(key) for key in keys} for row in manifest["records"]]}


def select_test_records(path, seed=42, expected_count=7264):
    """Use every unique annotation in official order; never deduplicate caption ids."""
    return select_records(path, seed, expected_count, split="test")


def select_records(path, seed=42, expected_count=None, *, split="train", groups=None):
    """Shared pair validation; train sampling never reads test or VERITE."""
    path = Path(path)
    if split not in ("train", "val", "test") or path.name.lower() != f"{split}.json" or "verite" in (p.lower() for p in path.parts):
        raise ValueError("Unexpected NewsCLIPpings split.")
    doc = json.loads(path.read_text(encoding="utf-8"))
    rows = doc["annotations"]
    sources = doc["source_datasets"]
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} test records; found {len(rows)}.")
    if not rows or len(rows) % 2:
        raise ValueError("Annotations must contain complete adjacent pairs.")
    pairs, relations = [], Counter()
    for index in range(0, len(rows), 2):
        pristine, falsified = rows[index:index + 2]
        for row in (pristine, falsified):
            if any(type(row.get(key)) is not int for key in ("id", "image_id", "source_dataset")):
                raise ValueError(f"Invalid identifiers at annotation {index}.")
            if str(row["source_dataset"]) not in sources:
                raise ValueError(f"Unknown source_dataset at annotation {index}.")
        shared_caption = pristine["id"] == falsified["id"]
        shared_image = pristine["image_id"] == falsified["image_id"]
        if not (
            pristine["falsified"] is False and falsified["falsified"] is True
            and pristine["id"] == pristine["image_id"]
            and falsified["id"] != falsified["image_id"]
            and pristine["source_dataset"] == falsified["source_dataset"]
            and shared_caption != shared_image
        ):
            raise ValueError(f"Unverified adjacent pair at annotations {index}, {index + 1}; no fallback pairing.")
        relation = "shared_caption" if shared_caption else "shared_image"
        relations[relation] += 1
        pairs.append({"pair_index": index // 2, "relation": relation, "rows": [pristine, falsified]})
    verified_pairs = len(pairs)
    if groups is not None:
        if split not in ("train", "val") or not 0 < groups <= len(pairs):
            raise ValueError("Sampling is allowed only for valid train/val groups.")
        pairs = random.Random(seed).sample(pairs, groups)
    selected = []
    for pair in pairs:
        for offset, row in enumerate(pair["rows"]):
            selected.append({
                **row, "annotation_index": 2 * pair["pair_index"] + offset,
                "pair_index": pair["pair_index"], "pair_relation": pair["relation"],
                "true_label": "MISMATCH" if row["falsified"] else "MATCH",
            })
    return {
        "schema_version": 1, "split": split, "seed": seed, "pair_count": len(pairs),
        "annotations_sha256": sha256_file(path),
        "source_datasets": sources,
        "audit": {"records": len(rows), "verified_pairs": verified_pairs, "relations": dict(relations)},
        "selected_pair_indices": [p["pair_index"] for p in pairs],
        "class_counts": dict(Counter(r["true_label"] for r in selected)),
        "records": selected,
    }


def safe_member_name(name):
    if "\\" in name or ":" in name or "\x00" in name:
        raise ValueError(f"Unsafe archive path: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe archive path: {name!r}")
    return path.as_posix()


def archive_identity(path):
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def scan_archive(path, wanted=None):
    """Read headers, seek over image bytes; never extractall or retain all members."""
    found = {}
    with tarfile.open(path, "r:", stream=True) as archive:
        for index, member in enumerate(archive, 1):
            if index % 100000 == 0:
                print(f"Archive: inspected {index:,} headers", flush=True)
            if not member.isfile():
                continue
            name = member.name
            interested = PurePosixPath(name).name == "data.json" if wanted is None else name.lstrip("./") in wanted
            if not interested:
                continue
            name = safe_member_name(name)
            if wanted is not None and name not in wanted:
                continue
            if name in found:
                raise ValueError(f"Duplicate archive member: {name}")
            if member.sparse is not None:
                raise ValueError(f"Sparse archive members are unsupported: {name}")
            found[name] = {"name": name, "offset": member.offset_data, "size": member.size}
    return found


def iter_json_array(handle, size):
    """Read a large UTF-8 JSON array with bounded memory, including split UTF-8 characters."""
    decoder = json.JSONDecoder()
    utf8 = codecs.getincrementaldecoder("utf-8-sig")()
    buffer, position, remaining = "", 0, size

    def refill():
        nonlocal buffer, position, remaining
        if not remaining:
            return False
        raw = handle.read(min(65536, remaining))
        if not raw:
            raise ValueError("Truncated VisualNews metadata.")
        remaining -= len(raw)
        buffer = buffer[position:] + utf8.decode(raw, final=remaining == 0)
        position = 0
        if len(buffer) > 16 * 1024 * 1024:
            raise ValueError("Oversized or invalid VisualNews JSON record.")
        return True

    def peek():
        nonlocal position
        while True:
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if position < len(buffer):
                return buffer[position]
            if not refill():
                return ""

    if peek() != "[":
        raise ValueError("Expected a VisualNews JSON array.")
    position += 1
    if peek() != "]":
        while True:
            if peek() != "{":
                raise ValueError("Expected a VisualNews metadata object.")
            while True:
                try:
                    record, end = decoder.raw_decode(buffer, position)
                    position = end
                    break
                except json.JSONDecodeError:
                    if not refill():
                        raise ValueError("Invalid VisualNews JSON.") from None
            yield record
            separator = peek()
            position += 1
            if separator == "]":
                break
            if separator != ",":
                raise ValueError("Expected a comma between VisualNews records.")
    else:
        position += 1
    if peek():
        raise ValueError("Unexpected trailing data in VisualNews metadata.")


def read_metadata(archive, member, needed):
    selected, total = {}, 0
    with archive.open("rb") as handle:
        handle.seek(member["offset"])
        for total, record in enumerate(iter_json_array(handle, member["size"]), 1):
            identifier = record.get("id")
            if identifier in needed:
                if identifier in selected:
                    raise ValueError(f"Duplicate VisualNews id: {identifier}")
                selected[identifier] = {k: record.get(k) for k in ("id", "caption", "image_path")}
            if total % 250000 == 0:
                print(f"VisualNews: inspected {total:,} metadata records", flush=True)
    missing = needed - selected.keys()
    if missing:
        raise ValueError(f"VisualNews metadata is missing ids: {sorted(missing)}")
    return selected, total


def image_candidates(metadata_name, image_path):
    if not isinstance(image_path, str) or not image_path:
        raise ValueError("Missing image_path in VisualNews.")
    relative = safe_member_name(image_path)
    parent = PurePosixPath(metadata_name).parent
    return {relative, (parent / relative).as_posix()}


def copy_member(archive, member, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    digest = hashlib.sha256()
    remaining = member["size"]
    with archive.open("rb") as source, temporary.open("wb") as output:
        source.seek(member["offset"])
        while remaining:
            chunk = source.read(min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError(f"Truncated archive at {member['name']}")
            remaining -= len(chunk)
            output.write(chunk)
            digest.update(chunk)
    temporary.replace(target)
    return digest.hexdigest()


def prepare(root, config, selection=None):
    if dataset_name(config) == "verite":
        manifest = verite_records(root, config)
        write_json(local_path(root, config["prepared_dir"]) / "manifest.json", manifest)
        print(f"Prepared VERITE adapted binary evaluation: {len(manifest['records'])} records, {manifest['unique_images']} existing images; no image copies.")
        return manifest
    output = local_path(root, config["prepared_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if selection is None:
        selection = select_test_records(local_path(root, config["annotations"]), config["seed"], config["expected_records"])
    write_json(output / "selection.json", selection)
    archive = local_path(root, config["archive"])
    if not archive.is_file():
        raise FileNotFoundError(f"Missing {archive}. Need VisualNews origin.tar containing data.json and images; selection.json is saved. Nothing downloaded or fabricated.")
    identity = archive_identity(archive)
    cache_path = output / "archive_metadata_index.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    if cache.get("archive") != identity:
        cache = {"archive": identity, "metadata_members": scan_archive(archive)}
        write_json(cache_path, cache)
    candidates = cache["metadata_members"]
    if "origin/data.json" in candidates:
        metadata = candidates["origin/data.json"]
    elif len(candidates) == 1:
        metadata = next(iter(candidates.values()))
    else:
        raise ValueError(f"Cannot uniquely locate VisualNews data.json: {list(candidates)}")
    print(f"VisualNews metadata: {metadata['name']} ({metadata['size']:,} bytes)", flush=True)
    needed = {r[k] for r in selection["records"] for k in ("id", "image_id")}
    subset_path = output / "visualnews_subset.json"
    subset = json.loads(subset_path.read_text(encoding="utf-8")) if subset_path.exists() else {}
    if subset.get("archive") == identity and subset.get("metadata_member") == metadata and subset.get("ids") == sorted(needed):
        mapping = {r["id"]: r for r in subset["records"]}
        metadata_count = subset["metadata_count"]
    else:
        mapping, metadata_count = read_metadata(archive, metadata, needed)
        write_json(subset_path, {"archive": identity, "metadata_member": metadata, "metadata_count": metadata_count,
                                "ids": sorted(needed), "records": list(mapping.values())})
    image_ids = sorted({r["image_id"] for r in selection["records"]})
    paths = {identifier: image_candidates(metadata["name"], mapping[identifier]["image_path"]) for identifier in image_ids}
    wanted = set().union(*paths.values())
    image_index_path = output / "archive_image_index.json"
    image_index = json.loads(image_index_path.read_text(encoding="utf-8")) if image_index_path.exists() else {}
    if image_index.get("archive") != identity or image_index.get("wanted") != sorted(wanted):
        image_index = {"archive": identity, "wanted": sorted(wanted), "members": scan_archive(archive, wanted)}
        write_json(image_index_path, image_index)
    members = image_index["members"]
    resolved = {}
    for identifier, possibilities in paths.items():
        matches = possibilities & members.keys()
        if len(matches) != 1:
            raise ValueError(f"Expected one image for image_id={identifier}; found {sorted(matches)}; searched {sorted(possibilities)}")
        resolved[identifier] = members[matches.pop()]
    records = []
    extracted = {}
    for identifier, member in resolved.items():
        target = output / "images" / f"{identifier}{PurePosixPath(member['name']).suffix.lower()}"
        # Recopy only the selected bytes; no image recompression or resizing.
        digest = copy_member(archive, member, target)
        extracted[identifier] = {"image_path": target.relative_to(root).as_posix() if target.is_relative_to(root) else str(target), "image_sha256": digest,
                                "archive_member": member["name"], "image_bytes": member["size"]}
    for row in selection["records"]:
        caption = mapping[row["id"]]["caption"]
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError(f"Missing caption for id={row['id']}")
        records.append({**row, "caption": caption, **extracted[row["image_id"]]})
    if archive_identity(archive) != identity:
        raise ValueError("Archive changed during preparation; rerun with a stable archive.")
    manifest = {**selection, "archive": identity, "metadata_member": metadata,
                "metadata_records": metadata_count, "unique_images": len(extracted),
                "extracted_bytes": sum(x["image_bytes"] for x in extracted.values()), "records": records}
    write_json(output / "manifest.json", manifest)
    print(f"Prepared {len(records)} records / {len(extracted)} images in {output}", flush=True)
    return manifest

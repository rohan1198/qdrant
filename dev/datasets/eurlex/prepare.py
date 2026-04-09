"""Prepare Multi-EurLex dataset for testbench use.

Extracts English EU legal documents with EUROVOC hierarchical labels (3 levels).
L1: 21 concepts (e.g., "trade", "finance", "international relations")
L2: 127 concepts
L3: 556 concepts

Source: nlpaueb/multi_eurlex on HuggingFace
EUROVOC: http://eurovoc.europa.eu/

Usage:
    python prepare.py
"""

import json
import zipfile
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
ZIP_FILE = DATA_DIR / "multi_eurlex_translated.zip"
DESCRIPTORS_FILE = DATA_DIR / "eurovoc_descriptors.json"
DOCUMENTS_FILE = DATA_DIR / "documents.jsonl"
HIERARCHY_FILE = DATA_DIR / "hierarchy.json"

SPLITS_IN_ZIP = ["train.jsonl", "dev.jsonl", "test.jsonl"]


def load_descriptors() -> dict[str, str]:
    """Load EUROVOC concept ID → English name mapping."""
    print("Loading EUROVOC descriptors...")
    with open(DESCRIPTORS_FILE) as f:
        raw = json.load(f)

    # Map concept ID → English descriptor name
    desc = {}
    for concept_id, names in raw.items():
        en_name = names.get("en", concept_id)
        desc[concept_id] = en_name

    print(f"  Loaded {len(desc)} descriptors")
    return desc


def build_hierarchy(descriptors: dict[str, str]) -> dict:
    """Build EUROVOC hierarchy from the level structure.

    Since EUROVOC is a DAG and we only have level assignments (not parent-child edges)
    from the dataset, we infer the hierarchy by observing which L3 concepts co-occur
    with which L2 concepts, and which L2 with which L1.

    Returns hierarchy info dict.
    """
    print("\nBuilding hierarchy from co-occurrence in documents...")

    # We'll build this from the actual documents (see build_documents)
    # For now, return a placeholder that gets filled during document processing
    return {
        "l1_to_l2": defaultdict(set),  # L1 concept → set of L2 concepts
        "l2_to_l3": defaultdict(set),  # L2 concept → set of L3 concepts
    }


def parse_documents(descriptors: dict[str, str], hierarchy: dict) -> list[dict]:
    """Parse Multi-EurLex JSONL files from zip, extracting English text + EUROVOC labels."""
    print(f"\nParsing documents from {ZIP_FILE.name}...")

    documents = []

    with zipfile.ZipFile(ZIP_FILE) as zf:
        for split_file in SPLITS_IN_ZIP:
            split_name = split_file.replace(".jsonl", "")
            if split_name == "dev":
                split_name = "dev"

            print(f"\n  Parsing {split_file} (split={split_name})...")
            count = 0

            with zf.open(split_file) as f:
                for line in f:
                    record = json.loads(line)

                    # Extract English text
                    text = record.get("text", {}).get("en", "")
                    if not text:
                        continue

                    celex_id = record.get("celex_id", "")
                    ec = record.get("eurovoc_concepts", {})

                    # Get labels at each level
                    l1_ids = ec.get("level_1", [])
                    l2_ids = ec.get("level_2", [])
                    l3_ids = ec.get("level_3", [])
                    all_ids = ec.get("all_levels", [])

                    if not l1_ids:
                        continue

                    # Build hierarchy paths from labels
                    # Map concept IDs to names
                    l1_names = [descriptors.get(cid, cid) for cid in l1_ids]
                    l2_names = [descriptors.get(cid, cid) for cid in l2_ids]
                    l3_names = [descriptors.get(cid, cid) for cid in l3_ids]

                    # Track hierarchy co-occurrence
                    for l1 in l1_ids:
                        for l2 in l2_ids:
                            hierarchy["l1_to_l2"][l1].add(l2)
                    for l2 in l2_ids:
                        for l3 in l3_ids:
                            hierarchy["l2_to_l3"][l2].add(l3)

                    # Primary path: deepest available
                    if l3_names:
                        primary_path = f"{l1_names[0]}/{l2_names[0]}/{l3_names[0]}" if l2_names else f"{l1_names[0]}/{l3_names[0]}"
                        depth = 3
                    elif l2_names:
                        primary_path = f"{l1_names[0]}/{l2_names[0]}"
                        depth = 2
                    else:
                        primary_path = l1_names[0]
                        depth = 1

                    # All paths: all L1/L2/L3 combinations
                    all_paths = []
                    for l1n in l1_names:
                        all_paths.append(l1n)
                        for l2n in l2_names:
                            all_paths.append(f"{l1n}/{l2n}")
                            for l3n in l3_names:
                                all_paths.append(f"{l1n}/{l2n}/{l3n}")

                    # Deduplicate
                    all_paths = list(dict.fromkeys(all_paths))

                    domain = l1_names[0] if l1_names else ""
                    area = l2_names[0] if l2_names else ""

                    documents.append({
                        "id": celex_id,
                        "text": text,
                        "tier": "content",
                        "depth": depth,
                        "domain": domain,
                        "area": area,
                        "hierarchy_path": primary_path,
                        "all_paths": all_paths,
                        "has_abstract": True,
                    })
                    count += 1

            print(f"    Parsed {count} documents")

    print(f"\n  Total documents: {len(documents)}")
    return documents


def main():
    if DOCUMENTS_FILE.exists():
        print(f"documents.jsonl already exists at {DOCUMENTS_FILE}")
        print(f"Delete it to regenerate.")
        return

    if not ZIP_FILE.exists():
        print(f"ERROR: {ZIP_FILE} not found. Download from HuggingFace first.")
        return

    if not DESCRIPTORS_FILE.exists():
        print(f"ERROR: {DESCRIPTORS_FILE} not found.")
        print("Download: curl -L https://raw.githubusercontent.com/nlpaueb/multi-eurlex/master/data/eurovoc_descriptors.json -o data/eurovoc_descriptors.json")
        return

    # Load EUROVOC descriptors
    descriptors = load_descriptors()

    # Parse documents
    hierarchy = build_hierarchy(descriptors)
    documents = parse_documents(descriptors, hierarchy)

    if not documents:
        print("ERROR: No documents parsed!")
        return

    # Summary stats
    print(f"\n=== Summary ===")
    print(f"  Total documents: {len(documents)}")

    depth_counts = defaultdict(int)
    for doc in documents:
        depth_counts[doc["depth"]] += 1
    print(f"  Depth distribution: {dict(sorted(depth_counts.items()))}")

    domain_counts = defaultdict(int)
    for doc in documents:
        domain_counts[doc["domain"]] += 1
    print(f"  Domains (L1): {len(domain_counts)}")
    for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    {domain}: {count}")

    unique_areas = len(set(doc["area"] for doc in documents if doc["area"]))
    print(f"  Unique areas (L2): {unique_areas}")

    multi_path = sum(1 for d in documents if len(d["all_paths"]) > 1)
    print(f"  Multi-path documents: {multi_path} ({multi_path/len(documents)*100:.1f}%)")

    # Generate intermediate tier documents
    print(f"\n=== Generating intermediate tier documents ===")
    category_docs: dict[int, set] = defaultdict(set)

    for doc in documents:
        parts = doc["hierarchy_path"].split("/")
        for level in range(1, len(parts)):
            category_path = "/".join(parts[:level])
            category_docs[level].add(category_path)

    tier_names = {1: "narrative", 2: "story"}
    tier_doc_count = 0

    for level in sorted(category_docs.keys()):
        tier_name = tier_names.get(level, f"tier_L{level}")
        for category_path in sorted(category_docs[level]):
            parts = category_path.split("/")
            domain = parts[0] if parts else ""
            area = parts[1] if len(parts) >= 2 else ""

            text = " - ".join(parts)
            documents.append({
                "id": f"tier_{tier_name}_{category_path.replace('/', '_').replace(' ', '_')}",
                "text": text,
                "tier": tier_name,
                "depth": level,
                "domain": domain,
                "area": area,
                "hierarchy_path": category_path,
                "all_paths": [category_path],
                "has_abstract": False,
            })
            tier_doc_count += 1

    print(f"  Generated {tier_doc_count} intermediate tier documents")

    # Final tier breakdown
    tier_counts = defaultdict(int)
    for doc in documents:
        tier_counts[doc["tier"]] += 1
    print(f"\n  Final tier breakdown: {dict(sorted(tier_counts.items()))}")
    print(f"  Total documents: {len(documents)}")

    # Finalize hierarchy info
    hierarchy_out = {
        "source": "Multi-EurLex (EUROVOC)",
        "levels": {"L1": 21, "L2": 127, "L3": 556},
        "max_depth": 3,
        "document_count": len(documents),
        "tier_counts": dict(tier_counts),
        "depth_distribution": dict(depth_counts),
        "l1_to_l2_edges": sum(len(v) for v in hierarchy["l1_to_l2"].values()),
        "l2_to_l3_edges": sum(len(v) for v in hierarchy["l2_to_l3"].values()),
    }

    # Save
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DOCUMENTS_FILE, "w") as f:
        for doc in documents:
            f.write(json.dumps(doc) + "\n")
    print(f"\nSaved {len(documents)} documents to {DOCUMENTS_FILE}")

    with open(HIERARCHY_FILE, "w") as f:
        json.dump(hierarchy_out, f, indent=2)
    print(f"Saved hierarchy info to {HIERARCHY_FILE}")

    print(f"\nDone. Run: cd ../../testbench && python embed.py --dataset eurlex")


if __name__ == "__main__":
    main()

"""Transform raw SciHTC data into testbench-ready documents.jsonl.

Pipeline:
1. Load raw papers + labels from download.py output
2. Discover hierarchy structure (depth, branching, poly-hierarchy)
3. Build hierarchy tree from observed label paths
4. Assign tiers based on discovered depth
5. Output documents.jsonl + hierarchy.json

Usage:
    python prepare.py
"""

import json
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
RAW_PAPERS_FILE = DATA_DIR / "raw_papers.jsonl"
RAW_LABELS_FILE = DATA_DIR / "raw_labels.jsonl"
DOCUMENTS_FILE = DATA_DIR / "documents.jsonl"
HIERARCHY_FILE = DATA_DIR / "hierarchy.json"


def load_raw_data():
    """Load raw papers and labels, join on paper_id."""
    print("Loading raw papers...")
    papers = {}
    with open(RAW_PAPERS_FILE) as f:
        for line in f:
            d = json.loads(line)
            papers[d["paper_id"]] = d
    print(f"  Loaded {len(papers)} papers")

    print("Loading raw labels...")
    labels_by_id = defaultdict(list)
    with open(RAW_LABELS_FILE) as f:
        for line in f:
            d = json.loads(line)
            pid = d["paper_id"]
            for label in d["labels"]:
                if label not in labels_by_id[pid]:
                    labels_by_id[pid].append(label)
    print(f"  Loaded labels for {len(labels_by_id)} papers")

    return papers, labels_by_id


def discover_hierarchy(labels_by_id: dict) -> dict:
    """Build hierarchy tree from all observed label paths.

    Discovers: separator, depth, nodes per level, poly-hierarchy.
    Returns hierarchy info dict.
    """
    print("\n=== Discovering hierarchy structure ===")

    # Collect all unique label strings
    all_labels = set()
    for labels in labels_by_id.values():
        all_labels.update(labels)

    print(f"  Unique label strings: {len(all_labels)}")
    sample_labels = sorted(all_labels)[:10]
    print(f"  Samples: {sample_labels}")

    # Discover path separator
    # SciHTC uses tilde (~) based on the paper's CCS format: "Computing methodologies~Machine learning~..."
    separators = {"~": 0, "/": 0, "->": 0, " > ": 0}
    for label in all_labels:
        for sep in separators:
            if sep in label:
                separators[sep] += 1

    print(f"  Separator counts: {separators}")
    best_sep = max(separators, key=separators.get)
    if separators[best_sep] == 0:
        print(f"  WARNING: No separator found in labels. They may be flat categories.")
        best_sep = None
    else:
        print(f"  Using separator: '{best_sep}'")

    # Split labels into path components
    path_depths = []
    all_nodes_by_level = defaultdict(set)
    tree_edges = set()  # (parent, child) pairs

    for label in all_labels:
        if best_sep:
            parts = [p.strip() for p in label.split(best_sep) if p.strip()]
        else:
            parts = [label.strip()]

        path_depths.append(len(parts))

        for level, node in enumerate(parts):
            all_nodes_by_level[level].add(node)
            if level > 0:
                tree_edges.add((parts[level - 1], node))

    max_depth = max(path_depths) if path_depths else 0
    print(f"  Max hierarchy depth: {max_depth}")
    print(f"  Nodes per level:")
    for level in sorted(all_nodes_by_level.keys()):
        print(f"    L{level}: {len(all_nodes_by_level[level])} nodes")

    # Check for poly-hierarchy (nodes with multiple parents)
    child_parents = defaultdict(set)
    for parent, child in tree_edges:
        child_parents[child].add(parent)
    multi_parent = {child: parents for child, parents in child_parents.items() if len(parents) > 1}
    print(f"  Multi-parent nodes (poly-hierarchy): {len(multi_parent)}")
    if multi_parent:
        sample = list(multi_parent.items())[:3]
        for child, parents in sample:
            print(f"    '{child}' has parents: {parents}")

    return {
        "separator": best_sep,
        "max_depth": max_depth,
        "nodes_per_level": {str(k): len(v) for k, v in all_nodes_by_level.items()},
        "total_unique_labels": len(all_labels),
        "multi_parent_count": len(multi_parent),
        "total_edges": len(tree_edges),
    }


def assign_tier(depth: int, max_depth: int) -> str:
    """Assign tier name based on depth relative to max hierarchy depth.

    Deepest level = 'content' (leaves).
    Shallowest = depends on max_depth:
      depth 3: narrative, story, content
      depth 4: theme, narrative, story, content
      depth 5: theme, narrative, sub_narrative, story, content
      depth 6: theme, narrative, sub_narrative, story, sub_story, content
    """
    tier_maps = {
        1: ["content"],
        2: ["story", "content"],
        3: ["narrative", "story", "content"],
        4: ["theme", "narrative", "story", "content"],
        5: ["theme", "narrative", "sub_narrative", "story", "content"],
        6: ["theme", "narrative", "sub_narrative", "story", "sub_story", "content"],
        7: ["root", "theme", "narrative", "sub_narrative", "story", "sub_story", "content"],
    }

    # Clamp to supported range
    effective_max = min(max_depth, max(tier_maps.keys()))
    tiers = tier_maps.get(effective_max, tier_maps[6])

    # depth is 1-indexed (1 = shallowest)
    idx = min(depth - 1, len(tiers) - 1)
    return tiers[idx]


def build_documents(papers: dict, labels_by_id: dict, hierarchy_info: dict) -> list[dict]:
    """Build documents.jsonl entries from papers + labels."""
    separator = hierarchy_info["separator"]
    max_depth = hierarchy_info["max_depth"]

    print(f"\n=== Building documents (max_depth={max_depth}) ===")

    documents = []
    skipped = 0

    for pid, paper in papers.items():
        labels = labels_by_id.get(pid, [])
        if not labels:
            skipped += 1
            continue

        # Build text
        title = paper.get("title") or ""
        abstract = paper.get("abstract") or ""
        has_abstract = bool(abstract)
        text = f"{title}. {abstract}".strip() if abstract else title.strip()
        if not text:
            skipped += 1
            continue

        # Parse all label paths
        all_paths = []
        for label in labels:
            if separator:
                path = "/".join(p.strip() for p in label.split(separator) if p.strip())
            else:
                path = label.strip()
            if path:
                all_paths.append(path)

        if not all_paths:
            skipped += 1
            continue

        # Primary path = deepest (most specific)
        primary_path = max(all_paths, key=lambda p: p.count("/") + 1)
        parts = primary_path.split("/")
        depth = len(parts)

        # Extract domain (L1) and area (L2) from primary path
        domain = parts[0] if len(parts) >= 1 else ""
        area = parts[1] if len(parts) >= 2 else ""

        # Assign tier based on leaf depth (papers are always content tier)
        tier = "content"

        documents.append({
            "id": pid,
            "text": text,
            "tier": tier,
            "depth": depth,
            "domain": domain,
            "area": area,
            "hierarchy_path": primary_path,
            "all_paths": all_paths,
            "has_abstract": has_abstract,
        })

    print(f"  Documents created: {len(documents)}")
    print(f"  Skipped (no text/labels): {skipped}")

    # Stats
    if documents:
        depths = [d["depth"] for d in documents]
        with_abs = sum(1 for d in documents if d["has_abstract"])
        unique_domains = len(set(d["domain"] for d in documents))
        unique_areas = len(set(d["area"] for d in documents))
        print(f"  Depth distribution: min={min(depths)}, max={max(depths)}, "
              f"mean={sum(depths)/len(depths):.1f}")
        print(f"  With abstract: {with_abs} ({with_abs/len(documents)*100:.1f}%)")
        print(f"  Unique domains (L1): {unique_domains}")
        print(f"  Unique areas (L2): {unique_areas}")

    # Now generate intermediate tier documents (centroids will be computed by embed.py,
    # but we need tier marker documents for the hierarchy)
    # For each unique category at each level, create a synthetic document
    print("\n  Generating intermediate tier documents...")
    category_docs = defaultdict(set)  # level -> set of category paths

    for doc in documents:
        parts = doc["hierarchy_path"].split("/")
        for level in range(1, len(parts)):  # Skip leaf level (that's the paper)
            category_path = "/".join(parts[:level])
            category_docs[level].add(category_path)

    tier_doc_count = 0
    for level in sorted(category_docs.keys()):
        tier_name = assign_tier(level, max_depth)
        for category_path in sorted(category_docs[level]):
            parts = category_path.split("/")
            domain = parts[0] if parts else ""
            area = parts[1] if len(parts) >= 2 else ""

            # Synthetic text from category names
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
    print(f"  Total documents: {len(documents)}")

    # Tier breakdown
    tier_counts = defaultdict(int)
    for doc in documents:
        tier_counts[doc["tier"]] += 1
    print(f"  Tier breakdown: {dict(sorted(tier_counts.items()))}")

    return documents


def main():
    if DOCUMENTS_FILE.exists():
        print(f"documents.jsonl already exists at {DOCUMENTS_FILE}")
        print(f"Delete it to regenerate.")
        return

    # Load raw data
    papers, labels_by_id = load_raw_data()

    # Discover hierarchy
    hierarchy_info = discover_hierarchy(labels_by_id)

    # Build documents
    documents = build_documents(papers, labels_by_id, hierarchy_info)

    if not documents:
        print("ERROR: No documents generated. Check raw data and discovery logs.")
        return

    # Save documents
    with open(DOCUMENTS_FILE, "w") as f:
        for doc in documents:
            f.write(json.dumps(doc) + "\n")
    print(f"\nSaved {len(documents)} documents to {DOCUMENTS_FILE}")

    # Save hierarchy info
    with open(HIERARCHY_FILE, "w") as f:
        json.dump(hierarchy_info, f, indent=2)
    print(f"Saved hierarchy info to {HIERARCHY_FILE}")

    print("\nDone. Run embed.py --dataset scihtc next.")


if __name__ == "__main__":
    main()

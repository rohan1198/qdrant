"""Prepare the Hierarchical WikiVitals (HWV) dataset for testbench use.

Downloads the HWV dataset from the revisitingHTC repo (CoNLL 2024, Plaud et al.)
and converts it into our standard documents.jsonl format with hierarchy metadata.

HWV structure:
- 10,012 Wikipedia article abstracts (train/val/test splits)
- 1,186 category nodes across 6 depth levels (plus virtual Root at depth 0)
- Variable-depth paths (2-6 levels)
- Single-path leaf (SPL) policy: each document has exactly one path
- Taxonomy: tab-separated parent -> children edges (hwv.taxonomy)

Source: https://github.com/RomanPlaud/revisitingHTC (MIT license)

Output: data/documents.jsonl + data/hierarchy.json

Usage:
    python prepare.py
"""

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
DOCUMENTS_FILE = DATA_DIR / "documents.jsonl"
HIERARCHY_FILE_OUT = DATA_DIR / "hierarchy.json"

# Cloned repo location
REPO_URL = "https://github.com/RomanPlaud/revisitingHTC.git"
REPO_DIR = SCRIPT_DIR / "revisitingHTC"

# Data files inside cloned repo
HWV_DIR = REPO_DIR / "data" / "HWV"
SPLITS = {
    "train": "hwv_train.json",
    "val": "hwv_val.json",
    "test": "hwv_test.json",
}
TAXONOMY_FILE = "hwv.taxonomy"

# Tier mapping by depth from root
# depth 0 = Root (virtual), depth 1 = top-level domains, etc.
# For internal nodes we map depth -> tier name:
#   0 -> "theme" (Root)
#   1 -> "narrative" (top-level: Technology, History, People, ...)
#   2 -> "story" (second-level subtopics)
#   3+ -> "content" (leaf documents and deep internal nodes)
TIER_MAP = {
    0: "theme",
    1: "narrative",
    2: "story",
}


def clone_repo():
    """Clone the revisitingHTC repo (shallow) if not already present."""
    if HWV_DIR.exists():
        print(f"Repo data already present at {HWV_DIR}")
        return

    print(f"Cloning {REPO_URL} (shallow)...")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)],
            check=True,
        )
        print(f"  Cloned to {REPO_DIR}")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: git clone failed: {e}")
        sys.exit(1)

    if not HWV_DIR.exists():
        print(f"ERROR: Expected data directory not found at {HWV_DIR}")
        print(f"  Repo contents: {list(REPO_DIR.iterdir())}")
        sys.exit(1)


def build_hierarchy_tree() -> dict:
    """Parse hwv.taxonomy into a tree structure.

    Taxonomy format: each line is tab-separated:
      parent<TAB>child1<TAB>child2<TAB>...

    Returns dict with:
      - tree: {parent: [children]}
      - parent_map: {child: parent}
      - roots: set of root nodes
      - depth: {node: depth_from_root}
      - leaves: set of leaf nodes (no children)
      - node_count: int
    """
    print("\n=== Building hierarchy tree from hwv.taxonomy ===")

    taxonomy_path = HWV_DIR / TAXONOMY_FILE
    if not taxonomy_path.exists():
        print(f"  ERROR: {taxonomy_path} not found")
        sys.exit(1)

    tree: dict[str, list[str]] = {}
    parent_map: dict[str, str] = {}
    all_nodes: set[str] = set()

    with open(taxonomy_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                print(f"  SKIP line with <2 fields: {line[:100]}")
                continue
            parent = parts[0]
            children = parts[1:]
            tree[parent] = children
            all_nodes.add(parent)
            for child in children:
                parent_map[child] = parent
                all_nodes.add(child)

    # Roots = nodes with no parent
    children_set = set(parent_map.keys())
    roots = all_nodes - children_set
    print(f"  Total nodes: {len(all_nodes)}")
    print(f"  Edges (parent-child): {sum(len(v) for v in tree.values())}")
    print(f"  Root nodes: {roots}")

    # Compute depth via BFS
    depth: dict[str, int] = {}
    queue = [(r, 0) for r in roots]
    while queue:
        node, d = queue.pop(0)
        if node in depth:
            continue
        depth[node] = d
        for child in tree.get(node, []):
            queue.append((child, d + 1))

    max_depth = max(depth.values()) if depth else 0
    print(f"  Max depth: {max_depth}")

    # Depth distribution
    level_counts: dict[int, int] = defaultdict(int)
    for d in depth.values():
        level_counts[d] += 1
    for lvl in sorted(level_counts):
        print(f"    Depth {lvl}: {level_counts[lvl]} nodes")

    # Leaf nodes
    leaves = all_nodes - set(tree.keys())
    print(f"  Leaf nodes: {len(leaves)}")
    print(f"  Internal nodes: {len(all_nodes) - len(leaves)}")

    return {
        "tree": tree,
        "parent_map": parent_map,
        "roots": roots,
        "depth": depth,
        "leaves": leaves,
        "node_count": len(all_nodes),
    }


def get_node_path(node: str, parent_map: dict[str, str]) -> list[str]:
    """Trace path from root to node using parent_map.

    Returns list of node names from root to this node (inclusive).
    """
    path = [node]
    current = node
    while current in parent_map:
        current = parent_map[current]
        path.append(current)
    path.reverse()
    return path


def parse_documents(hierarchy_info: dict) -> list[dict]:
    """Parse HWV train/val/test JSON files into document dicts.

    Each line in the source files is a JSON object:
      {"token": ["text..."], "label": ["Level1", "Level2", ...]}

    Returns list of document dicts.
    """
    depth_map = hierarchy_info["depth"]
    parent_map = hierarchy_info["parent_map"]
    tree = hierarchy_info["tree"]

    all_documents = []
    doc_idx = 0

    for split_name, filename in SPLITS.items():
        filepath = HWV_DIR / filename
        print(f"\nParsing {filepath.name} (split={split_name})...")

        if not filepath.exists():
            print(f"  ERROR: {filepath} not found")
            continue

        count = 0
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"  SKIP invalid JSON: {e}")
                    continue

                # Extract text (token field is a list with one string)
                tokens = record.get("token", [])
                if not tokens:
                    continue
                text = tokens[0] if isinstance(tokens, list) else str(tokens)
                text = text.strip()
                if not text:
                    continue

                # Extract label path
                labels = record.get("label", [])
                if not labels:
                    continue

                # The label list IS the path from root to leaf
                # e.g., ["History", "Post-classical history (History)", "Asia (...)"]
                hierarchy_path = "/".join(labels)

                # Depth = number of labels (the labels start at depth 1,
                # since depth 0 is Root which is not in the label list)
                leaf_label = labels[-1]
                leaf_depth = depth_map.get(leaf_label, len(labels))

                # Domain = first label (depth 1 category)
                domain = labels[0] if len(labels) >= 1 else ""
                # Area = second label (depth 2 category)
                area = labels[1] if len(labels) >= 2 else ""

                # Build parent/child IDs for this document's leaf node
                parent_ids = []
                if leaf_label in parent_map:
                    parent_ids.append(
                        f"hwv_node_{leaf_label}"
                    )

                doc_id = f"hwv_doc_{doc_idx}"
                all_documents.append({
                    "id": doc_id,
                    "text": text,
                    "tier": "content",
                    "depth": leaf_depth,
                    "domain": domain,
                    "area": area,
                    "hierarchy_path": hierarchy_path,
                    "all_paths": [hierarchy_path],
                    "parent_ids": parent_ids,
                    "child_ids": [],
                    "label_path": labels,
                })
                doc_idx += 1
                count += 1

        print(f"  Parsed {count} documents")

    return all_documents


def generate_internal_nodes(hierarchy_info: dict) -> list[dict]:
    """Generate synthetic documents for every internal tree node.

    Each internal node gets a text formed from its name + child names.
    This gives the testbench centroid targets at every hierarchy level.

    Returns list of node document dicts.
    """
    print("\n=== Generating internal node documents ===")

    tree = hierarchy_info["tree"]
    depth_map = hierarchy_info["depth"]
    parent_map = hierarchy_info["parent_map"]
    leaves = hierarchy_info["leaves"]
    all_nodes = set(depth_map.keys())

    node_docs = []
    node_idx = 0

    for node in sorted(all_nodes):
        # Skip the virtual Root node
        if node == "Root":
            continue

        d = depth_map.get(node, 0)
        children = tree.get(node, [])

        # Build synthetic text: node name + list of children
        if children:
            child_names = ", ".join(children)
            text = f"{node}: {child_names}"
        else:
            # Leaf node in taxonomy (no children in tree)
            text = node

        # Determine tier
        tier = TIER_MAP.get(d, "content")

        # Build full path from root
        path_parts = get_node_path(node, parent_map)
        # Remove "Root" from path display
        display_parts = [p for p in path_parts if p != "Root"]
        hierarchy_path = "/".join(display_parts)

        # Domain and area from path
        domain = display_parts[0] if len(display_parts) >= 1 else ""
        area = display_parts[1] if len(display_parts) >= 2 else ""

        # Parent and child IDs
        parent_ids = []
        if node in parent_map and parent_map[node] != "Root":
            parent_ids.append(f"hwv_node_{parent_map[node]}")

        child_ids = [f"hwv_node_{c}" for c in children]

        node_id = f"hwv_node_{node}"
        node_docs.append({
            "id": node_id,
            "text": text,
            "tier": tier,
            "depth": d,
            "domain": domain,
            "area": area,
            "hierarchy_path": hierarchy_path,
            "all_paths": [hierarchy_path],
            "parent_ids": parent_ids,
            "child_ids": child_ids,
            "label_path": display_parts,
        })
        node_idx += 1

    print(f"  Generated {len(node_docs)} internal+leaf node documents")

    # Tier breakdown
    tier_counts: dict[str, int] = defaultdict(int)
    for doc in node_docs:
        tier_counts[doc["tier"]] += 1
    for tier, count in sorted(tier_counts.items()):
        print(f"    {tier}: {count}")

    return node_docs


def main():
    if DOCUMENTS_FILE.exists():
        print(f"documents.jsonl already exists at {DOCUMENTS_FILE}")
        print("Delete it to regenerate.")
        return

    # Step 1: Clone the repo
    clone_repo()

    # Step 2: Build hierarchy tree
    hierarchy_info = build_hierarchy_tree()

    # Step 3: Parse document splits
    documents = parse_documents(hierarchy_info)
    if not documents:
        print("ERROR: No documents parsed!")
        return

    # Step 4: Generate internal node documents
    node_docs = generate_internal_nodes(hierarchy_info)

    # Combine
    all_documents = documents + node_docs

    # Summary
    print(f"\n=== Summary ===")
    print(f"  Content documents: {len(documents)}")
    print(f"  Node documents: {len(node_docs)}")
    print(f"  Total documents: {len(all_documents)}")

    # Depth distribution (content docs only)
    depth_counts: dict[int, int] = defaultdict(int)
    for doc in documents:
        depth_counts[doc["depth"]] += 1
    print(f"  Content depth distribution: {dict(sorted(depth_counts.items()))}")

    # Domain distribution
    domain_counts: dict[str, int] = defaultdict(int)
    for doc in documents:
        domain_counts[doc["domain"]] += 1
    print(f"  Domains (depth 1): {len(domain_counts)}")
    for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"    {domain}: {count}")

    # Final tier breakdown
    tier_counts: dict[str, int] = defaultdict(int)
    for doc in all_documents:
        tier_counts[doc["tier"]] += 1
    print(f"\n  Final tier breakdown: {dict(sorted(tier_counts.items()))}")

    # Save documents
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DOCUMENTS_FILE, "w", encoding="utf-8") as f:
        for doc in all_documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(all_documents)} documents to {DOCUMENTS_FILE}")

    # Build and save hierarchy info
    tree = hierarchy_info["tree"]
    depth_map = hierarchy_info["depth"]

    # Convert tree for JSON serialization (remove Root as top-level,
    # but keep it in the structure)
    hierarchy_out = {
        "source": "Hierarchical WikiVitals (HWV)",
        "paper": "Plaud et al., CoNLL 2024",
        "repo": REPO_URL,
        "license": "MIT",
        "max_depth": max(depth_map.values()) if depth_map else 0,
        "tree": tree,
        "roots": list(hierarchy_info["roots"]),
        "node_count": hierarchy_info["node_count"],
        "leaf_count": len(hierarchy_info["leaves"]),
        "document_count": len(documents),
        "total_count": len(all_documents),
        "tier_counts": dict(tier_counts),
        "depth_distribution": dict(sorted(depth_counts.items())),
        "domains": sorted(domain_counts.keys()),
    }
    with open(HIERARCHY_FILE_OUT, "w", encoding="utf-8") as f:
        json.dump(hierarchy_out, f, indent=2, ensure_ascii=False)
    print(f"Saved hierarchy info to {HIERARCHY_FILE_OUT}")

    print(f"\nDone. Run: cd ../../testbench && python embed.py --dataset hwv")


if __name__ == "__main__":
    main()

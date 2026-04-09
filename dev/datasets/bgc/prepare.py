"""Prepare the BlurbGenreCollection for testbench use.

Parses BGC XML-like text files + hierarchy.txt into our standard
documents.jsonl format with multi-label hierarchy paths.

BGC structure:
- 91,894 book blurbs (train/dev/test splits)
- 4-level genre hierarchy (d0 → d1 → d2 → d3)
- Multi-label: books can belong to multiple genres
- hierarchy.txt: tab-separated parent-child edges

Output: data/documents.jsonl + data/hierarchy.json

Usage:
    python prepare.py
"""

import json
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
DOCUMENTS_FILE = DATA_DIR / "documents.jsonl"
HIERARCHY_FILE_OUT = DATA_DIR / "hierarchy.json"

# BGC data files
SPLITS = {
    "train": DATA_DIR / "BlurbGenreCollection_EN_train.txt",
    "dev": DATA_DIR / "BlurbGenreCollection_EN_dev.txt",
    "test": DATA_DIR / "BlurbGenreCollection_EN_test.txt",
}
HIERARCHY_FILE = DATA_DIR / "hierarchy.txt"

# Tier mapping for 4-level hierarchy
# d0 = broadest (Fiction/Nonfiction), d3 = most specific
TIER_MAP = {
    0: "narrative",     # d0: Fiction, Nonfiction, Children's Books, etc.
    1: "story",         # d1: Historical Fiction, Mystery & Thriller, etc.
    2: "sub_story",     # d2: World War II Fiction, Cozy Mystery, etc.
    3: "content",       # d3: most specific genre label
}


def build_hierarchy_tree() -> dict:
    """Parse hierarchy.txt (tab-separated parent→child edges) into a tree.

    Returns dict with:
      - tree: {parent: [children]}
      - roots: set of root nodes (no parent)
      - depth: {node: depth_from_root}
      - node_count: int
    """
    print("=== Building hierarchy tree ===")

    if not HIERARCHY_FILE.exists():
        print(f"  WARNING: {HIERARCHY_FILE} not found, building from data only")
        return {"tree": {}, "roots": set(), "depth": {}, "node_count": 0}

    tree: dict[str, list[str]] = defaultdict(list)
    children_set: set[str] = set()
    all_nodes: set[str] = set()

    with open(HIERARCHY_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                print(f"  SKIP malformed line: {line[:100]}")
                continue
            parent, child = parts[0].strip(), parts[1].strip()
            tree[parent].append(child)
            children_set.add(child)
            all_nodes.add(parent)
            all_nodes.add(child)

    roots = all_nodes - children_set
    print(f"  Total nodes: {len(all_nodes)}")
    print(f"  Total edges: {sum(len(v) for v in tree.values())}")
    print(f"  Root nodes: {roots}")

    # Compute depth for each node via BFS
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

    # Nodes per depth level
    level_counts = defaultdict(int)
    for d in depth.values():
        level_counts[d] += 1
    for lvl in sorted(level_counts):
        print(f"    Depth {lvl}: {level_counts[lvl]} nodes")

    return {
        "tree": dict(tree),
        "roots": roots,
        "depth": depth,
        "node_count": len(all_nodes),
    }


def _build_path(genres: list[tuple[int, str]], hierarchy_depth: dict[str, int]) -> str:
    """Build a hierarchy path from a list of (depth, genre_name) tuples.

    Sorts by depth level, joins with '/'.
    """
    sorted_genres = sorted(genres, key=lambda x: x[0])
    return "/".join(g[1] for g in sorted_genres)


def parse_split(filepath: Path, split_name: str) -> list[dict]:
    """Parse a BGC split file (XML-like format) into document dicts.

    BGC format is not valid XML — books are concatenated without a root element.
    We wrap in a root tag and parse.
    """
    print(f"\nParsing {filepath.name} (split={split_name})...")

    if not filepath.exists():
        print(f"  ERROR: {filepath} not found")
        return []

    # Read raw content and wrap in root element for XML parsing
    raw = filepath.read_text(encoding="utf-8", errors="replace")

    # Fix common XML issues in BGC files
    # Some & chars aren't escaped
    raw = raw.replace("&amp;", "__AMP__")
    raw = raw.replace("&", "&amp;")
    raw = raw.replace("__AMP__", "&amp;")

    wrapped = f"<root>{raw}</root>"

    try:
        root = ET.fromstring(wrapped)
    except ET.ParseError as e:
        print(f"  XML parse error: {e}")
        print(f"  Trying fallback regex parser...")
        return _parse_split_regex(filepath, split_name)

    documents = []
    for book in root.findall("book"):
        title_el = book.find("title")
        body_el = book.find("body")
        metadata = book.find("metadata")

        title = title_el.text.strip() if title_el is not None and title_el.text else ""
        body = body_el.text.strip() if body_el is not None and body_el.text else ""
        text = f"{title}. {body}".strip() if body else title.strip()

        if not text:
            continue

        # Extract ISBN as ID
        isbn_el = metadata.find("isbn") if metadata is not None else None
        isbn = isbn_el.text.strip() if isbn_el is not None and isbn_el.text else ""

        # Extract genre hierarchy from <topics>
        topics = metadata.find("topics") if metadata is not None else None
        genres: list[tuple[int, str]] = []
        all_paths: list[str] = []

        if topics is not None:
            # Collect all d0, d1, d2, d3 tags
            for depth_level in range(4):
                tag = f"d{depth_level}"
                for el in topics.findall(tag):
                    if el.text:
                        genres.append((depth_level, el.text.strip()))

        # Build hierarchy paths
        # A book can have multiple labels at the same depth (multi-label)
        # Group by building paths from d0 down
        if genres:
            # Get all genres at each level
            by_level: dict[int, list[str]] = defaultdict(list)
            for depth_level, name in genres:
                if name not in by_level[depth_level]:
                    by_level[depth_level].append(name)

            # Build paths: for each d0, chain with d1, d2, d3
            # Simple approach: build the deepest path as primary
            max_level = max(by_level.keys())
            primary_parts = []
            for lvl in range(max_level + 1):
                if by_level[lvl]:
                    primary_parts.append(by_level[lvl][0])

            primary_path = "/".join(primary_parts)

            # All paths: cartesian product would be complex,
            # just list each unique partial path
            for lvl in range(max_level + 1):
                for name in by_level[lvl]:
                    # Build path from d0 to this level
                    parts = []
                    for l in range(lvl):
                        if by_level[l]:
                            parts.append(by_level[l][0])
                    parts.append(name)
                    path = "/".join(parts)
                    if path not in all_paths:
                        all_paths.append(path)
        else:
            primary_path = ""
            all_paths = []

        if not primary_path:
            continue

        # Determine tier from deepest genre level
        max_genre_depth = max(d for d, _ in genres) if genres else 0
        tier = TIER_MAP.get(max_genre_depth, "content")

        # For our testbench, all books are "content" tier (leaves)
        # The intermediate tiers will be generated as centroids by embed.py
        tier = "content"

        # Extract domain (d0) and area (d1)
        domain = by_level[0][0] if by_level.get(0) else ""
        area = by_level[1][0] if by_level.get(1) else ""

        documents.append({
            "id": isbn or f"{split_name}_{len(documents)}",
            "text": text,
            "tier": tier,
            "depth": max_genre_depth + 1,  # 1-indexed
            "domain": domain,
            "area": area,
            "hierarchy_path": primary_path,
            "all_paths": all_paths,
            "has_abstract": bool(body),
        })

    print(f"  Parsed {len(documents)} books")
    return documents


def _parse_split_regex(filepath: Path, split_name: str) -> list[dict]:
    """Fallback regex-based parser if XML parsing fails."""
    print(f"  Using regex fallback for {filepath.name}...")

    raw = filepath.read_text(encoding="utf-8", errors="replace")

    # Extract books with regex
    book_pattern = re.compile(
        r'<book[^>]*>.*?<title>(.*?)</title>.*?<body>(.*?)</body>.*?'
        r'<topics>(.*?)</topics>.*?(?:<isbn>(.*?)</isbn>)?.*?</book>',
        re.DOTALL,
    )

    documents = []
    for match in book_pattern.finditer(raw):
        title = match.group(1).strip()
        body = match.group(2).strip()
        topics_raw = match.group(3).strip()
        isbn = match.group(4).strip() if match.group(4) else ""

        text = f"{title}. {body}".strip() if body else title.strip()
        if not text:
            continue

        # Parse genre tags from topics
        by_level: dict[int, list[str]] = defaultdict(list)
        for d_match in re.finditer(r'<d(\d+)>(.*?)</d\1>', topics_raw):
            depth_level = int(d_match.group(1))
            name = d_match.group(2).strip()
            if name and name not in by_level[depth_level]:
                by_level[depth_level].append(name)

        if not by_level:
            continue

        max_level = max(by_level.keys())
        primary_parts = []
        for lvl in range(max_level + 1):
            if by_level[lvl]:
                primary_parts.append(by_level[lvl][0])
        primary_path = "/".join(primary_parts)

        all_paths = []
        for lvl in range(max_level + 1):
            for name in by_level[lvl]:
                parts = []
                for l in range(lvl):
                    if by_level[l]:
                        parts.append(by_level[l][0])
                parts.append(name)
                path = "/".join(parts)
                if path not in all_paths:
                    all_paths.append(path)

        domain = by_level[0][0] if by_level.get(0) else ""
        area = by_level[1][0] if by_level.get(1) else ""

        documents.append({
            "id": isbn or f"{split_name}_{len(documents)}",
            "text": text,
            "tier": "content",
            "depth": max_level + 1,
            "domain": domain,
            "area": area,
            "hierarchy_path": primary_path,
            "all_paths": all_paths,
            "has_abstract": bool(body),
        })

    print(f"  Regex parsed {len(documents)} books")
    return documents


def main():
    if DOCUMENTS_FILE.exists():
        print(f"documents.jsonl already exists at {DOCUMENTS_FILE}")
        print(f"Delete it to regenerate.")
        return

    # Build hierarchy tree
    hierarchy_info = build_hierarchy_tree()

    # Parse all splits
    all_documents = []
    for split_name, filepath in SPLITS.items():
        docs = parse_split(filepath, split_name)
        all_documents.extend(docs)

    if not all_documents:
        print("ERROR: No documents parsed!")
        return

    print(f"\n=== Summary ===")
    print(f"  Total documents: {len(all_documents)}")

    # Depth distribution
    depth_counts = defaultdict(int)
    for doc in all_documents:
        depth_counts[doc["depth"]] += 1
    print(f"  Depth distribution: {dict(sorted(depth_counts.items()))}")

    # Domain distribution
    domain_counts = defaultdict(int)
    for doc in all_documents:
        domain_counts[doc["domain"]] += 1
    print(f"  Domains (d0): {len(domain_counts)}")
    for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    {domain}: {count}")

    # Area distribution
    unique_areas = len(set(doc["area"] for doc in all_documents if doc["area"]))
    print(f"  Unique areas (d1): {unique_areas}")

    # Multi-label stats
    multi_path = sum(1 for d in all_documents if len(d["all_paths"]) > 1)
    print(f"  Multi-path documents: {multi_path} ({multi_path/len(all_documents)*100:.1f}%)")

    # Has abstract (body text)
    has_body = sum(1 for d in all_documents if d["has_abstract"])
    print(f"  With body text: {has_body} ({has_body/len(all_documents)*100:.1f}%)")

    # Generate intermediate tier documents from hierarchy
    print(f"\n=== Generating intermediate tier documents ===")
    category_docs: dict[str, set] = defaultdict(set)  # level -> set of paths

    for doc in all_documents:
        parts = doc["hierarchy_path"].split("/")
        for level in range(1, len(parts)):
            category_path = "/".join(parts[:level])
            category_docs[level].add(category_path)

    tier_names = {1: "narrative", 2: "story", 3: "sub_story"}
    tier_doc_count = 0

    for level in sorted(category_docs.keys()):
        tier_name = tier_names.get(level, f"tier_L{level}")
        for category_path in sorted(category_docs[level]):
            parts = category_path.split("/")
            domain = parts[0] if parts else ""
            area = parts[1] if len(parts) >= 2 else ""

            text = " - ".join(parts)

            all_documents.append({
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
    for doc in all_documents:
        tier_counts[doc["tier"]] += 1
    print(f"\n  Final tier breakdown: {dict(sorted(tier_counts.items()))}")
    print(f"  Total documents: {len(all_documents)}")

    # Save documents
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DOCUMENTS_FILE, "w") as f:
        for doc in all_documents:
            f.write(json.dumps(doc) + "\n")
    print(f"\nSaved {len(all_documents)} documents to {DOCUMENTS_FILE}")

    # Save hierarchy info
    hierarchy_out = {
        "source": "BlurbGenreCollection-EN",
        "max_depth": 4,
        "tree": hierarchy_info["tree"],
        "roots": list(hierarchy_info["roots"]),
        "node_count": hierarchy_info["node_count"],
        "document_count": len(all_documents),
        "tier_counts": dict(tier_counts),
        "depth_distribution": dict(depth_counts),
    }
    with open(HIERARCHY_FILE_OUT, "w") as f:
        json.dump(hierarchy_out, f, indent=2)
    print(f"Saved hierarchy info to {HIERARCHY_FILE_OUT}")

    print(f"\nDone. Run: cd ../../testbench && python embed.py --dataset bgc")


if __name__ == "__main__":
    main()

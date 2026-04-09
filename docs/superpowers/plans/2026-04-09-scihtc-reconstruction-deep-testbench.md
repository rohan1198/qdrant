# SciHTC Reconstruction + 6-Level Deep Testbench

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reconstruct the SciHTC dataset (186K ACM papers, 6-level hierarchy) via Semantic Scholar API and integrate into the testbench to validate hyperbolic embeddings at production hierarchy depth.

**Architecture:** Two new scripts in `dev/datasets/scihtc/` handle reconstruction (download.py for API I/O, prepare.py for transformation). The testbench gets a `--dataset` flag so embed.py and benchmark.py work with both WOS and SciHTC. Discovery-first approach — scripts inspect data formats at runtime rather than assuming structure.

**Tech Stack:** Python (requests, gdown/gdownload for Google Drive, semanticscholar or raw API), existing testbench infrastructure (qdrant-client, numpy, scikit-learn)

**Spec:** `docs/superpowers/specs/2026-04-09-scihtc-reconstruction-deep-testbench-design.md`

**Important:** Do NOT run Python scripts — tell the user the command and they run it.

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `dev/datasets/scihtc/download.py` | Create | Download SciHTC IDs+labels from Google Drive, fetch text from Semantic Scholar API |
| `dev/datasets/scihtc/prepare.py` | Create | Transform raw data to testbench JSONL, discover hierarchy structure, assign tiers |
| `dev/datasets/scihtc/README.md` | Create | Dataset card with source, license, reconstruction instructions |
| `dev/testbench/embed.py` | Modify | Add `--dataset` flag, make paths/collection names configurable, adaptive tier assignment |
| `dev/testbench/benchmark.py` | Modify | Generalize collection discovery for any prefix, multi-label evaluation |

---

### Task 1: Create `dev/datasets/scihtc/download.py`

**Files:**
- Create: `dev/datasets/scihtc/download.py`

This is the most critical and uncertain task — it involves external APIs and data we haven't seen. The implementer must inspect formats at each step.

- [ ] **Step 1: Create directory structure and skeleton**

```bash
mkdir -p /home/rohan/projects/qdrant/dev/datasets/scihtc/data
```

Create `dev/datasets/scihtc/download.py` with:

```python
"""Download and reconstruct the SciHTC dataset.

Pipeline:
1. Download paper IDs + labels from SciHTC Google Drive
2. Inspect format (discover delimiters, ID types, label encoding)
3. Fetch titles + abstracts from Semantic Scholar batch API
4. Save raw outputs with manifest

SciHTC: 186K ACM papers, 1,233 categories, 6-level ACM CCS hierarchy.
Paper: https://aclanthology.org/2022.emnlp-main.610.pdf
Public data: https://drive.google.com/drive/folders/1uRh5A-GpFRxA_QLzgN_D-y8G5j6JpZPJ

Usage:
    python download.py
"""

import json
import os
import time
from pathlib import Path

import requests

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
RAW_PAPERS_FILE = DATA_DIR / "raw_papers.jsonl"
RAW_LABELS_FILE = DATA_DIR / "raw_labels.jsonl"
MANIFEST_FILE = DATA_DIR / "manifest.json"

# SciHTC Google Drive folder
SCIHTC_GDRIVE_FOLDER = "1uRh5A-GpFRxA_QLzgN_D-y8G5j6JpZPJ"

# Semantic Scholar batch API
S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
S2_BATCH_SIZE = 500  # Max per request
S2_FIELDS = "title,abstract,externalIds"
S2_DELAY = 0.5  # seconds between batch requests (conservative)


def download_scihtc_labels():
    """Download SciHTC IDs + labels from Google Drive.

    The public data contains paper IDs and category labels for train/test/dev splits.
    We need to discover the file format (CSV? TSV? columns? delimiters?).
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Try using gdown to download the Google Drive folder
    try:
        import gdown
    except ImportError:
        print("Installing gdown for Google Drive downloads...")
        os.system("pip install gdown")
        import gdown

    gdrive_dir = DATA_DIR / "gdrive_raw"
    if gdrive_dir.exists() and any(gdrive_dir.iterdir()):
        print(f"Google Drive data already downloaded at {gdrive_dir}")
    else:
        gdrive_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading SciHTC data from Google Drive folder {SCIHTC_GDRIVE_FOLDER}...")
        gdown.download_folder(
            id=SCIHTC_GDRIVE_FOLDER,
            output=str(gdrive_dir),
            quiet=False,
        )

    # --- DISCOVERY PHASE ---
    # Inspect what we downloaded
    print("\n=== Discovery: Inspecting downloaded files ===")
    for root, dirs, files in os.walk(gdrive_dir):
        for f in sorted(files):
            fpath = Path(root) / f
            size = fpath.stat().st_size
            print(f"  {fpath.relative_to(gdrive_dir)} ({size:,} bytes)")

            # Show first few lines to discover format
            if fpath.suffix in ('.csv', '.tsv', '.txt', '') or size < 1_000_000:
                try:
                    with open(fpath) as fp:
                        lines = [fp.readline() for _ in range(5)]
                    print(f"    First lines:")
                    for i, line in enumerate(lines):
                        if line.strip():
                            print(f"      [{i}] {line.strip()[:200]}")
                except Exception as e:
                    print(f"    Could not read: {e}")

    return gdrive_dir


def discover_and_parse_labels(gdrive_dir: Path) -> list[dict]:
    """Inspect downloaded files and extract paper IDs + labels.

    Returns list of dicts: {"paper_id": str, "labels": list[str], "split": str}

    IMPORTANT: This function must discover the format, not assume it.
    Log what you find, warn on unexpected formats.
    """
    all_entries = []

    # Find all data files
    data_files = sorted(gdrive_dir.rglob("*"))
    data_files = [f for f in data_files if f.is_file() and not f.name.startswith(".")]

    print(f"\n=== Discovery: Parsing {len(data_files)} files ===")

    for fpath in data_files:
        # Try to determine the split from filename
        fname = fpath.stem.lower()
        if "train" in fname:
            split = "train"
        elif "test" in fname:
            split = "test"
        elif "dev" in fname or "val" in fname:
            split = "dev"
        else:
            split = "unknown"

        print(f"\nParsing {fpath.name} (split={split})...")

        try:
            with open(fpath) as f:
                first_line = f.readline().strip()
                f.seek(0)
                all_lines = f.readlines()
        except Exception as e:
            print(f"  SKIP: Could not read: {e}")
            continue

        if not all_lines:
            print(f"  SKIP: Empty file")
            continue

        # Discover delimiter
        # Try tab, comma, pipe
        for delim_name, delim in [("tab", "\t"), ("comma", ","), ("pipe", "|")]:
            n_cols = len(first_line.split(delim))
            if n_cols >= 2:
                print(f"  Delimiter '{delim_name}' gives {n_cols} columns on first line")

        # Parse based on discovered structure
        # We expect at minimum: paper_id, category_path(s)
        parsed_count = 0
        for line in all_lines:
            line = line.strip()
            if not line:
                continue

            # Try common delimiters
            for delim in ["\t", ","]:
                parts = line.split(delim)
                if len(parts) >= 2:
                    paper_id = parts[0].strip()
                    labels = [p.strip() for p in parts[1:] if p.strip()]
                    if paper_id and labels:
                        all_entries.append({
                            "paper_id": paper_id,
                            "labels": labels,
                            "split": split,
                        })
                        parsed_count += 1
                        break

        print(f"  Parsed {parsed_count} entries from {fpath.name}")

        # Show sample entries
        file_entries = [e for e in all_entries if e["split"] == split]
        if file_entries:
            sample = file_entries[0]
            print(f"  Sample: id='{sample['paper_id']}', labels={sample['labels'][:3]}...")

    print(f"\n=== Total entries: {len(all_entries)} ===")

    # Discover paper ID format
    if all_entries:
        sample_ids = [e["paper_id"] for e in all_entries[:10]]
        print(f"Sample paper IDs: {sample_ids}")

        # Detect if they look like DOIs, ACM IDs, or numeric
        doi_like = sum(1 for sid in sample_ids if "10." in sid and "/" in sid)
        numeric = sum(1 for sid in sample_ids if sid.isdigit())
        print(f"  DOI-like: {doi_like}/10, Numeric: {numeric}/10")

        # Discover label format
        sample_labels = all_entries[0]["labels"]
        print(f"Sample labels: {sample_labels}")
        has_tilde = any("~" in l for l in sample_labels)
        has_arrow = any("->" in l for l in sample_labels)
        has_slash = any("/" in l for l in sample_labels)
        print(f"  Tilde-separated: {has_tilde}, Arrow: {has_arrow}, Slash: {has_slash}")

    return all_entries


def fetch_from_semantic_scholar(paper_ids: list[str]) -> dict[str, dict]:
    """Batch-fetch paper metadata from Semantic Scholar API.

    Args:
        paper_ids: List of paper identifiers (DOIs, ACM IDs, etc.)

    Returns:
        Dict mapping paper_id -> {"title": str, "abstract": str|None, "s2_id": str}
    """
    print(f"\n=== Fetching {len(paper_ids)} papers from Semantic Scholar ===")

    # Determine ID format for S2 API
    # S2 accepts: DOI:xxx, CorpusID:xxx, ACL:xxx, PMID:xxx, or raw S2 ID
    sample = paper_ids[0]
    if "10." in sample and "/" in sample:
        # Looks like a DOI
        id_prefix = "DOI:"
        print(f"  ID format: DOI (prefix with 'DOI:')")
    elif sample.isdigit():
        # Numeric — could be ACM internal ID, try as CorpusID
        id_prefix = "CorpusID:"
        print(f"  ID format: Numeric (trying as CorpusID)")
    else:
        id_prefix = ""
        print(f"  ID format: Unknown, trying raw")

    results = {}
    total_batches = (len(paper_ids) + S2_BATCH_SIZE - 1) // S2_BATCH_SIZE
    found = 0
    with_abstract = 0
    title_only = 0
    missing = 0

    for batch_idx in range(total_batches):
        start = batch_idx * S2_BATCH_SIZE
        end = min(start + S2_BATCH_SIZE, len(paper_ids))
        batch_ids = paper_ids[start:end]

        # Format IDs for S2 API
        formatted_ids = [f"{id_prefix}{pid}" for pid in batch_ids]

        try:
            resp = requests.post(
                S2_BATCH_URL,
                json={"ids": formatted_ids},
                params={"fields": S2_FIELDS},
                timeout=30,
            )

            if resp.status_code == 200:
                data = resp.json()
                for orig_id, paper_data in zip(batch_ids, data):
                    if paper_data is None:
                        missing += 1
                        continue

                    title = paper_data.get("title") or ""
                    abstract = paper_data.get("abstract")
                    s2_id = paper_data.get("paperId", "")

                    results[orig_id] = {
                        "title": title,
                        "abstract": abstract,
                        "s2_id": s2_id,
                    }
                    found += 1
                    if abstract:
                        with_abstract += 1
                    else:
                        title_only += 1
            elif resp.status_code == 429:
                print(f"  Rate limited at batch {batch_idx}, waiting 10s...")
                time.sleep(10)
                continue  # Retry this batch
            else:
                print(f"  Batch {batch_idx} failed: HTTP {resp.status_code}")
                missing += len(batch_ids)

        except Exception as e:
            print(f"  Batch {batch_idx} error: {e}")
            missing += len(batch_ids)

        # Progress
        if (batch_idx + 1) % 50 == 0 or batch_idx == total_batches - 1:
            pct = (batch_idx + 1) / total_batches * 100
            print(f"  [{pct:.0f}%] Batch {batch_idx+1}/{total_batches}: "
                  f"found={found}, abstract={with_abstract}, title_only={title_only}, missing={missing}")

        time.sleep(S2_DELAY)

    print(f"\n=== Semantic Scholar Results ===")
    print(f"  Total requested: {len(paper_ids)}")
    print(f"  Found: {found} ({found/len(paper_ids)*100:.1f}%)")
    print(f"    With abstract: {with_abstract}")
    print(f"    Title only: {title_only}")
    print(f"  Missing: {missing} ({missing/len(paper_ids)*100:.1f}%)")

    coverage = found / len(paper_ids) if paper_ids else 0
    if coverage < 0.8:
        print(f"\n  WARNING: Coverage {coverage:.1%} is below 80% threshold!")
        print(f"  The dataset may not be usable. Check ID format and API responses.")

    return results


def main():
    """Main reconstruction pipeline."""
    if RAW_PAPERS_FILE.exists() and RAW_LABELS_FILE.exists():
        print(f"Raw data already exists at {DATA_DIR}")
        print(f"Delete {RAW_PAPERS_FILE} to re-download.")
        return

    # Step 1: Download labels from Google Drive
    print("=== Step 1: Download SciHTC labels from Google Drive ===")
    gdrive_dir = download_scihtc_labels()

    # Step 2: Discover and parse label format
    print("\n=== Step 2: Discover and parse labels ===")
    entries = discover_and_parse_labels(gdrive_dir)

    if not entries:
        print("ERROR: No entries parsed. Check the downloaded files and format discovery logs.")
        return

    # Step 3: Extract unique paper IDs
    unique_ids = sorted(set(e["paper_id"] for e in entries))
    print(f"\nUnique paper IDs: {len(unique_ids)}")

    # Step 4: Fetch from Semantic Scholar
    print("\n=== Step 3: Fetch text from Semantic Scholar ===")
    s2_data = fetch_from_semantic_scholar(unique_ids)

    # Step 5: Save raw outputs
    print("\n=== Step 4: Save raw data ===")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save labels
    with open(RAW_LABELS_FILE, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    print(f"  Saved {len(entries)} label entries to {RAW_LABELS_FILE}")

    # Save papers
    paper_count = 0
    with open(RAW_PAPERS_FILE, "w") as f:
        for pid, pdata in s2_data.items():
            f.write(json.dumps({"paper_id": pid, **pdata}) + "\n")
            paper_count += 1
    print(f"  Saved {paper_count} papers to {RAW_PAPERS_FILE}")

    # Save manifest
    manifest = {
        "source": "SciHTC (EMNLP 2022) + Semantic Scholar API",
        "scihtc_url": "https://github.com/msadat3/SciHTC",
        "total_label_entries": len(entries),
        "unique_paper_ids": len(unique_ids),
        "s2_found": len(s2_data),
        "s2_with_abstract": sum(1 for d in s2_data.values() if d.get("abstract")),
        "s2_title_only": sum(1 for d in s2_data.values() if not d.get("abstract")),
        "s2_missing": len(unique_ids) - len(s2_data),
        "coverage": len(s2_data) / len(unique_ids) if unique_ids else 0,
        "splits": {s: sum(1 for e in entries if e["split"] == s)
                   for s in set(e["split"] for e in entries)},
    }
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Saved manifest to {MANIFEST_FILE}")
    print(f"\n  Coverage: {manifest['coverage']:.1%}")

    print("\nDone. Run prepare.py next to build documents.jsonl.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify syntax**

```bash
cd /home/rohan/projects/qdrant/dev/datasets/scihtc
python3 -c "import ast; ast.parse(open('download.py').read()); print('OK')"
```

- [ ] **Step 3: Run the download script**

```bash
cd /home/rohan/projects/qdrant/dev/datasets/scihtc
python3 download.py
```

This will: download from Google Drive, print discovery logs (format, delimiters, ID types, label encoding), then batch-fetch from Semantic Scholar. **Read the output carefully** — the discovery phase tells us what format we're actually working with, which informs the next task.

Expected output: `data/raw_labels.jsonl`, `data/raw_papers.jsonl`, `data/manifest.json`.

---

### Task 2: Create `dev/datasets/scihtc/prepare.py`

**Files:**
- Create: `dev/datasets/scihtc/prepare.py`

This task depends on Task 1's discovery output. The implementer should read `data/manifest.json` and sample the raw files to understand the actual format before writing the transformation logic.

- [ ] **Step 1: Create `prepare.py`**

```python
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
```

- [ ] **Step 2: Verify syntax**

```bash
cd /home/rohan/projects/qdrant/dev/datasets/scihtc
python3 -c "import ast; ast.parse(open('prepare.py').read()); print('OK')"
```

- [ ] **Step 3: Run prepare.py** (after download.py has completed)

```bash
cd /home/rohan/projects/qdrant/dev/datasets/scihtc
python3 prepare.py
```

Expected output: `data/documents.jsonl`, `data/hierarchy.json` with discovery logs.

---

### Task 3: Create README

**Files:**
- Create: `dev/datasets/scihtc/README.md`

- [ ] **Step 1: Create dataset card**

```markdown
# SciHTC Dataset Reconstruction

Reconstructed version of the SciHTC dataset (EMNLP 2022) for hyperbolic embedding benchmarking.

## Source

- Paper: [Hierarchical Multi-Label Classification of Scientific Documents](https://aclanthology.org/2022.emnlp-main.610.pdf)
- IDs + Labels: [Google Drive](https://drive.google.com/drive/folders/1uRh5A-GpFRxA_QLzgN_D-y8G5j6JpZPJ)
- Text: Reconstructed via [Semantic Scholar API](https://api.semanticscholar.org/)
- Hierarchy: ACM Computing Classification System (CCS 2012)

## Reconstruction

```bash
python download.py    # Fetch IDs+labels from Google Drive, text from Semantic Scholar
python prepare.py     # Build documents.jsonl with hierarchy discovery
```

## Dataset Properties

- ~186K ACM CS papers
- 6-level ACM CCS hierarchy (poly-hierarchical DAG)
- 1,233 categories
- Multi-label: papers can belong to multiple branches
- Text: title + abstract (some title-only where abstract unavailable)

## Output Format

`data/documents.jsonl` — one JSON object per line:
- `id`: paper identifier
- `text`: title + abstract
- `tier`: content | sub_story | story | sub_narrative | narrative | theme
- `depth`: hierarchy depth of primary path
- `domain`: L1 category
- `area`: L2 category
- `hierarchy_path`: primary (deepest) path
- `all_paths`: list of ALL valid hierarchy paths
- `has_abstract`: boolean

## Citation

```bibtex
@inproceedings{sadat-caragea-2022-hierarchical,
    title = "Hierarchical Multi-Label Classification of Scientific Documents",
    author = "Sadat, Mobashir and Caragea, Cornelia",
    booktitle = "EMNLP 2022",
    year = "2022",
}
```

## License

SciHTC labels are provided by the authors for research use. Text reconstructed from Semantic Scholar (open access metadata).
```

---

### Task 4: Add `--dataset` flag to `embed.py`

**Files:**
- Modify: `dev/testbench/embed.py`

- [ ] **Step 1: Make paths configurable**

Replace the hardcoded path block at the top of `embed.py` (lines 35-71) with a dataset configuration system. The key change: all paths and collection names derive from a `DATASET` variable.

Replace lines 35-60 (from `SCRIPT_DIR` through `UNIFIED_COLLECTION`):

```python
SCRIPT_DIR = Path(__file__).parent

# Dataset configuration — set by --dataset flag
DATASET = "wos"  # default, overridden in main()

def _dataset_config(dataset_name: str) -> dict:
    """Return paths and collection names for a dataset."""
    if dataset_name == "wos":
        data_dir = SCRIPT_DIR / "data" / "wos"
        documents_file = data_dir / "documents.jsonl"
    elif dataset_name == "scihtc":
        data_dir = SCRIPT_DIR / "data" / "scihtc"
        documents_file = data_dir / "documents.jsonl"
        # Symlink or copy from datasets dir if not present
        source = SCRIPT_DIR.parent / "datasets" / "scihtc" / "data" / "documents.jsonl"
        if source.exists() and not documents_file.exists():
            data_dir.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(source, documents_file)
            print(f"Copied SciHTC data from {source}")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    prefix = dataset_name
    return {
        "data_dir": data_dir,
        "documents_file": documents_file,
        "embeddings_1024_file": data_dir / "embeddings_1024d.npy",
        "embeddings_128_file": data_dir / "embeddings_128d.npy",
        "focal_direction_file": data_dir / "focal_direction.npy",
        "busemann_depths_file": data_dir / "busemann_depths.npy",
        "cosine_collection": f"{prefix}_cosine",
        "unified_collection": f"{prefix}_unified",
    }

MODEL_PATH = Path.home() / "projects" / "pythia" / "models" / "pplx-embed-v1-0.6b"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CURVATURE = 5.0
VECTOR_DIM = 128
```

- [ ] **Step 2: Update main() to accept --dataset and use config**

Add `--dataset` argument to the argument parser (after `--cleanup-legacy`):

```python
    parser.add_argument(
        "--dataset",
        default="wos",
        choices=["wos", "scihtc"],
        help="Which dataset to use (default: wos)",
    )
```

At the top of main(), after `args = parser.parse_args()`, add:

```python
    # Load dataset configuration
    cfg = _dataset_config(args.dataset)
    data_dir = cfg["data_dir"]
    documents_file = cfg["documents_file"]
    embeddings_1024_file = cfg["embeddings_1024_file"]
    embeddings_128_file = cfg["embeddings_128_file"]
    cosine_collection = cfg["cosine_collection"]
    unified_collection = cfg["unified_collection"]
```

Then replace all references to the old global variables throughout main():
- `DOCUMENTS_FILE` → `documents_file`
- `EMBEDDINGS_1024_FILE` → `embeddings_1024_file`
- `EMBEDDINGS_128_FILE` → `embeddings_128_file`
- `DATA_DIR` → `data_dir`
- `COSINE_COLLECTION` → `cosine_collection`
- `UNIFIED_COLLECTION` → `unified_collection`

Also update `FOCAL_DIRECTION_FILE`, `BUSEMANN_DEPTHS_FILE` to derive from `data_dir`.

**Note:** The `load_documents()`, `embed_texts()`, `reduce_pca()`, `compute_busemann()` functions currently use module-level globals. Update them to accept path parameters, OR have main() set module-level variables from the config before calling them. The latter is simpler:

```python
    # Set module-level paths from config
    global DATA_DIR, DOCUMENTS_FILE, EMBEDDINGS_1024_FILE, EMBEDDINGS_128_FILE
    global FOCAL_DIRECTION_FILE, BUSEMANN_DEPTHS_FILE, COSINE_COLLECTION, UNIFIED_COLLECTION
    DATA_DIR = data_dir
    DOCUMENTS_FILE = documents_file
    EMBEDDINGS_1024_FILE = embeddings_1024_file
    EMBEDDINGS_128_FILE = embeddings_128_file
    FOCAL_DIRECTION_FILE = data_dir / "focal_direction.npy"
    BUSEMANN_DEPTHS_FILE = data_dir / "busemann_depths.npy"
    COSINE_COLLECTION = cosine_collection
    UNIFIED_COLLECTION = unified_collection
```

- [ ] **Step 3: Add adaptive tier centroid generation**

Update `generate_tier_centroids` to work with any tier names, not just WOS's hardcoded root/mid/leaf. The function should:
- Discover unique tiers from the documents (excluding "content")
- For each tier, find the documents at that tier level
- Generate centroids via Einstein midpoint

This may require refactoring `generate_tier_centroids` to accept tier names dynamically rather than hardcoding `area_groups` and `domain_groups`.

The simplest approach: generalize the function to group by `hierarchy_path` at each non-content tier level, then compute centroids for each group.

- [ ] **Step 4: Add multi-label payload**

When upserting to the unified collection, include `all_paths` in the payload:

```python
payload={
    "tier": doc["tier"],
    "busemann_depth": float(busemann_depths[i]),
    "domain": doc["domain"],
    "area": doc["area"],
    "hierarchy_path": doc["hierarchy_path"],
    "all_paths": doc.get("all_paths", [doc["hierarchy_path"]]),
    "source_ids": [],
},
```

- [ ] **Step 5: Verify syntax**

```bash
cd /home/rohan/projects/qdrant/dev/testbench
python3 -c "import ast; ast.parse(open('embed.py').read()); print('OK')"
```

---

### Task 5: Update `benchmark.py` for multi-dataset support

**Files:**
- Modify: `dev/testbench/benchmark.py`

- [ ] **Step 1: Generalize collection discovery**

Update `discover_collections` to detect any prefix, not just `wos_`:

```python
def discover_collections(client: QdrantClient, qdrant_url: str) -> list[dict]:
    """Auto-discover benchmark collections (wos_*, scihtc_*, etc.)."""
    import requests

    KNOWN_PREFIXES = ["wos_", "scihtc_"]

    existing = client.get_collections().collections
    collections = []

    for col_info in existing:
        name = col_info.name
        # Check if collection matches any known prefix
        prefix = None
        for p in KNOWN_PREFIXES:
            if name.startswith(p):
                prefix = p.rstrip("_")
                break
        if prefix is None:
            continue

        # ... rest of discovery logic, using prefix instead of hardcoded "wos"
```

For the unified collection detection, generalize:
```python
        if name == f"{prefix}_unified":
            is_poincare = True
            strategy = "unified"
            curvature = DEFAULT_CURVATURE
        elif name.startswith(f"{prefix}_unified_c"):
            # Curvature sweep collection
            ...
        elif name == f"{prefix}_cosine":
            strategy = None
            curvature = None
```

- [ ] **Step 2: Update UNIFIED_COLLECTION references**

Replace the hardcoded `UNIFIED_COLLECTION = "wos_unified"` constant. Instead, detect the unified collection name from the discovered collections. In main(), find the unified collection dynamically:

```python
    unified_name = None
    cosine_name = None
    for c in collection_configs:
        if c.get("strategy") == "unified" and "_c" not in c["name"]:
            unified_name = c["name"]
        if c.get("strategy") is None and c["name"].endswith("_cosine"):
            cosine_name = c["name"]
```

- [ ] **Step 3: Add multi-label evaluation**

In `benchmark_retrieval_quality` and `benchmark_retrieval_modes`, update the scoring logic. Currently it checks `nb_pt["area"] == query_pt["area"]`. For multi-label, check if ANY of the result's paths share a common ancestor with ANY of the query's paths:

```python
def _paths_match(query_pt, result_pt, level="area"):
    """Check if result matches query at given level, considering all_paths."""
    q_paths = query_pt.get("all_paths", [query_pt.get("hierarchy_path", "")])
    r_paths = result_pt.get("all_paths", [result_pt.get("hierarchy_path", "")])

    if level == "area":
        # Match if they share an L2 category
        q_areas = {"/".join(p.split("/")[:2]) for p in q_paths if "/" in p}
        r_areas = {"/".join(p.split("/")[:2]) for p in r_paths if "/" in p}
        return bool(q_areas & r_areas)
    elif level == "domain":
        # Match if they share an L1 category
        q_domains = {p.split("/")[0] for p in q_paths}
        r_domains = {p.split("/")[0] for p in r_paths}
        return bool(q_domains & r_domains)
    return False
```

Then in the scoring loops, replace:
```python
if nb_pt["area"] == query_pt["area"]:
```
with:
```python
if _paths_match(query_pt, nb_pt, level="area"):
```

- [ ] **Step 4: Update tier name mapping in hierarchy separation**

The `tier_map` in `benchmark_hierarchy_separation` currently maps `narrative/story/content → root/mid/leaf`. For SciHTC with 6 tiers, this needs to be dynamic. Replace the hardcoded map with discovery from the data:

```python
    # Discover tier names and map to canonical ordering
    unique_tiers = sorted(set(pt["tier"] for pt in points))
    # Sort by average depth (shallowest first = "root"-like)
    # ... or use the depth field if available
```

- [ ] **Step 5: Verify syntax**

```bash
cd /home/rohan/projects/qdrant/dev/testbench
python3 -c "import ast; ast.parse(open('benchmark.py').read()); print('OK')"
```

---

### Task 6: Run SciHTC Pipeline + Benchmarks

- [ ] **Step 1: Run the full reconstruction**

```bash
cd /home/rohan/projects/qdrant/dev/datasets/scihtc
python3 download.py
python3 prepare.py
```

Read the discovery logs. Verify coverage > 80%.

- [ ] **Step 2: Run embed pipeline on SciHTC**

```bash
cd /home/rohan/projects/qdrant/dev/testbench
python3 embed.py --qdrant-url http://localhost:6334 --dataset scihtc
```

This will embed ~186K papers (takes ~1-2 hours on GPU), project to Poincaré, compute Busemann depths across 6 levels, generate tier centroids, and upsert to `scihtc_cosine` + `scihtc_unified`.

- [ ] **Step 3: Run benchmarks**

```bash
cd /home/rohan/projects/qdrant/dev/testbench
python3 benchmark.py --qdrant-url http://localhost:6334
```

Should discover both `wos_*` and `scihtc_*` collections and benchmark both.

- [ ] **Step 4: Record results**

Save output to `dev/testbench/results/analysis_scihtc_round1.md`.
Key comparison: Euclidean vs Hyperbolic at 3 levels (WOS) vs 6 levels (SciHTC).

---

### Task 7: Commit

- [ ] **Step 1: Verify all Python files parse**

```bash
cd /home/rohan/projects/qdrant
for f in dev/datasets/scihtc/download.py dev/datasets/scihtc/prepare.py dev/testbench/embed.py dev/testbench/benchmark.py; do
    python3 -c "import ast; ast.parse(open('$f').read()); print('$f OK')"
done
```

- [ ] **Step 2: Commit**

```bash
git add dev/datasets/scihtc/download.py dev/datasets/scihtc/prepare.py dev/datasets/scihtc/README.md
git add dev/testbench/embed.py dev/testbench/benchmark.py
git add docs/superpowers/specs/2026-04-09-scihtc-reconstruction-deep-testbench-design.md
git add docs/superpowers/plans/2026-04-09-scihtc-reconstruction-deep-testbench.md
git add -f dev/testbench/results/analysis_scihtc_round1.md
git commit -m "feat(testbench): SciHTC 6-level dataset reconstruction + multi-dataset support

- Add dev/datasets/scihtc/ with download.py (Semantic Scholar API) and prepare.py
- Discovery-first pipeline: inspects data format before processing
- 186K ACM papers, 1,233 categories, 6-level ACM CCS hierarchy
- Multi-label support: papers can belong to multiple hierarchy branches
- Add --dataset flag to embed.py (wos | scihtc)
- Adaptive tier assignment based on discovered hierarchy depth
- Generalize benchmark.py collection discovery for any dataset prefix
- Multi-label evaluation: match against ANY valid path

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

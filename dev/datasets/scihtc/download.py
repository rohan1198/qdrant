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
S2_BASE_DELAY = 1.0  # seconds between batch requests
S2_API_KEY = os.environ.get("S2_API_KEY", "")  # Optional: set for higher rate limits


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

    Discovery-first: inspects format before parsing, skips non-data files.
    Uses only *_all_categories.csv files (full hierarchy depth).
    """
    import csv

    all_entries = []

    # Find CSV data files — only _all_categories.csv (full depth, not truncated _83_)
    data_files = sorted(gdrive_dir.rglob("*_all_categories.csv"))
    print(f"\n=== Discovery: Found {len(data_files)} all_categories CSV files ===")

    if not data_files:
        # Fallback: try any CSV
        data_files = sorted(f for f in gdrive_dir.rglob("*.csv") if f.is_file())
        print(f"  Fallback: found {len(data_files)} CSV files total")

    for fpath in data_files:
        # Determine split from filename
        fname = fpath.stem.lower()
        if "train" in fname:
            split = "train"
        elif "test" in fname:
            split = "test"
        elif "dev" in fname or "val" in fname:
            split = "dev"
        else:
            split = "unknown"
            print(f"  SKIP {fpath.name}: cannot determine split")
            continue

        print(f"\nParsing {fpath.name} (split={split})...")

        try:
            with open(fpath, newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                print(f"  Header: {header}")

                parsed_count = 0
                for row in reader:
                    if len(row) < 2:
                        continue

                    paper_id = row[0].strip()
                    # Category is the second column (may contain commas if quoted)
                    category = row[1].strip() if len(row) == 2 else ",".join(row[1:]).strip()

                    # Skip obviously non-numeric IDs (header rows, junk)
                    if not paper_id.isdigit():
                        continue

                    if category:
                        all_entries.append({
                            "paper_id": paper_id,
                            "labels": [category],
                            "split": split,
                        })
                        parsed_count += 1

                print(f"  Parsed {parsed_count} entries")

        except Exception as e:
            print(f"  ERROR reading {fpath.name}: {e}")
            continue

        # Show sample
        file_entries = [e for e in all_entries if e["split"] == split]
        if file_entries:
            sample = file_entries[0]
            print(f"  Sample: id='{sample['paper_id']}', label='{sample['labels'][0][:100]}...'")

    # Aggregate labels per paper (a paper can have multiple rows = multiple labels)
    print(f"\n=== Aggregating labels ===")
    paper_labels: dict[str, dict] = {}
    for entry in all_entries:
        pid = entry["paper_id"]
        if pid not in paper_labels:
            paper_labels[pid] = {"paper_id": pid, "labels": [], "split": entry["split"]}
        label = entry["labels"][0]
        if label not in paper_labels[pid]["labels"]:
            paper_labels[pid]["labels"].append(label)

    aggregated = list(paper_labels.values())
    multi_label = sum(1 for e in aggregated if len(e["labels"]) > 1)

    print(f"  Total entries (rows): {len(all_entries)}")
    print(f"  Unique papers: {len(aggregated)}")
    print(f"  Multi-label papers: {multi_label} ({multi_label/len(aggregated)*100:.1f}%)")

    # Discover formats
    if aggregated:
        sample_ids = [e["paper_id"] for e in aggregated[:10]]
        print(f"\nSample paper IDs: {sample_ids}")
        numeric = sum(1 for sid in sample_ids if sid.isdigit())
        print(f"  Numeric: {numeric}/10")

        sample_labels = aggregated[0]["labels"]
        print(f"Sample labels: {sample_labels[:3]}")
        has_arrow = any("->" in l for l in sample_labels)
        has_tilde = any("~" in l for l in sample_labels)
        print(f"  Arrow-separated (->): {has_arrow}, Tilde (~): {has_tilde}")

        # Show depth distribution
        depths = []
        for e in aggregated:
            for label in e["labels"]:
                parts = label.split("->")
                depths.append(len(parts))
        if depths:
            from collections import Counter
            depth_counts = Counter(depths)
            print(f"  Label depth distribution: {dict(sorted(depth_counts.items()))}")

    return aggregated


def _probe_s2_id_format(sample_ids: list[str]) -> str:
    """Try different ID formats against S2 API to find what works.

    ACM numeric IDs need special handling. Try formats in order:
    1. URL:https://dl.acm.org/doi/10.1145/{id} (ACM DL URL)
    2. CorpusID:{id}
    3. Raw numeric
    4. DOI:10.1145/{id}

    Returns the format string that works, or empty string.
    """
    test_ids = sample_ids[:5]
    formats = [
        ("ACM_URL", lambda pid: f"URL:https://dl.acm.org/doi/10.1145/{pid}"),
        ("CorpusID", lambda pid: f"CorpusID:{pid}"),
        ("ACM_DOI", lambda pid: f"DOI:10.1145/{pid}"),
        ("raw", lambda pid: pid),
    ]

    for fmt_name, fmt_fn in formats:
        formatted = [fmt_fn(pid) for pid in test_ids]
        print(f"  Probing format '{fmt_name}' with IDs: {formatted[:2]}...")

        try:
            resp = requests.post(
                S2_BATCH_URL,
                json={"ids": formatted},
                params={"fields": S2_FIELDS},
                timeout=15,
            )
            if resp.status_code == 200:
                data = resp.json()
                found = sum(1 for d in data if d is not None)
                print(f"    Result: {found}/{len(test_ids)} found")
                if found > 0:
                    # Show a sample
                    for d in data:
                        if d is not None:
                            print(f"    Sample: '{d.get('title', '?')[:80]}...'")
                            break
                    return fmt_name
            else:
                print(f"    HTTP {resp.status_code}")
        except Exception as e:
            print(f"    Error: {e}")

        time.sleep(1)

    print("  WARNING: No ID format worked!")
    return ""


def fetch_from_semantic_scholar(paper_ids: list[str]) -> dict[str, dict]:
    """Batch-fetch paper metadata from Semantic Scholar API.

    Args:
        paper_ids: List of paper identifiers (ACM numeric IDs, DOIs, etc.)

    Returns:
        Dict mapping paper_id -> {"title": str, "abstract": str|None, "s2_id": str}
    """
    print(f"\n=== Fetching {len(paper_ids)} papers from Semantic Scholar ===")

    # Probe to find the right ID format
    print("  Probing ID formats...")
    fmt_name = _probe_s2_id_format(paper_ids[:10])

    # Set up formatter based on probe result
    format_fns = {
        "ACM_URL": lambda pid: f"URL:https://dl.acm.org/doi/10.1145/{pid}",
        "CorpusID": lambda pid: f"CorpusID:{pid}",
        "ACM_DOI": lambda pid: f"DOI:10.1145/{pid}",
        "raw": lambda pid: pid,
    }
    format_fn = format_fns.get(fmt_name, format_fns["raw"])
    print(f"  Using format: {fmt_name}")

    # Build request headers
    headers = {}
    if S2_API_KEY:
        headers["x-api-key"] = S2_API_KEY
        print(f"  Using API key (1 RPS dedicated rate limit)")
    else:
        print(f"  No API key — using shared unauthenticated pool (slower, may hit 429s)")
        print(f"  Set S2_API_KEY env var for reliable fetching")

    results = {}
    total_batches = (len(paper_ids) + S2_BATCH_SIZE - 1) // S2_BATCH_SIZE
    found = 0
    with_abstract = 0
    title_only = 0
    missing = 0
    consecutive_429s = 0

    for batch_idx in range(total_batches):
        start = batch_idx * S2_BATCH_SIZE
        end = min(start + S2_BATCH_SIZE, len(paper_ids))
        batch_ids = paper_ids[start:end]

        # Format IDs for S2 API
        formatted_ids = [format_fn(pid) for pid in batch_ids]

        # Retry with exponential backoff
        max_retries = 5
        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    S2_BATCH_URL,
                    json={"ids": formatted_ids},
                    params={"fields": S2_FIELDS},
                    headers=headers,
                    timeout=30,
                )

                if resp.status_code == 200:
                    consecutive_429s = 0
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
                    break  # Success, exit retry loop

                elif resp.status_code == 429:
                    consecutive_429s += 1
                    backoff = min(2 ** attempt * 5, 120)  # 5s, 10s, 20s, 40s, 80s
                    print(f"  Rate limited (429) at batch {batch_idx}, attempt {attempt+1}/{max_retries}, "
                          f"waiting {backoff}s...")
                    time.sleep(backoff)
                    continue  # Retry

                else:
                    print(f"  Batch {batch_idx} failed: HTTP {resp.status_code}")
                    missing += len(batch_ids)
                    break  # Don't retry non-429 errors

            except Exception as e:
                print(f"  Batch {batch_idx} error: {e}")
                missing += len(batch_ids)
                break

        else:
            # All retries exhausted
            print(f"  Batch {batch_idx}: all {max_retries} retries failed, skipping")
            missing += len(batch_ids)

        # If we're getting hammered with 429s, slow down permanently
        if consecutive_429s >= 3:
            print(f"  3+ consecutive 429s — increasing base delay to 3s")
            delay = 3.0
        else:
            delay = S2_BASE_DELAY

        # Progress
        if (batch_idx + 1) % 50 == 0 or batch_idx == total_batches - 1:
            pct = (batch_idx + 1) / total_batches * 100
            print(f"  [{pct:.0f}%] Batch {batch_idx+1}/{total_batches}: "
                  f"found={found}, abstract={with_abstract}, title_only={title_only}, missing={missing}")

        time.sleep(delay)

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

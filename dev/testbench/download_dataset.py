"""Download and prepare the WOS hierarchical dataset for benchmarking.

Creates a 3-tier hierarchy:
- Root (~10 domain documents) — synthetic, from area descriptions
- Mid (~336 area documents) — synthetic, from paper titles in each area
- Leaf (~46K paper documents) — real paper abstracts (test+val splits)

Output: data/wos/documents.jsonl
"""

import json
import os
import random
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

OUTPUT_DIR = Path("data/wos")
OUTPUT_FILE = OUTPUT_DIR / "documents.jsonl"

# Use test + validation splits (smaller, faster; train is 106K)
SPLITS = ["test", "validation"]

# Max papers to sample for synthetic area/domain documents
TITLES_PER_AREA = 20
AREAS_PER_DOMAIN_DESC = 10


def main():
    if OUTPUT_FILE.exists():
        print(f"Dataset already exists at {OUTPUT_FILE}, skipping download.")
        print(f"Delete {OUTPUT_FILE} to re-download.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading WOS dataset from HuggingFace...")
    ds = load_dataset(
        "marcelsun/wos_hierarchical_multi_label_text_classification",
    )

    # Collect papers grouped by (domain, area)
    papers_by_area: dict[tuple[int, int], list[dict]] = defaultdict(list)
    all_papers = []

    for split_name in SPLITS:
        split = ds[split_name]
        for sample in tqdm(split, desc=f"Processing {split_name}"):
            text = sample["token"]
            labels = sample["label"]
            domain_id = labels[0]
            area_id = labels[1]

            paper = {
                "text": text,
                "domain": domain_id,
                "area": area_id,
            }
            all_papers.append(paper)
            papers_by_area[(domain_id, area_id)].append(paper)

    print(f"Total papers: {len(all_papers)}")
    print(f"Unique domains: {len(set(p['domain'] for p in all_papers))}")
    print(f"Unique areas: {len(set(p['area'] for p in all_papers))}")

    # Build documents for all tiers
    documents = []
    doc_id = 0

    # --- Leaf tier: individual papers ---
    for paper in all_papers:
        documents.append({
            "id": f"leaf_{doc_id}",
            "text": paper["text"],
            "tier": "leaf",
            "domain": paper["domain"],
            "area": paper["area"],
            "hierarchy_path": f"{paper['domain']}/{paper['area']}",
        })
        doc_id += 1

    print(f"Leaf documents: {doc_id}")

    # --- Mid tier: area representatives ---
    areas_by_domain: dict[int, list[int]] = defaultdict(list)
    mid_count = 0

    for (domain_id, area_id), papers in papers_by_area.items():
        # Sample paper titles/abstracts to create area description
        sampled = random.sample(papers, min(TITLES_PER_AREA, len(papers)))
        # Use first 200 chars of each paper as a snippet
        snippets = [p["text"][:200] for p in sampled]
        area_text = (
            f"Research area {area_id} in domain {domain_id}. "
            f"Representative works: {' | '.join(snippets)}"
        )

        documents.append({
            "id": f"mid_{domain_id}_{area_id}",
            "text": area_text,
            "tier": "mid",
            "domain": domain_id,
            "area": area_id,
            "hierarchy_path": f"{domain_id}/{area_id}",
        })
        mid_count += 1
        areas_by_domain[domain_id].append(area_id)

    print(f"Mid (area) documents: {mid_count}")

    # --- Root tier: domain representatives ---
    root_count = 0

    for domain_id, area_ids in areas_by_domain.items():
        # Build domain description from area descriptions
        sampled_areas = random.sample(
            area_ids, min(AREAS_PER_DOMAIN_DESC, len(area_ids))
        )
        area_descs = []
        for aid in sampled_areas:
            area_papers = papers_by_area[(domain_id, aid)]
            snippets = [p["text"][:100] for p in area_papers[:5]]
            area_descs.append(f"Area {aid}: {' | '.join(snippets)}")

        domain_text = (
            f"Research domain {domain_id} spanning {len(area_ids)} areas. "
            f"Sample areas: {' '.join(area_descs)}"
        )

        documents.append({
            "id": f"root_{domain_id}",
            "text": domain_text,
            "tier": "root",
            "domain": domain_id,
            "area": -1,  # root spans all areas
            "hierarchy_path": f"{domain_id}",
        })
        root_count += 1

    print(f"Root (domain) documents: {root_count}")
    print(f"Total documents: {len(documents)}")

    # Write output
    with open(OUTPUT_FILE, "w") as f:
        for doc in documents:
            f.write(json.dumps(doc) + "\n")

    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

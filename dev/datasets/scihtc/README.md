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

`data/documents.jsonl` -- one JSON object per line:
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

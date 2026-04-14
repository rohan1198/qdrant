"""A/B comparison framework for benchmarking pipeline configurations.

Runs the same queries through different PipelineConfigs and produces
side-by-side comparison tables.
"""

import time
import numpy as np
from scipy import stats
from pipeline_config import PipelineConfig


def compute_recall_at_k(retrieved_ids: list[int], ground_truth_ids: list[int], k: int = 10) -> float:
    """Compute recall@k: fraction of ground truth found in top-k retrieved."""
    retrieved_set = set(retrieved_ids[:k])
    gt_set = set(ground_truth_ids[:k])
    if not gt_set:
        return 0.0
    return len(retrieved_set & gt_set) / len(gt_set)


def compute_rank_correlation(ordering_a: list[int], ordering_b: list[int]) -> float:
    """Spearman rank correlation between two orderings of the same IDs.

    Returns correlation coefficient in [-1, 1]. Higher = more similar ordering.
    """
    common = set(ordering_a) & set(ordering_b)
    if len(common) < 3:
        return 0.0

    ranks_a = {pid: rank for rank, pid in enumerate(ordering_a) if pid in common}
    ranks_b = {pid: rank for rank, pid in enumerate(ordering_b) if pid in common}

    ids = sorted(common)
    ra = [ranks_a[pid] for pid in ids]
    rb = [ranks_b[pid] for pid in ids]

    corr, _ = stats.spearmanr(ra, rb)
    return float(corr) if not np.isnan(corr) else 0.0


def compare_report(results: dict[str, list[dict]], metric_keys: list[str] = None) -> str:
    """Generate a side-by-side comparison report.

    Args:
        results: {config_name: [per_query_metrics_dict, ...]}
        metric_keys: which metrics to include (default: all numeric keys)

    Returns:
        Formatted comparison table as string
    """
    if not results:
        return "No results to compare."

    config_names = list(results.keys())
    sample = results[config_names[0]][0]

    if metric_keys is None:
        metric_keys = [k for k, v in sample.items() if isinstance(v, (int, float))]

    lines = []
    header = f"{'Metric':<30}" + "".join(f"{name:<20}" for name in config_names)
    lines.append(header)
    lines.append("-" * len(header))

    for key in metric_keys:
        row = f"{key:<30}"
        values = []
        for name in config_names:
            vals = [q[key] for q in results[name] if key in q]
            if vals:
                avg = np.mean(vals)
                values.append(avg)
                row += f"{avg:<20.4f}"
            else:
                values.append(None)
                row += f"{'N/A':<20}"

        if values and all(v is not None for v in values):
            is_latency = "latency" in key.lower() or "time" in key.lower()
            best_idx = int(np.argmin(values)) if is_latency else int(np.argmax(values))
            row += f"  <- {config_names[best_idx]}"

        lines.append(row)

    return "\n".join(lines)

# Phase 7: Chatbot Query Optimization Benchmarks

**Date:** 2026-04-14
**Status:** Design
**Branch:** feat/hyperbolic-vector-support
**Depends on:** Phase 6 (commit f0f041698)

---

## 1. Problem Statement

Pythia's chatbot (Lens) currently requires multi-hop, multi-database retrieval for every query:

1. **Qdrant** — cosine similarity search across per-type collections (networks_posts, networks_stories, networks_narratives)
2. **Neo4j** — graph traversal for hierarchy (Story→Narrative BELONGS_TO, evolution edges) and relationships
3. **MongoDB** — full document enrichment
4. **LLM** — tokens spent constructing and explaining hierarchy structure in prompts

Real user queries from Pythia's `NetworksLensHistory` (67 queries across 3 clients: mayor_of_london, fcdo_iraq, italian_mod_libya) reveal that many queries are inherently hierarchical:

- "Tell me about the narratives around crime in London" — explicitly requests narrative-tier results with supporting evidence
- "Shia perception of UK following US/Israel strikes on Iran. Please also show how the UK compares to other Western countries and what different Shia militias are saying" — multi-level briefing with cross-branch comparison
- "Disorder following teen riots in Clapham, London" — event-level input requiring upward hierarchy traversal
- "view of british in iraq currently" — asks for narrative-level perception, not flat post retrieval

**Thesis:** A unified per-client collection with hyperbolic embeddings makes hierarchy implicit in the retrieval geometry, reducing the chatbot pipeline from 5-7 cross-database hops to 1-2 Qdrant queries while producing results that are already hierarchy-ordered.

Phase 7 validates this thesis by benchmarking hierarchy-aware retrieval against flat cosine retrieval on real chatbot query patterns.

---

## 2. Goals

1. **Measure retrieval quality difference** between hyperbolic-informed and flat cosine retrieval for hierarchy-dependent chatbot queries
2. **Measure operational efficiency** — query count, cumulative latency for equivalent retrieval tasks
3. **Validate on two datasets** — BGC (4-level, 92K docs) and HWV (6-level, 10K docs, variable depth)
4. **Produce a comparison report** with per-pattern, per-strategy metrics

### Non-Goals

- Not implementing a chatbot or LLM integration
- Not integrating with Pythia (that's a later phase)
- Not modifying the Qdrant fork (all work is Python-side in the testbench)
- Not benchmarking sparse vectors (focus is dense vs hyperbolic geometry)

---

## 3. Datasets

### BGC (Blurb Genre Collection)
- 4 levels: domain (7) → area (44) → content (91,894) + sub_story (45) + tier_L4 (12)
- 92,021 documents total
- Already embedded and populated at c=0.25 from Phase 6
- Strength: large scale, validates throughput
- Weakness: 99.9% leaf tier, shallow hierarchy

### HWV (Hierarchical WikiVitals)
- 6 levels: 11 → 109 → 381 → 437 → 244 → 4 nodes per level
- 10,013 documents
- Already downloaded, used in Phase 5 benchmarks
- Strength: deeper hierarchy, variable-depth paths (2-6), class imbalance
- Weakness: smaller scale

Both datasets have ground truth hierarchy paths per document, enabling exact evaluation of all hierarchy-awareness metrics.

---

## 4. Three Retrieval Strategies

Every query pattern runs through all three strategies, producing comparable result sets.

### Strategy 1: Cosine-Only (flat baseline)

Uses the `dense` named vector (1024d Cosine) on the unified collection. Payload filters (tier, busemann_depth range) applied where the query pattern requires tier-specific results. Represents: "everything in one collection, flat retrieval."

### Strategy 2: Hyperbolic (the thesis)

Uses `tangent` named vector (128d Euclid HNSW) for hierarchy-aware candidate retrieval. Busemann depth payloads for tier identification. parent_ids / child_ids for traversal. RRF fusion of dense + tangent where appropriate. Represents: "unified collection with hierarchy implicit in geometry."

### Strategy 3: Multi-Hop (current Pythia simulation)

Separate per-tier collections: `{dataset}_narratives`, `{dataset}_stories`, `{dataset}_content`. Sequential queries: search the target tier first, then traverse to related tiers via payload IDs. Results stitched together in Python. Measures total query count and cumulative latency. Represents: "the current multi-database approach."

The per-tier collections are created during setup from the same data as the unified collection — same documents, same embeddings, just split by tier.

---

## 5. Six Query Patterns

Each pattern simulates a real chatbot interaction. Ground truth is derived from the dataset's hierarchy labels (hierarchy_path, parent_ids, child_ids, tier).

### Pattern 1: Drill-Up

**Real query example:** "Disorder following teen riots in Clapham, London"

- **Input:** Random leaf-tier document
- **Task:** Find its parent (story-tier), grandparent (narrative-tier), and so on up the hierarchy
- **Ground truth:** The document's `hierarchy_path` ancestors — e.g., if a BGC content doc has path "Literature/Fiction", ground truth ancestors are area "Fiction" and domain "Literature"
- **Primary metric:** Ancestor precision — fraction of returned results that are actual ancestors of the query document
- **Secondary metric:** Hop efficiency — number of Qdrant queries needed to reach the root

**How each strategy handles it:**
- Cosine: Search for similar docs, filter by tier=story then tier=narrative. May find wrong branch.
- Hyperbolic: Tangent HNSW retrieves hierarchy-nearby docs; filter by shallower busemann_depth. Geometry naturally favors the correct subtree.
- Multi-hop: Look up parent_ids from payload → query stories collection → look up parent_ids → query narratives collection. Guaranteed correct but requires sequential queries.

### Pattern 2: Context Assembly

**Real query example:** "Central bank narratives", "Give me an update on Benghazi"

- **Input:** Random narrative/root-tier document
- **Task:** Return a complete hierarchy slice — the narrative, its stories, and sample content from each story
- **Ground truth:** All documents sharing the same subtree root
- **Primary metric:** Subtree coherence — fraction of results that belong to the query's subtree
- **Secondary metric:** Depth coverage — fraction of hierarchy levels represented in the top-k results (e.g., if the hierarchy has 4 levels and results contain items from 3 levels, coverage = 0.75)

**How each strategy handles it:**
- Cosine: Search from narrative embedding, hope semantic similarity pulls in child stories/content. Likely returns mostly same-tier results.
- Hyperbolic: Tangent HNSW from narrative position retrieves the surrounding subtree naturally — children are geometrically close in the Poincaré ball. Single query returns multi-tier results.
- Multi-hop: Drill down via child_ids payload from narrative → stories → content. Correct but requires 2-3 sequential queries.

### Pattern 3: Hierarchy-Aware Similarity

**Real query example:** "Knife Crime in London" (find related content within the same story/narrative)

- **Input:** Random leaf document + its parent ID as constraint
- **Task:** Find sibling documents — same parent, semantically similar
- **Ground truth:** Documents sharing the same parent node in the hierarchy
- **Primary metric:** Sibling recall — fraction of true siblings in top-k results
- **Secondary metric:** Intruder rate — fraction of results from a different parent (different branch)

**How each strategy handles it:**
- Cosine: Search with parent_id filter. Works but filter is doing all the hierarchy work; the embeddings don't help.
- Hyperbolic: Tangent HNSW naturally clusters siblings (they share a subtree in the Poincaré ball). InCone geometric filter can further constrain to the parent's directional subtree.
- Multi-hop: Query the content collection with parent_id filter. Equivalent to cosine with filter.

### Pattern 4: Multi-Level Briefing

**Real query example:** "Tell me about the narratives around crime in London"

- **Input:** A domain/top-level category (by keyword or embedding)
- **Task:** Return a depth-diverse result set — narratives for overview, stories for detail, content for evidence
- **Ground truth:** Documents from the target domain subtree, with representation from every hierarchy level
- **Primary metric:** Depth distribution entropy — higher entropy means better spread across levels (max entropy = uniform distribution across tiers). Specifically: `H = -Σ p_tier × log(p_tier)` where p_tier is fraction of results at each tier.
- **Secondary metric:** Subtree precision — fraction of results from the correct domain
- **Tertiary metric:** Level completeness — binary: are ALL hierarchy levels represented (yes/no)?

**How each strategy handles it:**
- Cosine: Returns 10 results, almost certainly all from the dominant tier (content). No depth diversity.
- Hyperbolic: Single tangent HNSW query returns results at multiple depths because the Poincaré ball encodes the full hierarchy. Busemann depth in results naturally spans tiers.
- Multi-hop: Query each tier's collection separately, merge. Guaranteed depth diversity but requires N queries (one per tier).

### Pattern 5: Cross-Branch Discovery

**Real query example:** "What other narratives are related to this one?" (from the "King Charles narratives" query)

- **Input:** A narrative-tier document
- **Task:** Find other narrative-tier documents from different branches
- **Ground truth:** Narratives from other subtrees (different parent) at the same depth
- **Primary metric:** Depth precision — fraction of results at narrative tier (not leaf or root)
- **Secondary metric:** Branch diversity — number of distinct parent branches represented in results

**How each strategy handles it:**
- Cosine: Finds semantically similar docs, but many will be content-tier (wrong depth). No depth control.
- Hyperbolic: Tangent HNSW + busemann_depth filter restricts to narrative-tier. Poincaré geometry preserves branch structure, so results come from distinct subtrees.
- Multi-hop: Query the narratives collection directly. Correct tier guaranteed, but only cosine similarity for branch selection.

### Pattern 6: Narrative Landscape

**Real query example:** "view of british in iraq currently", "UK in Iraq"

- **Input:** A broad topic query (embedding of the query text)
- **Task:** Surface the distinct narrative branches about this topic — not the most popular narrative repeated 10 times, but structurally different narratives
- **Ground truth:** Distinct top-level subtrees that contain documents matching the query topic
- **Primary metric:** Branch diversity — number of distinct narrative branches (unique root or 2nd-level ancestors) in top-k
- **Secondary metric:** Branch coverage — of all ground truth branches containing relevant docs, what fraction is represented?
- **Tertiary metric:** Redundancy ratio — fraction of results that share the same narrative branch as another result (lower is better)

**How each strategy handles it:**
- Cosine: Returns semantically nearest results, heavily biased toward the single most popular narrative. High redundancy.
- Hyperbolic: Poincaré geometry encodes branch structure. Combined with depth-band filtering (narrative tier) and diversity-aware retrieval, surfaces distinct subtrees. Cross-branch query pattern from Phase 6 (suite 5) showed 6.0 unique domains.
- Multi-hop: Query narratives collection, get top-k. Same cosine bias toward popular narrative. No branch diversity advantage.

---

## 6. Hierarchy-Awareness Metrics

All metrics are computed per-query and averaged across the query set (50 queries per pattern per dataset).

| Metric | Definition | Range | Better |
|--------|-----------|-------|--------|
| **Ancestor precision** | Fraction of results that are actual hierarchy ancestors of query | [0, 1] | Higher |
| **Subtree coherence** | Fraction of results belonging to query's subtree | [0, 1] | Higher |
| **Depth coverage** | Fraction of hierarchy levels represented in results | [0, 1] | Higher |
| **Depth distribution entropy** | Shannon entropy of tier distribution in results | [0, log(n_tiers)] | Higher |
| **Sibling recall** | Fraction of true siblings in results | [0, 1] | Higher |
| **Intruder rate** | Fraction of results from a different parent branch | [0, 1] | Lower |
| **Depth precision** | Fraction of results at the target tier | [0, 1] | Higher |
| **Branch diversity** | Number of distinct narrative branches in results | [1, k] | Higher |
| **Branch coverage** | Fraction of relevant branches represented | [0, 1] | Higher |
| **Redundancy ratio** | Fraction of results sharing a branch with another result | [0, 1] | Lower |

### Operational Metrics (per-pattern)

| Metric | Definition |
|--------|-----------|
| **Query count** | Number of Qdrant API calls to complete the retrieval task |
| **Total latency (ms)** | Cumulative wall-clock time for all queries in the pattern |
| **Result count** | Total documents retrieved |

---

## 7. Module Structure

```
dev/testbench/chatbot/
├── __init__.py
├── eval.py              # Main evaluator entry point
├── baselines.py         # Three retrieval strategies (cosine, hyperbolic, multi_hop)
├── metrics.py           # Hierarchy-awareness metric computations
├── query_patterns.py    # 6 chatbot query pattern implementations
└── results/             # JSON output (gitignored)
```

### Dependencies on Parent Testbench

Imports from `dev/testbench/`:
- `hyperbolic_math.py` — Poincaré distance, Busemann depth, log_map_origin
- `tangent.py` — tangent space projection for queries
- `queries.py` — tangent_pipeline_query, alpha_pipeline_query
- `fusion.py` — rrf_fuse
- `collection_builder.py` — create_unified_collection (for per-tier setup)
- `embed.py` — reuse embed pipeline for HWV population

No modifications to existing testbench files.

### Entry Point

```bash
cd dev/testbench
python -m chatbot.eval --dataset bgc [--patterns drill_up context_assembly ...]
python -m chatbot.eval --dataset hwv
python -m chatbot.eval --dataset bgc --report  # comparison report
```

### Output Format

JSON results file per run:
```json
{
  "dataset": "bgc",
  "timestamp": "2026-04-15T...",
  "patterns": {
    "drill_up": {
      "cosine": { "ancestor_precision": 0.XX, "query_count": N, "latency_ms": XX, ... },
      "hyperbolic": { "ancestor_precision": 0.XX, "query_count": N, "latency_ms": XX, ... },
      "multi_hop": { "ancestor_precision": 0.XX, "query_count": N, "latency_ms": XX, ... }
    },
    "context_assembly": { ... },
    ...
  }
}
```

---

## 8. Setup Requirements

### Unified Collection (already exists for BGC)

The Phase 6 unified collection (`bgc_unified`) with named vectors dense + poincaré + tangent, payloads (tier, busemann_depth, parent_ids, child_ids, hierarchy_path, domain, area), at c=0.25.

HWV needs the same treatment: embed, project at c=0.25, populate unified collection with tangent vectors.

### Per-Tier Collections (new, for multi-hop baseline)

Created during setup by splitting the unified collection data by tier:
- `{dataset}_narratives` — narrative/root-tier docs only, dense vector, Cosine
- `{dataset}_stories` — story/mid-tier docs only, dense vector, Cosine
- `{dataset}_content` — content/leaf-tier docs only, dense vector, Cosine

Each has payload indexes on parent_ids and child_ids to enable ID-based traversal.

---

## 9. Query Generation

For each pattern, 50 queries are generated per dataset using deterministic random sampling (seed=42):

- **Drill-Up:** Sample 50 leaf-tier documents. Use their embeddings as query vectors.
- **Context Assembly:** Sample 50 narrative/root-tier documents. Use their embeddings as query vectors.
- **Hierarchy-Aware Similarity:** Sample 50 leaf-tier documents. Query = (embedding, parent_id) pair.
- **Multi-Level Briefing:** Sample 50 domain/top-level categories. Query = category embedding (Einstein midpoint of all docs in domain).
- **Cross-Branch Discovery:** Sample 50 narrative-tier documents. Use their embeddings as query vectors.
- **Narrative Landscape:** Sample 50 documents (any tier), one per domain. Use their embeddings as broad topic queries. Ground truth = all distinct narrative branches (2nd-level subtree roots) within that domain.

---

## 10. Success Criteria

Phase 7 validates the thesis if:

1. **Hyperbolic strategy shows measurably higher hierarchy-awareness metrics** than cosine-only on at least 4 of 6 patterns (subtree coherence, depth coverage, sibling recall, branch diversity)
2. **Hyperbolic strategy uses fewer queries** than multi-hop for equivalent or better result quality on at least 4 of 6 patterns
3. **Results are consistent across both BGC and HWV** — the advantage isn't dataset-specific
4. **Multi-Level Briefing (Pattern 4)** shows clear depth distribution entropy advantage for hyperbolic — this is the core chatbot use case

If hyperbolic shows no advantage over cosine-only, that's also a valid finding — it means the unified collection architecture alone (without hyperbolic geometry) is sufficient, and the value is in consolidation rather than geometry.

---

## 11. Reference Queries from Pythia Production

These real user queries from `NetworksLensHistory` motivated the 6 patterns and serve as reference examples:

| Query | Client | Maps to Pattern |
|-------|--------|-----------------|
| "Tell me about the narratives around crime in London" | mayor_of_london | Multi-Level Briefing |
| "Shia perception of UK following US/Israel strikes on Iran. Please also show how the UK compares to other Western countries and what different Shia militias are saying" | fcdo_iraq | Multi-Level Briefing + Cross-Branch |
| "Disorder following teen riots in Clapham, London" | mayor_of_london | Drill-Up + Context Assembly |
| "Central bank narratives" | italian_mod_libya | Context Assembly |
| "Give me an update on Benghazi" | italian_mod_libya | Context Assembly |
| "King Charles narratives" | mayor_of_london | Context Assembly + Cross-Branch |
| "view of british in iraq currently" | fcdo_iraq | Narrative Landscape |
| "UK in Iraq" | fcdo_iraq | Narrative Landscape |
| "how shia's in Iraq view the UK after the US/Israeli strikes on Iran" | fcdo_iraq | Drill-Up + Narrative Landscape |
| "anti-dog agenda" | mayor_of_london | Narrative Landscape |
| "Is the UK coming up in the media" | fcdo_iraq | Narrative Landscape |

# Graph Report - /Users/chrissong/research_su26/02_projects/mp-factory  (2026-07-28)

## Corpus Check
- Corpus is ~1,961 words - fits in a single context window. You may not need a graph.

## Summary
- 27 nodes · 23 edges · 8 communities (7 shown, 1 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 1 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Python Segmentation Scripts
- Mask Evaluation Pipeline
- Project Core Concepts
- Audit SLURM Script

## God Nodes (most connected - your core abstractions)
1. `TotalSegmentator` - 6 edges
2. `evaluate_single_subject()` - 4 edges
3. `mp-factory` - 4 edges
4. `compute_betti_0()` - 3 edges
5. `compute_hd95()` - 3 edges
6. `main()` - 2 edges
7. `evaluate_gi_masks.py` - 2 edges
8. `Compute Betti-0 (number of connected components).` - 1 edges
9. `Compute 95th percentile Hausdorff Distance (HD95).` - 1 edges
10. `submit_audit.sh script` - 1 edges

## Surprising Connections (you probably didn't know these)
- `mp-factory` --references--> `TotalSegmentator`  [EXTRACTED]
  AGENTS.md → AGENTS.md  _Bridges community 2 → community 0_

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Automated 3D Segmentation Workflows** — agents_run_totalsegmentator_batch_py, agents_run_totalseg_py, agents_run_totalseg_bdmap_py, agents_run_totalseg_gi_array_py, agents_totalsegmentator [EXTRACTED 1.00]

## Communities (8 total, 1 thin omitted)

### Community 0 - "Python Segmentation Scripts"
Cohesion: 0.29
Nodes (7): evaluate_gi_masks.py, run_totalseg_bdmap.py, run_totalseg_gi_array.py, run_totalseg.py, run_totalsegmentator_batch.py, submit_audit.sh, TotalSegmentator

### Community 1 - "Mask Evaluation Pipeline"
Cohesion: 0.43
Nodes (6): compute_betti_0(), compute_hd95(), evaluate_single_subject(), main(), Compute Betti-0 (number of connected components)., Compute 95th percentile Hausdorff Distance (HD95).

### Community 2 - "Project Core Concepts"
Cohesion: 0.50
Nodes (4): BodyMaps, CancerVerse, mp-factory, SyntheticTumors

## Knowledge Gaps
- **9 isolated node(s):** `submit_audit.sh script`, `CancerVerse`, `BodyMaps`, `SyntheticTumors`, `submit_audit.sh` (+4 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `TotalSegmentator` connect `Python Segmentation Scripts` to `Project Core Concepts`?**
  _High betweenness centrality (0.117) - this node is a cross-community bridge._
- **Why does `mp-factory` connect `Project Core Concepts` to `Python Segmentation Scripts`?**
  _High betweenness centrality (0.074) - this node is a cross-community bridge._
- **What connects `submit_audit.sh script`, `CancerVerse`, `BodyMaps` to the rest of the system?**
  _9 weakly-connected nodes found - possible documentation gaps or missing edges._
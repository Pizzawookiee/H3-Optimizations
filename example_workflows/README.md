# H3 Optimizations example workflows

- 01_h3_memory_optimization.json — stock MiniMax H3 structure + H3 Memory Optimization defaults.
- 02_h3_recommended_15pct.json — stock MiniMax H3 subgraph + Memory Optimization + simple Sparse Attention at 15% KV with denser first/last 20% windows (50% edges). Only Video attention budget is exposed as an H3 control on the main graph.
- 03_h3_advanced_15_50_50.json — flattened workflow + Memory Optimization + Advanced Sparse Attention at 15% middle, 50% first 4 steps, 50% last 4 steps, Kitchen INT8 backend.

Source structure: Comfy-Org/workflow_templates/templates/video_minimax_h3_t2v.json (current stock MiniMax H3 template when generated).

# Master-Thesis-Experiment
The purpose of this research project is to evaluate the usefulness, usability, and analytical value of a visualization tool designed to support a process analyst during exploratory process analysis.

**Tutorial Binder:** [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lauradidden/Master-Thesis-Experiment/main?urlpath=%2Fdoc%2Ftree%2FTutorial_binder.ipynb)  

---

## Project Structure

#### `utils.py`
- `format_seconds` – format durations.  
- `format_metric_value` – format metrics consistently.  
- Metric enrichment functions (`add_case_durations`, `add_event_counts`, `add_avg_time_between_events`).  
- `METRIC_CONFIG` – central config for metrics.  
- `build_query_maps` – build query name to object/expression mappings.  
- `highlight_main_path` – mark lineage labels with 🟡.  
- `print_summary` – print case/metric summaries.  

#### `lineage_core.py`
- `get_lineage` – reconstruct lineage from a result set.  
- `get_sibling_subsets` – get parent log/query for pie chart.  

#### `lineage_filters.py`
- `compute_case_stats` – aggregate stats per subset.  
- `split_subsets` – divide subsets into filtered vs complement.  
- `recursively_apply_filters` – apply filters step-by-step and collect stats.  

#### `chart_helpers.py`
- `build_case_paths` – reconstruct pass/fail filter paths per case.  
- `get_normalized_colors` – normalize values and sample from a colorscale.  
- `format_slice_labels` – add readable labels, highlight final path with 🟡.  

#### `visualizations.py`
- `query_exploration_icicle` – builds the icicle chart.  
- `query_breakdown_pie` – builds the pie chart.  


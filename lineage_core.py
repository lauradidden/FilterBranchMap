import pandas as pd
from typing import Dict, List
from utils import build_query_maps


def get_lineage(registry: Dict[str, pd.DataFrame], result_set_name: str) -> pd.DataFrame:
    """Trace the lineage of a result_set by walking backwards through evaluations."""
    evaluations = registry["evaluations"]
    lineage_rows: List[pd.Series] = []

    def trace_back(current: str) -> None:
        for _, row in evaluations.iterrows():
            if row["result_set"] == current:
                lineage_rows.append(row)
                trace_back(row["source_log"])

    trace_back(result_set_name)
    return pd.DataFrame(lineage_rows[::-1])


def get_sibling_subsets(result_set_name, log_view):
    """Find sibling subsets by getting the parent log and query of a lineage leaf."""
    lineage_df = get_lineage(log_view.query_registry.summary(), result_set_name)
    if lineage_df.empty:
        raise ValueError("Lineage not found.")

    last_row = lineage_df.iloc[-1]
    parent_log, query_name, label, step_idx = (
        last_row["source_log"],
        last_row["query"],
        last_row["labels"],
        len(lineage_df) - 1,
    )
    query_map, _ = build_query_maps(log_view.query_registry)
    query_obj = query_map.get(query_name)

    if not query_obj:
        raise ValueError(f"Query object '{query_name}' not found.")
    return parent_log, query_obj, label, step_idx, lineage_df

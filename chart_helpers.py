import pandas as pd
from plotly.colors import sample_colorscale
from utils import format_metric_value


def compute_hover_data(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Hover info for icicle charts."""
    path_cols = [c for c in df.columns if c.startswith("Level")]

    def clean(label: str) -> str:
        return label.replace("🟡 ", "").strip()

    cleaned = df[path_cols].map(clean)
    df["id"] = cleaned.agg(" → ".join, axis=1)
    df["parent_id"] = cleaned.shift(axis=1).fillna("").agg(" → ".join, axis=1)
    df.loc[df["parent_id"].str.strip() == "", "parent_id"] = None

    parent_lookup = df.set_index("id")[[metric, "num_cases"]].to_dict("index")

    hover_cases, hover_pct, hover_metric, hover_delta = [], [], [], []

    for _, row in df.iterrows():
        cur_val, cur_cases = row[metric], row["num_cases"]
        parent_info = parent_lookup.get(row["parent_id"])

        hover_cases.append(f"{cur_cases:,}")
        hover_pct.append(
            f"{(cur_cases / parent_info['num_cases'] * 100):.1f}%" if parent_info else "100%"
        )
        hover_metric.append(format_metric_value(metric, cur_val))

        if parent_info:
            diff = cur_val - parent_info[metric]
            sign = "+" if diff >= 0 else "-"
            hover_delta.append(f"{sign}{format_metric_value(metric, abs(diff))}")
        else:
            hover_delta.append("—")

    df["hover_cases"], df["hover_pct"], df["hover_metric"], df["hover_delta"] = (
        hover_cases,
        hover_pct,
        hover_metric,
        hover_delta,
    )
    return df


def build_case_paths(lineage_df, full_log, filtered, log_view):
    """Build case to path mappings and return case_paths + final_result_path."""
    from utils import build_query_maps

    case_paths = {}
    query_map, query_expr_map = build_query_maps(log_view.query_registry)

    for _, row in lineage_df.iterrows():
        qname = row["query"]
        qexpr = query_expr_map.get(qname, qname)
        step_obj = query_map.get(qname)

        if step_obj is not None:
            passed_df, _ = log_view.query_evaluator.evaluate(full_log, step_obj)
            passed_cases = set(passed_df["case:concept:name"])
            for cid in filtered["case:concept:name"].unique():
                case_paths.setdefault(cid, [])
                case_paths[cid].append(f"{qexpr} ✅" if cid in passed_cases else f"{qexpr} ❌")

    final_result_path = " → ".join(
        [f"{query_expr_map.get(row['query'], row['query'])} ✅" for _, row in lineage_df.iterrows()]
    )
    return case_paths, final_result_path


def get_normalized_colors(values: pd.Series, color_scheme: str):
    """Normalize values and return a list of colors from the scheme."""
    min_val, max_val = values.min(), values.max()
    normed = (values - min_val) / (max_val - min_val + 1e-9)
    return sample_colorscale(color_scheme, normed.tolist())


def format_slice_labels(grouped: pd.DataFrame, final_result_path: str) -> pd.DataFrame:
    """Format slice labels for pie charts."""
    total_cases = grouped["num_cases"].sum()

    def label_with_dot(row):
        label = f"{int(row['num_cases']):,} cases ({row['num_cases'] / total_cases:.0%})"
        return f"🟡 {label}" if row["path_label"] == final_result_path else label

    grouped = grouped.copy()
    grouped.loc[:, "slice_label"] = grouped.apply(label_with_dot, axis=1)
    return grouped

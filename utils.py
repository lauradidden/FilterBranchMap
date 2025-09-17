import pandas as pd
from typing import Callable, Dict, Tuple, List

CASE_ID_COL = "case:concept:name"

def format_seconds(seconds: float) -> str:
    """Format seconds into a readable string (Xd Yh Zm)."""
    if pd.isna(seconds):
        return "N/A"
    seconds = int(seconds)
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes = rem // 60
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    return " ".join(parts) if parts else "0m"


def format_metric_value(metric: str, value: float) -> str:
    """Format a metric value using METRIC_CONFIG."""
    return format_seconds(value) if METRIC_CONFIG[metric]["is_time"] else f"{value:.2f}"


def add_case_durations(df: pd.DataFrame) -> pd.DataFrame:
    """Add case duration in seconds to each row."""
    grouped = df.groupby(CASE_ID_COL)["time:timestamp"]
    durations = (grouped.max() - grouped.min()).dt.total_seconds()
    df = df.copy()
    df["case_duration"] = df[CASE_ID_COL].map(durations)
    return df


def add_event_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Add number of events per case to each row."""
    counts = df.groupby(CASE_ID_COL).size()
    df = df.copy()
    df["num_events"] = df[CASE_ID_COL].map(counts)
    return df


def add_avg_time_between_events(df: pd.DataFrame) -> pd.DataFrame:
    """Add average time between events (seconds) to each row."""
    def avg_diff(x: pd.Series) -> float:
        diffs = x.sort_values().diff().dropna()
        return diffs.mean().total_seconds() if not diffs.empty else 0

    avg_diffs = df.groupby(CASE_ID_COL)["time:timestamp"].apply(avg_diff)
    df = df.copy()
    df["avg_time_between_events"] = df[CASE_ID_COL].map(avg_diffs)
    return df


METRIC_CONFIG = {
    "avg_case_duration_seconds": {
        "column": "case_duration",
        "enrich_fn": add_case_durations,
        "color_scheme": "Blues",
        "label": "Avg Case Duration (s)",
        "is_time": True,
    },
    "avg_events_per_case": {
        "column": "num_events",
        "enrich_fn": add_event_counts,
        "color_scheme": "Reds",
        "label": "Avg Events/Case",
        "is_time": False,
    },
    "avg_time_between_events": {
        "column": "avg_time_between_events",
        "enrich_fn": add_avg_time_between_events,
        "color_scheme": "Greens",
        "label": "Avg Time Between Events (s)",
        "is_time": True,
    },
}


def build_query_maps(query_registry) -> Tuple[Dict[str, object], Dict[str, str]]:
    """
    Build:
      (1) a map from query name to query object
      (2) a map from query name to its string expression
    """
    query_map, query_expr_map = {}, {}
    for rs_id in query_registry.get_registered_result_set_ids():
        ev = query_registry.get_evaluation(rs_id)
        name = ev["query"].name
        query_map[name] = ev["query"]
        query_expr_map[name] = ev["query"].as_string()
    return query_map, query_expr_map


def highlight_main_path(label_path: List[str], main_labels: set) -> List[str]:
    """Add 🟡 to labels along the main path for display purposes."""
    return [
        ("🟡 " + label if tuple(label_path[:i+1]) in main_labels else label)
        for i, label in enumerate(label_path)
    ]


def print_summary(rows: pd.DataFrame, path_cols: List[str], final_result_path: str,
                  metric: str, color_title: str) -> None:
    """Print a textual summary of case counts and metric values."""
    for _, row in rows.iterrows():
        path_parts = [str(row[c]).replace("🟡 ", "") for c in path_cols if pd.notna(row[c])]
        current_path = " → ".join(path_parts)
        is_final = current_path == final_result_path
        prefix = "🟡 " if is_final else ""
        metric_val = format_metric_value(metric, row[metric])
        path_with_emojis = current_path.replace("✔", "✅").replace("✘", "❌")
        print(
            f"- {prefix}{int(row['num_cases']):,} cases ({path_with_emojis}) | {color_title}: {metric_val}"
        )

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# utils still contains metric config and formatting helpers
from utils import format_metric_value, METRIC_CONFIG, print_summary

# chart_helpers contains visualization-specific helpers
from chart_helpers import build_case_paths, get_normalized_colors, format_slice_labels

from lineage_core import get_lineage, get_sibling_subsets
from lineage_filters import recursively_apply_filters



def query_exploration_icicle(
    result_set_name, log_view, metric="avg_case_duration_seconds", details=True
):
    """Build an icicle chart for query exploration lineage."""
    # Get the lineage of transformations leading to this result set
    lineage = get_lineage(log_view.query_registry.summary(), result_set_name)

    # Apply the sequence of filters and compute stats for each branch
    icicle_df, main_path = recursively_apply_filters(lineage, log_view, metric)

    # All hierarchy levels are stored in Level1, Level2, ...
    path_cols = [c for c in icicle_df.columns if c.startswith("Level")]

    # Mark leaves (no children further down the tree)
    icicle_df["is_leaf"] = ~icicle_df.duplicated(subset=path_cols, keep=False)

    # Build the actual icicle chart
    fig = px.icicle(
        icicle_df,
        path=path_cols,
        values="num_cases",
        color=metric,
        color_continuous_scale=METRIC_CONFIG[metric]["color_scheme"],
        title=f"Icicle Chart for: {result_set_name}",
    )
    fig.update_layout(
        margin=dict(t=40, l=0, r=0, b=0),
        coloraxis_colorbar=dict(title=METRIC_CONFIG[metric]["label"]),
    )
    fig.show()

    # Print textual summary if requested
    if details:
        print("\nSummary with Metrics:\n")
        final_path = " → ".join(main_path) if main_path else ""
        print_summary(icicle_df, path_cols, final_path, metric, METRIC_CONFIG[metric]["label"])





def query_breakdown_pie(result_set_name, log_view, metric="avg_case_duration_seconds", details=True):
    """Build a pie chart showing breakdown of cases along filter paths."""
    # Locate the query that produced this result and its siblings
    parent_log, query_obj, label, step_index, lineage_df = get_sibling_subsets(
        result_set_name, log_view
    )

    # Configure which metric to compute and how it should be displayed
    config = METRIC_CONFIG[metric]
    full_log = log_view.query_registry.get_initial_source_log()
    full_log = config["enrich_fn"](full_log)
    value_col, color_scheme, color_title = (
        config["column"],
        config["color_scheme"],
        config["label"],
    )

    # Apply the last query in the lineage
    filtered, _ = log_view.query_evaluator.evaluate(full_log, query_obj)
    if filtered.empty:
        print("No cases passed the final filter, pie chart cannot be built.")
        return

    # Reconstruct the sequence of filter passes/fails for each case
    case_paths, final_result_path = build_case_paths(lineage_df, full_log, filtered, log_view)

    # Attach the path string to each case
    filtered = filtered.copy()
    filtered["path_label"] = filtered["case:concept:name"].map(
        lambda cid: " → ".join(case_paths.get(cid, []))
    )

    # Aggregate stats: number of cases + average metric per path
    grouped = (
        filtered.groupby("path_label")[[ "case:concept:name", value_col ]]
        .agg(num_cases=("case:concept:name", "nunique"), avg_metric=(value_col, "mean"))
        .reset_index()
    )
    grouped = grouped.copy()
    grouped["wrapped_path"] = grouped["path_label"].str.replace(" → ", " →<br>")
    grouped = grouped.rename(columns={"avg_metric": metric})

    # Compute colors and format slice labels
    color_values = get_normalized_colors(grouped[metric], color_scheme)
    grouped = format_slice_labels(grouped, final_result_path)

    # Build pie chart
    fig = go.Figure(
        data=[
            go.Pie(
                labels=grouped["slice_label"],
                values=grouped["num_cases"],
                textinfo="label",
                customdata=grouped[["wrapped_path"]],  # show full path on hover
                hovertemplate="<b>%{customdata[0]}</b><extra></extra>",
                marker=dict(colors=color_values),
            )
        ],
        layout=go.Layout(
            title=dict(text=f"Breakdown of Filter: {query_obj.as_string()}", x=0.5),
            width=800,
            height=700,
            showlegend=False,
            paper_bgcolor="white",
            plot_bgcolor="white",
        ),
    )

    # Add colorbar for the metric scale
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(
                colorscale=color_scheme,
                cmin=grouped[metric].min(),
                cmax=grouped[metric].max(),
                colorbar=dict(title=color_title, len=0.8, thickness=15),
                color=[grouped[metric].min()],
                showscale=True,
            ),
            hoverinfo="none",
            showlegend=False,
        )
    )

    fig.show()

    # Print textual summary if requested
    if details:
        print("\nFilter Paths:\n")
        print_summary(grouped, ["path_label"], final_result_path, metric, color_title)

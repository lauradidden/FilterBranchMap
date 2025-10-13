import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display, clear_output
from utils import format_metric_value, METRIC_CONFIG, print_summary
from chart_helpers import build_case_paths, get_normalized_colors, format_slice_labels
from lineage_core import get_lineage, get_sibling_subsets
from lineage_filters import apply_filters

def query_exploration_icicle(
    result_set_name, log_view, metric="avg_case_duration_seconds", details=True
):
    """Build an icicle chart for query exploration."""

    # Get the lineage of transformations leading to this result set
    lineage = get_lineage(log_view.query_registry.summary(), result_set_name)

    # Apply the sequence of filters and compute stats for each branch
    icicle_df, main_path = apply_filters(lineage, log_view, metric)

    # All hierarchy levels are stored in Level1, Level2, ...
    path_cols = [c for c in icicle_df.columns if c.startswith("Level")]

    # Hover labels
    icicle_df["hover_text"] = icicle_df.apply(
        lambda row: (
            f"<b>{int(row['num_cases']):,} cases</b><br>"
            f"{METRIC_CONFIG[metric]['label']}: {format_metric_value(metric, row[metric])}"
        ),
        axis=1,
    )

    # Build the actual icicle chart
    fig = px.icicle(
        icicle_df,
        path=path_cols,
        values="num_cases",
        color=metric,
        custom_data=["hover_text"],   # attach hover text
        color_continuous_scale=METRIC_CONFIG[metric]["color_scheme"],
        title=f"Icicle Chart for: {result_set_name}",
    )

    # Apply hovertemplate
    fig.update_traces(
        hovertemplate="%{customdata[0]}<extra></extra>"
    )

    fig.update_layout(
        margin=dict(t=40, l=0, r=0, b=0),
        coloraxis_colorbar=dict(title=METRIC_CONFIG[metric]["label"]),
    )
    fig.show()

    # Print textual summary
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

    # Aggregate number of cases + average metric per path
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

    # Print textual summary
    if details:
        print("\nFilter Paths:\n")
        print_summary(grouped, ["path_label"], final_result_path, metric, color_title)



def chart_selecting(result_set_name, log_view):
    """
    Interactive selector for Icicle and Pie charts across all metrics.
    User selects via checkboxes and clicks 'Go' to render.
    """

    # --- Build checkbox list ---
    checkboxes = []
    for metric, config in METRIC_CONFIG.items():
        checkboxes.append((
            widgets.Checkbox(value=False, description=f"Icicle – {config['label']}"),
            ("icicle", metric)
        ))
        checkboxes.append((
            widgets.Checkbox(value=False, description=f"Pie – {config['label']}"),
            ("pie", metric)
        ))

    checkbox_widgets = [cb for cb, _ in checkboxes]

    # Button to trigger rendering
    go_button = widgets.Button(
        description="Go",
        button_style="success",
        tooltip="Generate selected visualizations",
        icon="play"
    )

    # Output area
    out = widgets.Output()

    def on_click(b):
        with out:
            clear_output(wait=True)

            # Collect selected items
            selected = [info for cb, info in checkboxes if cb.value]
            if not selected:
                print("⚠️ Please select at least one visualization.")
                return

            # Disable inputs while running
            for cb in checkbox_widgets:
                cb.disabled = True
            go_button.disabled = True

            # List all charts being generated
            labels = [f"{kind.title()} – {METRIC_CONFIG[metric]['label']}" for kind, metric in selected]
            print(f"⏳ Loading charts: {', '.join(labels)}...\n")

            try:
                # Render all selected charts
                for kind, metric in selected:
                    if kind == "icicle":
                        query_exploration_icicle(result_set_name, log_view, metric=metric, details=False)
                    elif kind == "pie":
                        query_breakdown_pie(result_set_name, log_view, metric=metric, details=False)
            finally:
                # Re-enable inputs after execution
                for cb in checkbox_widgets:
                    cb.disabled = False
                go_button.disabled = False

    go_button.on_click(on_click)

    # Layout
    ui = widgets.VBox([
        widgets.HTML("<h3>Select Visualizations (you can tick multiple)</h3>"),
        widgets.VBox(checkbox_widgets),
        go_button,
        out
    ])

    display(ui)


def interactive_icicle(result_set_name, log_view):
    """Interactive icicle chart with dropdown to switch between metrics."""
    lineage = get_lineage(log_view.query_registry.summary(), result_set_name)

    path_cols = None
    traces = []
    buttons = []

    for metric in METRIC_CONFIG.keys():
        # Compute per-metric icicle data
        icicle_df, _ = apply_filters(lineage, log_view, metric)
        path_cols = [c for c in icicle_df.columns if c.startswith("Level")]

        # Build hover text (cases + formatted metric)
        icicle_df["hover_text"] = icicle_df.apply(
            lambda row: (
                f"<b>{int(row['num_cases']):,} cases</b><br>"
                f"{METRIC_CONFIG[metric]['label']}: {format_metric_value(metric, row[metric])}"
            ),
            axis=1,
        )

        # Create icicle trace for this metric
        fig_metric = px.icicle(
            icicle_df,
            path=path_cols,
            values="num_cases",
            color=metric,
            custom_data=["hover_text"],
            color_continuous_scale=METRIC_CONFIG[metric]["color_scheme"],
        )

        trace = fig_metric.data[0]
        trace.visible = (metric == "avg_case_duration_seconds")  # default visible
        traces.append(trace)

        # Add dropdown button
        buttons.append(
            dict(
                label=METRIC_CONFIG[metric]["label"],
                method="update",
                args=[
                    {"visible": [t == trace for t in traces]},
                    {"coloraxis": {"colorbar": {"title": METRIC_CONFIG[metric]["label"]}}},
                ],
            )
        )

    # Build final figure with all traces
    fig = go.Figure(data=traces)
    fig.update_traces(hovertemplate="%{customdata[0]}<extra></extra>")

    fig.update_layout(
        title=f"Icicle Chart for: {result_set_name}",
        margin=dict(t=40, l=0, r=0, b=0),
        updatemenus=[dict(
            buttons=buttons,
            direction="down",
            showactive=True,
            x=1.05,
            y=1.0,
        )],
    )

    fig.show()


def interactive_pie(result_set_name, log_view):
    """Interactive pie chart with dropdown to switch between metrics."""
    traces = []
    buttons = []

    for metric, config in METRIC_CONFIG.items():
        # Metric setup
        full_log = log_view.query_registry.get_initial_source_log()
        full_log = config["enrich_fn"](full_log)
        value_col, color_scheme, color_title = (
            config["column"],
            config["color_scheme"],
            config["label"],
        )

        # Get siblings + apply query
        parent_log, query_obj, label, step_index, lineage_df = get_sibling_subsets(
            result_set_name, log_view
        )
        filtered, _ = log_view.query_evaluator.evaluate(full_log, query_obj)
        if filtered.empty:
            continue

        # Case paths
        case_paths, final_result_path = build_case_paths(
            lineage_df, full_log, filtered, log_view
        )

        # Path labels
        filtered = filtered.copy()
        filtered["path_label"] = filtered["case:concept:name"].map(
            lambda cid: " → ".join(case_paths.get(cid, []))
        )

        # Aggregate stats
        grouped = (
            filtered.groupby("path_label")[[ "case:concept:name", value_col ]]
            .agg(num_cases=("case:concept:name", "nunique"), avg_metric=(value_col, "mean"))
            .reset_index()
        )
        grouped = grouped.copy()
        grouped["wrapped_path"] = grouped["path_label"].str.replace(" → ", " →<br>")
        grouped = grouped.rename(columns={"avg_metric": metric})

        # Colors + labels
        color_values = get_normalized_colors(grouped[metric], color_scheme)
        grouped = format_slice_labels(grouped, final_result_path)

        # Create trace for this metric
        pie_trace = go.Pie(
            labels=grouped["slice_label"],
            values=grouped["num_cases"],
            textinfo="label",
            customdata=grouped[["wrapped_path"]],
            hovertemplate="<b>%{customdata[0]}</b><extra></extra>",
            marker=dict(colors=color_values),
            visible=(metric == "avg_case_duration_seconds"),  # default visible
        )

        traces.append(pie_trace)

        # Dropdown button for this metric
        buttons.append(
            dict(
                label=config["label"],
                method="update",
                args=[
                    {"visible": [t == pie_trace for t in traces]},
                    {"title": f"Pie Chart for: {query_obj.as_string()}<br>({config['label']})"},
                ],
            )
        )

    # Build final figure
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"Pie Chart for: {result_set_name}",
        width=800,
        height=700,
        showlegend=False,
        paper_bgcolor="white",
        plot_bgcolor="white",
        updatemenus=[dict(
            buttons=buttons,
            direction="down",
            showactive=True,
            x=1.05,
            y=1.0,
        )],
    )

    fig.show()

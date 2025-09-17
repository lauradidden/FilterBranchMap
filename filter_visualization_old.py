import pandas as pd
import pm4py
from logview.utils import LogViewBuilder
from logview.predicate import Query, EqToConstant, NotEqToConstant, GreaterEqualToConstant, LessThanConstant, StartWith, EndWith, DurationWithin
import plotly.express as px
import time
from datetime import timedelta
import string
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
import numpy as np

def get_lineage(registry, result_set_name):
    """
    Given a registry and a result_set name, return a filtered DataFrame
    showing the lineage of how that result_set was derived.
    """
    evaluations = registry['evaluations']
    
    lineage_rows = []

    def trace_back(current_result_set):
        for _, row in evaluations.iterrows():
            if row['result_set'] == current_result_set:
                lineage_rows.append(row)
                trace_back(row['source_log'])

    trace_back(result_set_name)
    
    # Reverse the result to show forward lineage
    lineage_df = pd.DataFrame(lineage_rows[::-1])
    return lineage_df

def add_case_durations(df):
    grouped = df.groupby("case:concept:name")["time:timestamp"]
    durations = (grouped.max() - grouped.min()).dt.total_seconds()
    df = df.copy()
    df["case_duration"] = df["case:concept:name"].map(durations)
    return df

def add_event_counts(df):
    event_counts = df.groupby("case:concept:name").size()
    df = df.copy()
    df["num_events"] = df["case:concept:name"].map(event_counts)
    return df

def add_avg_time_between_events(df):
    def avg_diff(x):
        diffs = x.sort_values().diff().dropna()
        return diffs.mean().total_seconds() if len(diffs) > 0 else 0
    avg_diffs = df.groupby("case:concept:name")["time:timestamp"].apply(avg_diff)
    df = df.copy()
    df["avg_time_between_events"] = df["case:concept:name"].map(avg_diffs)
    return df

def precompute_case_durations(log_df):
    """
    Adds a 'case_duration' column to the original log dataframe.
    """
    case_durations = (
        log_df.groupby("case:concept:name")["time:timestamp"]
        .agg(["min", "max"])
        .apply(lambda row: (row["max"] - row["min"]).total_seconds(), axis=1)
    )
    log_df = log_df.copy()
    log_df["case_duration"] = log_df["case:concept:name"].map(case_durations)
    return log_df

def format_seconds(seconds):
    if pd.isna(seconds):
        return "N/A"
    seconds = int(seconds)
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes = rem // 60
    parts = []
    if days: parts.append(f"{days}d")
    if hours: parts.append(f"{hours}h")
    if minutes: parts.append(f"{minutes}m")
    return " ".join(parts) if parts else "0m"

def compute_hover_data(df, metric):
    import numpy as np

    path_cols = [col for col in df.columns if col.startswith("Level")]
    
    # Compute ID and parent ID using cleaned labels
    def clean(label):
        return label.replace("🟡 ", "").strip()

    cleaned_path = df[path_cols].map(clean)
    df["id"] = cleaned_path.agg(" → ".join, axis=1)
    df["parent_id"] = cleaned_path.shift(axis=1).fillna("").agg(" → ".join, axis=1)
    
    df.loc[df["parent_id"].str.strip() == "", "parent_id"] = None

    # Create lookup table for parent metrics
    parent_lookup = df.set_index("id")[[metric, "num_cases"]].to_dict("index")

    hover_cases = []
    hover_pct = []
    hover_metric = []
    hover_delta = []

    for _, row in df.iterrows():
        cur_val = row[metric]
        cur_cases = row["num_cases"]
        parent_info = parent_lookup.get(row["parent_id"])

        # Cases
        hover_cases.append(f"{cur_cases:,}")

        # Percentage of parent
        if parent_info and parent_info["num_cases"] > 0:
            pct = (cur_cases / parent_info["num_cases"]) * 100
            hover_pct.append(f"{pct:.1f}%")
        else:
            hover_pct.append("100%")

        # Metric
        if "duration" in metric or "time" in metric:
            # Format time in human readable form
            sec = cur_val
            d = int(sec // 86400)
            h = int((sec % 86400) // 3600)
            m = int((sec % 3600) // 60)
            hover_metric.append(f"{d}d {h}h {m}m")
        else:
            hover_metric.append(f"{cur_val:.2f}")

        # Delta
        if parent_info:
            diff = cur_val - parent_info[metric]
            if "duration" in metric or "time" in metric:
                sign = "-" if diff < 0 else "+"
                diff = abs(diff)
                d = int(diff // 86400)
                h = int((diff % 86400) // 3600)
                m = int((diff % 3600) // 60)
                hover_delta.append(f"{sign}{d}d {h}h {m}m")
            else:
                sign = "-" if diff < 0 else "+"
                hover_delta.append(f"{sign}{abs(diff):.2f}")
        else:
            hover_delta.append("—")

    df["hover_cases"] = hover_cases
    df["hover_pct"] = hover_pct
    df["hover_metric"] = hover_metric
    df["hover_delta"] = hover_delta
    return df

def compute_case_stats(df, name, label_path, metric):
    if df.empty:
        return {
            "subset_name": name,
            "label_path": " → ".join(label_path),
            "num_cases": 0,
            metric: 0
        }

    df = df.drop_duplicates("case:concept:name")

    column = {
        "avg_case_duration_seconds": "case_duration",
        "avg_events_per_case": "num_events",
        "avg_time_between_events": "avg_time_between_events"
    }[metric]

    return {
        "subset_name": name,
        "label_path": " → ".join(label_path),
        "num_cases": df["case:concept:name"].nunique(),
        metric: df[column].mean()
    }

def split_subsets(subsets, query_obj, filter_label, step_index, query_evaluator, filter_cache):
    new_subsets = []

    for subset in subsets:
        subset_df = subset["df"]
        subset_name = subset["name"]
        path = subset["label_path"]
        order_path = subset.get("order_path", []) 

        cache_key = (subset_name, query_obj.name)
        if cache_key in filter_cache:
            df_filtered, df_complement = filter_cache[cache_key]
        else:
            df_filtered, df_complement = query_evaluator.evaluate(subset_df, query_obj)
            filter_cache[cache_key] = (df_filtered, df_complement)

        new_subsets.append({
            "df": df_filtered,
            "name": f"{subset_name}_F{step_index+1}",
            "label_path": path + [f"{filter_label} ✓"],
            "order_path": order_path + [0]
        })
        new_subsets.append({
            "df": df_complement,
            "name": f"{subset_name}_C{step_index+1}",
            "label_path": path + [f"{filter_label} ✗"],
            "order_path": order_path + [1]
        })

    return new_subsets

def recursively_apply_filters(selected_sequence_df, log_view, metric):
    initial_log_name = selected_sequence_df.iloc[0]['source_log']
    base_df = log_view.result_set_name_cache[initial_log_name]

    if metric == "avg_case_duration_seconds":
        initial_df = add_case_durations(base_df)
    elif metric == "avg_events_per_case":
        initial_df = add_event_counts(base_df)
    elif metric == "avg_time_between_events":
        initial_df = add_avg_time_between_events(base_df)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    main_path_leaf = None

    current_subsets = [{
        "df": initial_df,
        "name": initial_log_name,
        "label_path": ["Initial Source"],
        "order_path": [],
        "is_main_path": True
    }]

    filter_cache = {}

    query_map = {
        evaluation["query"].name: evaluation["query"]
        for result_set_id in log_view.query_registry.get_registered_result_set_ids()
        for evaluation in [log_view.query_registry.get_evaluation(result_set_id)]
    }

    query_expression_map = {
        evaluation["query"].name: evaluation["query"].as_string()
        for result_set_id in log_view.query_registry.get_registered_result_set_ids()
        for evaluation in [log_view.query_registry.get_evaluation(result_set_id)]
    }

    for i, row in selected_sequence_df.iterrows():
        query_obj = query_map.get(row["query"])
        query_expr = query_expression_map.get(row["query"], row["labels"])
        next_subsets = []

        for subset in current_subsets:
            df = subset["df"]
            if df.empty:
                continue

            path = subset["label_path"]
            order_path = subset["order_path"]
            is_main = subset["is_main_path"]

            cache_key = (subset["name"], query_obj.name)
            if cache_key in filter_cache:
                df_filtered, df_complement = filter_cache[cache_key]
            else:
                df_filtered, df_complement = log_view.query_evaluator.evaluate(df, query_obj)
                filter_cache[cache_key] = (df_filtered, df_complement)

            next_subsets.append({
                "df": df_filtered,
                "name": f"{subset['name']}_F{i+1}",
                "label_path": path + [f"{query_expr} ✔"],
                "order_path": order_path + [0],
                "is_main_path": is_main
            })
            next_subsets.append({
                "df": df_complement,
                "name": f"{subset['name']}_C{i+1}",
                "label_path": path + [f"{query_expr} ✘"],
                "order_path": order_path + [1],
                "is_main_path": False
            })

        current_subsets = next_subsets

    for subset in current_subsets:
        if subset["is_main_path"] and not subset["df"].empty:
            main_path_leaf = subset
            break

    main_path_label_set = set()
    if main_path_leaf:
        for i in range(len(main_path_leaf["label_path"])):
            main_path_label_set.add(tuple(main_path_leaf["label_path"][:i + 1]))

    result_rows = []
    for subset in current_subsets:
        df = subset["df"]
        if df.empty:
            continue

        label_path = subset["label_path"]
        display_path = []

        for i in range(len(label_path)):
            path_prefix = tuple(label_path[:i + 1])
            if path_prefix in main_path_label_set:
                display_path.append("🟡 " + label_path[i])
            else:
                display_path.append(label_path[i])

        stats = compute_case_stats(df, subset["name"], display_path, metric)
        row = {
            **{f"Level{i+1}": label for i, label in enumerate(display_path)},
            "num_cases": stats["num_cases"],
            metric: stats[metric],
            "order_path": subset["order_path"]
        }
        result_rows.append(row)

    result_df = pd.DataFrame(result_rows)
    result_df = result_df.sort_values(by="order_path")

    return result_df, main_path_leaf["label_path"] if main_path_leaf else []

def query_exploration_icicle(result_set_name, log_view, metric="avg_case_duration_seconds", show_time=False, details=True):

    def format_seconds(seconds):
        if pd.isna(seconds):
            return "N/A"
        seconds = int(seconds)
        days, rem = divmod(seconds, 86400)
        hours, rem = divmod(rem, 3600)
        minutes = rem // 60
        parts = []
        if days: parts.append(f"{days}d")
        if hours: parts.append(f"{hours}h")
        if minutes: parts.append(f"{minutes}m")
        return " ".join(parts) if parts else "0m"

    times = {}
    start_all = time.time()

    # Get lineage
    t0 = time.time()
    lineage = get_lineage(log_view.query_registry.summary(), result_set_name)
    times['lineage'] = time.time() - t0

    # Compute filtered subsets and stats
    t1 = time.time()
    icicle_df, main_path = recursively_apply_filters(lineage, log_view, metric=metric)

    # Filter to only final-level rows (leaves)
    path_cols = [col for col in icicle_df.columns if col.startswith("Level")]
    icicle_df["is_leaf"] = ~icicle_df.duplicated(subset=path_cols, keep=False)

    # Add hover label data only for leaves
    hover_cases = []
    hover_metric = []
    for _, row in icicle_df.iterrows():
        if row["is_leaf"]:
            hover_cases.append(f"{row['num_cases']:,}")
            if "duration" in metric or "time" in metric:
                sec = row[metric]
                d = int(sec // 86400)
                h = int((sec % 86400) // 3600)
                m = int((sec % 3600) // 60)
                hover_metric.append(f"{d}d {h}h {m}m")
            else:
                hover_metric.append(f"{row[metric]:.2f}")
        else:
            hover_cases.append("")
            hover_metric.append("")

    icicle_df["hover_cases"] = hover_cases
    icicle_df["hover_metric"] = hover_metric

    times['apply_filters'] = time.time() - t1

    # Define color settings
    color_schemes = {
        "avg_case_duration_seconds": "Blues",
        "avg_events_per_case": "Reds",
        "avg_time_between_events": "Greens"
    }
    color_labels = {
        "avg_case_duration_seconds": "Avg Case Duration (s)",
        "avg_events_per_case": "Avg Events per Case",
        "avg_time_between_events": "Avg Time Between Events (s)"
    }

    # Plot icicle
    fig = px.icicle(
        icicle_df,
        path=[col for col in icicle_df.columns if col.startswith("Level")],
        values="num_cases",
        color=metric,
        custom_data=["hover_cases", "hover_metric"],
        color_continuous_scale=color_schemes.get(metric, "Blues"),
        title=f"Icicle Chart for: {result_set_name}"
    )

    # Disable all hover labels
    fig.update_traces(hoverinfo="skip", hovertemplate=None)

    # Update layout
    fig.update_layout(
        margin=dict(t=40, l=0, r=0, b=0),
        coloraxis_colorbar=dict(title=color_labels.get(metric, metric))
    )

    fig.show()
    times['plot'] = time.time() - t1
    times['total'] = time.time() - start_all

    if details:
        print("\nSummary with Metrics:\n")

        # Build final result path (without yellow dots)
        final_result_path = " → ".join(main_path) if main_path else ""

        for _, row in icicle_df.iterrows():
            # Extract path parts
            path_parts = [
                str(row[col]).replace("🟡 ", "") for col in path_cols if pd.notna(row[col])
            ]
            current_path = " → ".join(path_parts)

            # Check if this row is the final result
            is_final_result = current_path == final_result_path

            path_with_emojis = current_path.replace("✔", "✅").replace("✘", "❌").replace("✓", "✅").replace("✗", "❌")

            # Format the metric
            if "duration" in metric or "time" in metric:
                metric_val = format_seconds(row[metric])
            else:
                metric_val = f"{row[metric]:.2f}"

            # Yellow dot only for the final row
            prefix = "🟡 " if is_final_result else ""

            print(f"- {prefix}{int(row['num_cases']):,} cases ({path_with_emojis}) | {color_labels.get(metric, metric)}: {metric_val}")


    if show_time:
        print("\n--- Execution Time (seconds) ---")
        for k, v in times.items():
            print(f"{k:25}: {v:.4f}")

def get_sibling_subsets(result_set_name, log_view):
    lineage_df = get_lineage(log_view.query_registry.summary(), result_set_name)
    if len(lineage_df) < 1:
        raise ValueError("Lineage not found.")

    last_query_row = lineage_df.iloc[-1]
    parent_log = last_query_row["source_log"]
    query_name = last_query_row["query"]
    label = last_query_row["labels"]
    step_index = len(lineage_df) - 1

    # Lookup actual query object from registry
    query_obj = None
    for rs_id in log_view.query_registry.get_registered_result_set_ids():
        evaluation = log_view.query_registry.get_evaluation(rs_id)
        if evaluation["query"].name == query_name:
            query_obj = evaluation["query"]
            break

    if query_obj is None:
        raise ValueError(f"Query object for '{query_name}' not found.")

    return parent_log, query_obj, label, step_index, lineage_df

def query_breakdown_pie(result_set_name, log_view, metric="avg_case_duration_seconds", details=True):
    from plotly.colors import sample_colorscale
    import plotly.graph_objects as go

    def format_seconds(seconds):
        seconds = int(seconds)
        days, rem = divmod(seconds, 86400)
        hours, rem = divmod(rem, 3600)
        minutes = rem // 60
        parts = []
        if days: parts.append(f"{days}d")
        if hours: parts.append(f"{hours}h")
        if minutes: parts.append(f"{minutes}m")
        return " ".join(parts) if parts else "0m"

    # Get lineage and query
    parent_log, query_obj, label, step_index, lineage_df = get_sibling_subsets(result_set_name, log_view)

    # Metric setup
    full_log = log_view.query_registry.get_initial_source_log()
    if metric == "avg_case_duration_seconds":
        full_log = add_case_durations(full_log)
        value_col = "case_duration"
        color_scheme = "Blues"
        color_title = "Avg Duration (s)"
    elif metric == "avg_events_per_case":
        full_log = add_event_counts(full_log)
        value_col = "num_events"
        color_scheme = "Reds"
        color_title = "Avg Events/Case"
    elif metric == "avg_time_between_events":
        full_log = add_avg_time_between_events(full_log)
        value_col = "avg_time_between_events"
        color_scheme = "Greens"
        color_title = "Avg Time Between Events (s)"
    else:
        raise ValueError("Unsupported metric")

    # Evaluate last query
    filtered, _ = log_view.query_evaluator.evaluate(full_log, query_obj)

    if filtered.empty:
        print("No cases passed the final filter, pie chart cannot be build.")
        return

    # Get query expressions
    query_expr_map = {
        ev["query"].name: ev["query"].as_string()
        for rs_id in log_view.query_registry.get_registered_result_set_ids()
        for ev in [log_view.query_registry.get_evaluation(rs_id)]
    }

    # Assign lineage path to cases
    case_paths = {}
    for _, row in lineage_df.iterrows():
        qname = row["query"]
        qexpr = query_expr_map.get(qname, qname)
        step_obj = next(
            (log_view.query_registry.get_evaluation(rs)["query"]
             for rs in log_view.query_registry.get_registered_result_set_ids()
             if log_view.query_registry.get_evaluation(rs)["query"].name == qname),
            None
        )
        if step_obj:
            passed_df, _ = log_view.query_evaluator.evaluate(full_log, step_obj)
            passed_cases = set(passed_df["case:concept:name"])
            for cid in filtered["case:concept:name"].unique():
                case_paths.setdefault(cid, [])
                case_paths[cid].append(f"{qexpr} ✅" if cid in passed_cases else f"{qexpr} ❌")

    filtered = filtered.copy()
    filtered["path_label"] = filtered["case:concept:name"].map(
        lambda cid: " → ".join(case_paths.get(cid, []))
    )

    # Identify final result path
    final_result_path = " → ".join([
        f"{query_expr_map.get(row['query'], row['query'])} ✅"
        for _, row in lineage_df.iterrows()
    ])


    # Aggregate
    grouped = (
        filtered.groupby("path_label")[["case:concept:name", value_col]]
        .agg(num_cases=("case:concept:name", "nunique"), avg_metric=(value_col, "mean"))
        .reset_index()
    )

    # Add line breaks for hover display
    grouped["wrapped_path"] = grouped["path_label"].str.replace(" → ", " →<br>")

    # Normalize color
    min_val, max_val = grouped["avg_metric"].min(), grouped["avg_metric"].max()
    normed = (grouped["avg_metric"] - min_val) / (max_val - min_val + 1e-9)
    color_values = sample_colorscale(color_scheme, normed.tolist())

    # Slice labels
    total_cases = grouped["num_cases"].sum()

    def label_with_dot(row):
        label = f"{int(row['num_cases']):,} cases ({row['num_cases'] / total_cases:.0%})"
        return f"🟡 {label}" if row["path_label"] == final_result_path else label

    grouped["slice_label"] = grouped.apply(label_with_dot, axis=1)

    # Build chart
    fig = go.Figure(
        data=[go.Pie(
            labels=grouped["slice_label"],
            values=grouped["num_cases"],
            textinfo="label",
            customdata=grouped[["wrapped_path"]],
            hovertemplate="<b>%{customdata[0]}</b><extra></extra>",
            marker=dict(colors=color_values)
        )],
        layout=go.Layout(
            title=dict(
                text=f"Breakdown of Filter: {query_obj.as_string()}",
                x=0.5
            ),
            width=800,
            height=700,
            showlegend=False,
            paper_bgcolor='white',
            plot_bgcolor='white',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )

    # Add colorbar
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(
            colorscale=color_scheme,
            cmin=min_val,
            cmax=max_val,
            colorbar=dict(title=color_title, len=0.8, thickness=15),
            color=[min_val],
            showscale=True
        ),
        hoverinfo='none',
        showlegend=False
    ))

    fig.show()

    # Print textual summary
    if details:
        print("\nFilter Paths:\n")
        for _, row in grouped.iterrows():
            if "duration" in metric or "time" in metric:
                time_str = format_seconds(row["avg_metric"])
            else:
                time_str = f"{row['avg_metric']:.2f}"
            prefix = "🟡 " if row["path_label"] == final_result_path else ""
            print(f"- {prefix}{int(row['num_cases']):,} cases ({row['num_cases'] / total_cases:.0%}): {row['path_label']} | {color_title}: {time_str}")

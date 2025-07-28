"""
smart_process_app.py
---------------------

This module implements a simple web application that wraps the process mining
pipeline defined in the `smart_process_miner` function.  The goal of the
application is to provide an easy‑to‑use interface where a user can upload
either an Excel or CSV file, inspect the cleaned event log, and, if they
choose, proceed to discover a process model using algorithms from the
``pm4py`` library.  The chosen discovery algorithm depends on the number of
traces in the uploaded log:

* Alpha Miner for small datasets (≤ 100 traces)
* Inductive Miner for medium datasets (101–500 traces)
* Heuristics Miner for large datasets (> 500 traces)

The resulting process model is rendered as a Graphviz diagram and displayed
within the page.  Users unfamiliar with ``pm4py`` can run this application
locally after installing the necessary dependencies with ``pip install
pm4py streamlit``.

Note: this script is designed to be run with Streamlit.  To launch the
application, execute ``streamlit run smart_process_app.py`` from the
terminal.
"""

from __future__ import annotations

import io
from typing import Tuple, Optional

import pandas as pd
import numpy as np

try:
    import streamlit as st
except ImportError as e:
    raise ImportError(
        "streamlit is required to run this application. "
        "Install it via 'pip install streamlit' and try again."
    ) from e

try:
    import pm4py
    from pm4py.objects.log.util import dataframe_utils
    from pm4py.objects.conversion.log import converter as log_converter
    from pm4py.algo.discovery.alpha import algorithm as alpha_miner
    from pm4py.algo.discovery.inductive import algorithm as inductive_miner
    from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
    from pm4py.visualization.petri_net import visualizer as pn_visualizer
    from pm4py.visualization.heuristics_net import visualizer as hn_visualizer
    from pm4py.objects.conversion.process_tree import converter as pt_converter
except ImportError:
    pm4py = None



def clean_event_log(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform a series of cleaning steps on the raw event log DataFrame.

    This function standardises column names, removes entirely empty rows and
    columns, handles missing and invalid values, forwards/fills missing
    observations, and removes duplicate rows.  It also normalises object
    columns to lower‑case trimmed strings.  Additionally it constructs two
    columns required by pm4py: ``session_id`` (which acts as case
    identifier) and ``semantic_label`` (used as the activity name), and
    sorts the log by session and timestamp, removing immediate self loops.

    Args:
        df: Raw event log loaded from a user provided file.

    Returns:
        A cleaned DataFrame ready for conversion into an EventLog.
    """
    # Drop entirely empty rows and columns
    df = df.copy()
    df.dropna(how='all', inplace=True)
    df.dropna(axis=1, how='all', inplace=True)

    # Standardise column names: strip whitespace, lower case, replace spaces
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    # Replace common invalid tokens with proper NaNs
    invalid_values = ["", "NaN", "nan", "null", "unknown", np.nan]
    df.replace(invalid_values, np.nan, inplace=True)

    # Remove exact duplicate rows
    df.drop_duplicates(inplace=True)

    # Forward and backward fill missing values to preserve trace continuity
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # Clean up string columns: trim and lowercase
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip().str.lower()

    # Create a session identifier if it doesn't exist.  Use the combination
    # of any existing 'type' and 'content' columns to group sessions.  If
    # these columns are absent, group by the entire row to create unique
    # sessions.
    if 'session_id' not in df.columns:
        if {'type', 'content'}.issubset(df.columns):
            df['session_id'] = df.groupby(['type', 'content']).ngroup().astype(str)
        else:
            df['session_id'] = df.groupby(df.columns.tolist()).ngroup().astype(str)

    # Create a semantic label combining existing "type" and "content"
    if 'semantic_label' not in df.columns:
        if 'type' in df.columns and 'content' in df.columns:
            df['semantic_label'] = df['type'].astype(str) + " → " + df['content'].astype(str)
        elif 'concept:name' in df.columns:
            # If concept:name already exists use it directly
            df['semantic_label'] = df['concept:name']
        else:
            # Fallback: use the first object column as the label
            first_obj_col = df.select_dtypes(include='object').columns[0]
            df['semantic_label'] = df[first_obj_col]

    # Ensure there is a timestamp column; if not, attempt to infer from existing
    # columns named similar to 'timestamp'
    if 'timestamp' not in df.columns:
        # try to detect a timestamp column heuristically
        candidate_cols = [c for c in df.columns if 'time' in c or 'date' in c]
        if candidate_cols:
            df.rename(columns={candidate_cols[0]: 'timestamp'}, inplace=True)
        else:
            raise ValueError(
                "No timestamp column found. The uploaded file must contain a "
                "timestamp column or a column with 'time'/'date' in its name."
            )

    # Sort by session and timestamp
    df = df.sort_values(by=['session_id', 'timestamp'])

    # Remove immediate self-loops: successive events with identical labels in
    # the same case.  We shift the semantic_label within each session and
    # compare to the current label.
    df['prev_action'] = df.groupby('session_id')['semantic_label'].shift()
    df = df[df['semantic_label'] != df['prev_action']].copy()
    df.drop(columns=['prev_action'], inplace=True)

    # Set pm4py required column names
    df['case:concept:name'] = df['session_id']
    df['concept:name'] = df['semantic_label']
    df['time:timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df.dropna(subset=['time:timestamp'], inplace=True)

    return df


def discover_process_model(df: pd.DataFrame) -> Tuple[str, Optional[str]]:
    """
    Discover a process model from a cleaned event log and return the name of
    the mining algorithm used along with a Graphviz DOT representation of
    the resulting net or heuristics net.

    This function converts the cleaned DataFrame into an EventLog using
    ``pm4py`` and then selects an appropriate discovery algorithm based on
    the number of traces (cases) present.  It handles small, medium and
    large logs with Alpha Miner, Inductive Miner and Heuristics Miner
    respectively.  The resulting model is converted into a Graphviz object
    which is serialised to a DOT string for consumption by Streamlit.

    Args:
        df: A cleaned DataFrame ready for conversion into an EventLog.

    Returns:
        A tuple of (algorithm_name, dot_source) where dot_source is None in
        case of failure.

    Raises:
        RuntimeError: If pm4py is not available in the environment.
    """
    if pm4py is None:
        raise RuntimeError(
            "pm4py is not installed. Process modelling cannot be performed."
        )

    # Convert DataFrame into an EventLog
    df_pm4py = dataframe_utils.convert_timestamp_columns_in_df(df)
    event_log = log_converter.apply(df_pm4py, variant=log_converter.Variants.TO_EVENT_LOG)

    trace_count = len(event_log)

    try:
        if trace_count <= 100:
            algorithm_name = "Alpha Miner"
            net, im, fm = alpha_miner.apply(event_log)
            gviz = pn_visualizer.apply(net, im, fm)
            dot = gviz.source
            return algorithm_name, dot

        elif 100 < trace_count <= 500:
            algorithm_name = "Inductive Miner"
            process_tree = inductive_miner.apply(event_log)
            net, im, fm = pt_converter.apply(process_tree)
            gviz = pn_visualizer.apply(net, im, fm)
            dot = gviz.source
            return algorithm_name, dot

        else:
            algorithm_name = "Heuristics Miner"
            heu_net = heuristics_miner.apply_heu(event_log)
            gviz = hn_visualizer.apply(heu_net)
            dot = gviz.source
            return algorithm_name, dot

    except Exception as exc:
        # Return a failure indicator.  In a production application you might
        # propagate the exception or log it to an error monitoring system.
        return f"Failed to discover model: {exc}", None


def main() -> None:
    """
    Entrypoint for the Streamlit application.  Defines the page layout
    consisting of a file uploader, cleaned data preview and a process
    modelling section which is revealed after the user clicks a button.
    """
    st.set_page_config(page_title="Smart Process Miner", layout="wide")
    st.title("📈 Smart Process Mining Web App")

    st.markdown(
        "Upload an Excel or CSV file containing your event log. "
        "After cleaning the data, you can inspect the first few rows and, if desired, "
        "discover a process model based on the log size."
    )

    uploaded_file = st.file_uploader("Choose an Excel or CSV file", type=["xlsx", "xls", "csv"])

    if uploaded_file is not None:
        # Read the uploaded file into a DataFrame
        try:
            if uploaded_file.name.lower().endswith((".xlsx", ".xls")):
                df_raw = pd.read_excel(uploaded_file)
            elif uploaded_file.name.lower().endswith(".csv"):
                df_raw = pd.read_csv(uploaded_file)
            else:
                st.error("Unsupported file type. Please upload an Excel (.xlsx) or CSV file.")
                return
        except Exception as exc:
            st.error(f"Failed to read the uploaded file: {exc}")
            return

        # Clean the DataFrame
        try:
            df_clean = clean_event_log(df_raw)
        except Exception as exc:
            st.error(f"Error during data cleaning: {exc}")
            return

        # Show a preview of the cleaned data
        st.subheader("Cleaned Event Log (Preview)")
        st.dataframe(df_clean.head(100))
        st.write(f"Total events after cleaning: {len(df_clean)}")
        st.write(f"Total cases (traces): {df_clean['session_id'].nunique()}")

        # Provide a button to proceed to process modelling
        if st.button("Proceed to Process Modelling"):
            if pm4py is None:
                st.error("PM4Py is not installed. Please install it using: pip install pm4py")
                return
            with st.spinner("Discovering process model…"):
                algo_name, dot = discover_process_model(df_clean)
            st.success(f"Model discovered using {algo_name}.")
            if dot:
                st.subheader("Process Model")
                st.graphviz_chart(dot, use_container_width=True)

                # Render the PNG image from dot source using Graphviz
                import graphviz
                png_data = graphviz.Source(dot).pipe(format="png")

                # Add download button for the PNG image
                st.download_button(
                    label="📥 Download Model as PNG",
                    data=png_data,
                    file_name="process_model.png",
                    mime="image/png"
                )
            else:
                st.error(algo_name)

if __name__ == "__main__":
    main()
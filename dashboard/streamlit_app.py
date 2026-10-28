from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
EXTRA_ARCHITECTURE = ROOT / "output" / "images" / "PRESTO-R_System_Architecture.png"

st.set_page_config(
    page_title="PRESTO-R Evaluation Dashboard",
    page_icon=":material/monitoring:",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_episode_metrics(path: str, modified: float) -> pd.DataFrame:
    del modified
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_csv(path: str, modified: float) -> pd.DataFrame:
    del modified
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def read_text(path: str, modified: float) -> str:
    del modified
    return Path(path).read_text(encoding="utf-8", errors="replace")


def csv_files() -> list[Path]:
    return sorted(RESULTS.rglob("*.csv")) if RESULTS.exists() else []


def graph_files() -> list[Path]:
    files = sorted(RESULTS.rglob("*.png")) if RESULTS.exists() else []
    if EXTRA_ARCHITECTURE.exists():
        files.append(EXTRA_ARCHITECTURE)
    return files


def relative_label(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return path.name


def success_criteria() -> pd.DataFrame:
    path = RESULTS / "success_criteria.md"
    if not path.exists():
        return pd.DataFrame()
    body = read_text(str(path), path.stat().st_mtime)
    rows = []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("|"):
            continue
        match = re.match(r"[-*]\s+\[([ xX])\]\s*(.*)", stripped)
        if match:
            rows.append({"Status": "PASS" if match.group(1).strip() else "FAIL", "Criterion": match.group(2)})
        elif re.match(r"[-*]\s+", stripped):
            status = "FAIL" if re.search(r"\b(fail|below|not met)\b", stripped, re.I) else "RESULT"
            rows.append({"Status": status, "Criterion": re.sub(r"^[-*]\s+", "", stripped)})
    return pd.DataFrame(rows)


st.title("PRESTO-R Evaluation Dashboard")
st.caption("Interactive view of saved simulation graphs, evaluation tables, and episode-level results.")

episode_path = RESULTS / "episode_metrics.parquet"
if not episode_path.exists():
    st.error(f"Episode metrics file not found: {relative_label(episode_path)}")
    st.stop()

episodes = load_episode_metrics(str(episode_path), episode_path.stat().st_mtime)

with st.sidebar:
    st.header("Filters")
    topologies = sorted(episodes["topology"].dropna().astype(str).unique())
    scenarios = sorted(episodes["scenario"].dropna().astype(str).unique())
    methods = sorted(episodes["method"].dropna().astype(str).unique())
    topology_filter = st.multiselect("Topology", topologies, default=topologies)
    scenario_filter = st.multiselect("Scenario", scenarios, default=scenarios)
    method_filter = st.multiselect("Method", methods, default=methods)
    st.caption(f"{len(episodes):,} recorded episode rows available")

filtered = episodes[
    episodes["topology"].astype(str).isin(topology_filter)
    & episodes["scenario"].astype(str).isin(scenario_filter)
    & episodes["method"].astype(str).isin(method_filter)
].copy()

overview_tab, graphs_tab, tables_tab, episodes_tab = st.tabs(
    ["Overview", "Graphs", "Evaluation outputs", "Episode data"]
)

with overview_tab:
    st.subheader("Filtered evaluation overview")
    if filtered.empty:
        st.info("Select at least one topology, scenario, and method to view results.")
    else:
        grouped = filtered.groupby("method", dropna=False)
        summary = grouped[[c for c in ["pdr", "delay_mean", "throughput", "switches", "energy", "reward"] if c in filtered]].mean(numeric_only=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Episode records", f"{len(filtered):,}")
        with c2:
            st.metric("Mean PDR", f"{filtered['pdr'].mean():.3f}" if "pdr" in filtered else "N/A")
        with c3:
            st.metric("Mean delay", f"{filtered['delay_mean'].mean():.3f}" if "delay_mean" in filtered else "N/A")
        with c4:
            st.metric("Mean switches", f"{filtered['switches'].mean():.2f}" if "switches" in filtered else "N/A")

        left, right = st.columns(2)
        if "pdr" in summary.columns:
            with left:
                st.subheader("Packet delivery ratio by method")
                st.bar_chart(summary[["pdr"]], x_label="Method", y_label="Mean PDR")
        if "delay_mean" in summary.columns:
            with right:
                st.subheader("Mean delay by method")
                st.bar_chart(summary[["delay_mean"]], x_label="Method", y_label="Mean delay")
        if "switches" in summary.columns:
            st.subheader("Mean switching activity by method")
            st.bar_chart(summary[["switches"]], x_label="Method", y_label="Mean switches")

    criteria = success_criteria()
    st.subheader("Success criteria")
    if criteria.empty:
        st.info("No success criteria file was found in results.")
    else:
        st.dataframe(criteria, hide_index=True, width="stretch", height=min(360, 40 + len(criteria) * 35))
        st.download_button(
            "Download success criteria",
            criteria.to_csv(index=False).encode("utf-8"),
            file_name="success_criteria.csv",
            mime="text/csv",
        )

with graphs_tab:
    st.subheader("Saved graphs")
    images = graph_files()
    if not images:
        st.info("No PNG graphs found under results or output/images.")
    else:
        folders = sorted({str(p.parent.relative_to(ROOT)) for p in images if p.is_relative_to(ROOT)})
        selected_folders = st.multiselect("Graph folders", folders, default=folders)
        visible = [p for p in images if str(p.parent.relative_to(ROOT)) in selected_folders]
        st.caption(f"Showing {len(visible)} of {len(images)} available graphs")
        for start in range(0, len(visible), 2):
            cols = st.columns(2)
            for col, image_path in zip(cols, visible[start : start + 2]):
                with col:
                    st.image(
                        str(image_path),
                        caption=relative_label(image_path),
                        alt=f"Evaluation graph {image_path.stem.replace('_', ' ')}",
                        width="stretch",
                    )

with tables_tab:
    st.subheader("Evaluation output files")
    outputs = csv_files()
    if not outputs:
        st.info("No CSV evaluation outputs found under results.")
    else:
        labels = {relative_label(path): path for path in outputs}
        selected = st.selectbox("Select an output table", list(labels))
        table_path = labels[selected]
        table = load_csv(str(table_path), table_path.stat().st_mtime)
        st.caption(f"{len(table):,} rows · {len(table.columns):,} columns · {selected}")
        st.dataframe(table, hide_index=True, width="stretch", height=520)
        st.download_button(
            "Download selected table",
            table.to_csv(index=False).encode("utf-8"),
            file_name=table_path.name,
            mime="text/csv",
        )
    criteria = success_criteria()
    if not criteria.empty:
        st.subheader("Success criteria summary")
        st.dataframe(criteria, hide_index=True, width="stretch")

with episodes_tab:
    st.subheader("Episode-level results")
    st.caption("Filters from the sidebar apply here. The table is capped for browser responsiveness; download exports all filtered rows.")
    st.download_button(
        "Download filtered episode data",
        filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_episode_metrics.csv",
        mime="text/csv",
        disabled=filtered.empty,
    )
    st.dataframe(filtered.head(2000), hide_index=True, width="stretch", height=600)

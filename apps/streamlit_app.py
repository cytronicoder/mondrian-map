"""
Authentic Mondrian Map Explorer - Streamlit Application

This is the main Streamlit application for visualizing biological pathway data
using authentic Mondrian Map algorithms from the IEEE BIBM 2024 paper.
"""

import html
import io
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import from package
from mondrian_map.data_processing import (get_colors,
                                          get_mondrian_color_description,
                                          load_pathway_info)
from mondrian_map.visualization import (create_authentic_mondrian_map,
                                        create_canvas_grid,
                                        create_color_legend)

# Configuration
APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = APP_DIR / "data/case_study/pathways_prepared_for_visualization"
DATASETS = {
    "Aggressive R1": DATA_DIR / "wikipathway_aggressive_R1_TP.csv",
    "Aggressive R2": DATA_DIR / "wikipathway_aggressive_R2_TP.csv",
    "Baseline R1": DATA_DIR / "wikipathway_baseline_R1_TP.csv",
    "Baseline R2": DATA_DIR / "wikipathway_baseline_R2_TP.csv",
    "Nonaggressive R1": DATA_DIR / "wikipathway_nonaggressive_R1_TP.csv",
    "Nonaggressive R2": DATA_DIR / "wikipathway_nonaggressive_R2_TP.csv",
}

DEFAULTS = dict(up=1.25, down=0.75, fdr=0.05)

# ------------------------------------------------------------
# Unified Plotly configuration: interactive (hover disabled via JS) + no mode bar
# ------------------------------------------------------------
PLOT_CONFIG = {"displayModeBar": False}


@st.cache_data
def load_csv_cached(csv_path: str, mtime: float):
    """Load CSV from disk with caching based on mtime"""
    return pd.read_csv(csv_path)


@st.cache_data
def load_uploaded_csv_cached(file_bytes: bytes):
    """Load CSV from bytes with caching"""
    return pd.read_csv(io.BytesIO(file_bytes))


def enrich_dataset(df: pd.DataFrame, pathway_info: dict) -> pd.DataFrame:
    """Add pathway descriptions and metadata to the dataframe"""
    df["Description"] = df["GS_ID"].map(
        lambda x: pathway_info.get(x, {}).get("Description", "")
    )
    df["Ontology"] = df["GS_ID"].map(
        lambda x: pathway_info.get(x, {}).get("Pathway Ontology", "")
    )
    df["Disease"] = df["GS_ID"].map(
        lambda x: pathway_info.get(x, {}).get("Disease", "")
    )
    df["NAME"] = df["GS_ID"].map(lambda x: pathway_info.get(x, {}).get("NAME", x))
    return df


@st.cache_data
def load_pathway_info_cached():
    """Load pathway info with caching"""
    info_path = (
        APP_DIR / "data/case_study/pathway_details/annotations_with_summary.json"
    )
    return load_pathway_info(info_path)


@st.cache_data
def load_deg_data():
    """Load differential gene expression data"""
    deg_path = APP_DIR / "data/case_study/differentially_expressed_genes/DEGs.pkl"
    if deg_path.exists():
        with open(deg_path, "rb") as f:
            return pickle.load(f)
    return None


def update_clicked_pathway_info(clicked_data):
    """Refactored helper to update session state on click"""
    if (
        clicked_data
        and hasattr(clicked_data, "selection")
        and clicked_data.selection.points
    ):
        for pt in clicked_data.selection.points:
            if isinstance(pt, dict):
                customdata = pt.get("customdata")
            else:
                customdata = getattr(pt, "customdata", None)
            if customdata:
                # Plotly customdata for a single point is typically an array; unwrap first element.
                value = customdata
                try:
                    if (
                        isinstance(customdata, (list, tuple, np.ndarray))
                        and len(customdata) > 0
                    ):
                        value = customdata[0]
                except TypeError:
                    # customdata is not a sized/sequence type; leave it as-is.
                    pass
                st.session_state.clicked_pathway_info = value
                break


def display_pathway_tooltip(pathway_info: dict):
    """Display a squared tooltip with full pathway description (XSS safe)"""
    if not pathway_info:
        return

    # Escaping content for security (P1)
    name = html.escape(str(pathway_info.get("name", "")))
    pid = html.escape(str(pathway_info.get("pathway_id", "")))
    ontology = html.escape(str(pathway_info.get("ontology", "")))
    disease = html.escape(str(pathway_info.get("disease", "")))
    description = html.escape(str(pathway_info.get("description", "")))

    fc = pathway_info.get("fold_change", 0)
    pvalue = pathway_info.get("pvalue", 1)

    # Create a styled container for the tooltip
    with st.container():
        st.markdown(
            f"""
            <div style="
                border: 2px solid #333;
                border-radius: 8px;
                padding: 20px;
                background-color: #f8f9fa;
                margin: 20px 0;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            ">
                <h3 style="margin-top: 0; color: #333;">Pathway: {name}</h3>
                <p><strong>ID:</strong> {pid}</p>
                <p><strong>Fold Change:</strong> {fc:.3f}</p>
                <p><strong>P-value:</strong> {pvalue:.2e}</p>
                <p><strong>Ontology:</strong> {ontology}</p>
                <p><strong>Disease:</strong> {disease}</p>
                <hr style="margin: 15px 0;">
                <p><strong>Description:</strong></p>
                <p style="text-align: justify; line-height: 1.6;">{description}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Close button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Close description", key="close_tooltip"):
                st.session_state.clicked_pathway_info = None
                st.rerun()


def display_pathway_genes(
    pathway_id: str,
    deg_data: dict,
    fc_threshold: float,
    up_color: str = "red",
    down_color: str = "blue",
):
    """Display genes for a selected pathway with differential expression coloring"""
    st.markdown(f"### Genes in pathway: {pathway_id}")

    if deg_data is None:
        st.warning("No differential gene expression data available")
        return

    pathway_genes = pd.DataFrame()

    # Check if pathway_id exists in the DEG data (P0 fix)
    for dataset_name, dataset_data in deg_data.items():
        if isinstance(dataset_data, dict) and "genes" in dataset_data:
            genes_df = dataset_data["genes"]
            if isinstance(genes_df, pd.DataFrame) and "pathway_id" in genes_df.columns:
                mask = genes_df["pathway_id"].astype(str).eq(str(pathway_id))
                if mask.any():
                    pathway_genes = genes_df.loc[mask].copy()
                    break

    if len(pathway_genes) == 0:
        st.info(f"No gene data found for pathway {pathway_id}")
        return

    # Create gene table with color coding
    gene_display = pathway_genes.copy()

    def get_gene_color(fc, pval):
        if pval < 0.05:  # Significant (using standard p<0.05 here, could be parameter)
            if fc >= fc_threshold:
                return f"{up_color} Up-regulated"
            elif fc <= (1 / fc_threshold):
                return f"{down_color} Down-regulated"
            else:
                return "Moderate"
        else:
            return "Not significant"

    gene_display["Regulation"] = gene_display.apply(
        lambda row: get_gene_color(row.get("fold_change", 1), row.get("pvalue", 1)),
        axis=1,
    )

    # Display the gene table
    st.dataframe(
        gene_display[["gene_symbol", "fold_change", "pvalue", "Regulation"]],
        width="stretch",
        height=300,
    )

    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Genes", len(gene_display))
    with col2:
        up_genes = len(
            gene_display[gene_display["Regulation"].str.contains("Up-regulated")]
        )
        st.metric("Up-regulated", up_genes)
    with col3:
        down_genes = len(
            gene_display[gene_display["Regulation"].str.contains("Down-regulated")]
        )
        st.metric("Down-regulated", down_genes)
    with col4:
        sig_genes = len(
            gene_display[
                gene_display["Regulation"].str.contains(f"{up_color}|{down_color}")
            ]
        )
        st.metric("Significant", sig_genes)


def display_pathway_crosstalks(
    df_list: list, dataset_names: list, min_similarity: float, min_overlap: int
):
    """Display pathway-to-pathway interaction details by condition"""
    if not df_list or len(df_list) == 0:
        st.info("No datasets available for crosstalk analysis")
        return

    # Load pathway network data
    network_dir = APP_DIR / "data/case_study/pathway_networks"

    if len(df_list) > 1:
        # Create tabs for each dataset's crosstalks
        tabs = st.tabs(dataset_names)
        for i, (df, name) in enumerate(zip(df_list, dataset_names)):
            with tabs[i]:
                display_dataset_crosstalks(
                    df, name, network_dir, min_similarity, min_overlap
                )
    else:
        # Single dataset
        display_dataset_crosstalks(
            df_list[0], dataset_names[0], network_dir, min_similarity, min_overlap
        )


def display_dataset_crosstalks(
    df: pd.DataFrame,
    dataset_name: str,
    network_dir: Path,
    min_similarity: float,
    min_overlap: int,
):
    """Display crosstalks for a single dataset"""
    # Map dataset names to network files
    network_file_map = {
        "Aggressive R1": "wikipathway_aggressive_R1_TP.csv",
        "Aggressive R2": "wikipathway_aggressive_R2_TP.csv",
        "Baseline R1": "wikipathway_baseline_R1_TP.csv",
        "Baseline R2": "wikipathway_baseline_R2_TP.csv",
        "Nonaggressive R1": "wikipathway_nonaggressive_R1_TP.csv",
        "Nonaggressive R2": "wikipathway_nonaggressive_R2_TP.csv",
    }

    network_file = network_file_map.get(dataset_name)
    if not network_file:
        st.warning(f"No network data available for {dataset_name}")
        return

    network_path = network_dir / network_file
    if not network_path.exists():
        st.warning(f"Network file not found: {network_path}")
        return

    try:
        # Load network data (could be cached if large issues arise, but often small enough)
        network_df = pd.read_csv(network_path)

        # Get pathway IDs from the current dataset
        current_pathway_ids = set(df["GS_ID"].tolist())

        # Filter network data to only include pathways in current dataset
        filtered_network = network_df[
            (network_df["GS_A_ID"].isin(current_pathway_ids))
            & (network_df["GS_B_ID"].isin(current_pathway_ids))
        ].copy()

        if len(filtered_network) == 0:
            st.info(f"No pathway crosstalks found for {dataset_name}")
            return

        # Canonical pair fix (P0)
        def create_canonical_pair(row):
            id_a, id_b = str(row["GS_A_ID"]), str(row["GS_B_ID"])
            return tuple(sorted((id_a, id_b)))

        filtered_network["canonical_pair"] = filtered_network.apply(
            create_canonical_pair, axis=1
        )

        # Remove duplicates by keeping only the first occurrence of each canonical pair
        filtered_network = filtered_network.drop_duplicates(
            subset=["canonical_pair"], keep="first"
        )

        # Add pathway names for better readability
        pathway_name_map = dict(zip(df["GS_ID"], df["NAME"]))
        filtered_network["Pathway_A_Name"] = filtered_network["GS_A_ID"].map(
            pathway_name_map
        )
        filtered_network["Pathway_B_Name"] = filtered_network["GS_B_ID"].map(
            pathway_name_map
        )

        # Sort by similarity score (highest first)
        filtered_network = filtered_network.sort_values("SIMILARITY", ascending=False)

        # Display summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Interactions", len(filtered_network))
        with col2:
            high_sim = len(filtered_network[filtered_network["SIMILARITY"] >= 0.5])
            st.metric("High Similarity (≥0.5)", high_sim)
        with col3:
            # Semantic fix: PVALUE in this dataset context seems to be score, rename it
            significant = len(filtered_network[filtered_network["PVALUE"] >= 50])
            st.metric("High Evidence (Score≥50)", significant)
        with col4:
            avg_similarity = filtered_network["SIMILARITY"].mean()
            st.metric("Avg Similarity", f"{avg_similarity:.3f}")

        # Apply filters passed from sidebar/main area
        filtered_display = filtered_network[
            (filtered_network["SIMILARITY"] >= min_similarity)
            & (filtered_network["OLAP"] >= min_overlap)
        ].copy()

        if len(filtered_display) == 0:
            st.warning("No interactions match the current filters")
            return

        # Format the display dataframe
        display_cols = [
            "Pathway_A_Name",
            "GS_A_ID",
            "GS_A_SIZE",
            "Pathway_B_Name",
            "GS_B_ID",
            "GS_B_SIZE",
            "OLAP",
            "SIMILARITY",
            "PVALUE",
        ]

        display_df = filtered_display[display_cols].copy()
        display_df.columns = [
            "Pathway A",
            "ID A",
            "Size A",
            "Pathway B",
            "ID B",
            "Size B",
            "Overlap",
            "Similarity",
            "Evidence Score",
        ]

        # Round numeric columns
        display_df["Similarity"] = display_df["Similarity"].round(4)
        display_df["Evidence Score"] = display_df["Evidence Score"].round(2)

        # Display the crosstalks table
        st.markdown(f"#### Pathway interactions ({len(display_df)} shown)")
        st.dataframe(display_df, width="stretch", height=400)

        # Top interactions summary
        if len(display_df) > 0:
            st.markdown("#### Top 5 strongest interactions")
            top_interactions = display_df.head(5)[
                ["Pathway A", "Pathway B", "Similarity", "Overlap"]
            ]

            for idx, row in top_interactions.iterrows():
                st.markdown(f"**{row['Pathway A']}** ↔ **{row['Pathway B']}**")
                st.markdown(
                    f"&nbsp;&nbsp;&nbsp;&nbsp;Similarity: {row['Similarity']:.3f} | Overlap: {row['Overlap']} genes"
                )

    except Exception as e:
        st.error(f"Error loading network data for {dataset_name}: {str(e)}")


def create_detailed_popup(
    df: pd.DataFrame, dataset_name: str, up_th: float, down_th: float, fdr_th: float
):
    """Create a detailed popup view for a specific Mondrian map"""
    st.markdown(f"## Detailed view: {dataset_name}")

    # Create two columns for the popup
    col1, col2 = st.columns([2, 1])

    with col1:
        # Show maximized Mondrian map
        detailed_fig = create_authentic_mondrian_map(
            df, dataset_name, maximize=True, show_pathway_ids=True
        )
        # Use simple key to avoid duplicate key errors if rerun
        st.plotly_chart(
            detailed_fig,
            width="stretch",
            key=f"detailed_{dataset_name}_chart",
            on_select="rerun",
            config=PLOT_CONFIG,
        )

        st.info(
            "Click pathway tiles in the map above to select them (updates main selection)"
        )

    with col2:
        st.markdown("### Dataset statistics")

        # Basic stats
        total_pathways = len(df)
        up_reg = len(df[df["wFC"] >= up_th])
        down_reg = len(df[df["wFC"] <= down_th])
        significant = len(df[df["pFDR"] < fdr_th])

        st.metric("Total Pathways", total_pathways)
        st.metric("Up-regulated", up_reg)
        st.metric("Down-regulated", down_reg)
        st.metric("Significant", significant)

        # Color distribution
        st.markdown("### Color distribution")
        colors = get_colors(df, up_th, down_th)
        color_counts = {
            "Red (Up-reg)": colors.count("red"),
            "Blue (Down-reg)": colors.count("blue"),
            "Yellow (Moderate)": colors.count("yellow"),
            "Black (Neutral)": colors.count("black"),
        }

        for color, count in color_counts.items():
            if count > 0:
                st.write(f"• {color}: {count}")

        # Top pathways by fold change
        st.markdown("### Top pathways by |FC|")
        df_with_abs_fc = df.copy()
        df_with_abs_fc["abs_wFC"] = df_with_abs_fc["wFC"].abs()
        top_pathways = df_with_abs_fc.nlargest(5, "abs_wFC")[["NAME", "wFC", "pFDR"]]
        st.dataframe(top_pathways, width="stretch")


# Helper function for input validation
def is_valid_csv_file(filename):
    # Only allow .csv extension and safe characters
    return bool(re.match(r"^[\w,\s-]+\.csv$", filename))


def validate_csv_columns(df):
    required_columns = {"GS_ID", "wFC", "pFDR", "x", "y"}
    return required_columns.issubset(set(df.columns))


def main():
    """Main Streamlit application"""
    st.set_page_config(page_title="Authentic Mondrian Map Explorer", layout="wide")

    # Initialize all session state variables at the start
    if "show_detailed_view" not in st.session_state:
        st.session_state.show_detailed_view = None
    if "selected_pathway" not in st.session_state:
        st.session_state.selected_pathway = None
    if "clicked_pathway_info" not in st.session_state:
        st.session_state.clicked_pathway_info = None
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []

    # Inject JS to strip native SVG tooltips created by Plotly
    # This keeps click interactions while preventing hover pop-ups
    components.html(
        """
        <script>
          const removePlotlyTitles = () => {
            document.querySelectorAll('.plotly-graph-div title').forEach(el => el.remove());
          };
          // Run immediately and then at intervals as new charts render
          removePlotlyTitles();
          const observer = new MutationObserver(removePlotlyTitles);
          observer.observe(document.body, { subtree: true, childList: true });
        </script>
        """,
        height=0,
        scrolling=False,
    )

    st.title("Authentic Mondrian Map Explorer")
    st.markdown("*Faithful implementation of the IEEE BIBM 2024 paper algorithms*")

    # Sidebar controls
    st.sidebar.header("Dataset configuration")

    # Threshold Controls (P2)
    with st.sidebar.expander("Analysis thresholds", expanded=False):
        up_thr = st.number_input(
            "Up-reg threshold (wFC)", value=DEFAULTS["up"], step=0.05
        )
        down_thr = st.number_input(
            "Down-reg threshold (wFC)", value=DEFAULTS["down"], step=0.05
        )
        fdr_thr = st.number_input(
            "Significance (pFDR)", value=DEFAULTS["fdr"], step=0.01, format="%.3f"
        )

    # File upload option with security checks (P0 & P1)
    uploaded_files_raw = st.sidebar.file_uploader(
        "Upload CSV files",
        type=["csv"],
        accept_multiple_files=True,
        help="Upload CSV files with columns: GS_ID, wFC, pFDR, x, y. Max size 20MB per file.",
    )

    valid_files = []
    MAX_MB = 20

    if uploaded_files_raw:
        for file in uploaded_files_raw:
            if file.size > MAX_MB * 1024 * 1024:
                st.sidebar.warning(
                    f"File {file.name} exceeds {MAX_MB} MB limit and was skipped."
                )
                continue

            if not is_valid_csv_file(file.name):
                st.sidebar.warning(
                    f"File {file.name} has an invalid name and was skipped."
                )
                continue

            try:
                # Basic read to validate schema (caching happens later on bytes)
                df = pd.read_csv(file)
                if not validate_csv_columns(df):
                    st.sidebar.warning(
                        f"File {file.name} missing required columns {validate_csv_columns(df)}"
                    )
                    continue

                # Rewind file for later use
                file.seek(0)
                valid_files.append(file)
            except Exception as e:
                st.sidebar.warning(f"File {file.name} could not be read: {e}")

    st.session_state.uploaded_files = valid_files

    # Use validated files list
    current_uploaded_files = st.session_state.uploaded_files

    # Load pathway info and DEG data
    pathway_info = load_pathway_info_cached()
    deg_data = load_deg_data()

    # Dataset selection (multi-select)
    df_list = []
    dataset_names = []

    if not current_uploaded_files:
        selected_datasets = st.sidebar.multiselect(
            "Select datasets",
            list(DATASETS.keys()),
            default=["Aggressive R1", "Baseline R1"],
        )

        # Load selected datasets (P1 - Caching)
        for dataset_name in selected_datasets:
            path = DATASETS[dataset_name]
            if path.exists():
                df_raw = load_csv_cached(str(path), path.stat().st_mtime)
                df = enrich_dataset(df_raw.copy(), pathway_info)
                df_list.append(df)
                dataset_names.append(dataset_name)
    else:
        # Use valid uploaded files
        for uploaded_file in current_uploaded_files:
            # P1 - Cache by bytes content
            file_bytes = uploaded_file.getvalue()
            df_raw = load_uploaded_csv_cached(file_bytes)

            # Enforce numeric types (P1)
            for col in ["wFC", "pFDR", "x", "y"]:
                df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")
            df_raw = df_raw.dropna(subset=["GS_ID", "wFC", "pFDR", "x", "y"])

            df = enrich_dataset(df_raw.copy(), pathway_info)
            df_list.append(df)
            dataset_names.append(uploaded_file.name.replace(".csv", ""))

    # Canvas Grid Configuration
    st.sidebar.header("Canvas grid layout")
    if len(df_list) > 0:
        if len(df_list) == 1:
            canvas_cols = 1
            canvas_rows = 1
            st.sidebar.info("Single dataset - 1×1 canvas")
        else:
            max_cols = min(4, len(df_list))
            canvas_cols = st.sidebar.slider(
                "Canvas columns", 1, max_cols, min(2, len(df_list))
            )
            canvas_rows = int(np.ceil(len(df_list) / canvas_cols))
            st.sidebar.info(f"Canvas: {canvas_rows} rows × {canvas_cols} columns")

        # Display options
        show_legend = st.sidebar.checkbox("Show color legend", True)
        show_pathway_ids = st.sidebar.checkbox(
            "Show pathway IDs", False, help="Toggle pathway ID labels on tiles"
        )
        show_full_size = st.sidebar.checkbox("Show full-size maps", False)
        maximize_maps = st.sidebar.checkbox(
            "Maximize individual maps",
            False,
            help="Show larger, detailed individual maps",
        )

    # Main content
    if len(df_list) > 0:
        # Canvas Grid Overview
        st.subheader("Canvas grid overview")

        # Explicit triggers for Detailed View (P0 fix)
        st.markdown("**Filters & Views**")
        col_btns = st.columns(len(dataset_names) + 1)
        # We limit button columns if there are too many (e.g. max 4)
        num_btns = min(len(dataset_names), 4)
        btn_cols = st.columns(num_btns)

        for i, name in enumerate(dataset_names):
            with btn_cols[i % num_btns]:
                if st.button(f"Detail: {name}", key=f"open_detail_{i}"):
                    st.session_state.show_detailed_view = name
                    st.rerun()

        canvas_fig = create_canvas_grid(
            df_list, dataset_names, canvas_rows, canvas_cols, show_pathway_ids
        )

        # Display the canvas with click event handling
        canvas_container = st.container()
        with canvas_container:
            clicked_data = st.plotly_chart(
                canvas_fig,
                width="stretch",
                key="canvas_chart",
                on_select="rerun",
                config=PLOT_CONFIG,
            )
            # Unified click handler (P1)
            update_clicked_pathway_info(clicked_data)

        # Display pathway tooltip if clicked
        if st.session_state.clicked_pathway_info:
            st.markdown("---")
            display_pathway_tooltip(st.session_state.clicked_pathway_info)

        # Add clickable functionality info
        st.markdown("### Interactive maps")
        st.info(
            "Click on pathway tiles to see full descriptions. Use the buttons above to open detailed views."
        )

        # Check if any detailed view should be shown
        for i, (df, name) in enumerate(zip(df_list, dataset_names)):
            if st.session_state.show_detailed_view == name:
                st.markdown("---")
                create_detailed_popup(df, name, up_thr, down_thr, fdr_thr)
                if st.button("Close detailed view", key=f"close_dt_{i}"):
                    st.session_state.show_detailed_view = None
                    st.rerun()

        # Full-size individual maps
        if show_full_size:
            st.subheader("Full-size authentic Mondrian maps")

            # Create columns for full-size maps
            cols_per_row = 1 if maximize_maps else 2

            for i in range(0, len(df_list), cols_per_row):
                cols = st.columns(cols_per_row)
                for j in range(cols_per_row):
                    if i + j < len(df_list):
                        with cols[j]:
                            fig = create_authentic_mondrian_map(
                                df_list[i + j],
                                dataset_names[i + j],
                                maximize=maximize_maps,
                                show_pathway_ids=show_pathway_ids,
                            )
                            c_data = st.plotly_chart(
                                fig,
                                width="stretch",
                                key=f"full_map_{i+j}",
                                on_select="rerun",
                                config=PLOT_CONFIG,
                            )
                            update_clicked_pathway_info(c_data)

        # Color legend and info
        if show_legend:
            col1, col2 = st.columns([1, 2])

            with col1:
                st.subheader("Color legend")
                legend_fig = create_color_legend()
                st.plotly_chart(legend_fig, width="stretch", config=PLOT_CONFIG)

            with col2:
                st.subheader("Authentic algorithm")
                st.markdown(
                    f"""
                **Faithful Implementation of IEEE BIBM 2024 Paper:**
                
                **3-Stage Generation Process:**
                1. **Grid System**: 1001×1001 canvas with 20×20 block grid
                2. **Block Placement**: Pathways as rectangles sized by fold change
                3. **Line Generation**: Authentic Mondrian-style grid lines
                
                **Parameters:**
                - Up-regulation threshold: ≥{up_thr}
                - Down-regulation threshold: ≤{down_thr}
                """
                )

        # Dataset statistics
        st.subheader("Dataset statistics")
        stats_cols = st.columns(min(len(df_list), 4))
        for i, (df, name) in enumerate(zip(df_list, dataset_names)):
            with stats_cols[i % 4]:
                st.metric(f"{name} - Total", len(df))
                up_reg = len(df[df["wFC"] >= up_thr])
                down_reg = len(df[df["wFC"] <= down_thr])
                st.metric("Up-regulated", up_reg)
                st.metric("Down-regulated", down_reg)

        # Detailed pathway tables
        st.subheader("Pathway details")

        # Create tabs for each dataset
        if len(df_list) > 1:
            tabs = st.tabs(dataset_names)
            for i, (df, name) in enumerate(zip(df_list, dataset_names)):
                with tabs[i]:
                    df_display = df.copy()
                    df_display["Color"] = df_display.apply(
                        lambda row: get_mondrian_color_description(
                            row["wFC"], row["pFDR"]
                        ),
                        axis=1,
                    )

                    event = st.dataframe(
                        df_display[
                            [
                                "NAME",
                                "GS_ID",
                                "wFC",
                                "pFDR",
                                "Color",
                                "Description",
                                "Ontology",
                            ]
                        ].round(4),
                        width="stretch",
                        height=400,
                        on_select="rerun",
                        selection_mode="single-row",
                    )

                    if event.selection and len(event.selection.rows) > 0:
                        selected_row = event.selection.rows[0]
                        selected_pathway_id = df_display.iloc[selected_row]["GS_ID"]
                        st.session_state.selected_pathway = selected_pathway_id
        else:
            if len(df_list) > 0:
                df_display = df_list[0].copy()
                df_display["Color"] = df_display.apply(
                    lambda row: get_mondrian_color_description(row["wFC"], row["pFDR"]),
                    axis=1,
                )

                event = st.dataframe(
                    df_display[
                        [
                            "NAME",
                            "GS_ID",
                            "wFC",
                            "pFDR",
                            "Color",
                            "Description",
                            "Ontology",
                        ]
                    ].round(4),
                    width="stretch",
                    height=400,
                    on_select="rerun",
                    selection_mode="single-row",
                )

                if event.selection and len(event.selection.rows) > 0:
                    selected_row = event.selection.rows[0]
                    selected_pathway_id = df_display.iloc[selected_row]["GS_ID"]
                    st.session_state.selected_pathway = selected_pathway_id

        # Display gene details for selected pathway
        if st.session_state.selected_pathway:
            st.markdown("---")
            display_pathway_genes(st.session_state.selected_pathway, deg_data, up_thr)

            if st.button("Close gene details"):
                st.session_state.selected_pathway = None
                st.rerun()

        # Section 3: Pathway Crosstalks
        st.subheader("Pathway crosstalks")
        st.markdown("*Pathway-to-pathway interaction details by condition*")

        # Crosstalk Filters
        col1, col2 = st.columns(2)
        with col1:
            min_similarity = st.slider(
                "Minimum Similarity",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.05,
                help="Filter interactions by minimum similarity score",
            )
        with col2:
            min_overlap = st.slider(
                "Minimum Overlap",
                min_value=1,
                max_value=20,  # Dynamic range ideal but static is safe for init
                value=2,
                help="Filter interactions by minimum gene overlap",
            )

        display_pathway_crosstalks(df_list, dataset_names, min_similarity, min_overlap)

    else:
        st.info("Please select datasets or upload CSV files to begin visualization")

        # Show example data format
        st.subheader("Required CSV format")
        example_df = pd.DataFrame(
            {
                "GS_ID": ["WAG002659", "WAG002805"],
                "wFC": [1.1057, 1.0888],
                "pFDR": [3.5e-17, 5.3e-17],
                "x": [381.9, 971.2],
                "y": [468.9, 573.7],
            }
        )
        st.dataframe(example_df)


if __name__ == "__main__":
    main()

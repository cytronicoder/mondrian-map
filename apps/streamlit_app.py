"""
Authentic Mondrian Map Explorer - Streamlit Application

This is the main Streamlit application for visualizing biological pathway data
using authentic Mondrian Map algorithms from the bioRxiv paper.
"""

import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Add the src directory to the path to import our modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

from mondrian_map.core import Colors
from mondrian_map.data_processing import (
    get_colors,
    get_mondrian_color_description,
    load_dataset,
    load_pathway_info,
    load_uploaded_dataset,
)
from mondrian_map.visualization import (
    create_authentic_mondrian_map,
    create_canvas_grid,
    create_color_legend,
)

# Configuration
DATA_DIR = Path("data/case_study/pathways_prepared_for_visualization")
DATASETS = {
    "Aggressive R1": DATA_DIR / "wikipathway_aggressive_R1_TP.csv",
    "Aggressive R2": DATA_DIR / "wikipathway_aggressive_R2_TP.csv",
    "Baseline R1": DATA_DIR / "wikipathway_baseline_R1_TP.csv",
    "Baseline R2": DATA_DIR / "wikipathway_baseline_R2_TP.csv",
    "Nonaggressive R1": DATA_DIR / "wikipathway_nonaggressive_R1_TP.csv",
    "Nonaggressive R2": DATA_DIR / "wikipathway_nonaggressive_R2_TP.csv",
}

# ------------------------------------------------------------
# Unified Plotly configuration: interactive (hover disabled via JS) + no mode bar
# ------------------------------------------------------------
PLOT_CONFIG = {"displayModeBar": False}


@st.cache_data
def load_pathway_info_cached():
    """Load pathway info with caching"""
    info_path = Path("data/case_study/pathway_details/annotations_with_summary.json")
    return load_pathway_info(info_path)


@st.cache_data
def load_deg_data():
    """Load differential gene expression data"""
    deg_path = Path("data/case_study/differentially_expressed_genes/DEGs.pkl")
    if deg_path.exists():
        with open(deg_path, "rb") as f:
            return pickle.load(f)
    return None


def display_pathway_tooltip(pathway_info: dict):
    """Display a squared tooltip with full pathway description"""
    if not pathway_info:
        return

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
                <h3 style="margin-top: 0; color: #333;">🧬 {pathway_info['name']}</h3>
                <p><strong>ID:</strong> {pathway_info['pathway_id']}</p>
                <p><strong>Fold Change:</strong> {pathway_info['fold_change']:.3f}</p>
                <p><strong>P-value:</strong> {pathway_info['pvalue']:.2e}</p>
                <p><strong>Ontology:</strong> {pathway_info['ontology']}</p>
                <p><strong>Disease:</strong> {pathway_info['disease']}</p>
                <hr style="margin: 15px 0;">
                <p><strong>Description:</strong></p>
                <p style="text-align: justify; line-height: 1.6;">{pathway_info['description']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Close button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("❌ Close Description", key="close_tooltip"):
                st.session_state.clicked_pathway_info = None
                st.rerun()


def display_pathway_genes(pathway_id: str, deg_data: dict, fc_threshold: float = 1.25):
    """Display genes for a selected pathway with differential expression coloring"""
    st.markdown(f"### 🧬 Genes in Pathway: {pathway_id}")

    if deg_data is None:
        st.warning("No differential gene expression data available")
        return

    # Find genes for this pathway (assuming pathway_id is in the data)
    # This is a simplified version - you may need to adjust based on actual data structure
    pathway_genes = []

    # Check if pathway_id exists in the DEG data
    for dataset_name, dataset_data in deg_data.items():
        if isinstance(dataset_data, dict) and "genes" in dataset_data:
            genes_df = dataset_data["genes"]
            if pathway_id in genes_df.get("pathway_id", []):
                pathway_genes = genes_df[genes_df["pathway_id"] == pathway_id]
                break

    if len(pathway_genes) == 0:
        st.info(f"No gene data found for pathway {pathway_id}")
        return

    # Create gene table with color coding
    gene_display = pathway_genes.copy()

    def get_gene_color(fc, pval):
        if pval < 0.05:  # Significant
            if fc >= fc_threshold:
                return "🔴 Up-regulated"
            elif fc <= (1 / fc_threshold):
                return "🔵 Down-regulated"
            else:
                return "🟡 Moderate"
        else:
            return "⚪ Not significant"

    gene_display["Regulation"] = gene_display.apply(
        lambda row: get_gene_color(row.get("fold_change", 1), row.get("pvalue", 1)),
        axis=1,
    )

    # Display the gene table
    st.dataframe(
        gene_display[["gene_symbol", "fold_change", "pvalue", "Regulation"]],
        use_container_width=True,
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
        sig_genes = len(gene_display[gene_display["Regulation"].str.contains("🔴|🔵")])
        st.metric("Significant", sig_genes)


def display_pathway_crosstalks(df_list: list, dataset_names: list):
    """Display pathway-to-pathway interaction details by condition"""
    if not df_list or len(df_list) == 0:
        st.info("No datasets available for crosstalk analysis")
        return

    # Load pathway network data
    network_dir = Path("data/case_study/pathway_networks")

    if len(df_list) > 1:
        # Create tabs for each dataset's crosstalks
        tabs = st.tabs(dataset_names)
        for i, (df, name) in enumerate(zip(df_list, dataset_names)):
            with tabs[i]:
                display_dataset_crosstalks(df, name, network_dir)
    else:
        # Single dataset
        display_dataset_crosstalks(df_list[0], dataset_names[0], network_dir)


def display_dataset_crosstalks(df: pd.DataFrame, dataset_name: str, network_dir: Path):
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
        # Load network data
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

        # Remove duplicate interactions (A-B and B-A are the same interaction)
        # Create a canonical representation where smaller ID comes first
        def create_canonical_pair(row):
            id_a, id_b = row["GS_A_ID"], row["GS_B_ID"]
            if id_a < id_b:
                return f"{id_a}|{id_b}"
            else:
                return f"{id_b}|{id_a}"

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
            significant = len(
                filtered_network[filtered_network["PVALUE"] >= 50]
            )  # Assuming higher p-value means more significant in this context
            st.metric("Strong Evidence (p≥50)", significant)
        with col4:
            avg_similarity = filtered_network["SIMILARITY"].mean()
            st.metric("Avg Similarity", f"{avg_similarity:.3f}")

        # Interactive filters
        st.markdown("#### 🔍 Filter Crosstalks")
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
                min_value=int(filtered_network["OLAP"].min()),
                max_value=int(filtered_network["OLAP"].max()),
                value=int(filtered_network["OLAP"].min()),
                help="Filter interactions by minimum gene overlap",
            )

        # Apply filters
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
        st.markdown(f"#### 🔗 Pathway Interactions ({len(display_df)} shown)")
        st.dataframe(display_df, use_container_width=True, height=400)

        # Top interactions summary
        if len(display_df) > 0:
            st.markdown("#### 🏆 Top 5 Strongest Interactions")
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


def create_detailed_popup(df: pd.DataFrame, dataset_name: str):
    """Create a detailed popup view for a specific Mondrian map"""
    st.markdown(f"## 🔍 Detailed View: {dataset_name}")

    # Create two columns for the popup
    col1, col2 = st.columns([2, 1])

    with col1:
        # Show maximized Mondrian map
        detailed_fig = create_authentic_mondrian_map(
            df, dataset_name, maximize=True, show_pathway_ids=True
        )
        st.plotly_chart(
            detailed_fig,
            use_container_width=True,
            key=f"detailed_{dataset_name}",
            config=PLOT_CONFIG,
        )

        st.info(
            "💡 **Click pathway tiles** in the map above to see individual pathway details (hover disabled for clean view)"
        )

    with col2:
        st.markdown("### 📊 Dataset Statistics")

        # Basic stats
        total_pathways = len(df)
        up_reg = len(df[df["wFC"] >= 1.25])
        down_reg = len(df[df["wFC"] <= 0.75])
        significant = len(df[df["pFDR"] < 0.05])

        st.metric("Total Pathways", total_pathways)
        st.metric("Up-regulated", up_reg)
        st.metric("Down-regulated", down_reg)
        st.metric("Significant (p<0.05)", significant)

        # Color distribution
        st.markdown("### 🎨 Color Distribution")
        colors = get_colors(df, 1.25, 0.75)
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
        st.markdown("### 🔝 Top Pathways by |FC|")
        df_with_abs_fc = df.copy()
        df_with_abs_fc["abs_wFC"] = df_with_abs_fc["wFC"].abs()
        top_pathways = df_with_abs_fc.nlargest(5, "abs_wFC")[["NAME", "wFC", "pFDR"]]
        st.dataframe(top_pathways, use_container_width=True)


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
        st.session_state.show_detailed_view = False
    if "selected_pathway" not in st.session_state:
        st.session_state.selected_pathway = None
    if "clicked_pathway_info" not in st.session_state:
        st.session_state.clicked_pathway_info = None
    if "current_dataset" not in st.session_state:
        st.session_state.current_dataset = None
    if "grid_layout" not in st.session_state:
        st.session_state.grid_layout = None
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []

    # Inject JS to strip native SVG tooltips created by Plotly (<title> tags)
    # This keeps click interactions while preventing hover pop-ups
    import streamlit.components.v1 as components  # local import to avoid global dependency if not needed elsewhere

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

    st.title("🎨 Authentic Mondrian Map Explorer")
    st.markdown("*Faithful implementation of the bioRxiv paper algorithms*")

    # Sidebar controls
    st.sidebar.header("📊 Dataset Configuration")

    # File upload option with security checks
    uploaded_files = st.sidebar.file_uploader(
        "Upload CSV files",
        type=["csv"],
        accept_multiple_files=True,
        help="Upload CSV files with columns: GS_ID, wFC, pFDR, x, y",
    )
    valid_files = []
    for file in uploaded_files or []:
        if not is_valid_csv_file(file.name):
            st.sidebar.warning(
                f"File {file.name} has an invalid name or extension and was skipped."
            )
            continue
        try:
            df = pd.read_csv(file)
            if not validate_csv_columns(df):
                st.sidebar.warning(
                    f"File {file.name} is missing required columns and was skipped."
                )
                continue
            valid_files.append(file)
        except Exception as e:
            st.sidebar.warning(f"File {file.name} could not be read: {e}")
    st.session_state.uploaded_files = valid_files

    # Load pathway info and DEG data
    pathway_info = load_pathway_info_cached()
    deg_data = load_deg_data()

    # Dataset selection (multi-select)
    if not uploaded_files:
        selected_datasets = st.sidebar.multiselect(
            "Select datasets",
            list(DATASETS.keys()),
            default=["Aggressive R1", "Baseline R1"],
        )

        # Load selected datasets
        df_list = []
        dataset_names = []
        for dataset_name in selected_datasets:
            df = load_dataset(DATASETS[dataset_name], pathway_info)
            df_list.append(df)
            dataset_names.append(dataset_name)
    else:
        # Use uploaded files
        df_list = []
        dataset_names = []
        for uploaded_file in uploaded_files:
            df = load_uploaded_dataset(uploaded_file, pathway_info)
            if df is not None:
                df_list.append(df)
                dataset_names.append(uploaded_file.name.replace(".csv", ""))

    # Canvas Grid Configuration
    st.sidebar.header("🎯 Canvas Grid Layout")
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
            "🔍 Maximize individual maps",
            False,
            help="Show larger, detailed individual maps",
        )

    # Main content
    if len(df_list) > 0:
        # Canvas Grid Overview
        st.subheader("📋 Canvas Grid Overview")
        st.markdown(
            "*Click on individual map titles below to see detailed popup views*"
        )

        canvas_fig = create_canvas_grid(
            df_list, dataset_names, canvas_rows, canvas_cols, show_pathway_ids
        )

        # Display the canvas with click event handling
        canvas_container = st.container()
        with canvas_container:
            clicked_data = st.plotly_chart(
                canvas_fig,
                use_container_width=True,
                key="canvas_chart",
                on_select="rerun",
                config=PLOT_CONFIG,
            )

            # Handle pathway clicks for tooltip
            if (
                clicked_data
                and hasattr(clicked_data, "selection")
                and clicked_data.selection.points
            ):
                if len(clicked_data.selection.points) > 0:
                    point_data = clicked_data.selection.points[0]
                    if "customdata" in point_data and point_data["customdata"]:
                        st.session_state.clicked_pathway_info = point_data["customdata"]

        # Display pathway tooltip if clicked
        if st.session_state.clicked_pathway_info:
            st.markdown("---")
            display_pathway_tooltip(st.session_state.clicked_pathway_info)

        # Add clickable functionality info
        st.markdown("### 🖱️ Interactive Maps")
        st.info(
            "💡 **Click on pathway tiles** to see full descriptions, or click pathway table rows for gene details"
        )

        # Check if any detailed view should be shown
        for i, (df, name) in enumerate(zip(df_list, dataset_names)):
            if st.session_state.show_detailed_view == name:
                st.markdown("---")
                create_detailed_popup(df, name)
                if st.button("❌ Close Detailed View", key=f"close_{i}"):
                    st.session_state.show_detailed_view = None
                    st.rerun()

        # Full-size individual maps
        if show_full_size:
            st.subheader("🔍 Full-Size Authentic Mondrian Maps")
            st.markdown(
                "*Individual maps using the exact 3-stage algorithm from the notebooks*"
            )

            if maximize_maps:
                st.info(
                    "🔍 **Maximized View**: Larger maps with enhanced details for better analysis"
                )

            # Create columns for full-size maps
            if len(df_list) == 1:
                full_fig = create_authentic_mondrian_map(
                    df_list[0],
                    dataset_names[0],
                    maximize=maximize_maps,
                    show_pathway_ids=show_pathway_ids,
                )
                clicked_data = st.plotly_chart(
                    full_fig,
                    use_container_width=True,
                    key=f"full_map_0",
                    on_select="rerun",
                    config=PLOT_CONFIG,
                )

                # Handle pathway clicks for tooltip
                if (
                    clicked_data
                    and hasattr(clicked_data, "selection")
                    and clicked_data.selection.points
                ):
                    if len(clicked_data.selection.points) > 0:
                        point_data = clicked_data.selection.points[0]
                        if "customdata" in point_data and point_data["customdata"]:
                            st.session_state.clicked_pathway_info = point_data[
                                "customdata"
                            ]

                st.info(
                    "💡 **Click pathway tiles above** to see full descriptions (hover disabled for clean view)"
                )
            else:
                # Show maps in pairs or single column if maximized
                cols_per_row = 1 if maximize_maps else 2

                for i in range(0, len(df_list), cols_per_row):
                    if cols_per_row == 1:
                        # Single column for maximized view
                        full_fig = create_authentic_mondrian_map(
                            df_list[i],
                            dataset_names[i],
                            maximize=maximize_maps,
                            show_pathway_ids=show_pathway_ids,
                        )
                        clicked_data = st.plotly_chart(
                            full_fig,
                            use_container_width=True,
                            key=f"full_map_{i}",
                            on_select="rerun",
                            config=PLOT_CONFIG,
                        )

                        # Handle pathway clicks for tooltip
                        if (
                            clicked_data
                            and hasattr(clicked_data, "selection")
                            and clicked_data.selection.points
                        ):
                            if len(clicked_data.selection.points) > 0:
                                point_data = clicked_data.selection.points[0]
                                if (
                                    "customdata" in point_data
                                    and point_data["customdata"]
                                ):
                                    st.session_state.clicked_pathway_info = point_data[
                                        "customdata"
                                    ]

                        st.info(
                            "💡 **Click pathway tiles** for descriptions (hover disabled)"
                        )
                    else:
                        # Two columns for normal view
                        cols = st.columns(2)

                        with cols[0]:
                            full_fig = create_authentic_mondrian_map(
                                df_list[i],
                                dataset_names[i],
                                maximize=maximize_maps,
                                show_pathway_ids=show_pathway_ids,
                            )
                            clicked_data = st.plotly_chart(
                                full_fig,
                                use_container_width=True,
                                key=f"full_map_{i}",
                                on_select="rerun",
                                config=PLOT_CONFIG,
                            )

                            # Handle pathway clicks for tooltip
                            if (
                                clicked_data
                                and hasattr(clicked_data, "selection")
                                and clicked_data.selection.points
                            ):
                                if len(clicked_data.selection.points) > 0:
                                    point_data = clicked_data.selection.points[0]
                                    if (
                                        "customdata" in point_data
                                        and point_data["customdata"]
                                    ):
                                        st.session_state.clicked_pathway_info = (
                                            point_data["customdata"]
                                        )

                            st.info(
                                "💡 **Click tiles** for descriptions (hover disabled)"
                            )

                        if i + 1 < len(df_list):
                            with cols[1]:
                                full_fig = create_authentic_mondrian_map(
                                    df_list[i + 1],
                                    dataset_names[i + 1],
                                    maximize=maximize_maps,
                                    show_pathway_ids=show_pathway_ids,
                                )
                                clicked_data = st.plotly_chart(
                                    full_fig,
                                    use_container_width=True,
                                    key=f"full_map_{i+1}",
                                    on_select="rerun",
                                    config=PLOT_CONFIG,
                                )

                                # Handle pathway clicks for tooltip
                                if (
                                    clicked_data
                                    and hasattr(clicked_data, "selection")
                                    and clicked_data.selection.points
                                ):
                                    if len(clicked_data.selection.points) > 0:
                                        point_data = clicked_data.selection.points[0]
                                        if (
                                            "customdata" in point_data
                                            and point_data["customdata"]
                                        ):
                                            st.session_state.clicked_pathway_info = (
                                                point_data["customdata"]
                                            )

                                st.info(
                                    "💡 **Click tiles** for descriptions (hover disabled)"
                                )

        # Color legend and info
        if show_legend:
            col1, col2 = st.columns([1, 2])

            with col1:
                st.subheader("🎨 Color Legend")
                legend_fig = create_color_legend()
                st.plotly_chart(
                    legend_fig, use_container_width=True, config=PLOT_CONFIG
                )

            with col2:
                st.subheader("ℹ️ Authentic Algorithm")
                st.markdown(
                    """
                **Faithful Implementation of bioRxiv Paper:**
                
                **3-Stage Generation Process:**
                1. **Grid System**: 1001×1001 canvas with 20×20 block grid
                2. **Block Placement**: Pathways as rectangles sized by fold change
                3. **Line Generation**: Authentic Mondrian-style grid lines
                
                **Key Features:**
                - Exact `GridSystem`, `Block`, `Line`, `Corner` classes from notebooks
                - Authentic color scheme: Red/Blue/Yellow/Black/White
                - Area scaling: `abs(log2(wFC)) * 4000`
                - Proper line width and adjustments
                - Rectangle placement based on pathway coordinates
                
                **Algorithm Parameters:**
                - Canvas: 1001×1001 pixels
                - Block size: 20×20 pixels  
                - Line width: 5 pixels
                - Area scalar: 4000
                - Up-regulation threshold: ≥1.25
                - Down-regulation threshold: ≤0.75
                """
                )

        # Dataset Statistics
        st.subheader("📈 Dataset Statistics")
        stats_cols = st.columns(len(df_list))
        for i, (df, name) in enumerate(zip(df_list, dataset_names)):
            with stats_cols[i]:
                st.metric(f"{name} - Total", len(df))
                up_reg = len(df[df["wFC"] >= 1.25])
                down_reg = len(df[df["wFC"] <= 0.75])
                st.metric("Up-regulated", up_reg)
                st.metric("Down-regulated", down_reg)

        # Detailed pathway tables
        st.subheader("📋 Pathway Details")

        # Create tabs for each dataset
        if len(df_list) > 1:
            tabs = st.tabs(dataset_names)
            for i, (df, name) in enumerate(zip(df_list, dataset_names)):
                with tabs[i]:
                    # Add color coding to the dataframe
                    df_display = df.copy()
                    df_display["Color"] = df_display.apply(
                        lambda row: get_mondrian_color_description(
                            row["wFC"], row["pFDR"]
                        ),
                        axis=1,
                    )

                    # Make pathway table clickable
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
                                "Disease",
                            ]
                        ].round(4),
                        use_container_width=True,
                        height=400,
                        on_select="rerun",
                        selection_mode="single-row",
                    )

                    # Handle row selection for gene details
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

                # Make pathway table clickable
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
                            "Disease",
                        ]
                    ].round(4),
                    use_container_width=True,
                    height=400,
                    on_select="rerun",
                    selection_mode="single-row",
                )

                # Handle row selection for gene details
                if event.selection and len(event.selection.rows) > 0:
                    selected_row = event.selection.rows[0]
                    selected_pathway_id = df_display.iloc[selected_row]["GS_ID"]
                    st.session_state.selected_pathway = selected_pathway_id

        # Display gene details for selected pathway
        if st.session_state.selected_pathway:
            st.markdown("---")
            display_pathway_genes(st.session_state.selected_pathway, deg_data)

            if st.button("❌ Close Gene Details"):
                st.session_state.selected_pathway = None
                st.rerun()

        # Section 3: Pathway Crosstalks
        st.subheader("🔗 Pathway Crosstalks")
        st.markdown("*Pathway-to-pathway interaction details by condition*")

        display_pathway_crosstalks(df_list, dataset_names)

    else:
        st.info("👆 Please select datasets or upload CSV files to begin visualization")

        # Show example data format
        st.subheader("📝 Required CSV Format")
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

        st.subheader("🎯 Authentic Implementation")
        st.markdown(
            """
        **Faithful to bioRxiv Paper Algorithm:**
        
        This implementation uses the exact same classes and methods from the research notebooks:
        - `GridSystem(1001, 1001, 20, 20)` - Authentic grid system
        - `Block`, `Line`, `Corner` classes with exact parameters
        - 3-stage generation process as described in the paper
        - Authentic color mapping and area scaling
        - Proper line width, adjustments, and positioning
        
        **Canvas Grid System:**
        - Level 1: Canvas arranges multiple Mondrian maps
        - Level 2: Each map uses authentic algorithm from notebooks
        - Users can view overview or full-size individual maps
        """
        )


if __name__ == "__main__":
    main()

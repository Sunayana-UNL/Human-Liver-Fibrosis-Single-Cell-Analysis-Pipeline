"""
03_cell_type_annotation.py
Cell type annotation using curated marker genes for human liver cell types.
Based on Ramachandran et al. 2019 marker definitions.

Input:  data/processed/GSE136103_integrated.h5ad
Output: data/processed/GSE136103_annotated.h5ad

"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import os

sc.settings.verbosity = 2
sc.settings.figdir = "figures/annotation/"
os.makedirs("figures/annotation", exist_ok=True)

adata = sc.read("data/processed/GSE136103_integrated.h5ad")
print(f"Loaded: {adata.n_obs:,} cells x {adata.n_vars:,} genes")
print(f"Clusters in leiden_0.8: {sorted(adata.obs['leiden_0.8'].unique())}")

# Sanity check: print file label vs condition to confirm metadata matches filenames
print("\nFile label vs condition sanity check (first 10 rows):")
if "file_label" in adata.obs.columns:
    check = adata.obs[["gsm", "file_label", "condition", "donor"]].drop_duplicates()
    print(check.to_string())

# Curated marker gene sets from Ramachandran et al 2019 and published liver atlases
MARKER_GENES = {
    "SAMac_TREM2":         ["TREM2", "CD9", "SPP1", "GPNMB", "LGALS3", "TNFSF12"],
    "Kupffer_cells":       ["TIMD4", "MARCO", "VSIG4", "CD5L", "HMOX1"],
    "Tissue_monocytes":    ["CCR2", "CD14", "S100A8", "S100A9", "FCGR3A"],
    "cDC1":                ["CLEC9A", "XCR1", "CADM1", "IDO1"],
    "cDC2":                ["CD1C", "FCER1A", "CLEC10A"],
    "pDC":                 ["LILRA4", "CLEC4C", "GZMB", "IL3RA"],
    "SAMes_myofibroblast": ["PDGFRA", "ACTA2", "COL1A1", "COL1A2", "FN1", "TIMP1"],
    "Quiescent_HSC":       ["LRAT", "RELN", "CYGB", "HAND2", "VIPR1"],
    "SAEndo_ACKR1":        ["ACKR1", "PLVAP", "ICAM1", "VWF", "PECAM1"],
    "LSEC":                ["STAB2", "LYVE1", "FCN3", "CLEC4G", "CLEC1B"],
    "Large_vessel_EC":     ["GJA4", "GJA5", "HEY1", "NOTCH4"],
    "NK_cells":            ["NKG7", "GNLY", "KLRB1", "NCR1", "FCGR3A"],
    "CD8_T_cells":         ["CD8A", "CD8B", "GZMB", "PRF1", "NKG7"],
    "CD4_T_cells":         ["CD4", "IL7R", "CCR7", "FOXP3"],
    "B_cells":             ["CD79A", "CD79B", "MS4A1", "CD19"],
    "Plasma_cells":        ["IGHA1", "IGHG1", "MZB1", "SDC1"],
    "Cholangiocytes":      ["KRT19", "KRT7", "EPCAM", "CFTR"],
}

# Score each cell against every marker set using lognorm values in adata.X
# adata.raw is not set so score_genes reads from adata.X directly
print("\nScoring marker gene sets...")
for celltype, genes in MARKER_GENES.items():
    genes_present = [g for g in genes if g in adata.var_names]
    if len(genes_present) >= 2:
        sc.tl.score_genes(
            adata,
            gene_list=genes_present,
            score_name=f"score_{celltype}"
        )
        print(f"  {celltype}: {len(genes_present)}/{len(genes)} markers found")
    else:
        print(f"  WARNING {celltype}: only {len(genes_present)} markers found, skipping")

# Auto annotate each cell by taking the highest scoring cell type
score_cols = [c for c in adata.obs.columns if c.startswith("score_")]
score_df   = adata.obs[score_cols].copy()
score_df.columns = [c.replace("score_", "") for c in score_cols]
adata.obs["auto_celltype"] = score_df.idxmax(axis=1)

# Print cluster composition table to help you decide on manual labels
print("\nCluster to auto cell type summary (use this to fill in CLUSTER_LABELS below):")
ct_summary = adata.obs.groupby(
    ["leiden_0.8", "auto_celltype"]
).size().unstack(fill_value=0)
print(ct_summary.to_string())

# IMPORTANT: CLUSTER_LABELS must be reviewed and updated after inspecting the
# dotplot and ct_summary printout above. The numbers below are typical for this
# dataset but your actual cluster numbers may differ.
CLUSTER_LABELS = {
    "0":  "Kupffer_cells",
    "1":  "CD8_T_cells",
    "2":  "SAMac_TREM2",
    "3":  "LSEC",
    "4":  "SAMes_myofibroblast",
    "5":  "NK_cells",
    "6":  "Tissue_monocytes",
    "7":  "SAEndo_ACKR1",
    "8":  "CD4_T_cells",
    "9":  "B_cells",
    "10": "Quiescent_HSC",
    "11": "cDC2",
    "12": "SAMac_TREM2",
    "13": "Plasma_cells",
    "14": "cDC1",
    "15": "LSEC",
    "16": "Large_vessel_EC",
    "17": "Cholangiocytes",
    "18": "pDC",
}

adata.obs["cell_type"] = (
    adata.obs["leiden_0.8"]
    .map(CLUSTER_LABELS)
    .fillna("Unknown")
)

# Simplified compartment groupings
COMPARTMENT_MAP = {
    "SAMac_TREM2":         "Macrophage",
    "Kupffer_cells":       "Macrophage",
    "Tissue_monocytes":    "Macrophage",
    "cDC1":                "DC",
    "cDC2":                "DC",
    "pDC":                 "DC",
    "SAMes_myofibroblast": "Mesenchymal",
    "Quiescent_HSC":       "Mesenchymal",
    "SAEndo_ACKR1":        "Endothelial",
    "LSEC":                "Endothelial",
    "Large_vessel_EC":     "Endothelial",
    "CD8_T_cells":         "Lymphoid",
    "CD4_T_cells":         "Lymphoid",
    "NK_cells":            "Lymphoid",
    "B_cells":             "Lymphoid",
    "Plasma_cells":        "Lymphoid",
    "Cholangiocytes":      "Epithelial",
}
adata.obs["compartment"] = (
    adata.obs["cell_type"]
    .map(COMPARTMENT_MAP)
    .fillna("Other")
)

#final composition
print(f"\nCell type composition:\n{adata.obs['cell_type'].value_counts().to_string()}")
print(f"\nCompartment composition:\n{adata.obs['compartment'].value_counts().to_string()}")

# Save
adata.write("data/processed/GSE136103_annotated.h5ad")
print("\nSaved: data/processed/GSE136103_annotated.h5ad")

# UMAP plots
# We draw each panel separately using matplotlib so we can control
# figure size, legend position, and spacing precisely.
# scanpy auto-placement with ncols stacks panels too tightly and
# the legend for cell_type overlaps the UMAP when there are many categories.

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_umap_panel(adata, color_key, title, ax, palette=None,
                    legend_outside=True, point_size=8):
    """
    Draw one UMAP panel on a provided matplotlib Axes object.
    Puts the legend outside the plot to prevent overlap with the points.

    color_key      column in adata.obs to color by
    title          panel title string
    ax             matplotlib Axes to draw on
    palette        optional dict mapping category to hex color
    legend_outside if True, places legend to the right of the panel
    point_size     dot size (smaller = less overlap for large datasets)
    """
    import numpy as np

    umap_coords = adata.obsm["X_umap"]
    categories  = adata.obs[color_key].astype(str)
    unique_cats = sorted(categories.unique())

    # Build a color map
    if palette and all(c in palette for c in unique_cats):
        color_map = palette
    else:
        # Use scanpy default palette cycling
        import scanpy as sc
        sc_palette = sc.pl.palettes.default_102
        color_map  = {
            cat: sc_palette[i % len(sc_palette)]
            for i, cat in enumerate(unique_cats)
        }

    colors = [color_map[c] for c in categories]

    # Shuffle points so no single category always renders on top
    idx = np.random.RandomState(42).permutation(len(umap_coords))
    ax.scatter(
        umap_coords[idx, 0],
        umap_coords[idx, 1],
        c=[colors[i] for i in idx],
        s=point_size,
        linewidths=0,
        rasterized=True   # rasterize for smaller PDF file size
    )

    ax.set_title(title, fontsize=11, fontweight="normal", pad=8)
    ax.set_xlabel("UMAP 1", fontsize=9)
    ax.set_ylabel("UMAP 2", fontsize=9)
    ax.tick_params(labelsize=7)

    # Remove top and right spines for a cleaner look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if legend_outside:
        # Build legend handles and place outside the axes to the right
        handles = [
            mpatches.Patch(color=color_map[cat], label=cat)
            for cat in unique_cats
        ]
        # ncol adjusts number of legend columns based on category count
        n_cols = 1 if len(unique_cats) <= 10 else 2
        ax.legend(
            handles=handles,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),  # place to the right of the axes
            fontsize=7,
            frameon=False,
            ncol=n_cols,
            handlelength=1.0,
            handleheight=0.8,
            borderpad=0.5,
            labelspacing=0.4,
        )


# Define custom palettes for biological meaning
CONDITION_PALETTE = {
    "healthy":   "#1D9E75",
    "cirrhotic": "#E24B4A",
}
COMPARTMENT_PALETTE = {
    "Macrophage":  "#E24B4A",
    "Mesenchymal": "#7F77DD",
    "Endothelial": "#D85A30",
    "Lymphoid":    "#378ADD",
    "DC":          "#BA7517",
    "Epithelial":  "#888780",
    "Other":       "#CCCCCC",
}

# Panel 1: cell type most categories so needs the most space
fig1, ax1 = plt.subplots(
    figsize=(9, 6),
    constrained_layout=False
)
plt.subplots_adjust(left=0.08, right=0.68, top=0.92, bottom=0.10)
plot_umap_panel(
    adata, "cell_type",
    title="Cell type annotation",
    ax=ax1,
    legend_outside=True,
    point_size=6
)
fig1.savefig("figures/annotation/umap_cell_type.pdf",
             dpi=150, bbox_inches="tight")
fig1.savefig("figures/annotation/umap_cell_type.png",
             dpi=150, bbox_inches="tight")
plt.close(fig1)
print("  Saved: figures/annotation/umap_cell_type.pdf")

# Panels 2 to 4: compartment, condition, etiology
# These have fewer categories and fit cleanly in a 1x3 row
fig2, axes = plt.subplots(
    1, 3,
    figsize=(18, 5.5),
    constrained_layout=False
)
plt.subplots_adjust(
    left=0.04,
    right=0.92,
    top=0.90,
    bottom=0.10,
    wspace=0.55    # horizontal space between panels
)

plot_umap_panel(
    adata, "compartment",
    title="Compartment",
    ax=axes[0],
    palette=COMPARTMENT_PALETTE,
    legend_outside=True,
    point_size=6
)
plot_umap_panel(
    adata, "condition",
    title="Condition",
    ax=axes[1],
    palette=CONDITION_PALETTE,
    legend_outside=True,
    point_size=6
)
plot_umap_panel(
    adata, "etiology",
    title="Etiology",
    ax=axes[2],
    legend_outside=True,
    point_size=6
)

fig2.suptitle(
    "GSE136103 UMAP: compartment, condition, and etiology",
    fontsize=12,
    y=0.98
)
fig2.savefig("figures/annotation/umap_compartment_condition_etiology.pdf",
             dpi=150, bbox_inches="tight")
fig2.savefig("figures/annotation/umap_compartment_condition_etiology.png",
             dpi=150, bbox_inches="tight")
plt.close(fig2)
print("  Saved: figures/annotation/umap_compartment_condition_etiology.pdf")

# Dotplot to validate marker gene expression matches expected cell types

sc.pl.dotplot(
    adata,
    var_names=MARKER_GENES,
    groupby="cell_type",
    use_raw=False,
    figsize=(18, 7),       # wider so gene labels have room
    save="_marker_dotplot.pdf"
)
print("  Saved: figures/annotation/dotplot_marker_dotplot.pdf")

print("Done. Next step: python 04_de_analysis.py")
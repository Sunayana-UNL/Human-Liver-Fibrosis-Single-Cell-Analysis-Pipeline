"""
01_qc_preprocessing.py
GSE136103 Human Liver Fibrosis scRNA-seq Analysis

What this script does, in order:
  1. Load all 10 samples and attach metadata
  2. Run Scrublet (doublet detection) per sample
  3. Compute QC metrics per cell (genes, UMI, MT%)
  4. Generate UMAP before filtering to show raw data quality
  5. Apply QC filters (genes, UMI, MT%, doublets)
  6. Generate UMAP after filtering to show the effect
  7. Normalize using three methods and store all in layers:
       Layer lognorm  : normalize_total CPM plus log1p  (default)
       Layer scran    : scran pooling normalization     (more accurate)
       Layer pearson  : analytic Pearson residuals      (variance stabilizing)
  8. Select highly variable genes
  9. Save QC summary plots and final AnnData

Note on batch correction:
  Batch correction via Harmony is intentionally NOT applied here.
  It operates on PCA embeddings not raw counts so it belongs in
  02_integration_clustering.py. Applying it to counts would distort
  the raw signal we need for differential expression.
"""

import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
import scrublet as scr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

sc.settings.verbosity = 2
sc.settings.figdir = "figures/qc/"
os.makedirs("figures/qc", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# Sample metadata loaded from CSV produced by 00_download_data.sh
# This replaces the old hardcoded dictionary.
# The CSV is generated automatically from the GEO SOFT file during download
# so this script never needs to be edited when samples change.
METADATA_CSV = "data/raw/sample_metadata.csv"

def load_sample_metadata(csv_path=METADATA_CSV):
    """
    Read sample_metadata.csv and return a dict keyed by GSM ID.
    Falls back to hardcoded values if the CSV does not exist yet
    for example if running preprocessing before the download script finished.
    """
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col="gsm")
        samples = df.to_dict(orient="index")
        print(f"  Loaded metadata for {len(samples)} samples from {csv_path}")
        return samples
    else:
        print(f"  WARNING: {csv_path} not found. Using hardcoded fallback metadata.")
        print(f"  Run 00_download_data.sh first to generate this file.")
        return {
            "GSM4041161": {"condition": "healthy",   "etiology": "none", "donor": "H1"},
            "GSM4041162": {"condition": "healthy",   "etiology": "none", "donor": "H2"},
            "GSM4041163": {"condition": "healthy",   "etiology": "none", "donor": "H3"},
            "GSM4041164": {"condition": "healthy",   "etiology": "none", "donor": "H4"},
            "GSM4041165": {"condition": "healthy",   "etiology": "none", "donor": "H5"},
            "GSM4041166": {"condition": "cirrhotic", "etiology": "ALD",  "donor": "C1"},
            "GSM4041167": {"condition": "cirrhotic", "etiology": "NASH", "donor": "C2"},
            "GSM4041168": {"condition": "cirrhotic", "etiology": "PBC",  "donor": "C3"},
            "GSM4041169": {"condition": "cirrhotic", "etiology": "PSC",  "donor": "C4"},
            "GSM4041170": {"condition": "cirrhotic", "etiology": "ALD",  "donor": "C5"},
        }

SAMPLES = load_sample_metadata()

# QC thresholds
# Liver NPCs have elevated metabolic activity vs typical tissues.
# A strict 10 to 15 percent MT cutoff would discard real activated macrophages and HSCs.
# 25 percent is well supported by reanalysis papers for this specific dataset.
MIN_GENES       = 300    # below this is likely an empty droplet
MAX_GENES       = 6000   # above this is likely a doublet
MIN_UMI         = 500    # minimum total RNA molecules detected
MAX_MT_PCT      = 25.0   # maximum percent mitochondrial reads
SCRUBLET_THRESH = 0.25   # doublet score threshold between 0 and 1
N_HVG           = 3000   # highly variable genes to select


# SECTION 1
# Loading and per sample QC

def find_file(folder, suffixes):
    """
    Find a file in a folder by matching the END of the filename.
    For example suffix _matrix.mtx will match
    GSM4041161_cirrhotic1_cd45+_matrix.mtx regardless of what comes before.
    Also matches compressed versions ending in .gz automatically.
    Returns the full path of the first match found.
    Raises a descriptive error listing all files present if nothing matches.
    """
    files_in_folder = os.listdir(folder)
    for fname in files_in_folder:
        fname_lower = fname.lower()
        for suffix in suffixes:
            # Checking for plain suffix and also the gz compressed version
            if fname_lower.endswith(suffix.lower()) or                fname_lower.endswith(suffix.lower() + ".gz"):
                return os.path.join(folder, fname)
    raise FileNotFoundError(
        f"Could not find a file ending in {suffixes} inside {folder}"
        f"Files present in that folder: {files_in_folder}"
    )


def extract_label_from_filename(filepath):
    """
    Pull out the condition label embedded in the filename.
    For example GSM4041161_cirrhotic1_cd45+_matrix.mtx
    returns cirrhotic1_cd45+ as the embedded label.
    This is the text between the GSM accession and the file type suffix.
    It is stored in adata.obs for reference but the authoritative
    condition label still comes from sample_metadata.csv.
    """
    fname  = os.path.basename(filepath)
    # Remove known suffixes to isolate the middle label part
    for suffix in ["_matrix.mtx.gz", "_matrix.mtx",
                   "_barcodes.tsv.gz", "_barcodes.tsv",
                   "_genes.tsv.gz", "_genes.tsv",
                   "_features.tsv.gz", "_features.tsv"]:
        if fname.lower().endswith(suffix.lower()):
            fname = fname[: -len(suffix)]
            break
    # Remove the leading GSM accession number
    parts = fname.split("_", 1)
    label = parts[1] if len(parts) > 1 else fname
    return label


def load_sample(gsm_id, data_root="data/raw"):
    """
    Loading one GEO sample by searching the folder for files ending in
    the expected suffixes regardless of any prefix in the filename.

    Works with filenames like:
      GSM4041161_cirrhotic1_cd45+_matrix.mtx
      GSM4041161_cirrhotic1_cd45+_barcodes.tsv
      GSM4041161_cirrhotic1_cd45+_genes.tsv

    Also works with compressed versions ending in .gz and with
    features.tsv instead of genes.tsv for newer datasets.
    """
    import scipy.io
    import gzip

    sample_dir = os.path.join(data_root, gsm_id)

    if not os.path.isdir(sample_dir):
        raise FileNotFoundError(
            f"Sample folder not found: {sample_dir}"
            f"Make sure 00_download_data.sh completed successfully."
        )

    # Searching by suffix so the GSM prefix and condition label are ignored
    matrix_path   = find_file(sample_dir, ["_matrix.mtx"])
    barcodes_path = find_file(sample_dir, ["_barcodes.tsv"])
    genes_path    = find_file(sample_dir, ["_genes.tsv", "_features.tsv"])

    # Pulling the embedded label out of the filename for reference
    file_label = extract_label_from_filename(matrix_path)

    print(f"    matrix:      {os.path.basename(matrix_path)}")
    print(f"    barcodes:    {os.path.basename(barcodes_path)}")
    print(f"    genes:       {os.path.basename(genes_path)}")
    print(f"    file label:  {file_label}")

    # Read the matrix which is in Matrix Market sparse format
    # Transpose so rows are cells and columns are genes
    if matrix_path.endswith(".gz"):
        with gzip.open(matrix_path, "rb") as f:
            matrix = scipy.io.mmread(f).T.tocsr()
    else:
        matrix = scipy.io.mmread(matrix_path).T.tocsr()

    # Read barcodes one per line
    opener = gzip.open if barcodes_path.endswith(".gz") else open
    with opener(barcodes_path, "rt") as f:
        barcodes = [line.strip() for line in f if line.strip()]

    # Read gene file
    # Column 1 is Ensembl ID and column 2 is gene symbol in both
    # the older genes.tsv two column format and newer features.tsv three column format
    opener = gzip.open if genes_path.endswith(".gz") else open
    with opener(genes_path, "rt") as f:
        gene_rows = [line.strip().split("	") for line in f if line.strip()]

    gene_ids     = [row[0] for row in gene_rows]
    gene_symbols = [row[1] for row in gene_rows]

    # Validate that dimensions are consistent before building AnnData
    n_cells = matrix.shape[0]
    n_genes = matrix.shape[1]
    if n_cells != len(barcodes):
        raise ValueError(
            f"{gsm_id}: matrix has {n_cells} cells but barcodes file "
            f"has {len(barcodes)} entries. Files may be mismatched."
        )
    if n_genes != len(gene_symbols):
        raise ValueError(
            f"{gsm_id}: matrix has {n_genes} genes but genes file "
            f"has {len(gene_symbols)} entries. Files may be mismatched."
        )

    # Build the AnnData object
    adata = ad.AnnData(X=matrix)
    adata.obs_names        = barcodes
    adata.var_names        = gene_symbols
    adata.var["gene_ids"]  = gene_ids

    adata.var_names_make_unique()

    # Attach metadata from sample_metadata.csv as the authoritative source
    meta = SAMPLES[gsm_id]
    adata.obs["gsm"]        = gsm_id
    adata.obs["condition"]  = meta["condition"]
    adata.obs["etiology"]   = meta["etiology"]
    adata.obs["donor"]      = meta["donor"]
    # Also store the filename label as a sanity check column
    # so you can verify it matches the condition from metadata
    adata.obs["file_label"] = file_label

    # Prefix barcodes with donor ID so they stay unique after concatenation
    adata.obs.index = [f"{meta['donor']}_{bc}" for bc in adata.obs.index]
    return adata


def run_scrublet(adata):
    """
    Detect doublets per sample.
    Scrublet must be run per sample BEFORE concatenation.
    Running it on the merged object would mix donor specific doublet
    distributions and produce unreliable scores.
    """
    scrub = scr.Scrublet(adata.X, expected_doublet_rate=0.06)
    scores, predicted = scrub.scrub_doublets(
        min_counts=2,
        min_cells=3,
        n_prin_comps=30,
    )
    adata.obs["doublet_score"]     = scores
    adata.obs["predicted_doublet"] = predicted
    print(f"    Scrublet doublet rate: {predicted.mean()*100:.1f}%  "
          f"({predicted.sum()} cells flagged)")
    return adata


def compute_qc_metrics(adata):
    """Add per cell QC columns: n_genes, n_counts, pct_mt, pct_ribo."""
    # Mitochondrial genes start with MT in humans
    adata.var["mt"]   = adata.var_names.str.startswith("MT-")
    # Ribosomal genes start with RPS for small subunit or RPL for large subunit
    adata.var["ribo"] = adata.var_names.str.match(r"^RP[SL]\d")

    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt", "ribo"],
        percent_top=None,
        inplace=True,
    )
    return adata


def qc_filter(adata):
    """Apply QC thresholds. Returns filtered copy."""
    n_before = adata.n_obs
    adata = adata[
        (adata.obs.n_genes_by_counts  >= MIN_GENES)  &
        (adata.obs.n_genes_by_counts  <= MAX_GENES)  &
        (adata.obs.total_counts       >= MIN_UMI)    &
        (adata.obs.pct_counts_mt      <= MAX_MT_PCT) &
        (~adata.obs.predicted_doublet)
    ].copy()
    n_after  = adata.n_obs
    pct_kept = n_after / n_before * 100
    print(f"    Cells kept: {n_after:,}/{n_before:,}  ({pct_kept:.1f}%)")
    return adata


# SECTION 2
# UMAP before and after QC

def quick_umap(adata, label):
    """
    Compute a fast PCA and UMAP on raw log counts for visualization only.
    This is NOT the final UMAP used for analysis which is in script 02.
    Purpose is to show structure and quality of cells at this processing stage.
    """
    print(f"  Computing quick UMAP ({label})...")
    a = adata.copy()

    # Minimal normalization just for the visualization embedding
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=2000, flavor="seurat_v3")
    sc.pp.pca(a, n_comps=30, use_highly_variable=True)
    sc.pp.neighbors(a, n_neighbors=15, n_pcs=20)
    sc.tl.umap(a, min_dist=0.3, random_state=42)

    return a


def plot_before_after_umap(adata_raw, adata_filtered):
    """
    Side by side UMAP showing before QC colored by MT percent and doublet score
    and after QC colored by condition and donor.
    Saved as figures/qc/umap_before_after_qc.pdf
    """
    print("  Plotting before and after UMAP...")

    raw_umap      = quick_umap(adata_raw,      "before QC")
    filtered_umap = quick_umap(adata_filtered, "after QC")

    fig = plt.figure(figsize=(20, 12))
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)

    # Row 1 shows the data before QC filtering
    fig.text(0.02, 0.97, "BEFORE QC", fontsize=13, fontweight="bold",
             color="#A32D2D", va="top")

    # Plot 1 shows MT percent where high values indicate dying cells
    ax1 = fig.add_subplot(gs[0, 0])
    sc.pl.umap(raw_umap, color="pct_counts_mt", ax=ax1, show=False,
               title="MT percent (high means dying cell)", vmax=50,
               color_map="Reds", frameon=False)

    # Plot 2 shows doublet score where high values indicate two cells captured together
    ax2 = fig.add_subplot(gs[0, 1])
    sc.pl.umap(raw_umap, color="doublet_score", ax=ax2, show=False,
               title="Doublet score (high means two cells)", vmax=0.5,
               color_map="Oranges", frameon=False)

    # Plot 3 shows genes per cell where low values indicate empty droplets
    ax3 = fig.add_subplot(gs[0, 2])
    sc.pl.umap(raw_umap, color="n_genes_by_counts", ax=ax3, show=False,
               title="Genes per cell (low means empty droplet)", vmax=5000,
               color_map="Blues", frameon=False)

    # Plot 4 shows donor identity to reveal uncorrected batch structure
    ax4 = fig.add_subplot(gs[0, 3])
    sc.pl.umap(raw_umap, color="donor", ax=ax4, show=False,
               title="Donor (batch structure before Harmony)", frameon=False)

    # Row 2 shows the data after QC filtering
    fig.text(0.02, 0.50, "AFTER QC", fontsize=13, fontweight="bold",
             color="#0F6E56", va="top")

    # Plot 5 shows condition to check healthy vs cirrhotic separation
    ax5 = fig.add_subplot(gs[1, 0])
    sc.pl.umap(filtered_umap, color="condition", ax=ax5, show=False,
               title="Condition healthy vs cirrhotic", frameon=False,
               palette={"healthy": "#1D9E75", "cirrhotic": "#E24B4A"})

    # Plot 6 shows donor after filtering to confirm all donors are still present
    ax6 = fig.add_subplot(gs[1, 1])
    sc.pl.umap(filtered_umap, color="donor", ax=ax6, show=False,
               title="Donor (batch before Harmony correction)", frameon=False)

    # Plot 7 shows MT percent after filtering which should now be below 25 percent
    ax7 = fig.add_subplot(gs[1, 2])
    sc.pl.umap(filtered_umap, color="pct_counts_mt", ax=ax7, show=False,
               title="MT percent after filtering should be under 25", vmax=30,
               color_map="Reds", frameon=False)

    # Plot 8 shows genes per cell after filtering which should now be in a healthy range
    ax8 = fig.add_subplot(gs[1, 3])
    sc.pl.umap(filtered_umap, color="n_genes_by_counts", ax=ax8, show=False,
               title="Genes per cell after filtering", vmax=5000,
               color_map="Blues", frameon=False)

    plt.suptitle(
        "GSE136103 UMAP before and after QC filtering\n"
        "Note: batch correction with Harmony is applied in script 02 not here",
        fontsize=14, y=1.01
    )

    fig.savefig("figures/qc/umap_before_after_qc.pdf", bbox_inches="tight", dpi=150)
    fig.savefig("figures/qc/umap_before_after_qc.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("    Saved: figures/qc/umap_before_after_qc.pdf and .png")


# SECTION 3
# Normalization using three methods all stored as layers
#
# Why three methods?
#
# normalize_total plus log1p stored as lognorm
#   The field default. Fast, simple and interpretable. logFC values are intuitive.
#   Weakness: assumes all cells had the same amount of RNA before sequencing
#   which is the equal library size assumption and is often violated.
#
# scran pooling normalization stored as scran
#   Groups similar cells into pools, estimates size factors from pools,
#   then back calculates per cell size factors. Handles unequal RNA content
#   much better and is more accurate for DE analysis.
#   Weakness: slower and requires rpy2 plus the scran R package.
#
# Analytic Pearson residuals stored as pearson
#   A newer approach from Lause et al 2021. Models raw counts with a negative
#   binomial and computes residuals which are variance stabilized and
#   do not require a separate HVG step because residuals directly encode variability.
#   Strength: best for PCA and embedding. Weakness: not suited for fold change DE.
#
# Which one to use downstream:
#   For clustering, UMAP and PCA use Pearson residuals via layer pearson
#   For DESeq2 pseudobulk DE always use raw counts via layer counts
#   For Wilcoxon DE and plotting use lognorm which is the default adata.X
#   For comparing with published results use lognorm since most papers use this

def normalize_all_methods(adata, n_top_genes=N_HVG):
    """
    Apply three normalization methods and store each as a layer.
    Sets adata.X to lognorm which is the safe interpretable default.
    """

    # Always preserve raw integer counts because DESeq2 requires them
    adata.layers["counts"] = adata.X.copy()
    print("  Stored raw counts in adata.layers['counts']")

    # Method 1: normalize_total plus log1p stored as lognorm
    print("  Normalizing: Method 1 normalize_total plus log1p...")
    sc.pp.normalize_total(adata, target_sum=1e4, layer=None)
    sc.pp.log1p(adata)
    adata.layers["lognorm"] = adata.X.copy()
    # adata.X is now lognorm which is the default used by scanpy plots
    print("    Stored in adata.layers['lognorm'] and adata.X")

    # Method 2: scran pooling normalization
    print("  Normalizing: Method 2 scran pooling...")
    try:
        import rpy2.robjects as ro
        import rpy2.robjects.numpy2ri as numpy2ri
        numpy2ri.activate()

        # scran needs integer counts not already normalized values
        counts_mat = adata.layers["counts"]
        if hasattr(counts_mat, "toarray"):
            counts_mat = counts_mat.toarray()

        # scran expects genes as rows and cells as columns so we transpose
        ro.globalenv["counts_matrix"] = counts_mat.T

        ro.r("""
            suppressPackageStartupMessages(library(scran))
            sce <- SingleCellExperiment::SingleCellExperiment(
                assays = list(counts = counts_matrix)
            )
            # Quick pre clustering helps scran estimate size factors more accurately
            clusters     <- scran::quickCluster(sce, min.size=50)
            sce          <- scran::computeSumFactors(sce, clusters=clusters)
            size_factors <- scran::sizeFactors(sce)
        """)

        size_factors = np.array(ro.globalenv["size_factors"])

        # Divide each cell counts by its size factor then log transform
        from scipy.sparse import issparse
        raw = adata.layers["counts"]
        if issparse(raw):
            raw = raw.toarray()
        scran_norm = raw / size_factors[:, None]
        scran_norm = np.log1p(scran_norm)
        adata.layers["scran"] = scran_norm
        print("    Stored in adata.layers['scran']")

    except ImportError:
        print("    rpy2 not available so skipping scran.")
        print("    Install with: pip install rpy2 (requires R in PATH)")

    # Method 3: Analytic Pearson residuals
    print("  Normalizing: Method 3 analytic Pearson residuals...")
    try:
        # Work on a fresh copy so we do not corrupt adata.X
        adata_pr = adata.copy()
        # Reset X to raw counts for Pearson computation
        from scipy.sparse import issparse
        adata_pr.X = adata_pr.layers["counts"].copy()
        sc.experimental.pp.normalize_pearson_residuals(adata_pr)
        adata.layers["pearson"] = adata_pr.X.copy()
        print("    Stored in adata.layers['pearson']")
    except AttributeError:
        # Older scanpy versions do not have normalize_pearson_residuals
        # so we use a shifted logarithm as a variance stabilizing fallback
        print("    normalize_pearson_residuals not available in this scanpy version.")
        print("    Using shifted logarithm as fallback.")
        from scipy.sparse import issparse
        raw = adata.layers["counts"]
        if issparse(raw):
            raw = raw.toarray().astype(float)
        size_factors = raw.sum(axis=1, keepdims=True) / 1e4
        shifted = np.log1p(raw / size_factors)
        adata.layers["pearson"] = shifted
        print("    Stored in adata.layers['pearson'] using shifted log fallback")

    return adata


# SECTION 4
# HVG selection and cell cycle scoring

def select_hvg_and_cell_cycle(adata):
    """Select highly variable genes and score cell cycle phase."""

    # HVG selection runs on lognorm which is adata.X using the seurat v3 method
    # seurat v3 uses raw counts internally for variance estimation
    # batch_key ensures we find genes variable across donors
    # not just variable because of one outlier donor
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=N_HVG,
        flavor="seurat_v3",
        layer="counts",
        batch_key="donor"
    )
    print(f"  HVGs selected: {adata.var.highly_variable.sum()}")

    # Cell cycle scoring using the Regev lab canonical gene lists
    cell_cycle_url = (
        "https://raw.githubusercontent.com/scverse/scanpy_usage/"
        "master/180209_cell_cycle/data/regev_lab_cell_cycle_genes.txt"
    )
    try:
        cc_genes     = pd.read_csv(cell_cycle_url, header=None).iloc[:, 0].tolist()
        s_genes_cc   = [g for g in cc_genes[:43] if g in adata.var_names]
        g2m_genes_cc = [g for g in cc_genes[43:] if g in adata.var_names]
        sc.tl.score_genes_cell_cycle(
            adata, s_genes=s_genes_cc, g2m_genes=g2m_genes_cc
        )
        print(f"  Cell cycle scored with {len(s_genes_cc)} S genes "
              f"and {len(g2m_genes_cc)} G2M genes")
    except Exception as e:
        print(f"  Cell cycle scoring skipped due to: {e}")

    return adata


# SECTION 5
# QC summary plots

def plot_qc_summary(adata_raw, adata_filtered):
    """
    Violin plots comparing QC metrics before and after filtering per donor.
    Saved as figures/qc/qc_summary.pdf
    """
    print("  Plotting QC summary...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    metrics = [
        ("n_genes_by_counts", "Genes per cell", 6500),
        ("total_counts",      "UMI per cell",   30000),
        ("pct_counts_mt",     "MT percent",     50),
    ]

    for col, (metric, label, ymax) in enumerate(metrics):
        for row, (adata, title_prefix, color) in enumerate([
            (adata_raw,      "Before QC", "#E24B4A"),
            (adata_filtered, "After QC",  "#1D9E75"),
        ]):
            ax = axes[row, col]
            donors = sorted(adata.obs.donor.unique())
            data_by_donor = [
                adata.obs.loc[adata.obs.donor == d, metric].values
                for d in donors
            ]
            parts = ax.violinplot(data_by_donor, showmedians=True)
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.6)
            ax.set_xticks(range(1, len(donors) + 1))
            ax.set_xticklabels(donors, fontsize=8)
            ax.set_ylim(0, ymax)
            ax.set_title(f"{title_prefix}: {label}", fontsize=10)
            ax.set_xlabel("Donor")
            ax.set_ylabel(label)

            # Draw threshold reference lines so filters are visible on the plot
            if metric == "n_genes_by_counts":
                ax.axhline(MIN_GENES, color="gray", ls="--", lw=0.8,
                           label=f"min {MIN_GENES}")
                ax.axhline(MAX_GENES, color="gray", ls=":",  lw=0.8,
                           label=f"max {MAX_GENES}")
            elif metric == "pct_counts_mt":
                ax.axhline(MAX_MT_PCT, color="gray", ls="--", lw=0.8,
                           label=f"max {MAX_MT_PCT} percent")
            ax.legend(fontsize=7)

    plt.suptitle("QC metrics per donor before and after filtering", fontsize=13)
    plt.tight_layout()
    fig.savefig("figures/qc/qc_summary.pdf", bbox_inches="tight", dpi=150)
    fig.savefig("figures/qc/qc_summary.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("    Saved: figures/qc/qc_summary.pdf and .png")


def plot_normalization_comparison(adata):
    """
    Scatter plots comparing the three normalization methods.
    Shows mean vs variance per gene to validate normalization worked.
    After good normalization variance should not scale with mean.
    Saved as figures/qc/normalization_comparison.pdf
    """
    print("  Plotting normalization comparison...")
    from scipy.sparse import issparse

    available_layers = [l for l in ["lognorm", "scran", "pearson"]
                        if l in adata.layers]
    n    = len(available_layers)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    colors = {"lognorm": "#378ADD", "scran": "#1D9E75", "pearson": "#7F77DD"}

    for ax, layer in zip(axes, available_layers):
        mat = adata.layers[layer]
        if issparse(mat):
            mat = mat.toarray()
        gene_means = mat.mean(axis=0)
        gene_vars  = mat.var(axis=0)

        ax.scatter(gene_means, gene_vars, s=1, alpha=0.3,
                   color=colors.get(layer, "gray"), rasterized=True)
        ax.set_xlabel("Mean expression")
        ax.set_ylabel("Variance")
        ax.set_title(f"Mean vs variance for {layer}", fontsize=11)
        # Under perfect Poisson noise variance equals mean so we draw that line as reference
        lim = max(gene_means.max(), gene_vars.max())
        ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="Poisson reference var equals mean")
        ax.legend(fontsize=8)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

    plt.suptitle(
        "Normalization method comparison mean vs variance per gene\n"
        "After normalization variance should not scale with mean",
        fontsize=12
    )
    plt.tight_layout()
    fig.savefig("figures/qc/normalization_comparison.pdf", bbox_inches="tight", dpi=150)
    fig.savefig("figures/qc/normalization_comparison.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("    Saved: figures/qc/normalization_comparison.pdf and .png")


# MAIN

if __name__ == "__main__":

    # Step 1: Load and QC filter each sample individually
    print("=" * 60)
    print("STEP 1: Loading and QC filtering samples")
    print("=" * 60)

    adatas_raw      = []   # unfiltered copies kept for before and after UMAP
    adatas_filtered = []   # filtered copies used for downstream analysis

    for gsm in SAMPLES:
        donor = SAMPLES[gsm]["donor"]
        print(f"\n  [{gsm}] donor={donor} condition={SAMPLES[gsm]['condition']}")
        a = load_sample(gsm)
        a = run_scrublet(a)
        a = compute_qc_metrics(a)
        adatas_raw.append(a.copy())
        a = qc_filter(a)
        adatas_filtered.append(a)

    # Step 2: Concatenate all samples into one object
    print("\n" + "=" * 60)
    print("STEP 2: Concatenating samples")
    print("=" * 60)

    adata_raw = ad.concat(
        adatas_raw, join="outer",
        label="sample", keys=list(SAMPLES.keys())
    )
    adata_raw.obs_names_make_unique()
    print(f"  Raw unfiltered: {adata_raw.n_obs:,} cells")

    adata_all = ad.concat(
        adatas_filtered, join="outer",
        label="sample", keys=list(SAMPLES.keys())
    )
    adata_all.obs_names_make_unique()
    print(f"  Filtered: {adata_all.n_obs:,} cells  "
          f"({adata_all.n_obs/adata_raw.n_obs*100:.1f}% retained)")

    # Step 3: Generate UMAP before and after QC for visual comparison
    print("\n" + "=" * 60)
    print("STEP 3: UMAP before and after QC")
    print("=" * 60)
    print("  NOTE: This is a quick visualization UMAP only.")
    print("  The analysis UMAP with Harmony is produced in script 02.")
    plot_before_after_umap(adata_raw, adata_all)

    # Step 4: QC violin plots per donor
    print("\n" + "=" * 60)
    print("STEP 4: QC summary plots")
    print("=" * 60)
    plot_qc_summary(adata_raw, adata_all)

    # Step 5: Normalize using all three methods
    print("\n" + "=" * 60)
    print("STEP 5: Normalization using three methods")
    print("=" * 60)
    adata_all = normalize_all_methods(adata_all)

    # Step 6: Plot normalization comparison
    plot_normalization_comparison(adata_all)

    # Step 7: Select highly variable genes and score cell cycle
    print("\n" + "=" * 60)
    print("STEP 6: HVG selection and cell cycle scoring")
    print("=" * 60)
    adata_all = select_hvg_and_cell_cycle(adata_all)

    # Step 8: Print summary of available layers
    print("\n" + "=" * 60)
    print("SUMMARY: AnnData layers available for downstream analysis")
    print("=" * 60)
    for layer_name, use_for in [
        ("counts",  "DESeq2 pseudobulk DE always use raw counts for this"),
        ("lognorm", "Wilcoxon DE, plotting and most scanpy functions"),
        ("scran",   "More accurate DE with size factor aware normalization"),
        ("pearson", "PCA, UMAP and clustering with variance stabilization"),
    ]:
        if layer_name in adata_all.layers:
            print(f"  adata.layers['{layer_name}'] available for {use_for}")
        else:
            print(f"  adata.layers['{layer_name}'] NOT AVAILABLE see log above")

    print(f"\n  adata.X is set to lognorm which is the default for scanpy functions")
    print(f"  adata.obs columns: {list(adata_all.obs.columns)}")
    print(f"  adata.var columns: {list(adata_all.var.columns)}")

    # Step 9: Save the final processed object
    print("\n" + "=" * 60)
    print("STEP 7: Saving")
    print("=" * 60)
    out_path = "data/processed/GSE136103_qc_normalized.h5ad"
    adata_all.write(out_path)
    print(f"  Saved: {out_path}")
    print(f"  File contains {adata_all.n_obs:,} cells x {adata_all.n_vars:,} genes")
    print(f"  Figures saved to: figures/qc/")
    print("\nDone. Next step: python 02_integration_clustering.py")

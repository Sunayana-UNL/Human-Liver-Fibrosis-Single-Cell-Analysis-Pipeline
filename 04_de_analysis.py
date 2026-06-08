"""
04_de_analysis.py
Differential expression: cirrhosis vs healthy.
  Method 1: Wilcoxon rank sum per cell type (fast, liberal)
  Method 2: DESeq2 pseudobulk via R (rigorous, donor aware)

Input:  data/processed/GSE136103_annotated.h5ad
Output: results/de/

"""

import scanpy as sc
import pandas as pd
import numpy as np
import subprocess
import os

sc.settings.verbosity = 2
sc.settings.figdir = "figures/de/"
os.makedirs("figures/de", exist_ok=True)
os.makedirs("results/de", exist_ok=True)

adata = sc.read("data/processed/GSE136103_annotated.h5ad")
print(f"Loaded: {adata.n_obs:,} cells")
print(f"Layers available: {list(adata.layers.keys())}")

# Confirming counts layer exists before proceeding
# This is critical for the pseudobulk export
if "counts" not in adata.layers:
    raise ValueError(
        "adata.layers['counts'] not found. "
        "Make sure 01_qc_preprocessing.py completed successfully. "
        "The counts layer is required for DESeq2 pseudobulk export."
    )

# Cell types to run DE on
TARGET_CELLTYPES = [
    "SAMac_TREM2",
    "Kupffer_cells",
    "Tissue_monocytes",
    "SAMes_myofibroblast",
    "Quiescent_HSC",
    "SAEndo_ACKR1",
    "LSEC",
]

all_results = []

for ct in TARGET_CELLTYPES:
    sub = adata[adata.obs.cell_type == ct].copy()
    if sub.n_obs < 50:
        print(f"Skipping {ct}: only {sub.n_obs} cells (need at least 50)")
        continue

    n_healthy    = (sub.obs.condition == "healthy").sum()
    n_cirrhotic  = (sub.obs.condition == "cirrhotic").sum()
    print(f"\n{ct}: {sub.n_obs} cells ({n_healthy} healthy, {n_cirrhotic} cirrhotic)")

    if n_healthy == 0 or n_cirrhotic == 0:
        print(f"  Skipping: need both conditions present")
        continue

    # Wilcoxon test using lognorm values from adata.X
    
    
    sc.tl.rank_genes_groups(
        sub,
        groupby="condition",
        groups=["cirrhotic"],
        reference="healthy",
        method="wilcoxon",
        use_raw=False,
        layer="lognorm",
        n_genes=sub.n_vars
    )
    res = sc.get.rank_genes_groups_df(sub, group="cirrhotic")
    res["cell_type"] = ct
    res["method"]    = "wilcoxon"
    all_results.append(res)
    print(f"  Wilcoxon complete: {(res.pvals_adj < 0.05).sum()} significant genes")

    # Exporting pseudobulk counts for DESeq2
    # We use adata.layers["counts"] here not adata.X
    # because DESeq2 requires raw integer counts
    pb_dir = f"results/de/pseudobulk_{ct}"
    os.makedirs(pb_dir, exist_ok=True)

    from scipy.sparse import issparse
    raw_counts = sub.layers["counts"]
    if issparse(raw_counts):
        raw_counts = raw_counts.toarray()

    counts_df = pd.DataFrame(
        raw_counts,
        index=sub.obs_names,
        columns=sub.var_names
    )
    counts_df["donor"]     = sub.obs["donor"].values
    counts_df["condition"] = sub.obs["condition"].values

    # Aggregate to pseudobulk by summing all cells from the same donor
    pb   = counts_df.groupby(["donor", "condition"]).sum().T
    meta = (
        counts_df[["donor", "condition"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    pb.to_csv(f"{pb_dir}/pseudobulk_counts.csv")
    meta.to_csv(f"{pb_dir}/metadata.csv", index=False)
    print(f"  Pseudobulk exported: {pb.shape[1]} donors x {pb.shape[0]} genes")

# Save all Wilcoxon results
if all_results:
    wilcoxon_df = pd.concat(all_results, ignore_index=True)
    wilcoxon_df.to_csv("results/de/wilcoxon_all_celltypes.csv", index=False)
    print(f"\nWilcoxon results saved: {len(wilcoxon_df)} gene tests across all cell types")
else:
    print("\nWARNING: No Wilcoxon results produced. Check cell type labels.")

# Run DESeq2 pseudobulk via R
print("\nRunning DESeq2 pseudobulk analysis via R...")
result = subprocess.run(
    ["Rscript", "R/deseq2_pseudobulk.R"],
    capture_output=True, text=True
)
print(result.stdout)
if result.returncode != 0:
    print(f"DESeq2 stderr: {result.stderr}")

# Dotplot of top DE genes for SAMacs
print("\nGenerating SAMac DE visualization...")
sub_samac = adata[adata.obs.cell_type == "SAMac_TREM2"].copy()
if sub_samac.n_obs >= 50:
    sc.tl.rank_genes_groups(
        sub_samac,
        groupby="condition",
        groups=["cirrhotic"],
        reference="healthy",
        method="wilcoxon",
        use_raw=False,
        layer="lognorm"
    )
    sc.pl.rank_genes_groups_dotplot(
        sub_samac,
        n_genes=15,
        groupby="condition",
        use_raw=False,
        save="_SAMac_top_DE.pdf"
    )
    top_genes = sc.get.rank_genes_groups_df(sub_samac, group="cirrhotic")
    top_genes.to_csv("results/de/SAMac_cirrhotic_vs_healthy.csv", index=False)
    print("  SAMac DE results saved")

print("\nDone. Next steps:")
print("  Rscript R/deseq2_pseudobulk.R    (if not already run above)")
print("  Rscript R/gsea_clusterProfiler.R")
print("  python 05_pathway_analysis.py")

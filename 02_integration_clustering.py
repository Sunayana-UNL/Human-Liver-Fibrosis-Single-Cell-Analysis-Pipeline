"""
02_integration_clustering.py
Harmony batch correction, UMAP embedding, Leiden clustering.

Input:  data/processed/GSE136103_qc_normalized.h5ad
Output: data/processed/GSE136103_integrated.h5ad
"""

import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import os

sc.settings.verbosity = 2
sc.settings.figdir = "figures/clustering/"
os.makedirs("figures/clustering", exist_ok=True)

adata = sc.read("data/processed/GSE136103_qc_normalized.h5ad")
print(f"Loaded: {adata.n_obs:,} cells x {adata.n_vars:,} genes")
print(f"Layers available: {list(adata.layers.keys())}")
print(f"obs columns: {list(adata.obs.columns)}")

# Setting adata.X to Pearson residuals for PCA that was saved eralier
# Otherwise fall back to lognorm
if "pearson" in adata.layers:
    print("Setting adata.X to Pearson residuals for PCA")
    pearson = adata.layers["pearson"]
    adata.X = pearson.toarray() if sp.issparse(pearson) else np.array(pearson)
else:
    print("Pearson layer not found. Using lognorm for PCA")

# PCA
print("Running PCA...")
sc.tl.pca(adata, n_comps=50, use_highly_variable=True)
print(f"PCA complete. X_pca shape: {adata.obsm['X_pca'].shape}")
# Expected shape is (n_cells, 50) for example (34095, 50)
# If you see (50,) or (50, 34095) something went wrong above
sc.pl.pca_variance_ratio(adata, n_pcs=50, save="_variance_ratio.pdf")

# Harmony integration

print("Running Harmony integration by donor (calling harmonypy directly)...")
import harmonypy as hm

# harmonypy.run_harmony expects:
#   data_mat  shape (n_cells, n_components)  which is our X_pca
#   meta_data a DataFrame with one row per cell
#   vars_use  the column name in meta_data to correct for
pca_embedding = adata.obsm["X_pca"]
print(f"  Input to Harmony: {pca_embedding.shape}")
# Confirm shape is (n_cells, n_components) before passing to Harmony
assert pca_embedding.shape[0] == adata.n_obs, (
    f"PCA shape {pca_embedding.shape} does not match n_obs {adata.n_obs}. "
    f"Expected ({adata.n_obs}, 50)."
)

harmony_out = hm.run_harmony(
    pca_embedding,
    adata.obs,
    vars_use="donor",
    max_iter_harmony=20,
    random_state=42
)

# harmony_out.Z_corr has shape (n_components, n_cells)

corrected = harmony_out.Z_corr
print(f"  Harmony output shape (after transpose): {corrected.shape}")
assert corrected.shape == pca_embedding.shape, (
    f"Harmony output shape {corrected.shape} does not match "
    f"expected {pca_embedding.shape}"
)
adata.obsm["X_pca_harmony"] = corrected
print("Harmony complete.")

# kNN graph on corrected embedding
print("Building kNN graph...")
sc.pp.neighbors(adata, use_rep="X_pca_harmony", n_neighbors=30, n_pcs=30)

# UMAP
print("Computing UMAP...")
sc.tl.umap(adata, min_dist=0.3, spread=1.0, random_state=42)

# Leiden clustering at three resolutions
print("Running Leiden clustering...")
sc.tl.leiden(adata, resolution=0.5, random_state=42, key_added="leiden_0.5")
sc.tl.leiden(adata, resolution=0.8, random_state=42, key_added="leiden_0.8")
sc.tl.leiden(adata, resolution=1.2, random_state=42, key_added="leiden_1.2")

n_clusters = adata.obs["leiden_0.8"].nunique()
print(f"Leiden resolution 0.8 produced {n_clusters} clusters")

# Restore adata.X to lognorm before saving
# Downstream scripts expect lognorm in adata.X
if "lognorm" in adata.layers:
    print("Restoring adata.X to lognorm")
    lognorm = adata.layers["lognorm"]
    adata.X = lognorm.toarray() if sp.issparse(lognorm) else np.array(lognorm)

# Save
adata.write("data/processed/GSE136103_integrated.h5ad")
print("Saved: data/processed/GSE136103_integrated.h5ad")

# UMAP plots
sc.pl.umap(
    adata,
    color=["leiden_0.8", "condition", "donor", "etiology"],
    ncols=2,
    save="_leiden_condition_donor_etiology.pdf"
)
sc.pl.umap(
    adata,
    color=["leiden_0.5", "leiden_0.8", "leiden_1.2"],
    ncols=3,
    save="_resolution_comparison.pdf"
)

print("Done. Next step: python 03_cell_type_annotation.py")
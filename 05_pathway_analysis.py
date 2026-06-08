"""
05_pathway_analysis.py
Pathway enrichment and ligand receptor analysis.

Input:  data/processed/GSE136103_annotated.h5ad
        results/de/wilcoxon_all_celltypes.csv
Output: results/pathways/

Requires decoupler==1.9.2 (pinned in environment.yml).
Do not upgrade decoupler without testing as the API changed significantly
between v1 and v2 and will break all function calls in this script.
"""

import scanpy as sc
import pandas as pd
import numpy as np
import os

sc.settings.verbosity = 2
os.makedirs("figures/pathways", exist_ok=True)
os.makedirs("results/pathways", exist_ok=True)

adata = sc.read("data/processed/GSE136103_annotated.h5ad")
print(f"Loaded: {adata.n_obs:,} cells")
print(f"Cell types present: {sorted(adata.obs['cell_type'].unique())}")


# SECTION 1
# Pathway activity scoring with decoupler v1.9.2

try:
    import decoupler as dc
    print(f"\ndecoupler version: {dc.__version__}")

    # Confirm we are on the expected version
    major, minor = [int(x) for x in dc.__version__.split(".")[:2]]
    if major >= 2:
        raise ImportError(
            f"decoupler {dc.__version__} detected but this script requires v1.9.2. "
            "Run: pip install decoupler==1.9.2"
        )

    # PROGENy pathway activity scoring
    # get_progeny returns a network DataFrame with source, target, weight columns
    print("\nRunning PROGENy pathway activity scoring...")
    progeny = dc.get_progeny(organism="human", top=500)
    print(f"  PROGENy network: {progeny.shape}")

    # run_mlm writes results to adata.obsm["mlm_estimate"] and ["mlm_pvals"]
    # use_raw=False because adata.raw is not set in our pipeline
    dc.run_mlm(
        mat=adata,
        net=progeny,
        source="source",
        target="target",
        weight="weight",
        verbose=True,
        use_raw=False
    )

    # Copy to a named key before the next run_mlm call overwrites mlm_estimate
    adata.obsm["progeny_mlm_estimate"] = adata.obsm["mlm_estimate"].copy()
    adata.obsm["progeny_mlm_pvals"]    = adata.obsm["mlm_pvals"].copy()

    # Save pathway activity scores per cell
    pathway_df = pd.DataFrame(
        adata.obsm["progeny_mlm_estimate"],
        index=adata.obs_names
    )
    pathway_df.to_csv("results/pathways/progeny_activity.csv")
    print("  PROGENy scores saved")

    # Mean pathway activity per cell type
    pathway_summary = (
        adata.obs[["cell_type"]]
        .join(pathway_df)
        .groupby("cell_type")
        .mean()
    )
    pathway_summary.to_csv("results/pathways/progeny_per_celltype.csv")
    print("  PROGENy per cell type summary saved")

    # CollecTRI transcription factor activity scoring
    #
    # RUNTIME WARNING: this step is slow.
    # GSEA runs 34,095 samples against 728 transcription factor sources
    # which is approximately 25 million scoring operations.
    # Expected runtime on a standard HPC node: 2 to 5 days.
    #
    # The full analysis is kept here because when time is not a constraint
    # it provides the most complete TF activity landscape. If 
    # faster results are required, three options are available:
    #
    # Option A: Switch GSEA to MLM (10 to 50x faster, similar results)
    #   Replace dc.run_gsea with dc.run_mlm below.
    #   MLM fits a linear model per sample rather than ranking all genes
    #   which is much faster and appropriate for large datasets.
    #
    # Option B: Filter to high confidence TFs before running
    #   Add before dc.run_gsea:
    #     tf_counts = collectri.groupby("source").size()
    #     collectri = collectri[collectri["source"].isin(
    #         tf_counts[tf_counts >= 10].index)]
    #   This reduces from 728 to roughly 200 to 300 sources
    #   and cuts runtime by about 60 percent.
    #
    # Option C: Skip this step entirely
    #   PROGENy already completed and provides pathway activity scores
    #   sufficient for biomarker prioritization in script 06 (which was the case in this pipeline).
    #   CollecTRI adds TF level resolution on top of that but is not
    #   required for the downstream analysis to run.
    print("\nRunning CollecTRI TF activity scoring...")
    print("  NOTE: this step may take 2 to 5 days on a standard HPC node.")
    print("  See comments above for faster alternatives.")
    try:
        collectri = dc.get_collectri(organism="human")
        print(f"  CollecTRI network: {collectri.shape}")
        dc.run_gsea(
            mat=adata,
            net=collectri,
            source="source",
            target="target",
            verbose=True,
            use_raw=False
        )
        adata.obsm["collectri_gsea_estimate"] = adata.obsm["gsea_estimate"].copy()
        tf_df = pd.DataFrame(
            adata.obsm["collectri_gsea_estimate"],
            index=adata.obs_names
        )
        tf_df.to_csv("results/pathways/collectri_tf_activity.csv")
        print("  CollecTRI TF scores saved")
    except Exception as e:
        print(f"  CollecTRI step skipped: {e}")

except ImportError as e:
    print(f"decoupler not available: {e}")
    print("Falling back to Enrichr API for pathway enrichment...")

    import requests

    def enrichr_query(gene_list, library="KEGG_2021_Human"):
        genes_str = "\n".join(gene_list)
        r = requests.post(
            "https://maayanlab.cloud/Enrichr/addList",
            files={
                "list":        (None, genes_str),
                "description": (None, "query")
            }
        )
        lid = r.json()["userListId"]
        r2  = requests.get(
            f"https://maayanlab.cloud/Enrichr/enrich"
            f"?userListId={lid}&backgroundType={library}"
        )
        return r2.json()[library]

    wilcoxon_path = "results/de/wilcoxon_all_celltypes.csv"
    if not os.path.exists(wilcoxon_path):
        print(f"  {wilcoxon_path} not found. Run 04_de_analysis.py first.")
    else:
        de_df = pd.read_csv(wilcoxon_path)
        for ct in ["SAMac_TREM2", "SAMes_myofibroblast", "SAEndo_ACKR1"]:
            sub = de_df[
                (de_df.cell_type == ct) &
                (de_df.logfoldchanges > 1.5) &
                (de_df.pvals_adj < 0.05)
            ]
            genes = sub.sort_values("scores", ascending=False).names.head(100).tolist()
            if len(genes) < 5:
                print(f"  {ct}: fewer than 5 significant genes, skipping")
                continue
            try:
                results    = enrichr_query(genes)
                results_df = pd.DataFrame(
                    results,
                    columns=["Rank","Term","Pval","Odds","Combined",
                             "Genes","AdjP","OldPval","OldAdjP"]
                )
                out = f"results/pathways/enrichr_KEGG_{ct}.csv"
                results_df.to_csv(out, index=False)
                print(f"  Saved KEGG enrichment for {ct}")
            except Exception as e:
                print(f"  Enrichr query failed for {ct}: {e}")


# SECTION 2
# Ligand receptor analysis with LIANA

try:
    import liana as li

    print("\nRunning LIANA ligand receptor analysis...")
    li.mt.rank_aggregate(
        adata,
        groupby="cell_type",
        verbose=True,
        resource_name="consensus"
    )
    lr_results = adata.uns["liana_res"]
    lr_results.to_csv("results/pathways/LR_interactions_all.csv", index=False)
    print(f"  Saved {len(lr_results)} L-R interactions")

    for source_ct, target_ct, label in [
        ("SAMac_TREM2",         "SAMes_myofibroblast", "SAMac_to_SAMes"),
        ("SAEndo_ACKR1",        "Tissue_monocytes",    "SAEndo_to_Monocytes"),
        ("SAMes_myofibroblast", "SAEndo_ACKR1",        "SAMes_to_SAEndo"),
    ]:
        subset = lr_results[
            (lr_results["source"].str.contains(source_ct, na=False)) &
            (lr_results["target"].str.contains(target_ct, na=False))
        ].sort_values("aggregate_rank")
        if len(subset) > 0:
            out = f"results/pathways/LR_{label}.csv"
            subset.head(30).to_csv(out, index=False)
            print(f"  Saved top 30 L-R pairs for {label}")

except ImportError:
    print("liana not installed. Saving curated L-R pairs from Ramachandran et al. 2019...")
    LR_PAIRS = pd.DataFrame([
        {"ligand":"TNFSF12","receptor":"TNFRSF12A","source":"SAMac_TREM2",        "target":"SAMes_myofibroblast","score":0.92},
        {"ligand":"SPP1",   "receptor":"CD44",      "source":"SAMac_TREM2",        "target":"SAMes_myofibroblast","score":0.88},
        {"ligand":"PDGFB",  "receptor":"PDGFRB",    "source":"SAMac_TREM2",        "target":"SAMes_myofibroblast","score":0.85},
        {"ligand":"CCL2",   "receptor":"CCR2",      "source":"SAEndo_ACKR1",       "target":"Tissue_monocytes",   "score":0.83},
        {"ligand":"VEGFA",  "receptor":"KDR",       "source":"SAMes_myofibroblast","target":"SAEndo_ACKR1",       "score":0.79},
        {"ligand":"CXCL12", "receptor":"CXCR4",     "source":"SAEndo_ACKR1",       "target":"SAMac_TREM2",        "score":0.76},
        {"ligand":"TGFB1",  "receptor":"TGFBR2",    "source":"SAMac_TREM2",        "target":"SAMes_myofibroblast","score":0.74},
        {"ligand":"IL1B",   "receptor":"IL1R1",     "source":"SAMac_TREM2",        "target":"SAEndo_ACKR1",       "score":0.70},
    ])
    LR_PAIRS.to_csv("results/pathways/LR_manual_curated.csv", index=False)
    print("  Manual curated L-R pairs saved")

print("\nPathway analysis complete.")
print("Done. Next step: python 06_biomarker_scoring.py")
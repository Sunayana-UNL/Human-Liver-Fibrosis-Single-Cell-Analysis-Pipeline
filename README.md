# Human-Liver-Fibrosis-Single-Cell-Analysis-Pipeline
GSE136103 Human Liver Fibrosis
Single-Cell Analysis Pipeline
Sunayana Malla | University of Nebraska-Lincoln
Ramachandran et al., Nature 577, 531-543 (2020) | GEO: GSE136103
What This Pipeline Does
Starting from raw count matrices downloaded from GEO, this pipeline:
•	Cleans and filters cells using tissue-aware QC thresholds specific to liver non-parenchymal cells
•	Applies three normalization methods stored as separate layers for different downstream uses
•	Corrects for donor batch effects using Harmony at the PCA embedding level
•	Annotates 18 liver cell types including disease-relevant scar-associated populations
•	Runs pseudobulk differential expression using DESeq2 with donor as the replicate unit
•	Scores pathway activity with PROGENy and transcription factor activity with CollecTRI
•	Maps cell-cell ligand-receptor interactions using LIANA
•	Prioritizes 15 biomarker candidates using three independent scoring methods: rule-based expert scoring, ML ensemble (Elastic Net, Random Forest, GBM), and network-based scoring with data-driven novel seed discovery
•	Produces a consensus ranked list with translational relevance commentary per candidate

Repository Structure
GSE136103_fibrosis_analysis/ README.docx
  environment.yml                     pinned conda environment

  00_download_data.sh             download raw data from GEO + parse metadata
  01_qc_preprocessing.py         QC, filtering, normalization (3 methods)
  02_integration_clustering.py  PCA, Harmony, UMAP, Leiden clustering
  03_cell_type_annotation.py    marker scoring, cell type labeling
  04_de_analysis.py                  Wilcoxon DE + pseudobulk export for DESeq2
  05_pathway_analysis.py         PROGENy, CollecTRI, LIANA LR analysis
  06_biomarker_scoring.py        multi-method biomarker prioritization

  R/
    deseq2_pseudobulk.R         DESeq2 pseudobulk differential expression
    gsea_clusterProfiler.R      GO and KEGG pathway enrichment


Data folders created at runtime
data/
  raw/          downloaded GEO count matrices (one folder per GSM)
  processed/    AnnData h5ad files at each pipeline stage

results/
  de/           differential expression results
  pathways/     pathway activity and LR interaction results
  biomarker_*.csv

figures/
  qc/           before and after QC plots
  clustering/   UMAP and resolution comparison
  annotation/   cell type UMAPs and marker dotplots
  de/           DE visualization
  biomarkers/   rank heatmap, scatter, network, consensus bar chart
Requirements
System
•	Linux or macOS (HPC recommended for full dataset)
•	At least 32 GB RAM (64 GB recommended)
•	At least 150 GB free disk space
Software
•	conda (Miniconda or Anaconda)
•	R 4.3+
Installation
Step 1: Clone or download the repository
cd /your/work/directory
unzip GSE136103_fibrosis_analysis.zip
cd GSE136103_fibrosis_analysis

Step 2: Create the conda environment
Installing to a specific path is recommended on HPC systems to avoid home directory quota issues:
conda env create -f environment.yml \
  --prefix /path/to/your/work/envs/gse136103_fibrosis

# Or install to the default location:
conda env create -f environment.yml

Step 3: Activate the environment
conda activate /path/to/your/work/envs/gse136103_fibrosis
# or if installed to default location:
conda activate gse136103_fibrosis

Step 4 : Install R packages
Open R and run:
install.packages('BiocManager')
BiocManager::install(c('DESeq2', 'clusterProfiler', 'org.Hs.eg.db'))
install.packages(c('tidyverse', 'readr', 'tibble'))
q()

Note: R packages are also listed in environment.yml and will install automatically via bioconda if your channels are configured correctly.

Running on HPC (SLURM)
On HPC systems, call the environment Python directly rather than using conda activate inside batch scripts, which does not work reliably in non-interactive shells:
ENV=/path/to/your/work/envs/gse136103_fibrosis
$ENV/bin/python 01_qc_preprocessing.py

Example SLURM batch script
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --job-name=qc_preprocessing
#SBATCH --error=qc.%J.err
#SBATCH --output=qc.%J.out
#SBATCH --mem=64G

ENV=/path/to/your/work/envs/gse136103_fibrosis
cd /path/to/GSE136103_fibrosis_analysis

echo "Python: $($ENV/bin/python --version)"
$ENV/bin/python 01_qc_preprocessing.py

Parallel execution after Step 3
Scripts 04, 05, and 06 can run in parallel after script 03 completes:
JOB03=$(sbatch --parsable 03_annotation_slurm.sh)
sbatch --dependency=afterok:$JOB03 04_de_slurm.sh
sbatch --dependency=afterok:$JOB03 05_pathway_slurm.sh
sbatch --dependency=afterok:$JOB03 06_biomarker_slurm.sh
Running the Pipeline
Scripts must be run in order from 00 through 06. Each script reads the output of the previous one.
00_download_data.sh
        |
01_qc_preprocessing.py
        |
02_integration_clustering.py
        |
03_cell_type_annotation.py
        |  |  |
       04  05  06   (can run in parallel after 03)
        |
   R scripts (called by 04 or run separately)

Step 0 — Download data (~1 to 3 hours)
bash 00_download_data.sh
Downloads all 10 donor samples from GEO, parses metadata from the SOFT file, and organizes files into per-sample folders.

Step 1 — QC and normalization (~20 to 40 min)
$ENV/bin/python 01_qc_preprocessing.py
Filters cells, removes doublets with Scrublet, applies three normalization methods, selects highly variable genes, and generates before and after QC UMAPs.

Key QC thresholds (liver NPC specific)
Parameter	Value	Reason
Min genes	300	removes empty droplets
Max genes	6000	flags likely doublets
Max MT%	25%	liver NPCs have elevated metabolic activity
Doublet threshold	0.25	Scrublet score cutoff

Normalization layers stored
Layer	Used for
counts	DESeq2 pseudobulk DE (always use raw counts)
lognorm	Wilcoxon DE, plotting, most scanpy functions
scran	More accurate DE with size factor correction
pearson	PCA, UMAP, clustering

Step 2 — Integration and clustering (~30 to 90 min)
$ENV/bin/python 02_integration_clustering.py
Runs PCA on Pearson residuals, applies Harmony batch correction by donor, computes UMAP, and runs Leiden clustering at resolutions 0.5, 0.8, and 1.2.
Note: harmonypy is called directly rather than through the scanpy wrapper to avoid a known transpose bug in certain version combinations.

Step 3 — Cell type annotation (~15 min)
$ENV/bin/python 03_cell_type_annotation.py
Scores clusters against 17 curated liver cell type marker sets and assigns labels. Review the output dotplot before proceeding — cluster numbers are not stable across runs and the CLUSTER_LABELS dictionary in the script may need updating.

Disease-relevant populations identified
Cell type	Key markers	Disease relevance
SAMac_TREM2	TREM2, CD9, SPP1	expanded 210% in cirrhosis
SAMes_myofibroblast	PDGFRA, ACTA2, COL1A1	primary scar producers
SAEndo_ACKR1	ACKR1, PLVAP, ICAM1	leukocyte transmigration
Kupffer_cells	TIMD4, MARCO, VSIG4	depleted in cirrhosis
Quiescent_HSC	LRAT, RELN, CYGB	depleted as HSCs activate

Step 4 — Differential expression (~35 min)
$ENV/bin/python 04_de_analysis.py
Rscript R/deseq2_pseudobulk.R
Rscript R/gsea_clusterProfiler.R
Runs Wilcoxon DE per cell type for discovery, exports pseudobulk count matrices per cell type for DESeq2, and runs GO and KEGG enrichment.
Note: Why pseudobulk? Wilcoxon treats each cell as an independent observation, inflating sample size and producing false positives. DESeq2 aggregates cells by donor (5 vs 5 replicates) which is the statistically correct unit. Always report DESeq2 results as primary evidence.

Step 5 — Pathway and LR analysis (~2 hours to 5 days)
$ENV/bin/python 05_pathway_analysis.py
Runs PROGENy pathway activity scoring and CollecTRI transcription factor activity scoring via GSEA, then maps ligand-receptor interactions with LIANA.
Note: Runtime warning: The CollecTRI GSEA step runs 34,095 samples against 728 TF sources and may take 2 to 5 days. PROGENy completes in approximately 2 hours and is sufficient for downstream biomarker scoring. See comments in the script for faster alternatives.
Step 6 — Biomarker prioritization (~30 to 60 min)
$ENV/bin/python 06_biomarker_scoring.py
Scores 15 candidates across three methods and produces a consensus ranking.
Three scoring methods
Method	Approach	Primary strength
Rule-based	5 expert dimensions, max 10 pts	interpretable, literature-grounded
ML ensemble	Elastic Net + RF + GBM with LOOCV	integrates multiple evidence types
Network-based	STRING PPI + novel seed discovery	captures mechanistic centrality
Note: Does not require Step 5 to complete. Script 06 only reads DESeq2 results from Step 4 and runs independently of pathway analysis.

Key Outputs
File	Description
data/processed/GSE136103_annotated.h5ad	Final annotated AnnData object
results/biomarker_multimethod_ranked.csv	Full multi-method ranked list
results/biomarker_consensus_summary.csv	Consensus ranking table
results/de/DESeq2_*_cirrhotic_vs_healthy.csv	Per cell type DE results
results/pathways/progeny_per_celltype.csv	Pathway activity per cell type
results/pathways/LR_SAMac_to_SAMes.csv	Top SAMac to HSC interactions
results/network_seed_genes.csv	Established and novel seed genes
figures/biomarkers/consensus_ranking.pdf	Final ranked bar chart
figures/biomarkers/ppi_network.pdf	PPI network visualization
figures/biomarkers/rank_heatmap.pdf	Cross-method rank comparison

Top Biomarker Candidates
Rank	Gene	Score	Category	Compartment
1	TREM2	9.8/10	Dx + Tx	Macrophage
2	SPP1	9.4/10	Dx	Macrophage
3	COL1A1	9.2/10	Tx	Mesenchymal
4	PDGFRA	8.9/10	Tx	Mesenchymal
5	ACKR1	8.7/10	Dx + Tx	Endothelial
Full ranked list in results/biomarker_consensus_summary.csv.
Common Errors and Fixes (that I faced and fixed)
Error	Cause	Fix
ModuleNotFoundError: scanpy	conda env not active	conda activate gse136103_fibrosis
FileNotFoundError: matrix.mtx.gz	files not downloaded or wrong folder	run 00_download_data.sh first
ValueError: X_pca_harmony shape	scanpy Harmony wrapper transpose bug	script calls harmonypy directly, check version
AttributeError: get_progeny	wrong decoupler version	pip install decoupler==1.9.2
OSError: Disk quota exceeded	home directory full	set conda dirs to scratch in ~/.condarc
YAML AttributeError: str has no copy	tabs in environment.yml	recreate file using cat << EOF in terminal
CommandNotFoundError: conda activate	conda not initialized	source /path/to/conda/etc/profile.d/conda.sh

Reproducing the Environment
# Clean conda cache first if on HPC with quota limits
conda clean --all -y
rm -rf ~/.cache/pip

# Create environment
conda env create -f environment.yml \
  --prefix /scratch/your_id/envs/gse136103_fibrosis

# Verify key packages
ENV=/scratch/your_id/envs/gse136103_fibrosis
$ENV/bin/python -c "import scanpy; print('scanpy', scanpy.__version__)"
$ENV/bin/python -c "import decoupler; print('decoupler', decoupler.__version__)"
$ENV/bin/python -c "import harmonypy; print('harmonypy', harmonypy.__version__)"

Citation
the original dataset: Ramachandran P et al. (2019). Resolving the fibrotic niche of human liver cirrhosis at single-cell level. Nature 577, 531-543. https://doi.org/10.1038/s41586-019-1631-3

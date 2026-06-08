#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH --job-name=download_GSE136103
#SBATCH --error=download.%J.err
#SBATCH --output=download.%J.out
#SBATCH --mail-type=ALL
module load python
set -euo pipefail
# set -e  = stop immediately if any command fails
# set -u  = treat unset variables as errors
# set -o pipefail = catch errors inside pipes too


WORK_DIR=".../Karyon_Project/GSE136103_fibrosis_analysis"
RAW_DIR="$WORK_DIR/data/raw"
ENV_PYTHON="$WORK_DIR/envs/gse136103_fibrosis/bin/python"
ACCESSION="GSE136103"

mkdir -p "$RAW_DIR"
cd "$RAW_DIR"

echo "============================================"
echo "GSE136103 — Data Download Script"
echo "Working directory: $RAW_DIR"
echo "Started: $(date)"
echo "============================================"


echo ""
echo "[Step 1] Downloading SOFT metadata file..."
wget -q --show-progress \
  "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE136nnn/${ACCESSION}/soft/${ACCESSION}_family.soft.gz" \
  -O "${ACCESSION}_family.soft.gz"

gunzip -f "${ACCESSION}_family.soft.gz"
echo "  Done: ${ACCESSION}_family.soft"


echo ""
echo "[Step 2] Extracting sample IDs from metadata..."
grep "^!Sample_geo_accession" "${ACCESSION}_family.soft" \
  | awk '{print $3}' \
  | sort -u > all_gsm_ids.txt

N_SAMPLES=$(wc -l < all_gsm_ids.txt)
echo "  Found $N_SAMPLES samples:"
cat all_gsm_ids.txt | sed 's/^/    /'


echo ""
echo "[Step 3] Downloading RAW data archive..."
echo "  This may take 1-3 hours depending on connection speed."
wget --show-progress \
  "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE136nnn/${ACCESSION}/suppl/${ACCESSION}_RAW.tar" \
  -O "${ACCESSION}_RAW.tar"
echo "  Download complete: $(du -sh ${ACCESSION}_RAW.tar | cut -f1)"

echo ""
echo "[Step 4] Unpacking archive..."
tar -xvf "${ACCESSION}_RAW.tar"
echo "  Unpacked."

#Step 5: Organize files into per-sample folders 
echo ""
echo "[Step 5] Organizing files into per-sample folders..."
while read gsm; do
    n_files=$(ls ${gsm}_* 2>/dev/null | wc -l)
    if [ "$n_files" -gt 0 ]; then
        mkdir -p "$gsm"
        mv ${gsm}_* "$gsm/"
        # Decompress any .gz files inside the folder
        gunzip -f "$gsm"/*.gz 2>/dev/null || true
        echo "  $gsm: $n_files files moved and decompressed"
    else
        echo "  $gsm: WARNING — no files found for this sample"
    fi
done < all_gsm_ids.txt

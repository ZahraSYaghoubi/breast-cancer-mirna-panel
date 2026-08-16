# Breast Cancer miRNA Diagnostic Panel

A machine-learning pipeline that builds a **20-miRNA serum
diagnostic panel** for breast cancer detection from raw
GEO microarray data (GSE41922).

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source   | [GEO GSE41922](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE41922) |
| File     | `GSE41922_non-normalized.txt` (place in `data/`) |
| Platform | Affymetrix GeneChip miRNA 2.0 |

## 🚀 Usage
```bash
# 1) Install dependencies
pip install -r requirements.txt

# 2) Place raw data file into data/
#    (download from GEO link above)

# 3) Run pipeline in order
python scripts/01_preprocessing.py
python scripts/02_normalization.py
python scripts/03_lasso_selection.py
python scripts/04_rfe_optimization.py
python scripts/05_final_panel.py

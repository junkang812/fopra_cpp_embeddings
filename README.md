# Embedding-based CPP for Signal Peptide Prediction

This project investigates how protein language model (pLM) embeddings can be integrated into the **CPP (Comprehensive Physicochemical Property)** framework for interpretable signal peptide prediction.

Instead of relying solely on handcrafted amino acid scales, residue-level **ESM2 embeddings** are transformed into numerical pseudo-scales and processed by the CPP feature engineering pipeline.

---

## Overview

The project compares four sequence representations:

| Representation | Description |
|---------------|-------------|
| Whole-sequence embedding | Mean pooled ESM2 embedding over the full protein |
| Pre-CS embedding | Mean pooled ESM2 embedding before the cleavage site |
| Original CPP | CPP features extracted from physicochemical scales |
| Embedding-CPP | CPP features extracted from residue-level ESM2 embeddings |

---

## Pipeline

```
SignalP sequences
        │
        ▼
ESM2 residue embeddings
        │
        ▼
EmbeddingPreprocessor
   ├── encode()
   ├── build_scales()
   └── build_cat()
        │
        ▼
NumericalFeature
        │
        ▼
CPP.run_num()
        │
        ▼
Embedding-CPP feature matrix
        │
        ├── PCA / UMAP visualization
        └── Classification (ongoing)
```

---

## Repository Structure

```
scripts/
│
├── 01_make_subset.py
├── 02_fetch_fullseq_uniprot.py
├── 03_embedding_analysis.py
├── 04_original_cpp_features.py
├── 05_prepare_embedding_cpp.py
├── 06_build_embedding_cat.py
├── 07_embedding_cpp_features.py
├── 08_plot_umap_comparison.py
└── 09_plot_pca_comparison.py
```

---

## Current Results

Current visualization compares the feature spaces of

- Whole-sequence ESM2 embeddings
- Pre-CS ESM2 embeddings
- Original CPP features
- Embedding-based CPP features

using

- PCA
- UMAP

Example:

```
Raw Embedding
        ↓
Embedding CPP
        ↓
Improved class separation
```

---

## Current Status

- [x] Dataset preprocessing
- [x] Residue-level ESM2 embedding extraction
- [x] Original CPP feature extraction
- [x] Embedding-based CPP implementation
- [x] PCA visualization
- [x] UMAP visualization
- [ ] Binary SP classification
- [ ] Multi-class SP type classification
- [ ] Feature importance analysis (SHAP)

---

## Requirements

```
Python 3.10

PyTorch
AAanalysis
NumPy
Pandas
scikit-learn
Matplotlib
UMAP-learn
```

---

## Future Work

- Quantitative comparison of all representations
- Binary SP vs. NO_SP prediction
- Multi-class signal peptide type prediction
- Explainability analysis of embedding-derived CPP features
- Comparison with SignalP baselines

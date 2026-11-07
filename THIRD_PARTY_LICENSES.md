# Third-party components, data and licenses

This project's own code and the synthetic dataset it generates are released under the MIT License (see `LICENSE`).
Nothing below is vendored into the repository except where stated; dependencies are installed from PyPI.
License identifiers were read from the installed package metadata (versions used in development).
This is a good-faith summary, not legal advice.

## Software dependencies

| Package | Version | License |
|---|---|---|
| CatBoost | 1.2.10 | Apache-2.0 |
| scikit-learn | 1.8.0 | BSD-3-Clause |
| pandas | 3.0.1 | BSD-3-Clause |
| NumPy | 2.4.6 | BSD-3-Clause (plus bundled permissive components: 0BSD, MIT, Zlib, CC0-1.0) |
| SciPy | 1.17.1 | BSD-3-Clause |
| NetworkX | 3.6.1 | BSD-3-Clause |
| Plotly | 7.1.0 | MIT |
| Streamlit | 1.67.0 | Apache-2.0 |
| Matplotlib | 3.10.8 | Matplotlib License (PSF-based, permissive) |
| python-pptx | 1.0.2 | MIT |

All are permissive licenses compatible with MIT distribution of this project's code and with commercial use.

## Data

### Synthetic dataset (`data/transacciones_fraude_grafo.csv`, produced by `data/generar_datos_fraude_grafo.py`)
Generated entirely by this project's own code, from random draws. It contains no real personal data and no third-party
data. Released under MIT with the code.

### Elliptic Bitcoin Dataset (real data, NOT included in this repository)
* Used only for the validation in `modelo/validar_real.py`. It is downloaded by the user with
  `data/descargar_elliptic.py` from <https://data.pyg.org/datasets/elliptic/> (the same data is published at
  <https://www.kaggle.com/datasets/ellipticco/elliptic-data-set>).
* License: **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)**,
  as stated on the dataset's Kaggle page.
* Consequences honoured here: (1) attribution is given below; (2) the dataset is not redistributed in this repository
  (`data/elliptic/` is git-ignored), nor any modified copy of it; (3) the repository contains only aggregate evaluation
  metrics and figures, not the data; (4) use is for research and evaluation within a non-commercial competition.
  Anyone reusing this repository commercially must not use the Elliptic data under this license.
* Required citation:
  1. Elliptic, <https://www.elliptic.co>.
  2. M. Weber, G. Domeniconi, J. Chen, D. K. I. Weidele, C. Bellei, T. Robinson, C. E. Leiserson,
     "Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics",
     KDD '19 Workshop on Anomaly Detection in Finance, August 2019, Anchorage, AK, USA. arXiv:1908.02591.

## Models and generated artifacts
`modelo/model.cbm` is a CatBoost model trained only on the synthetic dataset above (no Elliptic data, no real data).
Figures, metrics and the presentation are original work of the author.

## Competition
Prepared for DataNova 2026 Global Data-thon (Devpost). Submitted work is original; third-party libraries are
acknowledged above.

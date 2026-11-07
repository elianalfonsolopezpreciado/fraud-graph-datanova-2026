# Fraud Graph — Real-Time Payment Fraud Detection

**DataNova 2026 · Global Data-thon · Track: Cybersecurity & Fraud Detection**

Most fraud models score each payment in isolation. Organised fraud (mule accounts, rings sharing devices/IPs)
looks ordinary payment-by-payment but is obvious as a **network**. Fraud Graph maintains a live
user ↔ device ↔ IP ↔ merchant graph and feeds that network context to a CatBoost scorer, with per-transaction
SHAP explanations and a three-level decision engine (HIGH → block · MEDIUM → manual review · LOW → approve).

> **Main data are synthetic** (own generator, `data/`); the approach is also **validated on a real public dataset**
> (Elliptic Bitcoin) and **stress-tested** to map where the graph helps and where it fails. Absolute numbers will not
> transfer to a real bank. See *Real-data validation*, *Stress benchmark* and *Limitations*.

## Results (held-out future, seed 42 · 3,586 transactions, 134 frauds)

| | CatBoost, tabular only | **CatBoost + graph** |
|---|---|---|
| PR-AUC | 0.613 | **0.776** |
| ROC-AUC | 0.931 | **0.967** |
| Recall of flagged (MEDIUM+HIGH), FPR ≈ 2.2% | 64.2% | **74.6%** |
| Precision of HIGH (block) | 73.8% | **79.6%** |
| Fraud rings with ≥1 flagged tx | – | 4 / 4 |

* Gain in PR-AUC: **+0.163, bootstrap 95% CI [+0.099, +0.222]**.
* Robustness over **5 independent synthetic datasets** (mean ± std): PR-AUC 0.663 ± 0.057 → **0.815 ± 0.032**.
* IsolationForest (unsupervised) reference: PR-AUC 0.562 ± 0.039.
* Decision buckets on test: HIGH 108 tx (79.6% fraud) · MEDIUM 67 tx (21% fraud) · LOW 3,411 tx (1.0% fraud).
* Graph features carry 38% of total SHAP attribution.

Everything above is produced by `python modelo/train_eval.py` and stored in `modelo/metrics.json` /
`modelo/ablation_seeds.csv`; the dashboard and the presentation read those files (nothing is hard-coded).

## Real-data validation (Elliptic Bitcoin dataset)

203,769 real transactions, 234,355 edges, 46,564 labelled (4,545 illicit); train = time steps 1-34, test = 35-49.
Neighbour labels are never used (edges only link transactions of the same step, so that would leak); we propagate the
predicted risk of a first-stage model. 5 seeds.

| Model | PR-AUC | Recall @ FPR 1% |
|---|---|---|
| Local features | 0.793 ± 0.002 | 0.712 |
| Local + graph context | 0.807 ± 0.004 | 0.723 |
| Local + aggregates of the original paper | 0.807 ± 0.002 | 0.735 |
| Aggregates + graph context | 0.816 ± 0.005 | 0.729 |

Graph context adds +0.014 PR-AUC (bootstrap 95% CI [+0.011, +0.021]): real but small on this sparse graph.
A real concept drift at step 43 (dark-market shutdown) collapses both models (PR-AUC about 0.92 before, 0.13-0.18 after);
the graph does not protect against changed criminal behaviour. Run: `python data/descargar_elliptic.py` then
`python modelo/validar_real.py`.

## Stress benchmark: what the graph knows and what it does not

`python modelo/benchmark_estres.py` (3 seeds). Gain = PR-AUC(graph) minus PR-AUC(tabular):

| Scenario | Gain |
|---|---|
| Baseline | +0.135 |
| Labels arrive after 7 days | +0.134 |
| Labels never arrive (no feedback) | +0.031 |
| 30% / 60% of fraud never labelled | +0.216 / +0.219 |
| Criminals rotate devices/IPs | +0.027 |
| Tactic shift after deployment | +0.034 |
| Innocent users share criminal infrastructure | +0.136 (but 9.7% false flags on those users vs 2.3% for the rest) |
| Only isolated fraud (no network) | +0.004 |
| Only fraud-history features / sharing counts / hops-to-fraud | +0.139 / +0.037 / +0.012 |

Reading: the graph is mostly a memory of confirmed fraud attached to devices, IPs and users. It needs label feedback,
tolerates delay and missing labels, and loses value when criminals rotate infrastructure or fraud is isolated.
Innocent bystanders on shared infrastructure suffer about 4x more false flags, which is why MEDIUM goes to a human.

## Evaluation protocol (why the numbers can be trusted)

* **Temporal split** 60% train · 10% validation (selects thresholds) · 30% test (the future, scored once).
* **No leakage:** features are built chronologically from past information only; fraud labels enter the graph
  **24 h after** the transaction (like chargebacks). The label of a transaction never enters its own features.
* **Self-audit:** our first prototype computed graph distance using each transaction's own label in training — a
  leak. It was found and removed; the reported results are the corrected ones.
* **Ablation** (identical model with vs without graph features), multi-seed, bootstrap CI, unsupervised baseline.
* **Thresholds from validation:** HIGH = FPR ≤ 0.3%, MEDIUM = FPR ≤ 2%.

## Project layout

```
data/generar_datos_fraude_grafo.py   synthetic generator (isolated fraud, mule accounts, rings sharing infrastructure)
data/transacciones_fraude_grafo.csv  dataset, seed 42 (11,952 tx, 3.6% fraud)
modelo/features.py                   leak-free tabular + graph feature engineering (incremental graph, BFS to confirmed fraud)
modelo/train_eval.py                 training, ablation, 5-seed robustness, thresholds, SHAP, ring detection
modelo/*.json|csv|cbm                metrics, predictions, SHAP importance, trained model
dashboard/app.py                     Streamlit dashboard (overview, alerts + SHAP, ring graph, what-if, stress benchmark, real data, methodology)
modelo/benchmark_estres.py           stress benchmark of graph knowledge
modelo/validar_real.py               validation on the real Elliptic dataset (data/descargar_elliptic.py downloads it)
generar_ppt.py                       builds the presentation (.pptx) from the metrics files
assets/                              figures used in the deck
```

## Run it

```bash
pip install -r requirements.txt
python data/generar_datos_fraude_grafo.py     # optional: regenerates the dataset
python modelo/train_eval.py                   # ~1–2 min: metrics, model, predictions
streamlit run dashboard/app.py                # http://localhost:8501
python generar_ppt.py                         # rebuild the deck (edit TEAM / links at the top first)
```

Deploy the demo: push this folder to GitHub → <https://share.streamlit.io> → main file `dashboard/app.py`.
(`modelo/` artifacts are committed, so the app starts without retraining.)

## Limitations

* Main data synthetic; on real Elliptic data the graph gain is small (+0.014) and drift hurts both models. Real deployment needs production data, drift monitoring and retraining.
* Isolated frauds without a network footprint are the hardest (≈65% flagged); rings and mules are almost fully caught.
* Few rings fall in the test window (4); hence we report multi-seed robustness rather than one run.
* Responsible AI: no personal data; SHAP explanations support human review and customer appeals.

## Stack & acknowledgements

Python · CatBoost · scikit-learn · NetworkX · Plotly · Streamlit · pandas · NumPy · python-pptx · matplotlib
(all open-source). Dataset generated by the team's own code.

## License and data

* Code and synthetic dataset: **MIT** (`LICENSE`). Dependencies are permissive (Apache-2.0 / BSD-3 / MIT / PSF-style).
* Real-data validation uses the **Elliptic Bitcoin Dataset, CC BY-NC-ND 4.0** (attribution, non-commercial, no
  derivatives). It is **not** included in this repository; download it with `python data/descargar_elliptic.py`.
  Cite: Elliptic; Weber et al., KDD '19 Workshop on Anomaly Detection in Finance (arXiv:1908.02591).
* Full table of components and obligations: [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

## Team

**Elian Alfonso López Preciado** — solo project: data generation, graph feature engineering, modeling and evaluation, dashboard, presentation.

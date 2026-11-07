"""
Validación en DATOS REALES: Elliptic Bitcoin Dataset (transacciones reales, 203,769 nodos,
234,355 aristas, 49 pasos de tiempo, clases: 1 = ilícito, 2 = lícito, unknown = sin etiqueta).
Fuente: https://data.pyg.org/datasets/elliptic/  (Weber et al., 2019, Elliptic / MIT-IBM).

Pregunta: ¿el contexto de grafo mejora la detección en datos reales, sin fuga?

Protocolo (idéntico en espíritu al de datos sintéticos)
  * Split temporal estándar: train = pasos 1-34, test = pasos 35-49 (el futuro).
  * Las aristas solo unen transacciones del MISMO paso, así que usar etiquetas de vecinos sería fuga.
    Por eso NO usamos etiquetas: propagamos el RIESGO PREDICHO de un modelo de 1ª etapa entrenado
    solo con el pasado (guilt-by-association sobre predicciones).
  * Etapa 1: CatBoost sobre features locales (93). Scores de train = out-of-fold (sin sobreajuste).
  * Features de grafo propias: grado, riesgo medio/máximo de vecinos (1 y 2 saltos), nº de vecinos de riesgo alto.
  * Variantes: A) local  B) local + grafo propio  C) local + agregados de vecinos de Weber et al. (165)
               D) 165 + grafo propio.   5 semillas; IC95% bootstrap de la ganancia.
Salidas: modelo/real_validation.json, modelo/real_validation_steps.csv
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, scipy.sparse as sp
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_score, recall_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(BASE, "data", "elliptic")
OUT = os.path.join(BASE, "modelo")
SEEDS = [42, 7, 123, 2024, 99]
TRAIN_MAX, N_STEPS = 34, 49


def cargar():
    # ids como int64 (float32 pierde precisión en enteros de 9 dígitos y rompe el cruce de archivos)
    dt = {0: np.int64, 1: np.int32}
    dt.update({i: np.float32 for i in range(2, 167)})
    f = pd.read_csv(os.path.join(D, "elliptic_txs_features.csv"), header=None, dtype=dt)
    ids = f[0].values
    step = f[1].values.astype(int)
    X = f.iloc[:, 2:].values.astype(np.float32)            # 93 locales + 72 agregados
    cls = pd.read_csv(os.path.join(D, "elliptic_txs_classes.csv"))
    cls["txId"] = cls["txId"].astype(np.int64)
    m = dict(zip(cls["txId"], cls["class"].astype(str)))
    y = np.array([{"1": 1, "2": 0}.get(m.get(i, "unknown"), -1) for i in ids])  # -1 = sin etiqueta
    e = pd.read_csv(os.path.join(D, "elliptic_txs_edgelist.csv")).astype(np.int64)
    idx = {v: i for i, v in enumerate(ids)}
    r = e["txId1"].map(idx).values; c = e["txId2"].map(idx).values
    ok = ~(pd.isna(r) | pd.isna(c))
    n = len(ids)
    A = sp.coo_matrix((np.ones(ok.sum()), (r[ok].astype(np.int64), c[ok].astype(np.int64))), shape=(n, n)).tocsr()
    return ids, step, X, y, A


def feats_grafo(A, s):
    """Features de grafo a partir de un score por nodo (sin etiquetas)."""
    U = ((A + A.T) > 0).astype(np.float32).tocsr()
    deg = np.asarray(U.sum(1)).ravel()
    indeg = np.asarray(A.sum(0)).ravel(); outdeg = np.asarray(A.sum(1)).ravel()
    inv = 1.0 / np.maximum(deg, 1)
    mean1 = (U @ s) * inv
    # máximo de vecinos
    mx = np.zeros_like(s); Uc = U.tocoo()
    np.maximum.at(mx, Uc.row, s[Uc.col])
    hi = U @ (s > 0.5).astype(np.float32)
    mean2 = (U @ mean1) * inv
    return np.column_stack([deg, indeg, outdeg, mean1, mx, hi, mean2, s * 0 + (mean1 - s)]).astype(np.float32)


GN = ["g_grado", "g_in", "g_out", "g_riesgo_medio_vecinos", "g_riesgo_max_vecinos",
      "g_n_vecinos_riesgo_alto", "g_riesgo_medio_2saltos", "g_vecinos_menos_propio"]


def clf(seed):
    return CatBoostClassifier(iterations=500, depth=6, learning_rate=0.06, random_seed=seed,
                              verbose=False, auto_class_weights="SqrtBalanced", thread_count=-1)


def recall_at_fpr(y, s, fpr=0.01):
    thr = np.quantile(s[y == 0], 1 - fpr)
    return float((s[y == 1] >= thr).mean())


def metr(y, s, thr):
    p = (s >= thr).astype(int)
    return {"pr_auc": float(average_precision_score(y, s)), "roc_auc": float(roc_auc_score(y, s)),
            "recall_fpr1": recall_at_fpr(y, s, 0.01),
            "f1_illicit": float(f1_score(y, p)), "precision": float(precision_score(y, p, zero_division=0)),
            "recall": float(recall_score(y, p))}


def main():
    print("Cargando Elliptic ...")
    ids, step, X, y, A = cargar()
    n = len(ids); lab = y >= 0
    tr = lab & (step <= TRAIN_MAX); te = lab & (step > TRAIN_MAX)
    print(f"nodos={n:,} aristas={A.nnz:,} etiquetados={lab.sum():,} ilícitos={int((y == 1).sum()):,} "
          f"| train={tr.sum():,} (ilícito {y[tr].mean():.1%}) test={te.sum():,} (ilícito {y[te].mean():.1%})")
    L = X[:, :93]

    filas, scores_test = [], {}
    for sd in SEEDS:
        # etapa 1: score por nodo con features locales
        s1 = np.zeros(n, dtype=np.float32)
        tri = np.where(tr)[0]
        for a, b in StratifiedKFold(5, shuffle=True, random_state=sd).split(tri, y[tri]):
            m = clf(sd).fit(L[tri[a]], y[tri[a]])
            s1[tri[b]] = m.predict_proba(L[tri[b]])[:, 1]
        m_full = clf(sd).fit(L[tr], y[tr])
        rest = ~tr
        s1[rest] = m_full.predict_proba(L[rest])[:, 1]
        G = feats_grafo(A, s1)
        # umbral operativo elegido SOLO con train (cuantil de FPR 1% sobre OOF)
        variantes = {"A_local": L, "B_local+grafo": np.hstack([L, G]),
                     "C_local+agregados(Weber)": X, "D_agregados+grafo": np.hstack([X, G])}
        for nombre, M in variantes.items():
            if nombre == "A_local":
                s_te = m_full.predict_proba(L[te])[:, 1]
                thr = np.quantile(s1[tr][y[tr] == 0], 0.99)
            else:
                oof = np.zeros(tr.sum())
                Mt = M[tr]; yt = y[tr]
                for a, b in StratifiedKFold(3, shuffle=True, random_state=sd).split(Mt, yt):
                    oof[b] = clf(sd).fit(Mt[a], yt[a]).predict_proba(Mt[b])[:, 1]
                mm = clf(sd).fit(Mt, yt)
                s_te = mm.predict_proba(M[te])[:, 1]
                thr = np.quantile(oof[yt == 0], 0.99)
            r = metr(y[te], s_te, thr); r.update(seed=sd, modelo=nombre)
            filas.append(r); scores_test[(sd, nombre)] = s_te
            print(f"seed {sd} {nombre:26s} PR-AUC={r['pr_auc']:.3f} recall@FPR1%={r['recall_fpr1']:.3f} F1={r['f1_illicit']:.3f}")

    df = pd.DataFrame(filas); df.to_csv(os.path.join(OUT, "real_validation_seeds.csv"), index=False)
    resumen = {nm: {k: {"mean": float(g[k].mean()), "std": float(g[k].std())}
                    for k in ("pr_auc", "roc_auc", "recall_fpr1", "f1_illicit", "precision", "recall")}
               for nm, g in df.groupby("modelo")}

    # bootstrap IC95 de la ganancia (B - A) y (D - C) con semilla 42
    yt = y[te]; rng = np.random.default_rng(0)
    def ic(a, b):
        d = []
        for _ in range(300):
            i = rng.integers(0, len(yt), len(yt))
            d.append(average_precision_score(yt[i], scores_test[(42, b)][i]) - average_precision_score(yt[i], scores_test[(42, a)][i]))
        return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
    ic_ba, ic_dc = ic("A_local", "B_local+grafo"), ic("C_local+agregados(Weber)", "D_agregados+grafo")

    # rendimiento por paso de tiempo (semilla 42) — muestra la deriva tras el paso 43
    st = step[te]; pasos = []
    for k in sorted(set(st)):
        mk = st == k
        if y[te][mk].sum() >= 5:
            pasos.append({"paso": int(k), "n_ilicitos": int(y[te][mk].sum()),
                          "pr_auc_local": float(average_precision_score(yt[mk], scores_test[(42, "A_local")][mk])),
                          "pr_auc_grafo": float(average_precision_score(yt[mk], scores_test[(42, "B_local+grafo")][mk]))})
    pd.DataFrame(pasos).to_csv(os.path.join(OUT, "real_validation_steps.csv"), index=False)

    out = {"dataset": "Elliptic Bitcoin Dataset (real)", "fuente": "https://data.pyg.org/datasets/elliptic/",
           "nodos": int(n), "aristas": int(A.nnz), "etiquetados": int(lab.sum()),
           "ilicitos": int((y == 1).sum()), "n_train": int(tr.sum()), "n_test": int(te.sum()),
           "tasa_ilicito_test": float(y[te].mean()), "split": "train pasos 1-34 / test pasos 35-49",
           "resumen_5_semillas": resumen,
           "ganancia_pr_auc_B_vs_A": resumen["B_local+grafo"]["pr_auc"]["mean"] - resumen["A_local"]["pr_auc"]["mean"],
           "ic95_B_vs_A_seed42": ic_ba,
           "ganancia_pr_auc_D_vs_C": resumen["D_agregados+grafo"]["pr_auc"]["mean"] - resumen["C_local+agregados(Weber)"]["pr_auc"]["mean"],
           "ic95_D_vs_C_seed42": ic_dc}
    json.dump(out, open(os.path.join(OUT, "real_validation.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\n=== RESUMEN (5 semillas, mean±std) ===")
    for nm, r in resumen.items():
        print(f"{nm:26s} PR-AUC {r['pr_auc']['mean']:.3f}±{r['pr_auc']['std']:.3f}  "
              f"recall@FPR1% {r['recall_fpr1']['mean']:.3f}  F1 {r['f1_illicit']['mean']:.3f}")
    print("ΔPR-AUC B-A", round(out["ganancia_pr_auc_B_vs_A"], 3), ic_ba, "| D-C", round(out["ganancia_pr_auc_D_vs_C"], 3), ic_dc)


if __name__ == "__main__":
    main()

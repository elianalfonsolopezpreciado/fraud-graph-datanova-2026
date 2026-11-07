"""
DN-FN-01 — Entrenamiento y evaluación (Fraud Graph) con protocolo sin fuga.

Protocolo
---------
* Split TEMPORAL: 60% train | 10% validación (fija umbrales) | 30% test (futuro).
* Features calculadas cronológicamente con retraso de etiquetas (ver features.py).
* Ablación: CatBoost solo-tabular  vs  CatBoost tabular+grafo  (misma configuración).
* Referencia no supervisada: IsolationForest.
* Robustez: se repite con 5 semillas de datos sintéticos distintas (media ± desv.).
* Umbrales operativos elegidos en VALIDACIÓN, evaluados una sola vez en TEST:
    ALTO  = menor umbral con FPR <= 0.3% en validación  -> bloqueo (casi sin falsos positivos)
    MEDIO = menor umbral con FPR <= 2% en validación    -> revisión manual
* Explicabilidad: valores SHAP nativos de CatBoost (por transacción).

Salidas: modelo/metrics.json, ablation_seeds.csv, predicciones_test.csv,
         feature_importance_shap.csv, model.cbm, model_meta.json
"""
import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, precision_recall_curve

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "modelo"))
sys.path.insert(0, os.path.join(BASE, "data"))
from features import construir_features, TABULAR_COLS, GRAPH_COLS, FEATURES_FULL  # noqa: E402
from generar_datos_fraude_grafo import generar  # noqa: E402

OUT = os.path.join(BASE, "modelo")
SEED_PRINCIPAL = 42
SEEDS_ROBUSTEZ = [42, 7, 123, 2024, 99]
FRAC_TRAIN, FRAC_VAL = 0.60, 0.10
FPR_ALTO, FPR_MEDIO = 0.003, 0.02


def entrenar(X, y, seed):
    clf = CatBoostClassifier(iterations=400, depth=5, learning_rate=0.05,
                             loss_function="Logloss", random_seed=seed, verbose=False,
                             auto_class_weights="SqrtBalanced")
    clf.fit(X, y)
    return clf


def umbral_precision(y, s, prec_min=0.90, min_pos=3):
    """Menor umbral cuya precisión (en validación) >= prec_min."""
    p, r, t = precision_recall_curve(y, s)
    ok = [(thr, pp) for thr, pp in zip(t, p[:-1]) if pp >= prec_min and (s >= thr).sum() >= min_pos]
    return float(min(ok)[0]) if ok else float(np.quantile(s, 0.995))


def umbral_fpr(y, s, fpr_max=0.01):
    """Umbral exacto: cuantil (1 - fpr_max) de los scores de los negativos en validación."""
    neg = np.asarray(s)[np.asarray(y) == 0]
    return float(np.quantile(neg, 1 - fpr_max))


def metricas(y, s, thr_alto, thr_medio):
    y = np.asarray(y); s = np.asarray(s)
    flag = s >= thr_medio
    alto = s >= thr_alto
    tp, fp = int((flag & (y == 1)).sum()), int((flag & (y == 0)).sum())
    fn, tn = int((~flag & (y == 1)).sum()), int((~flag & (y == 0)).sum())
    tpa, fpa = int((alto & (y == 1)).sum()), int((alto & (y == 0)).sum())
    return {
        "pr_auc": float(average_precision_score(y, s)),
        "roc_auc": float(roc_auc_score(y, s)),
        "recall_flag": tp / max(tp + fn, 1),
        "precision_flag": tp / max(tp + fp, 1),
        "fpr_flag": fp / max(fp + tn, 1),
        "alto_n": tpa + fpa, "alto_precision": tpa / max(tpa + fpa, 1),
        "alto_recall": tpa / max(tp + fn, 1),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }


def correr_semilla(seed, df=None, detalle=False):
    if df is None:
        df = generar(seed)
    f = construir_features(df)
    n = len(f)
    i1, i2 = int(n * FRAC_TRAIN), int(n * (FRAC_TRAIN + FRAC_VAL))
    tr, va, te = f.iloc[:i1], f.iloc[i1:i2], f.iloc[i2:]
    res = {}
    modelos = {}
    for nombre, cols in (("tabular", TABULAR_COLS), ("tabular+grafo", FEATURES_FULL)):
        clf = entrenar(tr[cols], tr["is_fraude"], seed)
        sv = clf.predict_proba(va[cols])[:, 1]
        st = clf.predict_proba(te[cols])[:, 1]
        ta, tm = umbral_fpr(va["is_fraude"], sv, FPR_ALTO), umbral_fpr(va["is_fraude"], sv, FPR_MEDIO)
        tm = min(tm, ta)
        res[nombre] = metricas(te["is_fraude"], st, ta, tm)
        res[nombre].update(thr_alto=ta, thr_medio=tm)
        modelos[nombre] = (clf, cols, st, ta, tm)
    iso = IsolationForest(n_estimators=200, contamination=0.04, random_state=seed, n_jobs=-1)
    iso.fit(tr[FEATURES_FULL])
    s_iso = -iso.score_samples(te[FEATURES_FULL])
    res["isolation_forest"] = {"pr_auc": float(average_precision_score(te["is_fraude"], s_iso)),
                               "roc_auc": float(roc_auc_score(te["is_fraude"], s_iso))}
    res["_info"] = {"n_train": len(tr), "n_val": len(va), "n_test": len(te),
                    "fraude_test": int(te["is_fraude"].sum()),
                    "fraude_rate_test": float(te["is_fraude"].mean())}
    if detalle:
        return res, f, (tr, va, te), modelos
    return res


def bootstrap_ic(y, s_a, s_b, n=300, seed=0):
    """IC95% de la diferencia de PR-AUC (grafo - tabular) por bootstrap de test."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); d = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() == 0:
            continue
        d.append(average_precision_score(y[idx], s_b[idx]) - average_precision_score(y[idx], s_a[idx]))
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def main():
    print("=" * 64)
    print("DN-FN-01 Fraud Graph — protocolo temporal sin fuga + ablación")
    print("=" * 64)

    # ---------------- robustez: 5 semillas ----------------
    filas = []
    for sd in SEEDS_ROBUSTEZ:
        r = correr_semilla(sd)
        for m in ("tabular", "tabular+grafo"):
            filas.append({"seed": sd, "modelo": m, **{k: v for k, v in r[m].items()}})
        filas.append({"seed": sd, "modelo": "isolation_forest", **r["isolation_forest"]})
        print(f"seed {sd}: PR-AUC tabular={r['tabular']['pr_auc']:.3f} "
              f"grafo={r['tabular+grafo']['pr_auc']:.3f} iso={r['isolation_forest']['pr_auc']:.3f}")
    abl = pd.DataFrame(filas)
    abl.to_csv(os.path.join(OUT, "ablation_seeds.csv"), index=False)
    resumen = (abl.groupby("modelo")[["pr_auc", "roc_auc", "recall_flag", "precision_flag",
                                      "fpr_flag", "alto_precision"]]
               .agg(["mean", "std"]).round(4))
    print("\n", resumen.to_string())

    # ---------------- semilla principal (CSV publicado) ----------------
    df = pd.read_csv(os.path.join(BASE, "data", "transacciones_fraude_grafo.csv"), parse_dates=["timestamp"])
    res, f, (tr, va, te), mod = correr_semilla(SEED_PRINCIPAL, df=df, detalle=True)
    clf, cols, s_graf, ta, tm = mod["tabular+grafo"]
    _, _, s_tab, _, _ = mod["tabular"]
    ic = bootstrap_ic(te["is_fraude"].values, s_tab, s_graf)

    # SHAP nativo (valores por transacción)
    shap = clf.get_feature_importance(Pool(te[cols]), type="ShapValues")
    shap_v = shap[:, :-1]
    imp = pd.DataFrame({"feature": cols, "shap_importance": np.abs(shap_v).mean(axis=0)}) \
        .sort_values("shap_importance", ascending=False)
    imp["es_grafo"] = imp["feature"].isin(GRAPH_COLS)
    imp.to_csv(os.path.join(OUT, "feature_importance_shap.csv"), index=False)

    # top razones por transacción
    def razones(row):
        idx = np.argsort(-row)[:3]
        return "; ".join(f"{cols[i]}({row[i]:+.2f})" for i in idx if row[i] > 0)

    pred = te.copy()
    pred["riesgo"] = s_graf
    pred["decision"] = np.where(s_graf >= ta, "ALTO", np.where(s_graf >= tm, "MEDIO", "BAJO"))
    pred["riesgo_sin_grafo"] = s_tab
    pred["razones"] = [razones(r) for r in shap_v]
    pred.to_csv(os.path.join(OUT, "predicciones_test.csv"), index=False)

    # detección de anillos (nivel anillo, no transacción)
    anillos = []
    for aid, g in pred[pred["anillo_id"] >= 0].groupby("anillo_id"):
        anillos.append({"anillo": int(aid), "tx_test": len(g),
                        "detectado": bool((g["decision"] != "BAJO").any()),
                        "tx_marcadas": int((g["decision"] != "BAJO").sum())})
    por_tipo = {t: float((g["decision"] != "BAJO").mean())
                for t, g in pred[pred["is_fraude"] == 1].groupby("fraude_tipo")}

    clf.save_model(os.path.join(OUT, "model.cbm"))
    json.dump({"features": cols, "thr_alto": ta, "thr_medio": tm,
               "descripciones": __import__("features").DESCRIPCIONES},
              open(os.path.join(OUT, "model_meta.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    graf_rank = int(list(imp["feature"]).index("g_dist_a_fraude") + 1) if "g_dist_a_fraude" in list(imp["feature"]) else None
    shap_graf_pct = float(imp.loc[imp["es_grafo"], "shap_importance"].sum() / imp["shap_importance"].sum())

    metrics = {
        "protocolo": "split temporal 60/10/30, features sin fuga, retraso de etiquetas 24h",
        "datos": "SINTÉTICOS (generador propio, semilla 42)",
        "n": res["_info"], "umbral_alto": ta, "umbral_medio": tm,
        "fpr_objetivo_alto": FPR_ALTO, "fpr_objetivo_medio": FPR_MEDIO,
        "principal_tabular": res["tabular"], "principal_grafo": res["tabular+grafo"],
        "principal_isolation_forest": res["isolation_forest"],
        "delta_pr_auc_grafo_vs_tabular": res["tabular+grafo"]["pr_auc"] - res["tabular"]["pr_auc"],
        "delta_pr_auc_IC95_bootstrap": ic,
        "anillos": {"total_en_test": len(anillos), "detectados": sum(a["detectado"] for a in anillos),
                    "detalle": anillos},
        "recall_por_tipo_fraude": por_tipo,
        "shap_top_feature": imp.iloc[0]["feature"], "shap_peso_features_grafo": shap_graf_pct,
        "shap_rank_dist_a_fraude": graf_rank,
        "decision_buckets": {b: {"n": int((pred.decision == b).sum()),
                                 "fraudes": int(pred.loc[pred.decision == b, "is_fraude"].sum())}
                             for b in ("ALTO", "MEDIO", "BAJO")},
        "robustez_5_seeds": {m: {k: {"mean": float(abl[abl.modelo == m][k].mean()),
                                     "std": float(abl[abl.modelo == m][k].std())}
                                 for k in ("pr_auc", "roc_auc", "recall_flag", "precision_flag", "fpr_flag")}
                             for m in ("tabular", "tabular+grafo")},
    }
    json.dump(metrics, open(os.path.join(OUT, "metrics.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    print("\n--- PRINCIPAL (seed 42, test) ---")
    for k in ("tabular", "tabular+grafo"):
        r = res[k]
        print(f"{k:14s} PR-AUC={r['pr_auc']:.3f} ROC={r['roc_auc']:.3f} "
              f"recall@FPR<=1%={r['recall_flag']:.3f} prec={r['precision_flag']:.3f} "
              f"fpr={r['fpr_flag']:.4f} | ALTO n={r['alto_n']} prec={r['alto_precision']:.3f}")
    print(f"ΔPR-AUC grafo-tabular = {metrics['delta_pr_auc_grafo_vs_tabular']:+.3f}  IC95 {ic}")
    print(f"Anillos detectados: {metrics['anillos']['detectados']}/{metrics['anillos']['total_en_test']}")
    print("Recall por tipo:", {k: round(v, 3) for k, v in por_tipo.items()})
    print("SHAP top:\n", imp.head(8).to_string(index=False))
    print(f"Peso SHAP de features de grafo: {shap_graf_pct:.1%}")


if __name__ == "__main__":
    main()

"""
Benchmark de estrés del conocimiento de grafo: ¿qué sabe el modelo y dónde falla?

Cada escenario deforma los datos sintéticos de una forma concreta y compara
CatBoost solo-tabular vs CatBoost tabular+grafo con el MISMO protocolo temporal
(60/10/30, umbrales por FPR en validación). Se promedia sobre 3 semillas.

Escenarios (de favorable a adverso)
  base            referencia sin deformación
  retraso_etiq    las etiquetas llegan 0 h / 24 h / 72 h / 7 d / nunca (el grafo vive de ellas)
  ruido_etiq      se pierde 30% / 60% de las etiquetas de fraude (fraude no detectado nunca etiquetado)
  rotacion_infra  los defraudadores NO comparten dispositivos/IP (rotan en cada transacción)
  cambio_tactica  rotación de infraestructura solo en el periodo de test (táctica nueva tras el despliegue)
  inocentes       10% de usuarios legítimos comparten dispositivos/IP con la infraestructura criminal
  solo_aislado    sin anillos ni mulas: solo fraude aislado (el grafo no debería ayudar)
Además: ablación por grupo de features de grafo (qué conocimiento concreto aporta valor).

Salidas: modelo/benchmark_estres.csv, modelo/benchmark_estres.json
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "modelo")); sys.path.insert(0, os.path.join(BASE, "data"))
from features import construir_features, TABULAR_COLS, GRAPH_COLS, FEATURES_FULL  # noqa
from generar_datos_fraude_grafo import generar  # noqa
from train_eval import entrenar, umbral_fpr, FPR_ALTO, FPR_MEDIO, FRAC_TRAIN, FRAC_VAL  # noqa

SEEDS = [42, 7, 123]
OUT = os.path.join(BASE, "modelo")


def pools_criminales(df):
    f = df[df["fraude_tipo"].isin(["anillo", "mula"])]
    return list(f["device_id"].unique()), list(f["ip"].unique())


def rotar_infra(df, rng, desde=None):
    """Los defraudadores de red usan dispositivo/IP aleatorios únicos (sin infraestructura compartida)."""
    d = df.copy()
    m = d["fraude_tipo"].isin(["anillo", "mula"])
    if desde is not None:
        m &= d["timestamp"] >= desde
    d.loc[m, "device_id"] = [f"DEV_ROT_{i:06d}" for i in range(m.sum())]
    d.loc[m, "ip"] = [f"198.51.100.{i % 250}.{i}" for i in range(m.sum())]
    return d


def inocentes(df, rng, frac_users=0.10, p_tx=0.30):
    """Usuarios legítimos que comparten infraestructura criminal (wifi público, NAT, dispositivo prestado)."""
    d = df.copy(); devs, ips = pools_criminales(d)
    legit_users = d.loc[d["is_fraude"] == 0, "user_id"].unique()
    victims = set(rng.choice(legit_users, int(len(legit_users) * frac_users), replace=False))
    m = d["user_id"].isin(victims) & (d["is_fraude"] == 0) & (rng.random(len(d)) < p_tx)
    d.loc[m, "device_id"] = rng.choice(devs, m.sum()); d.loc[m, "ip"] = rng.choice(ips, m.sum())
    d["inocente"] = m.astype(int)
    return d


def evaluar(f_all, y_true, cols_extra=None, grupo_graf=None, split_ts=None):
    """f_all: features (con etiquetas de grafo posiblemente ruidosas en 'is_fraude'); y_true: etiqueta real."""
    n = len(f_all); i1, i2 = int(n * FRAC_TRAIN), int(n * (FRAC_TRAIN + FRAC_VAL))
    tr, va, te = f_all.iloc[:i1], f_all.iloc[i1:i2], f_all.iloc[i2:]
    res = {}
    modelos = {"tabular": TABULAR_COLS, "grafo": TABULAR_COLS + (grupo_graf if grupo_graf is not None else GRAPH_COLS)}
    for nombre, cols in modelos.items():
        clf = entrenar(tr[cols], tr["is_fraude"], 0)
        sv = clf.predict_proba(va[cols])[:, 1]; st = clf.predict_proba(te[cols])[:, 1]
        ta, tm = umbral_fpr(va["is_fraude"], sv, FPR_ALTO), umbral_fpr(va["is_fraude"], sv, FPR_MEDIO)
        tm = min(tm, ta)
        yt = y_true.loc[te.index].values; flag = st >= tm
        r = {"pr_auc": float(average_precision_score(yt, st)),
             "recall": float((flag & (yt == 1)).sum() / max(yt.sum(), 1)),
             "fpr": float((flag & (yt == 0)).sum() / max((yt == 0).sum(), 1))}
        for tipo in ("aislado", "mula", "anillo"):
            mk = (te["fraude_tipo"] == tipo).values
            r[f"recall_{tipo}"] = float(flag[mk].mean()) if mk.any() else np.nan
        if "inocente" in te.columns:
            mi = (te["inocente"] == 1).values & (yt == 0)
            r["fpr_inocentes"] = float(flag[mi].mean()) if mi.any() else np.nan
            mo = (te["inocente"] == 0).values & (yt == 0)
            r["fpr_resto"] = float(flag[mo].mean())
        res[nombre] = r
    return res


def escenario(nombre, seed):
    rng = np.random.default_rng(seed)
    df = generar(seed)
    delay, ruido, extra_cols, grupo = 24, 0.0, None, None
    if nombre.startswith("retraso_"):
        delay = {"0h": 0, "24h": 24, "72h": 72, "7d": 168, "nunca": 10 ** 6}[nombre.split("_")[1]]
        nombre_base = "retraso"
    elif nombre.startswith("ruido_"):
        ruido = float(nombre.split("_")[1]) / 100
    elif nombre == "rotacion_infra":
        df = rotar_infra(df, rng)
    elif nombre == "cambio_tactica":
        df = df.sort_values("timestamp").reset_index(drop=True)
        df = rotar_infra(df, rng, desde=df["timestamp"].iloc[int(len(df) * (FRAC_TRAIN + FRAC_VAL))])
    elif nombre == "inocentes":
        df = inocentes(df, rng)
    elif nombre == "solo_aislado":
        df = df[df["fraude_tipo"].isin(["normal", "aislado"])].reset_index(drop=True)
    elif nombre.startswith("grupo_"):
        grupo = {"grupo_comparticion": ["g_user_n_dispositivos", "g_user_n_ips", "g_dev_n_usuarios", "g_ip_n_usuarios"],
                 "grupo_historial_fraude": ["g_dev_fraude_previo", "g_ip_fraude_previo", "g_user_fraude_previo", "g_merchant_tasa_fraude"],
                 "grupo_distancia": ["g_dist_a_fraude"]}[nombre]
    verdad = df.set_index("transaction_id")["is_fraude"].copy()
    dg = df.copy()
    if ruido > 0:  # fraude que nunca se etiqueta: el sistema lo ve como legítimo
        fr = dg.index[dg["is_fraude"] == 1]
        dg.loc[rng.choice(fr, int(len(fr) * ruido), replace=False), "is_fraude"] = 0
    f = construir_features(dg, label_delay_h=delay)
    y_true = pd.Series(verdad.loc[f["transaction_id"]].values, index=f.index)
    return evaluar(f, y_true, grupo_graf=grupo)


ESCENARIOS = ["base", "retraso_0h", "retraso_72h", "retraso_7d", "retraso_nunca", "ruido_30", "ruido_60",
              "rotacion_infra", "cambio_tactica", "inocentes", "solo_aislado",
              "grupo_comparticion", "grupo_historial_fraude", "grupo_distancia"]


def main():
    filas = []
    for esc in ESCENARIOS:
        for sd in SEEDS:
            r = escenario(esc, sd)
            for m in ("tabular", "grafo"):
                filas.append({"escenario": esc, "seed": sd, "modelo": m, **r[m]})
        d = pd.DataFrame([x for x in filas if x["escenario"] == esc])
        a, b = d[d.modelo == "tabular"], d[d.modelo == "grafo"]
        print(f"{esc:24s} PR-AUC tab={a.pr_auc.mean():.3f} graf={b.pr_auc.mean():.3f} "
              f"d={b.pr_auc.mean() - a.pr_auc.mean():+.3f} | recall {a.recall.mean():.2f}->{b.recall.mean():.2f} "
              f"fpr {a.fpr.mean():.3f}->{b.fpr.mean():.3f}", flush=True)
    df = pd.DataFrame(filas); df.to_csv(os.path.join(OUT, "benchmark_estres.csv"), index=False)
    agg = df.groupby(["escenario", "modelo"]).mean(numeric_only=True).round(4)
    json.dump({"semillas": SEEDS, "resultados": json.loads(agg.reset_index().to_json(orient="records"))},
              open(os.path.join(OUT, "benchmark_estres.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()

"""
DN-FN-01 — Fraud Graph: real-time payment fraud detection dashboard.

Run:  streamlit run dashboard/app.py
All numbers come from modelo/metrics.json (produced by modelo/train_eval.py).
The "What-if" simulator scores with the REAL trained CatBoost model + native SHAP.
"""
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from catboost import CatBoostClassifier, Pool

st.set_page_config(page_title="Fraud Graph · DataNova 2026", layout="wide")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = os.path.join(BASE, "modelo")
COLORS = {"LOW": "#2ecc71", "MEDIUM": "#f1c40f", "HIGH": "#e74c3c"}
DEC_MAP = {"BAJO": "LOW", "MEDIO": "MEDIUM", "ALTO": "HIGH"}


@st.cache_data
def load():
    pred = pd.read_csv(os.path.join(M, "predicciones_test.csv"), parse_dates=["timestamp"])
    pred["decision"] = pred["decision"].map(DEC_MAP)
    data = pd.read_csv(os.path.join(BASE, "data", "transacciones_fraude_grafo.csv"), parse_dates=["timestamp"])
    metrics = json.load(open(os.path.join(M, "metrics.json"), encoding="utf-8"))
    shap_df = pd.read_csv(os.path.join(M, "feature_importance_shap.csv"))
    meta = json.load(open(os.path.join(M, "model_meta.json"), encoding="utf-8"))
    abl = pd.read_csv(os.path.join(M, "ablation_seeds.csv"))
    return pred, data, metrics, shap_df, meta, abl


@st.cache_resource
def load_model():
    clf = CatBoostClassifier()
    clf.load_model(os.path.join(M, "model.cbm"))
    return clf


pred, data, metrics, shap_df, meta, abl = load()
clf = load_model()
FEATS, THR_H, THR_M, DESC = meta["features"], meta["thr_alto"], meta["thr_medio"], meta["descripciones"]
G, T = metrics["principal_grafo"], metrics["principal_tabular"]


def decide(score):
    return "HIGH" if score >= THR_H else ("MEDIUM" if score >= THR_M else "LOW")


# ------------------------------------------------------------------ header
st.title("Fraud Graph — Real-Time Payment Fraud Detection")
st.caption("DataNova 2026 · Cybersecurity & Fraud Detection · CatBoost + transaction-graph features + SHAP · "
           "Synthetic main data, stress-tested, plus validation on real data (see tabs)")

tab_over, tab_alerts, tab_graph, tab_sim, tab_stress, tab_real, tab_method = st.tabs(
    ["Overview", "Alerts & explanations", "Fraud-ring graph", "What-if simulator",
     "Stress benchmark", "Real-data validation", "Methodology"])

# ------------------------------------------------------------------ overview
with tab_over:
    st.subheader("Does the graph actually help? Ablation on the held-out future")
    a, b, c, d = st.columns(4)
    a.metric("PR-AUC  (graph)", f"{G['pr_auc']:.3f}", f"{G['pr_auc'] - T['pr_auc']:+.3f} vs no-graph")
    b.metric("Recall @ FPR≈1%", f"{G['recall_flag']:.1%}", f"{G['recall_flag'] - T['recall_flag']:+.1%}")
    c.metric("Precision of HIGH (block)", f"{G['alto_precision']:.1%}")
    d.metric("False-positive rate", f"{G['fpr_flag']:.2%}", f"{G['fpr_flag'] - T['fpr_flag']:+.2%}", delta_color="inverse")
    lo, hi = metrics["delta_pr_auc_IC95_bootstrap"]
    r5 = metrics["robustez_5_seeds"]
    st.info(
        f"**Held-out test = the future** ({metrics['n']['n_test']:,} transactions, {metrics['n']['fraude_test']} frauds). "
        f"Graph features raise PR-AUC by **{metrics['delta_pr_auc_grafo_vs_tabular']:+.3f}** "
        f"(bootstrap 95% CI [{lo:+.3f}, {hi:+.3f}]). Over 5 independent synthetic datasets: "
        f"PR-AUC {r5['tabular']['pr_auc']['mean']:.3f} ± {r5['tabular']['pr_auc']['std']:.3f} (no graph) → "
        f"**{r5['tabular+grafo']['pr_auc']['mean']:.3f} ± {r5['tabular+grafo']['pr_auc']['std']:.3f}** (graph)."
    )
    c1, c2 = st.columns(2)
    with c1:
        tbl = abl[abl.modelo.isin(["tabular", "tabular+grafo", "isolation_forest"])].copy()
        tbl["modelo"] = tbl["modelo"].map({"tabular": "CatBoost · tabular only",
                                           "tabular+grafo": "CatBoost · tabular + graph",
                                           "isolation_forest": "IsolationForest (unsupervised)"})
        fig = px.box(tbl, x="modelo", y="pr_auc", points="all", color="modelo",
                     labels={"pr_auc": "PR-AUC (test)", "modelo": ""},
                     color_discrete_sequence=["#95a5a6", "#2e86de", "#bdc3c7"])
        fig.update_layout(showlegend=False, title="PR-AUC across 5 dataset seeds", height=380)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        sh = shap_df.head(10).copy()
        sh["feature"] = sh["feature"].map(lambda f: DESC.get(f, f))
        fig2 = px.bar(sh[::-1], x="shap_importance", y="feature", orientation="h",
                      color="es_grafo", color_discrete_map={True: "#2e86de", False: "#95a5a6"},
                      labels={"shap_importance": "mean |SHAP|", "feature": "", "es_grafo": "graph feature"})
        fig2.update_layout(title=f"What drives the model "
                                 f"(graph features = {metrics['shap_peso_features_grafo']:.0%} of total SHAP)", height=380)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Decision engine")
    bk = metrics["decision_buckets"]
    cols = st.columns(3)
    for col, (k, lab, act) in zip(cols, [("ALTO", "HIGH", "Block"), ("MEDIO", "MEDIUM", "Manual review"), ("BAJO", "LOW", "Approve")]):
        n, f = bk[k]["n"], bk[k]["fraudes"]
        col.metric(f"{lab} → {act}", f"{n:,} tx", f"{f} frauds ({f / max(n, 1):.1%})", delta_color="off")
    rt = metrics["recall_por_tipo_fraude"]
    ra = metrics["anillos"]
    st.caption(f"Share of frauds flagged (MEDIUM+HIGH) by type — isolated: {rt.get('aislado', 0):.0%} · "
               f"mule accounts: {rt.get('mula', 0):.0%} · ring members: {rt.get('anillo', 0):.0%}. "
               f"Fraud rings with ≥1 flagged transaction: {ra['detectados']}/{ra['total_en_test']}. "
               f"Honest limitation: isolated frauds (no network footprint) remain the hardest.")

# ------------------------------------------------------------------ alerts
with tab_alerts:
    st.subheader("Alert queue")
    f1, f2 = st.columns([2, 1])
    sel_dec = f1.multiselect("Decision", ["HIGH", "MEDIUM", "LOW"], default=["HIGH", "MEDIUM"])
    only_f = f2.checkbox("Only real frauds (ground truth)")
    q = pred[pred["decision"].isin(sel_dec)]
    if only_f:
        q = q[q["is_fraude"] == 1]
    q = q.sort_values("riesgo", ascending=False)
    show = q[["transaction_id", "timestamp", "user_id", "merchant_id", "amount", "decision", "riesgo",
              "is_fraude", "fraude_tipo", "razones"]].rename(columns={
        "riesgo": "risk", "is_fraude": "is_fraud", "fraude_tipo": "fraud_type", "razones": "top SHAP reasons"})
    st.dataframe(show, use_container_width=True, height=330, hide_index=True)

    st.subheader("Why was it flagged? (per-transaction SHAP)")
    if len(q):
        tx = st.selectbox("Transaction", q["transaction_id"].head(200))
        row = pred[pred["transaction_id"] == tx].iloc[0]
        sv = clf.get_feature_importance(Pool(pd.DataFrame([row[FEATS]], columns=FEATS).astype(float)),
                                        type="ShapValues")[0][:-1]
        ex = pd.DataFrame({"feature": [DESC.get(f, f) for f in FEATS], "contribution": sv,
                           "value": [row[f] for f in FEATS]}).sort_values("contribution", key=abs, ascending=False).head(8)
        fig = px.bar(ex[::-1], x="contribution", y="feature", orientation="h",
                     color=ex[::-1]["contribution"] > 0,
                     color_discrete_map={True: "#e74c3c", False: "#2ecc71"},
                     hover_data=["value"], labels={"contribution": "SHAP (log-odds)", "feature": ""})
        fig.update_layout(showlegend=False, height=360,
                          title=f"{tx} · risk {row['riesgo']:.2f} → {row['decision']} · "
                                f"ground truth: {'FRAUD (' + row['fraude_tipo'] + ')' if row['is_fraude'] else 'legitimate'}")
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------ graph
with tab_graph:
    st.subheader("Network around an alert: users ↔ devices ↔ IPs")
    st.caption("Everything the account has touched, up to the moment of the alert (2 hops). "
               "Red = node linked to a confirmed fraud · Blue = no fraud · large outlined node = the flagged user.")
    cand = pred[pred["decision"].isin(["HIGH", "MEDIUM"])].sort_values("riesgo", ascending=False)
    if cand.empty:
        st.info("No alerts.")
    else:
        opts = (cand["transaction_id"] + " · " + cand["user_id"] + " · " + cand["fraude_tipo"]).tolist()
        pick = st.selectbox("Alert", opts[:150])
        r = cand[cand["transaction_id"] == pick.split(" · ")[0]].iloc[0]
        hist = data[data["timestamp"] <= r["timestamp"]]
        Gx = nx.Graph()
        fraud_nodes = set()
        for t in hist.itertuples(index=False):
            u, d, i = f"U:{t.user_id}", f"D:{t.device_id}", f"I:{t.ip}"
            Gx.add_edge(u, d); Gx.add_edge(u, i)
            if t.is_fraude == 1:
                fraud_nodes.update([u, d, i])
        center = f"U:{r['user_id']}"
        sub = nx.ego_graph(Gx, center, radius=2)
        if sub.number_of_nodes() > 120:
            sub = nx.ego_graph(Gx, center, radius=1)
        pos = nx.spring_layout(sub, seed=7, k=0.6)
        ex, ey = [], []
        for u, v in sub.edges():
            ex += [pos[u][0], pos[v][0], None]; ey += [pos[u][1], pos[v][1], None]
        sym = {"U": "circle", "D": "square", "I": "diamond"}
        figg = go.Figure(go.Scatter(x=ex, y=ey, mode="lines", line=dict(width=0.7, color="#bdc3c7"), hoverinfo="none", showlegend=False))
        for kind, name in (("U", "User"), ("D", "Device"), ("I", "IP")):
            ns = [n for n in sub.nodes if n.startswith(kind + ":")]
            figg.add_trace(go.Scatter(
                x=[pos[n][0] for n in ns], y=[pos[n][1] for n in ns], mode="markers", name=name,
                text=[n[2:] + (" (flagged)" if n == center else "") for n in ns], hoverinfo="text",
                marker=dict(symbol=sym[kind], size=[22 if n == center else 11 for n in ns],
                            color=["#e74c3c" if n in fraud_nodes else "#3498db" for n in ns],
                            line=dict(width=[3 if n == center else 1 for n in ns], color="#2c3e50"))))
        figg.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0), hovermode="closest",
                           xaxis=dict(visible=False), yaxis=dict(visible=False),
                           legend=dict(orientation="h", y=1.02))
        st.plotly_chart(figg, use_container_width=True)
        st.caption(f"{sub.number_of_nodes()} nodes · {sub.number_of_edges()} edges · "
                   f"risk {r['riesgo']:.2f} ({r['decision']}) · "
                   f"distance to confirmed fraud: {int(r['g_dist_a_fraude'])} hops · "
                   f"users sharing this device: {int(r['g_dev_n_usuarios'])}")

# ------------------------------------------------------------------ simulator
with tab_sim:
    st.subheader("What-if simulator — scored by the real model")
    st.caption("Start from a real test transaction, change what you like, and see the live risk score with SHAP. "
               "Graph context (shared devices, known-fraud neighbours) can be changed too.")
    base_id = st.selectbox("Base transaction", pred.sort_values("riesgo")["transaction_id"].tolist()[::50]
                           + pred[pred["decision"] == "HIGH"]["transaction_id"].head(10).tolist())
    b0 = pred[pred["transaction_id"] == base_id].iloc[0]
    avg1 = b0["amount"] / max(b0["amount_vs_user_avg"], 1e-9)  # avg histórico + 1
    c1, c2, c3 = st.columns(3)
    amount = c1.number_input("Amount", 1.0, 20000.0, float(round(b0["amount"], 2)), 10.0)
    hour = c1.slider("Hour of day", 0, 23, int(b0["hour"]))
    new_dev = c2.selectbox("Device new for this user?", [0, 1], index=int(b0["nuevo_dispositivo_para_user"]))
    new_ip = c2.selectbox("IP new for this user?", [0, 1], index=int(b0["nuevo_ip_para_user"]))
    dev_fraud = c3.number_input("Prior confirmed frauds on this device", 0, 20, int(b0["g_dev_fraude_previo"]))
    dev_users = c3.number_input("Other users sharing this device", 0, 30, int(b0["g_dev_n_usuarios"]))
    x = b0[FEATS].astype(float).copy()
    x["amount"], x["hour"] = amount, hour
    x["hora_no_laboral"] = float(hour < 6 or hour > 22)
    x["amount_vs_user_avg"] = amount / avg1
    x["nuevo_dispositivo_para_user"], x["nuevo_ip_para_user"] = new_dev, new_ip
    x["g_dev_fraude_previo"], x["g_dev_n_usuarios"] = dev_fraud, dev_users
    x["g_dist_a_fraude"] = 0 if dev_fraud > 0 else b0["g_dist_a_fraude"]
    X = pd.DataFrame([x[FEATS]], columns=FEATS)
    score = float(clf.predict_proba(X)[:, 1][0])
    sv = clf.get_feature_importance(Pool(X), type="ShapValues")[0][:-1]
    d_ = decide(score)
    m1, m2 = st.columns(2)
    m1.metric("Risk score", f"{score:.3f}")
    m2.markdown(f"### Decision: **{d_}** "
                f"({'block' if d_ == 'HIGH' else 'manual review' if d_ == 'MEDIUM' else 'approve'})")
    ex = pd.DataFrame({"feature": [DESC.get(f, f) for f in FEATS], "contribution": sv}) \
        .sort_values("contribution", key=abs, ascending=False).head(8)
    fig = px.bar(ex[::-1], x="contribution", y="feature", orientation="h",
                 color=ex[::-1]["contribution"] > 0, color_discrete_map={True: "#e74c3c", False: "#2ecc71"},
                 labels={"contribution": "SHAP (log-odds)", "feature": ""})
    fig.update_layout(showlegend=False, height=340)
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------ stress benchmark
with tab_stress:
    st.subheader("Pushing the graph to its limits: what it knows and what it does not")
    st.caption("Each scenario deforms the synthetic data in one specific way. Same protocol as the main results; "
               "3 dataset seeds. Gain = PR-AUC(graph model) minus PR-AUC(tabular model).")
    sb = pd.read_csv(os.path.join(M, "benchmark_estres.csv"))
    pv = sb.groupby(["escenario", "modelo"])[["pr_auc", "recall", "fpr"]].mean().unstack()
    names = {"base": "Baseline", "retraso_7d": "Labels arrive after 7 days", "retraso_nunca": "Labels never arrive",
             "ruido_30": "30% of fraud never labelled", "ruido_60": "60% of fraud never labelled",
             "rotacion_infra": "Criminals rotate devices/IPs", "cambio_tactica": "Tactic shift after deployment",
             "inocentes": "Innocents share criminal infrastructure", "solo_aislado": "Only isolated fraud",
             "grupo_historial_fraude": "Only fraud-history features", "grupo_comparticion": "Only device/IP sharing features",
             "grupo_distancia": "Only hops-to-fraud feature"}
    rows = []
    for k, lab in names.items():
        rows.append({"Scenario": lab, "PR-AUC tabular": pv.loc[k, ("pr_auc", "tabular")],
                     "PR-AUC graph": pv.loc[k, ("pr_auc", "grafo")],
                     "Gain": pv.loc[k, ("pr_auc", "grafo")] - pv.loc[k, ("pr_auc", "tabular")],
                     "Recall tabular": pv.loc[k, ("recall", "tabular")], "Recall graph": pv.loc[k, ("recall", "grafo")]})
    tb = pd.DataFrame(rows)
    figs = px.bar(tb[::-1], x="Gain", y="Scenario", orientation="h", text=tb[::-1]["Gain"].map(lambda v: f"{v:+.3f}"),
                  labels={"Gain": "Gain in PR-AUC from graph features"})
    figs.update_traces(marker_color="#2e86de"); figs.update_layout(height=470, yaxis_title="")
    st.plotly_chart(figs, use_container_width=True)
    st.dataframe(tb.round(3), use_container_width=True, hide_index=True)
    ino = sb[(sb.escenario == "inocentes") & (sb.modelo == "grafo")]
    st.markdown(f"""
**What the graph knows**
- Fraud history attached to devices, IPs and users: it carries essentially the entire gain.
- Mule accounts and rings that reuse infrastructure: recall of mules rises from about 52% to 100%.
- It tolerates delayed and missing labels (up to 60% of fraud never labelled) and keeps its advantage.

**What it does not know, and how it fails**
- Without confirmed-fraud feedback the gain shrinks from +0.135 to +0.031: the graph is mostly a memory of past fraud.
- If criminals rotate devices/IPs, or only isolated fraud exists, the gain is close to zero (+0.027, +0.004).
- Hops-to-fraud alone adds almost nothing (+0.012); sharing counts alone add little (+0.037).
- Innocent users who share criminal infrastructure are flagged at {ino['fpr_inocentes'].mean():.1%} versus
  {ino['fpr_resto'].mean():.1%} for everyone else, about 4x: guilt by association. This is why MEDIUM goes to a human
  and only strong evidence blocks.
""")

# ------------------------------------------------------------------ real data
with tab_real:
    st.subheader("Validation on real data: Elliptic Bitcoin dataset")
    rv = json.load(open(os.path.join(M, "real_validation.json"), encoding="utf-8"))
    rs = rv["resumen_5_semillas"]
    st.caption(f"{rv['nodos']:,} real transactions, {rv['aristas']:,} edges, {rv['etiquetados']:,} labelled "
               f"({rv['ilicitos']:,} illicit). Source: {rv['fuente']}. Train = time steps 1-34, test = steps 35-49.")
    nm = {"A_local": "Local features", "B_local+grafo": "Local + graph context",
          "C_local+agregados(Weber)": "Local + neighbour aggregates (original paper)", "D_agregados+grafo": "Aggregates + graph context"}
    t = pd.DataFrame([{"Model": nm[k], "PR-AUC": f"{v['pr_auc']['mean']:.3f} ± {v['pr_auc']['std']:.3f}",
                       "Recall @ FPR 1%": f"{v['recall_fpr1']['mean']:.3f}", "F1 (illicit)": f"{v['f1_illicit']['mean']:.3f}"}
                      for k, v in rs.items()])
    st.dataframe(t, use_container_width=True, hide_index=True)
    lo, hi = rv["ic95_B_vs_A_seed42"]
    st.info(f"Graph context adds **{rv['ganancia_pr_auc_B_vs_A']:+.3f}** PR-AUC over local features "
            f"(bootstrap 95% CI [{lo:+.3f}, {hi:+.3f}]) and {rv['ganancia_pr_auc_D_vs_C']:+.3f} over the original paper's "
            f"neighbour aggregates. The effect is real but small on this sparse graph.")
    stp = pd.read_csv(os.path.join(M, "real_validation_steps.csv"))
    figr = go.Figure()
    figr.add_trace(go.Scatter(x=stp["paso"], y=stp["pr_auc_local"], name="Local features", mode="lines+markers"))
    figr.add_trace(go.Scatter(x=stp["paso"], y=stp["pr_auc_grafo"], name="Local + graph context", mode="lines+markers"))
    figr.add_vline(x=43, line_dash="dash", line_color="#e74c3c", annotation_text="step 43: dark-market shutdown")
    figr.update_layout(height=380, xaxis_title="Test time step", yaxis_title="PR-AUC (illicit)", yaxis_range=[0, 1.05])
    st.plotly_chart(figr, use_container_width=True)
    st.warning("Both models are strong before step 43 (PR-AUC about 0.92) and collapse after it (about 0.13-0.18): "
               "a real concept drift. The graph does not protect against a change in criminal behaviour, which is why a "
               "production system needs drift monitoring and retraining.")
    st.caption("Honest scope: Elliptic is Bitcoin transaction data, not card payments; graph edges only link transactions "
               "within the same time step, so labels of neighbours are never used (that would be leakage). "
               "We propagate the predicted risk of a first-stage model instead.")

# ------------------------------------------------------------------ methodology
with tab_method:
    st.subheader("Methodology & honesty notes")
    st.markdown(f"""
**Problem.** Rules and per-transaction models miss *coordinated* fraud (mule accounts, rings sharing devices/IPs)
because each payment looks ordinary in isolation.

**Idea.** Build a live user–device–IP–merchant graph and feed *network context* to the classifier:
shared devices/IPs, accounts per device, distance (hops) to a confirmed-fraud node, merchant fraud rate.

**Leak-free protocol.** Features are computed chronologically using only the past; fraud labels reach the graph
**24 h late** (as chargebacks/reviews do). Split is temporal: 60% train · 10% validation (picks thresholds) · 30% test
(the future, evaluated once). Thresholds (validation): HIGH = FPR ≤ 0.3% (block); MEDIUM = FPR ≤ 2% (manual review)
(HIGH = {THR_H:.3f}, MEDIUM = {THR_M:.3f}).

**Evidence.** Ablation with identical settings (tabular vs tabular+graph), repeated on 5 independent datasets,
bootstrap CI on the difference, SHAP attribution, unsupervised baseline (IsolationForest).

**Limitations (stated openly).**
- Main data are **synthetic** (own generator in `data/`): absolute numbers will not transfer to a real bank. On the
  **real** Elliptic dataset the graph gain is small (about +0.014 PR-AUC) and a real concept drift hurts both models;
  both are reported in their own tabs, together with a stress benchmark that shows where the graph stops helping.
- Isolated frauds with no network footprint are caught mostly by tabular signals (≈{rt.get('aislado', 0):.0%} flagged).
- Rings in the test period: {ra['total_en_test']} — small sample; hence we report 5-seed robustness instead of a single run.

**Stack.** Python · CatBoost · SHAP (CatBoost native) · scikit-learn · NetworkX/Plotly · pandas · Streamlit.
""")

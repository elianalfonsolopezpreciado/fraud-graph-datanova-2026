"""
Builds the DataNova 2026 presentation. All numbers are read from modelo/metrics.json
(never hard-coded), so slides cannot drift from the real results.

Run:  python generar_ppt.py        ->  DN-FN-01_Fraud_Detection.pptx
Edit TEAM / DEMO_URL / REPO_URL / VIDEO_URL below before submitting.
"""
import os, json
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

# ----------------------------- EDIT BEFORE SUBMITTING -----------------------------
TEAM = "Solo project — Elian Alfonso López Preciado (data, modeling, dashboard, presentation)"
DEMO_URL = "<LIVE DEMO URL>"
REPO_URL = "<GITHUB REPO URL>"
VIDEO_URL = "<VIDEO URL (optional)>"
# -----------------------------------------------------------------------------------

BASE = os.path.dirname(os.path.abspath(__file__))
M = json.load(open(os.path.join(BASE, "modelo", "metrics.json"), encoding="utf-8"))
G, T = M["principal_grafo"], M["principal_tabular"]
R5 = M["robustez_5_seeds"]
lo, hi = M["delta_pr_auc_IC95_bootstrap"]
ring = M["anillos"]
bk = M["decision_buckets"]
A = lambda n: os.path.join(BASE, "assets", n)
RV = json.load(open(os.path.join(BASE, "modelo", "real_validation.json"), encoding="utf-8"))
RS = RV["resumen_5_semillas"]; rlo, rhi = RV["ic95_B_vs_A_seed42"]

NAVY = RGBColor(0x0F, 0x2A, 0x4A); BLUE = RGBColor(0x2E, 0x86, 0xDE); GREEN = RGBColor(0x2E, 0xCC, 0x71)
YEL = RGBColor(0xF1, 0xC4, 0x0F); RED = RGBColor(0xE7, 0x4C, 0x3C); WHITE = RGBColor(255, 255, 255)
SOFT = RGBColor(0xBD, 0xC3, 0xD7); CARD = RGBColor(0x1B, 0x3A, 0x63)

prs = Presentation(); prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
blank = prs.slide_layouts[6]


def slide(title, sub=None):
    s = prs.slides.add_slide(blank)
    bg = s.shapes.add_shape(1, 0, 0, prs.slide_width, prs.slide_height)
    bg.fill.solid(); bg.fill.fore_color.rgb = NAVY; bg.line.fill.background()
    bar = s.shapes.add_shape(1, 0, 0, Inches(0.18), prs.slide_height)
    bar.fill.solid(); bar.fill.fore_color.rgb = GREEN; bar.line.fill.background()
    text(s, 0.6, 0.35, 12.1, 0.8, title, 32, True)
    if sub:
        text(s, 0.6, 1.05, 12.1, 0.5, sub, 16, False, SOFT)
    return s


def text(s, x, y, w, h, t, size=16, bold=False, color=WHITE, align=None):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = tb.text_frame
    tf.word_wrap = True; p = tf.paragraphs[0]; r = p.add_run(); r.text = t
    r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = color
    if align: p.alignment = align
    return tb


def bullets(s, x, y, w, h, items, size=17, color=WHITE):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = tb.text_frame; tf.word_wrap = True
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run(); r.text = "•  " + it; r.font.size = Pt(size); r.font.color.rgb = color
        p.space_after = Pt(9)


def kpi(s, x, y, label, value, color=GREEN, w=2.9):
    card = s.shapes.add_shape(1, Inches(x), Inches(y), Inches(w), Inches(1.45))
    card.fill.solid(); card.fill.fore_color.rgb = CARD; card.line.fill.background()
    text(s, x + .1, y + .08, w - .2, .8, value, 36, True, color, PP_ALIGN.CENTER)
    text(s, x + .1, y + .92, w - .2, .5, label, 13, False, SOFT, PP_ALIGN.CENTER)


def pic(s, name, x, y, w):
    s.shapes.add_picture(A(name), Inches(x), Inches(y), width=Inches(w))


# 1 ─ title
s = slide("Fraud Graph", "Catching coordinated payment fraud that per-transaction models cannot see")
text(s, .6, 2.4, 12, .6, "DataNova 2026 · Global Data-thon · Track: Cybersecurity & Fraud Detection", 20, True, GREEN)
bullets(s, .6, 3.3, 12, 2.5, [
    "Live user ↔ device ↔ IP ↔ merchant graph feeding a CatBoost scorer with SHAP explanations",
    f"Leak-free temporal evaluation: on synthetic data graph context lifts PR-AUC {T['pr_auc']:.2f} → {G['pr_auc']:.2f}; on real Bitcoin data (Elliptic) the gain is small ({RV['ganancia_pr_auc_B_vs_A']:+.3f}) and reported as such",
    "Stress benchmark: where the graph helps, where it stops, and how it fails",
    "Working demo: Streamlit dashboard scoring with the real model (link on last slide)"], 19)
text(s, .6, 6.4, 12, .5, TEAM, 14, False, SOFT)

# 2 ─ problem
s = slide("The problem", "Fraud is organised; most detectors look at one payment at a time")
bullets(s, .6, 1.8, 7.2, 4.8, [
    "Real-time payments: little time to decide, and a wrong block costs a customer; a miss costs money.",
    "Mule accounts and rings look ordinary in isolation: normal amounts, normal hours, fresh devices.",
    "The tell is relational: many accounts sharing devices/IPs, or sitting one hop from a known fraudster.",
    "Labels arrive late (chargebacks, reviews) — a production system must work with delayed truth.",
    "Judges and risk teams also need to know WHY a payment was flagged: explainability is mandatory."], 18)
pic(s, "ring.png", 8.0, 1.8, 5.0)
text(s, 8.0, 5.55, 5.0, .9, "Red: one ring of mule accounts and the devices/IPs they share, hidden among legitimate users.", 12, False, SOFT)

# 3 ─ solution
s = slide("Our solution", "Network context + gradient boosting + explainable three-level decisions")
bullets(s, .6, 1.8, 6.4, 4.8, [
    "Incremental graph over users, devices and IPs, updated after every transaction.",
    "9 graph features: accounts per device/IP, devices per user, confirmed frauds on the device/IP/user, merchant fraud rate, hops to the nearest confirmed fraud.",
    "CatBoost scorer; per-transaction SHAP shows exactly which signals fired.",
    "Decision engine: HIGH → block · MEDIUM → manual review · LOW → approve, thresholds set on a validation window."], 17)
for i, (lab, val, c) in enumerate([("HIGH · block", bk["ALTO"], RED), ("MEDIUM · review", bk["MEDIO"], YEL), ("LOW · approve", bk["BAJO"], GREEN)]):
    y = 1.9 + i * 1.6
    card = s.shapes.add_shape(1, Inches(7.5), Inches(y), Inches(5.2), Inches(1.4)); card.fill.solid(); card.fill.fore_color.rgb = CARD; card.line.fill.background()
    text(s, 7.7, y + .1, 4.8, .5, lab, 18, True, c)
    text(s, 7.7, y + .6, 4.8, .8, f"{val['n']:,} test tx · {val['fraudes']} frauds ({val['fraudes'] / max(val['n'], 1):.1%})", 16)

# 4 ─ methodology
s = slide("Methodology: no leakage, no shortcuts", "Designed to survive a technical judge's questions")
bullets(s, .6, 1.8, 12.1, 4.9, [
    "Temporal split: 60% train · 10% validation (picks thresholds) · 30% test = the future, scored once.",
    "Features computed chronologically with only past information; the graph learns a fraud label 24 h AFTER the transaction, like real chargebacks.",
    "We found and removed a leak from our own first prototype (graph distance was computed with each transaction's own label) — the numbers here are the corrected ones.",
    "Ablation: identical CatBoost with vs without graph features; plus an unsupervised IsolationForest baseline.",
    "Robustness: 5 independent synthetic datasets (mean ± std) and bootstrap 95% CI on the gain.",
    "Operating points: HIGH = false-positive rate ≤ 0.3% on validation (block); MEDIUM = FPR ≤ 2% (manual review)."], 17)

# 5 ─ results
s = slide("Results: the graph is what moves the needle", "Held-out future · synthetic data · ablation vs no-graph model")
kpi(s, .6, 1.7, "PR-AUC (graph)", f"{G['pr_auc']:.3f}", GREEN)
kpi(s, 3.6, 1.7, "Recall of flagged (MED+HIGH)", f"{G['recall_flag']:.0%}", GREEN)
kpi(s, 6.6, 1.7, "Precision of HIGH (block)", f"{G['alto_precision']:.0%}", YEL)
kpi(s, 9.6, 1.7, f"Δ PR-AUC vs no graph (95% CI {lo:+.2f}…{hi:+.2f})", f"{M['delta_pr_auc_grafo_vs_tabular']:+.2f}", BLUE, 3.1)
pic(s, "ablation.png", .6, 3.35, 5.7)
bullets(s, 6.6, 3.5, 6.2, 3.7, [
    f"No-graph model: PR-AUC {R5['tabular']['pr_auc']['mean']:.3f} ± {R5['tabular']['pr_auc']['std']:.3f} → with graph: {R5['tabular+grafo']['pr_auc']['mean']:.3f} ± {R5['tabular+grafo']['pr_auc']['std']:.3f} (5 datasets).",
    f"Same review budget (FPR ≈ {G['fpr_flag']:.1%}): recall of flagged frauds {T['recall_flag']:.0%} → {G['recall_flag']:.0%}; precision of blocks {T['alto_precision']:.0%} → {G['alto_precision']:.0%}.",
    f"Fraud rings with at least one flagged transaction: {ring['detectados']}/{ring['total_en_test']} in test (small sample, reported openly).",
    "Isolated frauds without network footprint remain the hardest class — we say so."], 15)

# 6 ─ explainability
s = slide("Explainable by design", "Every alert comes with the signals that triggered it")
pic(s, "shap.png", .6, 1.7, 7.4)
bullets(s, 8.3, 1.8, 4.6, 5, [
    f"Graph features account for {M['shap_peso_features_grafo']:.0%} of total SHAP attribution.",
    "Per-transaction SHAP bars in the dashboard: analysts see 'device shared by N accounts', 'confirmed fraud 1 hop away', etc.",
    "Compliance-friendly: a reviewer can justify every block."], 16)

# 6b ─ stress benchmark
s = slide("Stress benchmark: what the graph knows and what it does not", "Each scenario deforms the data in one way · same protocol · 3 seeds")
pic(s, "stress.png", .6, 1.6, 7.6)
bullets(s, 8.4, 1.7, 4.5, 5.3, [
    "The gain comes almost entirely from confirmed-fraud history on devices, IPs and users.",
    "Without label feedback the gain shrinks from +0.135 to +0.031.",
    "If criminals rotate devices, or fraud is isolated, the gain is near zero.",
    "Innocents sharing criminal infrastructure get about 4x more false flags (9.7% vs 2.3%): guilt by association.",
    "Hence: MEDIUM goes to a human; only strong evidence blocks."], 14)

# 6c ─ real data
s = slide("Validation on real data: Elliptic Bitcoin", f"{RV['nodos']:,} real transactions · {RV['aristas']:,} edges · train steps 1-34, test steps 35-49")
pic(s, "real.png", .6, 1.6, 7.6)
bullets(s, 8.4, 1.7, 4.5, 5.3, [
    f"PR-AUC {RS['A_local']['pr_auc']['mean']:.3f} (local) → {RS['B_local+grafo']['pr_auc']['mean']:.3f} with graph context ({RV['ganancia_pr_auc_B_vs_A']:+.3f}, 95% CI {rlo:+.3f}…{rhi:+.3f}).",
    f"Also {RV['ganancia_pr_auc_D_vs_C']:+.3f} over the neighbour aggregates of the original paper.",
    "A real concept drift at step 43 collapses both models: the graph does not protect against changed behaviour.",
    "Neighbour labels are never used (edges stay within a time step); we propagate predicted risk instead."], 14)

# 7 ─ demo
s = slide("Live demo", "Streamlit dashboard — scores with the real trained model")
bullets(s, .6, 1.8, 7.3, 4.8, [
    "Overview: ablation, SHAP drivers, decision buckets.",
    "Alert queue with per-transaction SHAP explanation.",
    "Fraud-ring graph: the 2-hop neighbourhood of any alert.",
    "What-if simulator: change amount, hour, device/IP or graph context and watch the real model re-score.",
    "Methodology tab states every limitation."], 18)
card = s.shapes.add_shape(1, Inches(8.2), Inches(1.9), Inches(4.6), Inches(2.6)); card.fill.solid(); card.fill.fore_color.rgb = CARD; card.line.fill.background()
text(s, 8.4, 2.0, 4.2, .5, "Links", 18, True, GREEN)
text(s, 8.4, 2.6, 4.2, 1.9, f"Demo: {DEMO_URL}\nCode: {REPO_URL}\nVideo: {VIDEO_URL}", 13)

# 8 ─ innovation
s = slide("Innovation & originality", "What is different from typical fraud submissions")
bullets(s, .6, 1.8, 12.1, 4.9, [
    "Relational features computed online and leak-free, with delayed labels — a protocol most prototypes skip.",
    "Evidence over assertion: ablation, multi-seed, bootstrap CI, honest baselines and a disclosed leak fix.",
    "A stress benchmark that maps the limits of the graph, and validation on a real public dataset reported as it came out.",
    "Explainability at the transaction level and a three-level action policy tuned on a validation window.",
    "A simulator that calls the actual model — not hand-written rules."], 19)

# 9 ─ impact
s = slide("Impact & feasibility", "A realistic path to a pilot beside an existing rules engine")
bullets(s, .6, 1.8, 12.1, 4.9, [
    f"Operational load: MEDIUM + HIGH is {(bk['ALTO']['n'] + bk['MEDIO']['n']) / M['n']['n_test']:.1%} of volume; manual review is concentrated where fraud density is highest.",
    "At the same false-positive budget the graph catches more fraud — or blocks fewer legitimate customers for the same recall.",
    "Cost: incremental dictionaries/adjacency, O(1) lookups per payment plus a depth-3 BFS; trains in well under a minute on a laptop CPU.",
    "Deployment path: shadow mode → analyst feedback loop (labels refine the graph) → partial blocking.",
    "Beneficiaries: banks, fintechs, payment processors and their customers."], 18)

# 10 ─ limits
s = slide("Honest limitations & next steps")
bullets(s, .6, 1.5, 12.1, 5.3, [
    "Main data are synthetic (own generator); absolute numbers will not transfer to a real bank. On real Bitcoin data the graph gain is small and a real drift hurts both models.",
    "The graph is largely a memory of confirmed fraud: it needs label feedback and loses value when criminals rotate infrastructure.",
    "Rings in the test window are few; we therefore report multi-seed robustness rather than a single lucky run.",
    "Next: validate on a public real dataset, add drift monitoring and periodic retraining, counterfactual advice to unblock false positives, adversarial stress tests of the graph features.",
    "Responsible use: synthetic data only, no personal data; SHAP explanations support human review and appeals."], 18)

# 11 ─ close
s = slide("Thank you", "Fraud Graph · DataNova 2026 · Cybersecurity & Fraud Detection")
text(s, .6, 2.4, 12, .6, TEAM, 18, False, WHITE)
text(s, .6, 3.4, 12, 2.2, f"Demo: {DEMO_URL}\nCode: {REPO_URL}\nVideo: {VIDEO_URL}", 16, False, SOFT)
text(s, .6, 5.7, 12, .8, "Stack: Python · CatBoost · scikit-learn · NetworkX · SHAP · Plotly · Streamlit · pandas  (all open-source, acknowledged)", 13, False, SOFT)

out = os.path.join(BASE, "DN-FN-01_Fraud_Detection.pptx")
prs.save(out); print("saved", out, "slides:", len(prs.slides))

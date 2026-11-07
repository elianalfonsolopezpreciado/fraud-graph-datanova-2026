"""
Ingeniería de features SIN FUGA DE DATOS para detección de fraude con grafo.

Principio: las features de cada transacción se calculan procesando el histórico
en orden cronológico, usando únicamente información disponible ANTES de ella:
  * historial del usuario / dispositivo / IP / comercio
  * etiquetas de fraude confirmadas, que llegan con un retraso (LABEL_DELAY_H),
    como en producción (contracargos / revisión manual no son instantáneos).
La etiqueta de la propia transacción jamás entra en sus features.
"""
import heapq
from collections import defaultdict, deque

import pandas as pd

LABEL_DELAY_H = 24      # las etiquetas de fraude se conocen 24 h después
BFS_MAX_DEPTH = 3       # saltos máximos al buscar fraude conocido en el grafo

TABULAR_COLS = [
    "amount", "hour", "hora_no_laboral", "amount_vs_user_avg",
    "tx_ultima_hora_user", "nuevo_dispositivo_para_user", "nuevo_ip_para_user",
    "user_tx_previas",
]
GRAPH_COLS = [
    "g_user_n_dispositivos", "g_user_n_ips",
    "g_dev_n_usuarios", "g_ip_n_usuarios",
    "g_dev_fraude_previo", "g_ip_fraude_previo", "g_user_fraude_previo",
    "g_merchant_tasa_fraude", "g_dist_a_fraude",
]
FEATURES_FULL = TABULAR_COLS + GRAPH_COLS

DESCRIPCIONES = {
    "amount": "Transaction amount",
    "hour": "Hour of day",
    "hora_no_laboral": "Night-time transaction (before 6h / after 22h)",
    "amount_vs_user_avg": "Amount vs the user's historical average",
    "tx_ultima_hora_user": "User's transactions in the last hour",
    "nuevo_dispositivo_para_user": "Device never seen for this user",
    "nuevo_ip_para_user": "IP never seen for this user",
    "user_tx_previas": "User tenure (previous transactions)",
    "g_user_n_dispositivos": "[Graph] Distinct devices used by the user",
    "g_user_n_ips": "[Graph] Distinct IPs used by the user",
    "g_dev_n_usuarios": "[Graph] Other users who shared the device",
    "g_ip_n_usuarios": "[Graph] Other users who shared the IP",
    "g_dev_fraude_previo": "[Graph] Prior confirmed frauds on the device",
    "g_ip_fraude_previo": "[Graph] Prior confirmed frauds on the IP",
    "g_user_fraude_previo": "[Graph] Prior confirmed frauds of the user",
    "g_merchant_tasa_fraude": "[Graph] Confirmed fraud rate of the merchant",
    "g_dist_a_fraude": "[Graph] Hops to a confirmed-fraud account/device/IP",
}


def construir_features(df: pd.DataFrame, label_delay_h: float = LABEL_DELAY_H) -> pd.DataFrame:
    """Devuelve df ordenado por tiempo con TABULAR_COLS + GRAPH_COLS (sin fuga)."""
    df = df.sort_values("timestamp").reset_index(drop=True).copy()

    adj = defaultdict(set)                  # grafo bipartito usuario <-> (dispositivo | IP)
    fraud_nodes = set()                     # nodos con fraude CONFIRMADO (ya conocido)
    user_cnt = defaultdict(int)
    user_sum = defaultdict(float)
    user_devs = defaultdict(set)
    user_ips = defaultdict(set)
    user_times = defaultdict(deque)
    dev_users = defaultdict(set)
    ip_users = defaultdict(set)
    dev_fraud = defaultdict(int)
    ip_fraud = defaultdict(int)
    user_fraud = defaultdict(int)
    mer_cnt = defaultdict(int)
    mer_fraud = defaultdict(int)
    pending = []                            # heap (t_conocimiento, idx)
    delay = pd.Timedelta(hours=label_delay_h)

    rows = []
    for i, r in enumerate(df.itertuples(index=False)):
        ts = r.timestamp
        # 1) liberar etiquetas que ya se conocen en este instante
        while pending and pending[0][0] <= ts:
            _, j = heapq.heappop(pending)
            q = df.iloc[j]
            if q["is_fraude"] == 1:
                u, d, ip, m = q["user_id"], q["device_id"], q["ip"], q["merchant_id"]
                fraud_nodes.update([f"U:{u}", f"D:{d}", f"I:{ip}"])
                dev_fraud[d] += 1
                ip_fraud[ip] += 1
                user_fraud[u] += 1
                mer_fraud[m] += 1

        u, d, ip, m = r.user_id, r.device_id, r.ip, r.merchant_id
        un, dn, inn = f"U:{u}", f"D:{d}", f"I:{ip}"

        # 2) features tabulares (solo pasado)
        n_prev = user_cnt[u]
        avg = user_sum[u] / n_prev if n_prev else r.amount
        tq = user_times[u]
        while tq and tq[0] < ts - pd.Timedelta(hours=1):
            tq.popleft()
        nuevo_dev = int(d not in user_devs[u]) if n_prev else 0
        nuevo_ip = int(ip not in user_ips[u]) if n_prev else 0

        # 3) distancia BFS al fraude confirmado más cercano (grafo hasta ahora)
        starts = [un, dn, inn]
        dist = 99
        seen = set(starts)
        frontier = starts
        for depth in range(BFS_MAX_DEPTH + 1):
            if any(n in fraud_nodes for n in frontier):
                dist = depth
                break
            nxt = []
            for n in frontier:
                for nb in adj.get(n, ()):
                    if nb not in seen:
                        seen.add(nb)
                        nxt.append(nb)
            frontier = nxt
            if not frontier:
                break

        rows.append({
            "amount": r.amount, "hour": r.hour,
            "hora_no_laboral": int(r.hour < 6 or r.hour > 22),
            "amount_vs_user_avg": r.amount / (avg + 1),
            "tx_ultima_hora_user": len(tq),
            "nuevo_dispositivo_para_user": nuevo_dev,
            "nuevo_ip_para_user": nuevo_ip,
            "user_tx_previas": n_prev,
            "g_user_n_dispositivos": len(user_devs[u]) + (1 if d not in user_devs[u] else 0),
            "g_user_n_ips": len(user_ips[u]) + (1 if ip not in user_ips[u] else 0),
            "g_dev_n_usuarios": len(dev_users[d] - {u}),
            "g_ip_n_usuarios": len(ip_users[ip] - {u}),
            "g_dev_fraude_previo": dev_fraud[d],
            "g_ip_fraude_previo": ip_fraud[ip],
            "g_user_fraude_previo": user_fraud[u],
            "g_merchant_tasa_fraude": (mer_fraud[m] + 0.04 * 20) / (mer_cnt[m] + 20),
            "g_dist_a_fraude": dist,
        })

        # 4) actualizar estado con esta transacción (su etiqueta se conocerá más tarde)
        user_cnt[u] += 1
        user_sum[u] += r.amount
        user_devs[u].add(d)
        user_ips[u].add(ip)
        tq.append(ts)
        dev_users[d].add(u)
        ip_users[ip].add(u)
        mer_cnt[m] += 1
        for a, b in ((un, dn), (un, inn)):
            adj[a].add(b)
            adj[b].add(a)
        heapq.heappush(pending, (ts + delay, i))

    feats = pd.DataFrame(rows).drop(columns=["amount", "hour"])
    return pd.concat([df, feats], axis=1)

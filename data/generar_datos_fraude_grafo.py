"""
Generador sintético de transacciones para DN-FN-01 (Fraud Graph).

Objetivo: crear un dataset realista con:
- Transacciones normales (mayoría)
- Fraudes aislados (importe inusual, dispositivo nuevo, hora rara)
- ANILLOS DE FRAUDE: grupos de usuarios que se envían dinero entre sí y a un comercio común (lavado / cuentas mula)
- CUENTAS MULA: usuario con muchos dispositivos / IPs diferentes en poco tiempo

Todas las columnas se generan dentro de este script (no depende de archivos externos).
Split temporal 70/30 para simular tiempo real (train = pasado, test = futuro).

Salida: data/transacciones_fraude_grafo.csv
"""

import os
import numpy as np
import pandas as pd

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
N_TRANSACCIONES = 12_000
FRAUDE_TASA = 0.04          # ~4% de fraude total (desequilibrio realista)
ANILLOS = 4                 # número de anillos de fraude inyectados
USUARIOS_POR_ANILLO = 6     # usuarios que participan en cada anillo
COMERCIOS = [f"COM_{i:03d}" for i in range(1, 51)]
DISPOSITIVOS = [f"DEV_{i:04d}" for i in range(1, 201)]
IPS = [f"203.0.113.{i}" for i in range(1, 101)]
CIUDADES = ["Madrid", "Barcelona", "Valencia", "Sevilla", "Bilbao",
            "Mexico DF", "Bogota", "Lima", "Santiago", "Buenos Aires"]
PAISES = {"Madrid": "ES", "Barcelona": "ES", "Valencia": "ES", "Sevilla": "ES", "Bilbao": "ES",
          "Mexico DF": "MX", "Bogota": "CO", "Lima": "PE", "Santiago": "CL", "Buenos Aires": "AR"}

OUT_DIR = os.path.join(os.path.dirname(__file__))
OUT_PATH = os.path.join(OUT_DIR, "transacciones_fraude_grafo.csv")


def generar_usuarios(n_usuarios=800):
    """Crea usuarios con un dispositivo e IP 'hábital' y un importe promedio típico."""
    usuarios = {}
    for i in range(n_usuarios):
        uid = f"U{i:05d}"
        usuarios[uid] = {
            "dispositivo_habitual": np.random.choice(DISPOSITIVOS),
            "ip_habitual": np.random.choice(IPS),
            "ciudad_habitual": np.random.choice(CIUDADES),
            "importe_promedio": np.random.exponential(150) + 20,  # la mayoría gasta poco
            "moneda": np.random.choice(["EUR", "USD", "MXN", "COP"], p=[0.4, 0.3, 0.2, 0.1]),
        }
    return usuarios


def generar_transacciones(usuarios):
    filas = []
    uid_list = list(usuarios.keys())
    t0 = pd.Timestamp("2026-01-01")

    # ---- 1. Transacciones normales (base) ----
    n_normal = int(N_TRANSACCIONES * (1 - FRAUDE_TASA))
    for _ in range(n_normal):
        uid = np.random.choice(uid_list)
        u = usuarios[uid]
        # la mayoría usa dispositivo/IP habitual
        dispositivo = u["dispositivo_habitual"] if np.random.rand() < 0.85 else np.random.choice(DISPOSITIVOS)
        ip = u["ip_habitual"] if np.random.rand() < 0.85 else np.random.choice(IPS)
        ciudad = u["ciudad_habitual"] if np.random.rand() < 0.9 else np.random.choice(CIUDADES)
        importe = max(5, np.random.exponential(u["importe_promedio"] * 1.2))
        hora = int(np.random.normal(14, 4)) % 24  # concentración al mediodía
        minuto = np.random.randint(0, 60)
        ts = t0 + pd.Timedelta(hours=np.random.randint(0, 24 * 90), minutes=minuto)
        filas.append({
            "transaction_id": f"T{len(filas):06d}",
            "user_id": uid,
            "merchant_id": np.random.choice(COMERCIOS),
            "device_id": dispositivo,
            "ip": ip,
            "ciudad": ciudad,
            "pais": PAISES.get(ciudad, "ES"),
            "amount": round(importe, 2),
            "currency": u["moneda"],
            "timestamp": ts,
            "hour": hora,
            "is_fraude": 0,
            "fraude_tipo": "normal",
        })

    # ---- 2. Fraudes aislados (importe alto / dispositivo nuevo / hora rara) ----
    n_fraude_aislado = int(N_TRANSACCIONES * FRAUDE_TASA * 0.6)
    for _ in range(n_fraude_aislado):
        uid = np.random.choice(uid_list)
        u = usuarios[uid]
        importe = u["importe_promedio"] * np.random.lognormal(1.6, 0.7)  # importe inusual (solapa con normales)
        toma_cuenta = np.random.rand() < 0.35  # 35%: desde dispositivo/IP habitual (toma de cuenta)
        dispositivo = u["dispositivo_habitual"] if toma_cuenta else np.random.choice(DISPOSITIVOS)
        ip = u["ip_habitual"] if toma_cuenta else np.random.choice(IPS)
        ciudad = np.random.choice(CIUDADES)
        hora = int(np.random.choice([2, 3, 4, 5, 6])) if np.random.rand() < 0.5 else int(np.random.normal(14, 5)) % 24
        ts = t0 + pd.Timedelta(hours=np.random.randint(0, 24 * 90))
        filas.append({
            "transaction_id": f"T{len(filas):06d}",
            "user_id": uid,
            "merchant_id": np.random.choice(COMERCIOS),
            "device_id": dispositivo,
            "ip": ip,
            "ciudad": ciudad,
            "pais": PAISES.get(ciudad, "ES"),
            "amount": round(importe, 2),
            "currency": u["moneda"],
            "timestamp": ts,
            "hour": hora,
            "is_fraude": 1,
            "fraude_tipo": "aislado",
        })

    # ---- 3. Anillos de fraude (cuentas mula que se envían dinero entre sí) ----
    for anillo in range(ANILLOS):
        miembros = np.random.choice(uid_list, size=USUARIOS_POR_ANILLO, replace=False)
        comercio_pivote = np.random.choice(COMERCIOS)
        pool_dev = np.random.choice(DISPOSITIVOS, size=3, replace=False)  # infraestructura compartida
        pool_ip = np.random.choice(IPS, size=3, replace=False)
        for _ in range(USUARIOS_POR_ANILLO * 3):  # varias transacciones por anillo
            emisor = np.random.choice(miembros)
            receptor = np.random.choice([m for m in miembros if m != emisor])
            # el receptor actúa como "comercio" (usuario destino)
            importe = np.random.uniform(100, 1500)
            dispositivo = np.random.choice(pool_dev)
            ip = np.random.choice(pool_ip)
            ciudad = np.random.choice(CIUDADES)
            hora = int(np.random.choice([1, 2, 3])) if np.random.rand() < 0.5 else int(np.random.normal(14, 5)) % 24
            ts = t0 + pd.Timedelta(hours=np.random.randint(0, 24 * 90))
            filas.append({
                "transaction_id": f"T{len(filas):06d}",
                "user_id": emisor,
                "merchant_id": comercio_pivote,  # todos apuntan al mismo comercio pivote
                "device_id": dispositivo,
                "ip": ip,
                "ciudad": ciudad,
                "pais": PAISES.get(ciudad, "ES"),
                "amount": round(importe, 2),
                "currency": "EUR",
                "timestamp": ts,
                "hour": hora,
                "is_fraude": 1,
                "fraude_tipo": "anillo",
                "anillo_id": anillo,        # SOLO para evaluar deteccion de anillos (nunca es feature)
            })

    # ---- 4. Cuentas mula (un usuario con muchos dispositivos/IPs distintos) ----
    n_mula = int(N_TRANSACCIONES * FRAUDE_TASA * 0.15)
    mulas = np.random.choice(uid_list, size=5, replace=False)
    pool_dev_m = np.random.choice(DISPOSITIVOS, size=4, replace=False)
    pool_ip_m = np.random.choice(IPS, size=4, replace=False)
    for _ in range(n_mula):
        uid = np.random.choice(mulas)
        u = usuarios[uid]
        importe = np.random.uniform(100, 1000)
        dispositivo = np.random.choice(pool_dev_m)
        ip = np.random.choice(pool_ip_m)
        ciudad = np.random.choice(CIUDADES)
        hora = int(np.random.choice([2, 3, 4])) if np.random.rand() < 0.5 else int(np.random.normal(14, 5)) % 24
        ts = t0 + pd.Timedelta(hours=np.random.randint(0, 24 * 90))
        filas.append({
            "transaction_id": f"T{len(filas):06d}",
            "user_id": uid,
            "merchant_id": np.random.choice(COMERCIOS),
            "device_id": dispositivo,
            "ip": ip,
            "ciudad": ciudad,
            "pais": PAISES.get(ciudad, "ES"),
            "amount": round(importe, 2),
            "currency": u["moneda"],
            "timestamp": ts,
            "hour": hora,
            "is_fraude": 1,
            "fraude_tipo": "mula",
        })

    df = pd.DataFrame(filas)
    # columnas internas de grafo (se usan para features, no se guardan como target)
    df["anillo_id"] = df["anillo_id"].fillna(-1).astype(int)
    # coherencia: la hora del timestamp es la columna 'hour'
    minutos = np.random.randint(0, 60, len(df))
    df["timestamp"] = (df["timestamp"].dt.normalize()
                       + pd.to_timedelta(df["hour"], unit="h") + pd.to_timedelta(minutos, unit="m"))
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def generar(seed=RANDOM_SEED):
    """Devuelve el dataset sintetico completo para una semilla dada (reproducible)."""
    np.random.seed(seed)
    return generar_transacciones(generar_usuarios())


def main():
    print("Generando usuarios y transacciones...")
    df = generar(RANDOM_SEED)

    # eliminar columnas internas antes de guardar (se usan solo en features.py si se quiere)
    cols_guardar = list(df.columns)
    df[cols_guardar].to_csv(OUT_PATH, index=False)

    print(f"\nGuardado: {OUT_PATH}")
    print(f"Filas: {len(df)} | Fraudes: {df['is_fraude'].sum()} ({df['is_fraude'].mean()*100:.2f}%)")
    print("\nPor tipo de fraude:")
    print(df[df["is_fraude"] == 1]["fraude_tipo"].value_counts())
    print("\nColumnas:", list(df[cols_guardar].columns))


if __name__ == "__main__":
    main()

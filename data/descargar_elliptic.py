"""Descarga el dataset real Elliptic (Bitcoin) desde data.pyg.org a data/elliptic/ (~153 MB comprimido).

Licencia del dataset: CC BY-NC-ND 4.0 (ver THIRD_PARTY_LICENSES.md). No se redistribuye en este repositorio.
Ejecutar manualmente:  python data/descargar_elliptic.py
"""
import os
import urllib.request
import zipfile


def main():
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "elliptic")
    os.makedirs(d, exist_ok=True)
    for n in ("elliptic_txs_features", "elliptic_txs_edgelist", "elliptic_txs_classes"):
        z = os.path.join(d, n + ".csv.zip")
        if not os.path.exists(os.path.join(d, n + ".csv")):
            urllib.request.urlretrieve(f"https://data.pyg.org/datasets/elliptic/{n}.csv.zip", z)
            zipfile.ZipFile(z).extractall(d)
            os.remove(z)
    print("listo:", os.listdir(d))


if __name__ == "__main__":
    main()

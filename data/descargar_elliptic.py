"""Descarga el dataset real Elliptic (Bitcoin) desde data.pyg.org a data/elliptic/ (~153 MB comprimido)."""
import os, urllib.request, zipfile
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "elliptic"); os.makedirs(D, exist_ok=True)
for n in ("elliptic_txs_features", "elliptic_txs_edgelist", "elliptic_txs_classes"):
    z = os.path.join(D, n + ".csv.zip")
    if not os.path.exists(os.path.join(D, n + ".csv")):
        urllib.request.urlretrieve(f"https://data.pyg.org/datasets/elliptic/{n}.csv.zip", z)
        zipfile.ZipFile(z).extractall(D); os.remove(z)
print("listo:", os.listdir(D))

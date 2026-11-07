"""Descarga el dataset real Elliptic (Bitcoin) desde data.pyg.org a data/elliptic/ (~153 MB comprimido).

Licencia del dataset: CC BY-NC-ND 4.0 (ver THIRD_PARTY_LICENSES.md). No se redistribuye en este repositorio.
Ejecutar manualmente:  python data/descargar_elliptic.py

Nota de despliegue: la app de Streamlit Community Cloud se creó con este archivo como "main file" y esa
configuración no se puede editar. Por eso, si lo ejecuta Streamlit, en lugar de descargar nada lanza el
dashboard (dashboard/app.py). Ejecutado desde la terminal con python, descarga el dataset.
"""
import os
import runpy
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    d = os.path.join(ROOT, "data", "elliptic")
    os.makedirs(d, exist_ok=True)
    for n in ("elliptic_txs_features", "elliptic_txs_edgelist", "elliptic_txs_classes"):
        z = os.path.join(d, n + ".csv.zip")
        if not os.path.exists(os.path.join(d, n + ".csv")):
            urllib.request.urlretrieve(f"https://data.pyg.org/datasets/elliptic/{n}.csv.zip", z)
            zipfile.ZipFile(z).extractall(d)
            os.remove(z)
    print("listo:", os.listdir(d))


def _bajo_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if _bajo_streamlit():
    runpy.run_path(os.path.join(ROOT, "dashboard", "app.py"), run_name="__main__")
elif __name__ == "__main__":
    main()

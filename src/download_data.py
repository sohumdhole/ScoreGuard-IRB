"""Download and verify raw datasets for ScoreGuard-IRB."""
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def download_uci_350():
    print("--- 1. Downloading UCI ID 350 (Default of Credit Card Clients) ---")
    zip_url = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
    zip_path = RAW_DIR / "default_of_credit_card_clients.zip"
    if not zip_path.exists():
        urllib.request.urlretrieve(zip_url, zip_path)
    print(f"Saved UCI zip to {zip_path} (SHA256: {sha256_file(zip_path)})")

    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()
        print("Zip contents:", names)
        xls_name = [n for n in names if n.endswith(".xls") or n.endswith(".xlsx")][0]
        extracted_path = RAW_DIR / xls_name
        if not extracted_path.exists():
            z.extract(xls_name, RAW_DIR)

    # Read the original XLS (header is on row 1 in 0-indexed, while row 0 has X1..X23, Y)
    df_xls_row0 = pd.read_excel(extracted_path, header=0, nrows=2)
    print("XLS Row 0 headers (X1..X23, Y):", list(df_xls_row0.columns))
    df_xls = pd.read_excel(extracted_path, header=1)
    print("XLS Row 1 real column names:", list(df_xls.columns))
    print("XLS shape:", df_xls.shape)

    # Also check ucimlrepo fetch_ucirepo(id=350) to see how it maps columns
    try:
        uci_ds = fetch_ucirepo(id=350)
        print("ucimlrepo features columns:", list(uci_ds.data.features.columns))
        print("ucimlrepo targets columns:", list(uci_ds.data.targets.columns))
        if uci_ds.variables is not None:
            print("ucimlrepo variables mapping:")
            print(uci_ds.variables[["name", "role", "type", "description"]].to_string())
    except Exception as e:
        print("ucimlrepo fetch notice:", e)

    csv_path = RAW_DIR / "uci_credit_card_default_raw.csv"
    df_xls.to_csv(csv_path, index=False)
    print(f"Saved canonical raw CSV with real column names to {csv_path} (SHA256: {sha256_file(csv_path)})")
    return df_xls


def download_cso_lrm01():
    print("\n--- 2. Downloading CSO Ireland LRM01 (Persons on Live Register) ---")
    cso_url = "https://ws.cso.ie/public/api.restful/PxStat.Data.Cube_API.ReadDataset/LRM01/CSV/1.0/en"
    cso_path = RAW_DIR / "cso_lrm01_raw.csv"
    if not cso_path.exists():
        req = urllib.request.Request(cso_url, headers={"User-Agent": "ScoreGuard-IRB/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        with open(cso_path, "wb") as f:
            f.write(data)
    print(f"Saved CSO LRM01 CSV to {cso_path} (SHA256: {sha256_file(cso_path)})")
    df_cso = pd.read_csv(cso_path)
    print("CSO LRM01 actual columns:", list(df_cso.columns))
    print("CSO LRM01 shape:", df_cso.shape)
    for col in df_cso.columns:
        if df_cso[col].nunique() < 20:
            print(f"  Unique values in '{col}':", df_cso[col].unique().tolist())
    return df_cso


if __name__ == "__main__":
    download_uci_350()
    download_cso_lrm01()

